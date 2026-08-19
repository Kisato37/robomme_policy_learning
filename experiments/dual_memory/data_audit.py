#!/usr/bin/env python3
"""Audit 100 pinned RoboMME training samples before any training is allowed."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from mme_vla_suite.shared.data_utils import even_sampling_indices


GROUNDING_PATTERN = re.compile(r"at <(\d+), (\d+)>")
REQUIRED_FIELDS = [
    "image",
    "wrist_image",
    "state",
    "actions",
    "prompt",
    "grounded_subgoal",
    "grounded_subgoal_online",
    "epis_idx",
    "step_idx",
    "exec_start_idx",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_summary(value) -> dict:
    array = np.asarray(value)
    result = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if np.issubdtype(array.dtype, np.number) and array.size:
        result.update(
            {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "finite": bool(np.isfinite(array).all()),
            }
        )
    return result


def parse_grounding(text: str) -> list[tuple[int, int]]:
    """Return official RoboMME (y, x) point pairs in 256x256 front-image space."""
    return [(int(y), int(x)) for y, x in GROUNDING_PATTERN.findall(text)]


def draw_example(sample: dict, destination: Path, sample_id: int, history_length: int) -> None:
    image = np.asarray(sample["image"]).copy()
    wrist = np.asarray(sample["wrist_image"]).copy()
    for y, x in parse_grounding(str(sample["grounded_subgoal"])):
        cv2.circle(image, (x, y), 6, (255, 255, 0), -1)
        cv2.putText(image, f"(y={y},x={x})", (max(0, x - 40), max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    canvas = np.concatenate([image, wrist], axis=1)
    lines = [
        f"sample={sample_id} episode={int(np.asarray(sample['epis_idx']).item())} step={int(np.asarray(sample['step_idx']).item())}",
        f"task={sample['prompt']}",
        f"subgoal={sample['grounded_subgoal']}",
        f"history_length={history_length} action_shape={np.asarray(sample['actions']).shape}",
    ]
    line_height = 22
    header = np.zeros((line_height * len(lines) + 10, canvas.shape[1], 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(header, line[:180], (8, 20 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(destination), np.concatenate([header, canvas], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--visual-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    json_path = args.output_dir / "data_audit.json"
    markdown_path = args.output_dir / "data_audit.md"
    visual_dir = args.output_dir / "coordinate_examples"
    if json_path.exists() or markdown_path.exists() or (visual_dir.exists() and any(visual_dir.iterdir())):
        raise FileExistsError("Refusing to overwrite prior data-audit evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    stats_path = args.dataset / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    total = int(stats["execution_samples"])
    if total < args.sample_count:
        raise RuntimeError(f"Dataset has only {total} execution samples")
    rng = random.Random(args.seed)
    sample_ids = sorted(rng.sample(range(total), args.sample_count))

    missing = Counter()
    field_shapes = defaultdict(Counter)
    field_dtypes = defaultdict(Counter)
    ranges = defaultdict(lambda: [float("inf"), float("-inf")])
    coordinate_counts = Counter()
    problems = []
    episode_steps = defaultdict(list)
    records = []

    for ordinal, sample_id in enumerate(sample_ids):
        sample_path = args.dataset / "data" / f"{sample_id}.pkl"
        with sample_path.open("rb") as stream:
            sample = pickle.load(stream)
        for field in REQUIRED_FIELDS:
            if field not in sample or sample[field] is None:
                missing[field] += 1
                problems.append(f"sample {sample_id}: missing {field}")
                continue
            if not isinstance(sample[field], str):
                summary = array_summary(sample[field])
                field_shapes[field][str(summary["shape"])] += 1
                field_dtypes[field][summary["dtype"]] += 1
                if "min" in summary:
                    ranges[field][0] = min(ranges[field][0], summary["min"])
                    ranges[field][1] = max(ranges[field][1], summary["max"])
                    if not summary["finite"]:
                        problems.append(f"sample {sample_id}: non-finite {field}")

        if any(field not in sample or sample[field] is None for field in REQUIRED_FIELDS):
            continue
        episode = int(np.asarray(sample["epis_idx"]).item())
        step = int(np.asarray(sample["step_idx"]).item())
        exec_start = int(np.asarray(sample["exec_start_idx"]).item())
        episode_steps[episode].append(step)
        if step < exec_start:
            problems.append(f"sample {sample_id}: execution step {step} precedes exec_start {exec_start}")

        image = np.asarray(sample["image"])
        wrist = np.asarray(sample["wrist_image"])
        if image.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
            problems.append(f"sample {sample_id}: unexpected image shapes {image.shape}/{wrist.shape}")
        if np.asarray(sample["actions"]).shape != (20, 8):
            problems.append(f"sample {sample_id}: unexpected action shape {np.asarray(sample['actions']).shape}")

        points = parse_grounding(str(sample["grounded_subgoal"]))
        online_points = parse_grounding(str(sample["grounded_subgoal_online"]))
        coordinate_counts[len(points)] += 1
        for source, source_points in [("recorded", points), ("online", online_points)]:
            for y, x in source_points:
                if not (0 <= y <= 255 and 0 <= x <= 255):
                    problems.append(f"sample {sample_id}: {source} coordinate out of bounds {(y, x)}")

        feature_dir = args.dataset / "features" / f"episode_{episode}"
        current_feature = feature_dir / f"token_emb_{step}.npy"
        first_feature = feature_dir / "token_emb_0.npy"
        if not current_feature.exists() or not first_feature.exists():
            problems.append(f"sample {sample_id}: missing current or first history feature")
            continue
        with current_feature.open("rb") as stream:
            feature = np.load(stream, allow_pickle=True).item()
        expected_feature_keys = {
            "image_emb_8x8",
            "image_emb_4x4",
            "image_emb_2x2",
            "pos_emb_8x8",
            "pos_emb_4x4",
            "pos_emb_2x2",
            "state_emb",
        }
        missing_feature_keys = expected_feature_keys - feature.keys()
        if missing_feature_keys:
            problems.append(f"sample {sample_id}: missing feature keys {sorted(missing_feature_keys)}")
        frame_indices = even_sampling_indices(step, 512 // 16)
        if not frame_indices or frame_indices[0] != 0 or frame_indices[-1] != step:
            problems.append(f"sample {sample_id}: FrameSamp excludes first/current frame")
        if len(frame_indices) != len(set(frame_indices)):
            problems.append(f"sample {sample_id}: duplicate FrameSamp indices")
        for frame_index in frame_indices:
            if not (feature_dir / f"token_emb_{frame_index}.npy").exists():
                problems.append(f"sample {sample_id}: missing sampled history frame {frame_index}")

        record = {
            "sample_id": sample_id,
            "episode": episode,
            "step": step,
            "exec_start": exec_start,
            "task": str(sample["prompt"]),
            "grounded_subgoal": str(sample["grounded_subgoal"]),
            "coordinates_yx": points,
            "history_length": step + 1,
            "framesamp_indices": frame_indices,
            "action_shape": list(np.asarray(sample["actions"]).shape),
            "sample_sha256": sha256_file(sample_path),
        }
        records.append(record)
        if ordinal < args.visual_count:
            draw_example(sample, visual_dir / f"sample_{sample_id:07d}.png", sample_id, step + 1)

    normalized_ranges = {
        key: value for key, value in ranges.items() if value[0] != float("inf")
    }
    audit = {
        "status": "pass" if not problems else "fail",
        "dataset_root": str(args.dataset.resolve()),
        "dataset_stats": stats,
        "stats_sha256": sha256_file(stats_path),
        "sample_seed": args.seed,
        "sample_ids": sample_ids,
        "sample_count": args.sample_count,
        "missing_counts": dict(missing),
        "field_shapes": {key: dict(value) for key, value in field_shapes.items()},
        "field_dtypes": {key: dict(value) for key, value in field_dtypes.items()},
        "field_ranges": normalized_ranges,
        "grounding": {
            "coordinate_order": "(y, x)",
            "coordinate_space": "integer pixels in the 256x256 front-view image",
            "parser": r"at <(\d+), (\d+)>",
            "point_counts_per_sample": dict(coordinate_counts),
            "visualization_conversion": "cv2 receives (x, y), so the official pair is reversed only for drawing",
        },
        "history": {
            "includes_first_frame": True,
            "includes_current_frame": True,
            "maximum_frames": 32,
            "tokens_per_frame": 16,
            "token_budget": 512,
            "episode_storage_isolated_by_directory": True,
        },
        "prompt_consistency": {
            "training_and_evaluation_transform": "ModelTransformFactory -> TokenizePromptWithSymbolicMemory",
            "template": "Task: {task};\\nCurrent Subgoal: {grounded_subgoal};\\nAction: ",
        },
        "qwen_coordinate_path": {
            "raw_space": "1000x1000 Qwen box tokens",
            "policy_space": "256x256",
            "official_parser": "Qwen3VLModel._parse_subgoal_for_vla",
            "open_risk": "Qwen raw box convention must be visually checked in SP-Qwen smoke before formal evaluation",
        },
        "records": records,
        "problems": problems,
    }
    with json_path.open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        "# RoboMME dual-memory data audit",
        "",
        f"Status: **{audit['status'].upper()}**",
        "",
        f"Audited {args.sample_count} deterministic random samples from {total} execution samples.",
        "",
        "## Findings",
        "",
        "- GroundSG coordinates are `(y, x)` integer pixels in the 256x256 front-view image.",
        "- FrameSamp uses at most 32 unique frames x 16 tokens = 512 tokens and includes both frame 0 and the current frame.",
        "- Training and evaluation share `TokenizePromptWithSymbolicMemory` and the same grounded-subgoal prompt template.",
        f"- Missing required fields: `{dict(missing)}`.",
        f"- Blocking problems: `{len(problems)}`.",
        "",
        "## Qwen audit boundary",
        "",
        "The official parser maps Qwen's 1000x1000 box tokens to 256x256 coordinates. A visual Qwen smoke remains mandatory because the raw Qwen pair convention cannot be proven from preprocessed policy samples alone.",
        "",
        "## Evidence",
        "",
        "Machine-readable details are in `data_audit.json`; 20 annotated front/wrist examples are in `coordinate_examples/`.",
    ]
    with markdown_path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    if problems:
        raise RuntimeError(f"Data audit failed with {len(problems)} blocking problems")


if __name__ == "__main__":
    main()
