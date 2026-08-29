#!/usr/bin/env python3
"""Real-checkpoint GPU architecture gate for causal memory selection.

The gate deliberately uses synthetic histories.  It exercises the released
policy and memory path without opening any benchmark split or scientific
outcome.  A report is published exactly once beneath an already prepared smoke
run root.
"""

# Lazy heavyweight imports keep --dry-run CPU-only. Private policy state is
# intentionally inspected here because clearing it is part of this gate.
# ruff: noqa: PLC0415, SLF001

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

from experiments.keyframe_oracle_sampling.artifacts import (
    SMOKE_ACTION_DTYPE,
    released_prepared_component_dtypes,
    validate_architecture_pass_report,
)
from experiments.keyframe_oracle_sampling.environment_contract import (
    validate_environment_contract,
)
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
    derive_smoke_random_seed,
    select_indices,
)

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_RELATIVE = Path(
    "runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"
)
CHECKPOINT_ARCHIVE_SHA256 = (
    "2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62"
)
EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "313e483ae32881e606402365a8b01a599eda22a367e436f9dfbd5456120dca26"
)
CHECKPOINT_CONTENT_TREE_ALGORITHM = "sha256-canonical-file-content-tree-v1"
POLICY_SEED = 7
TASK = "InsertPeg"
EPISODE_ID = 0
ARMS = ("U", "O", "OC", "R")
HISTORY_LENGTHS = (16, 64)
CHUNK_SIZE = 16
EXPECTED_MEMORY_TOKENS = 512
TOKENS_PER_FRAME = 16
EXPECTED_MEMORY_TOKEN_DIM = 1024
EXPECTED_COMPONENT_SHAPES = (
    (EXPECTED_MEMORY_TOKENS, 2048),
    (EXPECTED_MEMORY_TOKENS, 768),
    (EXPECTED_MEMORY_TOKENS, 8),
    (EXPECTED_MEMORY_TOKENS,),
)
EXPECTED_FINAL_MEMORY_DTYPE = "bfloat16"
EXPECTED_ACTION_SHAPE = (20, 8)
EXPECTED_ACTION_DTYPE = SMOKE_ACTION_DTYPE
REPORT_REQUIRED_FIELDS = (
    "passed",
    "repository_commit_sha",
    "checkpoint_unpacked_metadata_sha256",
    "checkpoint_content_tree_algorithm",
    "checkpoint_unpacked_content_tree_sha256",
    "run_root",
    "slurm_job_id",
    "case_count",
    "cases",
    "final_reset_evidence",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    import jax
    import numpy as np

    value_dtype = getattr(value, "dtype", None)
    if value_dtype is not None and jax.dtypes.issubdtype(
        value_dtype, jax.dtypes.prng_key
    ):
        # Typed PRNG keys intentionally reject direct NumPy conversion.
        value = jax.random.key_data(value)
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _action_contract_checks(value: Any) -> dict[str, bool]:
    """Return absolute frozen action-contract checks, never relative baselines."""
    import numpy as np
    import jax.numpy as jnp

    actions = np.asarray(value)
    return {
        "action_shape_is_frozen_20x8": actions.shape == EXPECTED_ACTION_SHAPE,
        "action_dtype_matches_frozen_released_contract": str(actions.dtype)
        == EXPECTED_ACTION_DTYPE,
        "action_dtype_is_floating": bool(
            jnp.issubdtype(actions.dtype, jnp.floating)
        ),
        "action_values_are_finite": bool(np.isfinite(actions).all()),
    }


def _checkpoint_metadata_digest(checkpoint_dir: Path) -> str:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint_dir}")
    entries = [
        {"path": str(path.relative_to(checkpoint_dir)), "size": path.stat().st_size}
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file()
    ]
    if not entries:
        raise RuntimeError(f"Frozen checkpoint directory is empty: {checkpoint_dir}")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_content_tree_identity(checkpoint_dir: Path) -> dict[str, int | str]:
    # Import the shared run-level implementation lazily so --dry-run remains
    # lightweight and the exact canonicalization cannot drift between gates.
    from experiments.keyframe_oracle_sampling.prepare_smoke import (
        checkpoint_content_tree_identity,
    )

    return checkpoint_content_tree_identity(checkpoint_dir)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite architecture report: {path}")
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _dry_contract_reset() -> dict[str, Any]:
    return {
        "history_frame_count": 0,
        "history_metadata_count": 0,
        "step_idx": -1,
        "selector_configuration_cleared": True,
        "selector_call_index": 0,
        "selector_rng_cleared": True,
        "pending_trace_cleared": True,
        "temporary_override_cleared": True,
        "policy_rng_reset": True,
        "passed": True,
    }


def _dry_contract_digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _dry_contract_case(arm: str, history_length: int) -> dict[str, Any]:
    boundaries = list(range(0, history_length, 7))
    seed = (
        derive_smoke_random_seed(TASK, EPISODE_ID, 0) if arm == "R" else None
    )
    selected = select_indices(
        arm,
        history_length - 1,
        boundary_indices=boundaries,
        random_seed=seed,
    )
    selected_digest = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    expected_component_dtypes = released_prepared_component_dtypes(len(selected))
    checks = {
        "history_accumulated_exactly": True,
        "memory_has_four_components": True,
        "component_shapes_match_frozen_released_contract": True,
        "component_dtypes_match_frozen_released_contract": True,
        "all_components_have_512_slots": True,
        "mask_is_boolean": True,
        "valid_token_count_matches": True,
        "valid_mask_is_prefix": True,
        "padding_mask_is_false": True,
        "trace_frame_count_matches": True,
        "trace_token_count_matches": True,
        "single_policy_call": True,
        "temporary_override_restored": True,
        "final_memory_shape_is_512x1024": True,
        "final_memory_dtype_is_floating": True,
        "final_memory_dtype_matches_frozen_released_contract": True,
        "final_memory_values_are_finite": True,
        "action_shape_is_frozen_20x8": True,
        "action_dtype_matches_frozen_released_contract": True,
        "action_dtype_is_floating": True,
        "action_values_are_finite": True,
    }
    if arm == "U":
        checks["literal_uniform_indices_match"] = True
    memory_digest = _dry_contract_digest(f"{arm}-{history_length}-memory")
    evidence = {
        "reset_before_run": _dry_contract_reset(),
        "selected_frame_indices": selected,
        "selected_indices_sha256": selected_digest,
        "visible_boundary_indices": boundaries,
        "valid_frame_count": len(selected),
        "valid_memory_token_count": 16 * len(selected),
        "padding_frame_count": 32 - len(selected),
        "component_shapes": [list(shape) for shape in EXPECTED_COMPONENT_SHAPES],
        "component_dtypes": list(expected_component_dtypes),
        "final_memory_tensor_shape": [1, 512, EXPECTED_MEMORY_TOKEN_DIM],
        "final_memory_tensor_dtype": EXPECTED_FINAL_MEMORY_DTYPE,
        "final_memory_tensor_finite": True,
        "final_memory_tensor_sha256": memory_digest,
        "prepared_memory_components_sha256": _dry_contract_digest(
            f"{arm}-{history_length}-components"
        ),
        "mask_sha256": _dry_contract_digest(f"{arm}-{history_length}-mask"),
        "action_shape": list(EXPECTED_ACTION_SHAPE),
        "action_dtype": EXPECTED_ACTION_DTYPE,
        "action_finite": True,
        "action_sha256": _dry_contract_digest(f"{arm}-{history_length}-action"),
        "compile_cache": {
            "vision_before": 1,
            "vision_after": 1,
            "memory_before": 1,
            "memory_after": 1,
            "sample_before": 1,
            "sample_after": 1,
        },
        "checks": checks,
    }
    return {
        "arm": arm,
        "history_length": history_length,
        "first": dict(evidence),
        "repeat": dict(evidence),
        "same_selected_indices": True,
        "same_memory_tensor_digest": True,
        "same_live_process_action_digest_audit": True,
        "released_shape_match": True,
        "released_dtype_match": True,
        "released_action_shape_match": True,
        "released_action_dtype_match": True,
        "compile_cache_stable_after_first_inference": True,
        "passed": True,
    }


def _dry_run_contract_example(run_root: Path) -> dict[str, Any]:
    return {
        "passed": True,
        "repository_commit_sha": "0" * 40,
        "checkpoint_unpacked_metadata_sha256": "0" * 64,
        "checkpoint_content_tree_algorithm": CHECKPOINT_CONTENT_TREE_ALGORITHM,
        "checkpoint_unpacked_content_tree_sha256": "0" * 64,
        "run_root": str(run_root),
        "slurm_job_id": "dry-run-job",
        "case_count": 8,
        "cases": [
            _dry_contract_case(arm, history_length)
            for arm in ARMS
            for history_length in HISTORY_LENGTHS
        ],
        "reference_component_shapes": [
            list(shape) for shape in EXPECTED_COMPONENT_SHAPES
        ],
        "reference_component_dtypes_by_padding": {
            "padded": list(released_prepared_component_dtypes(1)),
            "unpadded": list(released_prepared_component_dtypes(32)),
        },
        "reference_final_memory_dtype": EXPECTED_FINAL_MEMORY_DTYPE,
        "reference_action_shape": list(EXPECTED_ACTION_SHAPE),
        "reference_action_dtype": EXPECTED_ACTION_DTYPE,
        "stable_compile_cache": {
            "vision": 1,
            "perceptual_memory": 1,
            "sample_actions": 1,
        },
        "final_reset_evidence": _dry_contract_reset(),
    }


def _dry_run_payload() -> dict[str, Any]:
    cases = [
        {"arm": arm, "history_length": history_length}
        for arm in ARMS
        for history_length in HISTORY_LENGTHS
    ]
    if len(cases) != 8 or {case["arm"] for case in cases} != set(ARMS):
        raise AssertionError("Architecture case matrix is invalid")
    if not any(case["history_length"] < 32 for case in cases):
        raise AssertionError("Architecture matrix lacks a below-budget history")
    if not any(case["history_length"] > 32 for case in cases):
        raise AssertionError("Architecture matrix lacks an overflow history")
    dry_run_root = REPO / "runs" / "keyframe_oracle_sampling" / "dry-run-contract"
    contract_example = _dry_run_contract_example(dry_run_root)
    _validate_pass_report_contract(
        contract_example,
        run_root=dry_run_root,
        architecture_submission={
            "run_root": str(dry_run_root),
            "slurm_job_id": "dry-run-job",
        },
    )
    return {
        "valid": True,
        "report_contract_valid": True,
        "report_required_fields": list(REPORT_REQUIRED_FIELDS),
        "submits_jobs": False,
        "runs_gpu_inference": False,
        "checkpoint": str(CHECKPOINT_RELATIVE),
        "case_count": len(cases),
        "cases": cases,
    }


def _validate_pass_report_contract(
    report: dict[str, Any],
    *,
    run_root: Path,
    architecture_submission: dict[str, Any],
) -> None:
    missing = [field for field in REPORT_REQUIRED_FIELDS if field not in report]
    if missing:
        raise RuntimeError(f"Architecture PASS report lacks required fields: {missing}")
    if report["passed"] is not True:
        raise RuntimeError("Architecture PASS report must record passed=true")
    commit = str(report["repository_commit_sha"])
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("Architecture report has an invalid repository commit SHA")
    checkpoint_digest = str(report["checkpoint_unpacked_metadata_sha256"])
    if len(checkpoint_digest) != 64 or any(
        value not in "0123456789abcdef" for value in checkpoint_digest
    ):
        raise RuntimeError("Architecture report has an invalid checkpoint metadata digest")
    content_tree_digest = str(report["checkpoint_unpacked_content_tree_sha256"])
    if report["checkpoint_content_tree_algorithm"] != CHECKPOINT_CONTENT_TREE_ALGORITHM:
        raise RuntimeError("Architecture report has the wrong content-tree algorithm")
    if len(content_tree_digest) != 64 or any(
        value not in "0123456789abcdef" for value in content_tree_digest
    ):
        raise RuntimeError(
            "Architecture report has an invalid checkpoint content-tree digest"
        )
    validate_architecture_pass_report(
        report,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )


def _cache_size(function: Any) -> int:
    """Return the cache size of the inner jax.jit built by module_jit."""
    nonlocals = inspect.getclosurevars(function).nonlocals
    jitted = nonlocals.get("jitted_fn")
    if jitted is None or not hasattr(jitted, "_cache_size"):
        raise RuntimeError("Cannot audit the module_jit compilation cache")
    return int(jitted._cache_size())


def _reset_snapshot(policy: Any) -> dict[str, Any]:
    import jax

    expected_rng = jax.random.key(POLICY_SEED)
    observed_rng = policy._rng
    snapshot = {
        "history_frame_count": len(policy.mem_buffer._history_feats),
        "history_metadata_count": len(policy.mem_buffer._history_metadata),
        "step_idx": int(policy.step_idx),
        "selector_configuration_cleared": policy._keyframe_selector_config is None,
        "selector_call_index": int(policy._selector_call_index),
        "selector_rng_cleared": policy._selector_rng is None,
        "pending_trace_cleared": policy._pending_selector_trace is None,
        "temporary_override_cleared": policy.mem_buffer._frame_sampling_selector is None,
        "policy_rng_reset": _array_digest(observed_rng) == _array_digest(expected_rng),
    }
    snapshot["passed"] = all(
        (
            snapshot["history_frame_count"] == 0,
            snapshot["history_metadata_count"] == 0,
            snapshot["step_idx"] == -1,
            snapshot["selector_configuration_cleared"],
            snapshot["selector_call_index"] == 0,
            snapshot["selector_rng_cleared"],
            snapshot["pending_trace_cleared"],
            snapshot["temporary_override_cleared"],
            snapshot["policy_rng_reset"],
        )
    )
    return snapshot


def _synthetic_history(history_length: int) -> tuple[Any, Any, Any]:
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(2026082501 + history_length))
    images = rng.integers(
        0,
        256,
        size=(history_length, 1, 256, 256, 3),
        dtype=np.uint8,
    )
    phase = np.linspace(-0.25, 0.25, history_length, dtype=np.float32)[:, None]
    offsets = np.linspace(-0.04, 0.04, 8, dtype=np.float32)[None, :]
    states = (phase + offsets).astype(np.float32)
    stages = (np.arange(history_length, dtype=np.int64) // 7).astype(np.int64)
    return images, states, stages


def _expected_uniform(history_length: int) -> list[int]:
    import numpy as np

    if history_length <= 32:
        return list(range(history_length))
    return np.linspace(0, history_length - 1, 32, dtype=np.int32).tolist()


def _expected_boundary_count(stages: Any) -> int:
    import numpy as np

    stages = np.asarray(stages)
    return int(1 + np.count_nonzero(stages[1:] != stages[:-1]))


def _load_seed_configuration(run_root: Path) -> tuple[tuple[int, ...], dict[str, str]]:
    from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
    from experiments.keyframe_oracle_sampling.artifacts import (
        validate_smoke_formal_seed_disjointness,
    )

    seed_path = run_root / "protocol" / "seed_table.json"
    payload, lookup = load_seed_table(
        seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_payload, _ = load_seed_table(
        run_root / "protocol" / "formal_seed_audit_table.json",
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    validate_smoke_formal_seed_disjointness(payload, formal_payload)
    seeds = tuple(lookup[(TASK, EPISODE_ID, call_index)] for call_index in range(82))
    return seeds, {
        "seed_table_sha256": str(payload["entries_sha256"]),
        "seed_table_scope": str(payload["scope"]),
        "seed_table_dataset": str(payload["dataset"]),
    }


def _capture_memory_prepare(policy: Any) -> dict[str, Any]:
    import numpy as np

    capture: dict[str, Any] = {}
    original = policy._prepare_experiment_frame_sampling

    def audited_prepare(self, *args, **kwargs):
        prepared = original(*args, **kwargs)
        arrays = tuple(np.asarray(value) for value in prepared)
        capture["arrays"] = arrays
        capture["shapes"] = [list(value.shape) for value in arrays]
        capture["dtypes"] = [str(value.dtype) for value in arrays]
        return prepared

    policy._prepare_experiment_frame_sampling = types.MethodType(audited_prepare, policy)
    return capture


def _run_once(
    policy: Any,
    capture: dict[str, Any],
    *,
    arm: str,
    history_length: int,
    seeds: tuple[int, ...],
    seed_table_contract: dict[str, str],
) -> dict[str, Any]:
    import numpy as np

    policy.reset()
    reset = _reset_snapshot(policy)
    if not reset["passed"]:
        raise RuntimeError(f"Policy reset isolation failed before {arm}/{history_length}")
    policy.configure_keyframe_selector(
        {
            "arm": arm,
            "task": TASK,
            "episode_id": EPISODE_ID,
            "random_seeds": seeds,
            **seed_table_contract,
        }
    )
    images, states, stages = _synthetic_history(history_length)
    vision_cache_before = _cache_size(policy._vision_encode)
    for start in range(0, history_length, CHUNK_SIZE):
        stop = min(history_length, start + CHUNK_SIZE)
        policy.add_buffer(
            {
                "images": images[start:stop],
                "state": states[start:stop],
                "current_task_index": stages[start:stop],
                "exec_start_idx": 0,
            }
        )
    vision_cache_after = _cache_size(policy._vision_encode)
    memory_cache_before = _cache_size(policy._perceptual_memory_encode)
    sample_cache_before = _cache_size(policy._sample_actions)
    front = images[-1, 0]
    wrist = np.flip(front, axis=1).copy()
    output = policy.infer(
        {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": states[-1],
            "prompt": "insert the peg into the hole",
            "keyframe_environment_step": history_length - 1,
        }
    )
    memory_cache_after = _cache_size(policy._perceptual_memory_encode)
    sample_cache_after = _cache_size(policy._sample_actions)
    trace = dict(output["selector_trace"])
    arrays = capture.pop("arrays")
    shapes = capture.pop("shapes")
    dtypes = capture.pop("dtypes")
    mask = arrays[3]
    expected_frames = (
        min(32, _expected_boundary_count(stages))
        if arm == "O"
        else min(32, history_length)
    )
    expected_tokens = TOKENS_PER_FRAME * expected_frames
    expected_component_dtypes = released_prepared_component_dtypes(expected_frames)
    actions = np.asarray(output["actions"])
    final_memory_shape = trace["final_memory_tensor_shape"]
    final_memory_dtype = str(trace["final_memory_tensor_dtype"])
    checks = {
        "history_accumulated_exactly": policy.step_idx == history_length - 1,
        "memory_has_four_components": len(arrays) == 4,
        "component_shapes_match_frozen_released_contract": tuple(
            value.shape for value in arrays
        )
        == EXPECTED_COMPONENT_SHAPES,
        "component_dtypes_match_frozen_released_contract": tuple(
            str(value.dtype) for value in arrays
        )
        == expected_component_dtypes,
        "all_components_have_512_slots": all(
            value.shape[0] == EXPECTED_MEMORY_TOKENS for value in arrays
        ),
        "mask_is_boolean": mask.dtype == np.bool_,
        "valid_token_count_matches": int(mask.sum()) == expected_tokens,
        "valid_mask_is_prefix": bool(mask[:expected_tokens].all()),
        "padding_mask_is_false": bool((~mask[expected_tokens:]).all()),
        "trace_frame_count_matches": int(trace["valid_frame_count"]) == expected_frames,
        "trace_token_count_matches": int(trace["valid_memory_token_count"])
        == expected_tokens,
        "single_policy_call": policy._selector_call_index == 1,
        "temporary_override_restored": policy.mem_buffer._frame_sampling_selector is None,
        "final_memory_shape_is_512x1024": final_memory_shape
        == [1, EXPECTED_MEMORY_TOKENS, EXPECTED_MEMORY_TOKEN_DIM],
        "final_memory_dtype_is_floating": trace[
            "final_memory_tensor_is_floating"
        ]
        is True,
        "final_memory_dtype_matches_frozen_released_contract": final_memory_dtype
        == EXPECTED_FINAL_MEMORY_DTYPE,
        "final_memory_values_are_finite": trace["final_memory_tensor_finite"] is True,
        **_action_contract_checks(actions),
    }
    if arm == "U":
        checks["literal_uniform_indices_match"] = (
            trace["selected_frame_indices"] == _expected_uniform(history_length)
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Architecture checks failed for {arm}/{history_length}: {failed}")
    return {
        "reset_before_run": reset,
        "selected_frame_indices": trace["selected_frame_indices"],
        "selected_indices_sha256": trace["selected_indices_sha256"],
        "visible_boundary_indices": trace["visible_boundary_indices"],
        "valid_frame_count": int(trace["valid_frame_count"]),
        "valid_memory_token_count": int(trace["valid_memory_token_count"]),
        "padding_frame_count": int(trace["padding_frame_count"]),
        "component_shapes": shapes,
        "component_dtypes": dtypes,
        "final_memory_tensor_shape": final_memory_shape,
        "final_memory_tensor_dtype": final_memory_dtype,
        "final_memory_tensor_finite": trace["final_memory_tensor_finite"],
        "final_memory_tensor_sha256": trace["final_memory_tensor_sha256"],
        "prepared_memory_components_sha256": trace[
            "prepared_memory_components_sha256"
        ],
        "mask_sha256": trace["mask_sha256"],
        "action_shape": list(actions.shape),
        "action_dtype": str(actions.dtype),
        "action_finite": bool(np.isfinite(actions).all()),
        "action_sha256": _array_digest(actions),
        "compile_cache": {
            "vision_before": vision_cache_before,
            "vision_after": vision_cache_after,
            "memory_before": memory_cache_before,
            "memory_after": memory_cache_after,
            "sample_before": sample_cache_before,
            "sample_after": sample_cache_after,
        },
        "checks": checks,
    }


def _run_architecture(report: dict[str, Any], run_root: Path) -> None:
    import jax

    from mme_vla_suite.policies import policy_config
    import mme_vla_suite.training.config as training_config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"GPU architecture smoke requires a GPU, got {jax.devices()}")
    launch_manifest_path = run_root / "protocol" / "launch_manifest.json"
    submission_path = run_root / "protocol" / "architecture_submission_record.json"
    launch_manifest = json.loads(launch_manifest_path.read_text())
    architecture_submission = json.loads(submission_path.read_text())
    validate_environment_contract(launch_manifest)
    repository_sha = _git_output("rev-parse", "HEAD")
    # The prepared run root and the separately identity-checked frozen
    # checkpoint are intentionally untracked artifacts inside the repository.
    # Exclude only those namespaces while failing on every source/config change.
    dirty_status = _git_output(
        "status",
        "--porcelain=v1",
        "--",
        ".",
        ":(exclude)runs/keyframe_oracle_sampling",
        f":(exclude){CHECKPOINT_RELATIVE.as_posix()}",
    )
    if dirty_status:
        raise RuntimeError("GPU architecture smoke requires a clean committed worktree")
    if launch_manifest["repository"]["commit_sha"] != repository_sha:
        raise RuntimeError("Prepared run root repository SHA differs from the executing checkout")
    if launch_manifest["checkpoint_path"] != str(CHECKPOINT_RELATIVE):
        raise RuntimeError("Prepared run root does not name the frozen protocol checkpoint")
    if (
        launch_manifest["checkpoint_archive_sha256_expected"]
        != CHECKPOINT_ARCHIVE_SHA256
    ):
        raise RuntimeError("Prepared run root has the wrong checkpoint archive SHA-256")
    if (
        launch_manifest.get("checkpoint_archive_sha256_actual")
        != CHECKPOINT_ARCHIVE_SHA256
    ):
        raise RuntimeError("Prepared run root did not verify the actual checkpoint archive")
    seed_table_path = run_root / "protocol" / "seed_table.json"
    if _sha256_file(seed_table_path) != launch_manifest["seed_table_file_sha256"]:
        raise RuntimeError("Prepared seed-table file digest mismatch")
    formal_seed_audit_path = run_root / "protocol" / "formal_seed_audit_table.json"
    if (
        _sha256_file(formal_seed_audit_path)
        != launch_manifest["formal_seed_audit_file_sha256"]
    ):
        raise RuntimeError("Prepared formal seed-audit table digest mismatch")
    if architecture_submission.get("repository_commit_sha") != repository_sha:
        raise RuntimeError("Architecture submission record used a different commit")
    if (
        architecture_submission.get("checkpoint_content_tree_algorithm")
        != launch_manifest.get("checkpoint_content_tree_algorithm")
        or architecture_submission.get("checkpoint_unpacked_content_tree_sha256")
        != launch_manifest.get("checkpoint_unpacked_content_tree_sha256")
    ):
        raise RuntimeError(
            "Architecture submission record is not bound to the prepared content-tree digest"
        )
    active_job_id = os.environ.get("SLURM_JOB_ID")
    if not active_job_id or architecture_submission.get("slurm_job_id") != active_job_id:
        raise RuntimeError("Architecture gate is not running inside the recorded Slurm job")

    checkpoint = REPO / CHECKPOINT_RELATIVE
    checkpoint_metadata = _checkpoint_metadata_digest(checkpoint)
    if checkpoint_metadata != EXPECTED_CHECKPOINT_METADATA_SHA256:
        raise RuntimeError("Live checkpoint does not match the frozen unpacked identity")
    if checkpoint_metadata != launch_manifest["checkpoint_unpacked_metadata_sha256"]:
        raise RuntimeError("Checkpoint metadata changed after the smoke run was prepared")
    checkpoint_content_tree = _checkpoint_content_tree_identity(checkpoint)
    if (
        checkpoint_content_tree["algorithm"] != CHECKPOINT_CONTENT_TREE_ALGORITHM
        or checkpoint_content_tree["algorithm"]
        != launch_manifest.get("checkpoint_content_tree_algorithm")
        or checkpoint_content_tree["content_tree_sha256"]
        != launch_manifest.get("checkpoint_unpacked_content_tree_sha256")
    ):
        raise RuntimeError(
            "Checkpoint file content changed after the smoke run was prepared"
        )
    seeds, seed_table_contract = _load_seed_configuration(run_root)
    policy = policy_config.create_trained_policy(
        training_config.get_config("mme_vla_suite"),
        checkpoint,
        seed=POLICY_SEED,
    )
    capture = _capture_memory_prepare(policy)
    cases = report.setdefault("cases", [])
    stable_vision_cache: int | None = None
    stable_memory_cache: int | None = None
    stable_sample_cache: int | None = None

    for arm in ARMS:
        for history_length in HISTORY_LENGTHS:
            first = _run_once(
                policy,
                capture,
                arm=arm,
                history_length=history_length,
                seeds=seeds,
                seed_table_contract=seed_table_contract,
            )
            if stable_vision_cache is None:
                stable_vision_cache = first["compile_cache"]["vision_after"]
                stable_memory_cache = first["compile_cache"]["memory_after"]
                stable_sample_cache = first["compile_cache"]["sample_after"]
            repeat = _run_once(
                policy,
                capture,
                arm=arm,
                history_length=history_length,
                seeds=seeds,
                seed_table_contract=seed_table_contract,
            )
            compile_stable = all(
                (
                    first["compile_cache"]["vision_after"] == stable_vision_cache,
                    first["compile_cache"]["memory_after"] == stable_memory_cache,
                    first["compile_cache"]["sample_after"] == stable_sample_cache,
                    repeat["compile_cache"]["vision_after"] == stable_vision_cache,
                    repeat["compile_cache"]["memory_after"] == stable_memory_cache,
                    repeat["compile_cache"]["sample_after"] == stable_sample_cache,
                )
            )
            evidence = {
                "arm": arm,
                "history_length": history_length,
                "first": first,
                "repeat": repeat,
                "same_selected_indices": first["selected_frame_indices"]
                == repeat["selected_frame_indices"],
                "same_memory_tensor_digest": first["final_memory_tensor_sha256"]
                == repeat["final_memory_tensor_sha256"],
                "same_live_process_action_digest_audit": first["action_sha256"]
                == repeat["action_sha256"],
                "released_shape_match": first["component_shapes"]
                == [list(shape) for shape in EXPECTED_COMPONENT_SHAPES]
                and repeat["component_shapes"]
                == [list(shape) for shape in EXPECTED_COMPONENT_SHAPES],
                "released_dtype_match": first["component_dtypes"]
                == list(
                    released_prepared_component_dtypes(first["valid_frame_count"])
                )
                and repeat["component_dtypes"]
                == list(
                    released_prepared_component_dtypes(repeat["valid_frame_count"])
                ),
                "released_action_shape_match": first["action_shape"]
                == list(EXPECTED_ACTION_SHAPE)
                and repeat["action_shape"] == list(EXPECTED_ACTION_SHAPE),
                "released_action_dtype_match": first["action_dtype"]
                == EXPECTED_ACTION_DTYPE
                and repeat["action_dtype"] == EXPECTED_ACTION_DTYPE,
                "compile_cache_stable_after_first_inference": compile_stable,
            }
            evidence["passed"] = all(
                (
                    evidence["same_selected_indices"],
                    evidence["same_memory_tensor_digest"],
                    evidence["released_shape_match"],
                    evidence["released_dtype_match"],
                    evidence["released_action_shape_match"],
                    evidence["released_action_dtype_match"],
                    evidence["compile_cache_stable_after_first_inference"],
                )
            )
            cases.append(evidence)
            if not evidence["passed"]:
                raise RuntimeError(f"Repeat/isolation gate failed for {arm}/{history_length}")

    policy.reset()
    final_reset = _reset_snapshot(policy)
    if not final_reset["passed"]:
        raise RuntimeError("Final policy reset did not clear architecture-smoke state")
    report.update(
        {
            "repository_commit_sha": repository_sha,
            "repo_dirty": False,
            "checkpoint_relative_path": str(CHECKPOINT_RELATIVE),
            "checkpoint_archive_sha256_expected": CHECKPOINT_ARCHIVE_SHA256,
            "checkpoint_unpacked_metadata_sha256": checkpoint_metadata,
            "checkpoint_content_tree_algorithm": checkpoint_content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": checkpoint_content_tree[
                "content_tree_sha256"
            ],
            "slurm_job_id": active_job_id,
            "device_count": jax.device_count(),
            "devices": [str(device) for device in jax.devices()],
            "case_count": len(cases),
            "reference_component_shapes": [
                list(shape) for shape in EXPECTED_COMPONENT_SHAPES
            ],
            "reference_component_dtypes_by_padding": {
                "padded": list(released_prepared_component_dtypes(1)),
                "unpadded": list(released_prepared_component_dtypes(32)),
            },
            "reference_final_memory_dtype": EXPECTED_FINAL_MEMORY_DTYPE,
            "reference_action_shape": list(EXPECTED_ACTION_SHAPE),
            "reference_action_dtype": EXPECTED_ACTION_DTYPE,
            "stable_compile_cache": {
                "vision": stable_vision_cache,
                "perceptual_memory": stable_memory_cache,
                "sample_actions": stable_sample_cache,
            },
            "final_reset_evidence": final_reset,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(_dry_run_payload(), sort_keys=True))
        return
    if args.run_root is None:
        parser.error("--run-root is required unless --dry-run is used")

    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Prepared smoke run root does not exist: {run_root}")
    architecture_dir = run_root / "architecture_smoke"
    if architecture_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse architecture-smoke directory: {architecture_dir}"
        )
    architecture_dir.mkdir()
    report_path = architecture_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": _utc_now(),
        "run_root": str(run_root),
        "synthetic_only": True,
        "formal_test_outcomes_opened": False,
        "passed": False,
        "cases": [],
    }
    failure: BaseException | None = None
    try:
        _run_architecture(report, run_root)
        report["passed"] = True
        architecture_submission = json.loads(
            (
                run_root / "protocol" / "architecture_submission_record.json"
            ).read_text()
        )
        _validate_pass_report_contract(
            report,
            run_root=run_root,
            architecture_submission=architecture_submission,
        )
    except BaseException as exc:  # Preserve a write-once failure report before surfacing it.
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    report["finished_utc"] = _utc_now()
    _write_once_json(report_path, report)
    print(report_path)
    if failure is not None:
        raise failure
    if not report["passed"] or len(report["cases"]) != 8:
        raise RuntimeError(f"Architecture smoke report is incomplete: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
