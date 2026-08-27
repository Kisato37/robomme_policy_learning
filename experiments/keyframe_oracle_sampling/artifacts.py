"""Write-once artifacts, seed tables, and completeness audits.

The helpers here are intentionally independent from the simulator and model so
their failure behavior can be exercised on a CPU-only workspace.
"""

from __future__ import annotations

import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import tempfile
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    MASTER_SELECTOR_SEED,
    MAX_POLICY_CALLS,
    RANDOM_SELECTOR_LABEL,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
    SelectorArm,
    derive_random_seed,
    derive_smoke_random_seed,
    parse_arm,
    select_indices,
    validate_selector_output,
)


PROTOCOL_VERSION = "v0.9.1"
FORMAL_TASKS = (
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
ALL_ARMS = tuple(arm.value for arm in SelectorArm)
ARCHITECTURE_ARMS = ("U", "O", "OC", "R")
ARCHITECTURE_HISTORY_LENGTHS = (16, 64)
SEED_TABLE_SCHEMA_VERSION = 2
FORMAL_SEED_DERIVATION = (
    'SHA256(JSON_UTF8([2026082501,task_name,episode_id,policy_call_index,"RandomSamp"]))'
    "[:64bits-big-endian]"
)
SMOKE_SEED_DERIVATION = (
    'SHA256(JSON_UTF8([2026082501,"development-smoke-v1","val",task_name,'
    'episode_id,policy_call_index,"RandomSamp"]))[:64bits-big-endian]'
)
SMOKE_DATASET = "val"
SMOKE_EXECUTED_ACTION_HORIZON = 16
SMOKE_EVALUATION_POLICY_SEED = 7
SMOKE_CHECKPOINT_ID = 79999
SMOKE_CHECKPOINT_PATH = (
    "runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"
)
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
SMOKE_INITIAL_CONDITION_HASH_FIELDS = (
    "front_observations_sha256",
    "wrist_observations_sha256",
    "robot_states_sha256",
    "task_state_sha256",
    "task_instruction_sha256",
)
SMOKE_PREPARED_COMPONENT_SHAPES = (
    (512, 2048),
    (512, 768),
    (512, 8),
    (512,),
)
SMOKE_PREPARED_COMPONENT_DTYPES = (
    "bfloat16",
    "float32",
    "float32",
    "bool",
)
SMOKE_FINAL_MEMORY_SHAPE = (1, 512, 1024)
SMOKE_FINAL_MEMORY_DTYPE = "bfloat16"
SMOKE_ACTION_SHAPE = (20, 8)
SMOKE_ACTION_DTYPE = "float32"
ARCHITECTURE_COMMON_CHECKS = frozenset(
    {
        "history_accumulated_exactly",
        "memory_has_four_components",
        "component_shapes_match_frozen_released_contract",
        "component_dtypes_match_frozen_released_contract",
        "all_components_have_512_slots",
        "mask_is_boolean",
        "valid_token_count_matches",
        "valid_mask_is_prefix",
        "padding_mask_is_false",
        "trace_frame_count_matches",
        "trace_token_count_matches",
        "single_policy_call",
        "temporary_override_restored",
        "final_memory_shape_is_512x1024",
        "final_memory_dtype_is_floating",
        "final_memory_dtype_matches_frozen_released_contract",
        "final_memory_values_are_finite",
        "action_shape_is_frozen_20x8",
        "action_dtype_is_frozen_float32",
        "action_dtype_is_floating",
        "action_values_are_finite",
    }
)
ARCHITECTURE_RESET_FIELDS = {
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


class ArtifactContractError(RuntimeError):
    pass


def validate_architecture_pass_report(
    report: Mapping[str, Any],
    *,
    run_root: str | Path,
    architecture_submission: Mapping[str, Any],
) -> None:
    """Validate and bind the exact GPU architecture-smoke PASS contract.

    A truthy top-level PASS or ``all([])`` is deliberately insufficient.  The
    report must contain each arm/history cell exactly once, and it must have
    been produced for this run root by the Slurm job recorded at submission.
    """
    if report.get("passed") is not True:
        raise ArtifactContractError("GPU architecture smoke is not a recorded PASS")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ArtifactContractError("GPU architecture report cases must be a list")
    case_count = report.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int):
        raise ArtifactContractError("GPU architecture report case_count must be an integer")

    expected = {
        (arm, history_length)
        for arm in ARCHITECTURE_ARMS
        for history_length in ARCHITECTURE_HISTORY_LENGTHS
    }
    expected_reference_fields = {
        "reference_component_shapes": [
            list(shape) for shape in SMOKE_PREPARED_COMPONENT_SHAPES
        ],
        "reference_component_dtypes": list(SMOKE_PREPARED_COMPONENT_DTYPES),
        "reference_final_memory_dtype": SMOKE_FINAL_MEMORY_DTYPE,
        "reference_action_shape": list(SMOKE_ACTION_SHAPE),
        "reference_action_dtype": SMOKE_ACTION_DTYPE,
    }
    _require_exact_fields(
        report,
        expected_reference_fields,
        source="GPU architecture report references",
    )

    observed: list[tuple[Any, Any]] = []
    observed_stable_caches: list[tuple[int, int, int]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ArtifactContractError("GPU architecture report contains a non-object case")
        if case.get("passed") is not True:
            raise ArtifactContractError("GPU architecture report contains a failed case")
        arm = case.get("arm")
        history_length = case.get("history_length")
        observed.append((arm, history_length))
        if (arm, history_length) not in expected:
            raise ArtifactContractError(
                "GPU architecture report contains an unknown arm/history case"
            )
        required_case_flags = (
            "same_selected_indices",
            "same_memory_tensor_digest",
            "released_shape_match",
            "released_dtype_match",
            "released_action_shape_match",
            "released_action_dtype_match",
            "compile_cache_stable_after_first_inference",
        )
        if any(case.get(field) is not True for field in required_case_flags):
            raise ArtifactContractError(
                "GPU architecture case lacks required repeat/isolation evidence"
            )

        if type(history_length) is not int:
            raise ArtifactContractError("GPU architecture history_length must be an integer")
        expected_boundaries = list(range(0, history_length, 7))
        expected_seed = (
            derive_smoke_random_seed("InsertPeg", 0, 0)
            if arm == SelectorArm.RANDOM.value
            else None
        )
        expected_selected = select_indices(
            str(arm),
            history_length - 1,
            boundary_indices=expected_boundaries,
            random_seed=expected_seed,
        )
        expected_selected_digest = hashlib.sha256(
            json.dumps(expected_selected, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        expected_checks = set(ARCHITECTURE_COMMON_CHECKS)
        if arm == SelectorArm.OFFICIAL_UNIFORM.value:
            expected_checks.add("literal_uniform_indices_match")

        run_evidence = []
        for repetition in ("first", "repeat"):
            evidence = case.get(repetition)
            if not isinstance(evidence, Mapping):
                raise ArtifactContractError(
                    f"GPU architecture case lacks {repetition} evidence"
                )
            if evidence.get("reset_before_run") != ARCHITECTURE_RESET_FIELDS:
                raise ArtifactContractError(
                    "GPU architecture case lacks exact pre-run reset evidence"
                )
            checks = evidence.get("checks")
            if (
                not isinstance(checks, Mapping)
                or set(checks) != expected_checks
                or any(value is not True for value in checks.values())
            ):
                raise ArtifactContractError(
                    "GPU architecture case lacks the complete absolute check set"
                )
            _require_exact_fields(
                evidence,
                {
                    "selected_frame_indices": expected_selected,
                    "selected_indices_sha256": expected_selected_digest,
                    "visible_boundary_indices": expected_boundaries,
                    "valid_frame_count": len(expected_selected),
                    "valid_memory_token_count": 16 * len(expected_selected),
                    "padding_frame_count": 32 - len(expected_selected),
                    "component_shapes": [
                        list(shape) for shape in SMOKE_PREPARED_COMPONENT_SHAPES
                    ],
                    "component_dtypes": list(SMOKE_PREPARED_COMPONENT_DTYPES),
                    "final_memory_tensor_shape": list(SMOKE_FINAL_MEMORY_SHAPE),
                    "final_memory_tensor_dtype": SMOKE_FINAL_MEMORY_DTYPE,
                    "final_memory_tensor_finite": True,
                    "action_shape": list(SMOKE_ACTION_SHAPE),
                    "action_dtype": SMOKE_ACTION_DTYPE,
                    "action_finite": True,
                },
                source=f"GPU architecture {arm}/{history_length}/{repetition}",
            )
            for field in (
                "final_memory_tensor_sha256",
                "prepared_memory_components_sha256",
                "mask_sha256",
                "action_sha256",
            ):
                _require_sha256(
                    evidence.get(field),
                    field=f"GPU architecture {repetition} {field}",
                )
            compile_cache = evidence.get("compile_cache")
            if (
                not isinstance(compile_cache, Mapping)
                or set(compile_cache)
                != {
                    "vision_before",
                    "vision_after",
                    "memory_before",
                    "memory_after",
                    "sample_before",
                    "sample_after",
                }
                or any(
                    type(value) is not int or value < 0
                    for value in compile_cache.values()
                )
            ):
                raise ArtifactContractError(
                    "GPU architecture case has invalid compilation-cache evidence"
                )
            run_evidence.append(evidence)
        first, repeat = run_evidence
        if first["selected_frame_indices"] != repeat["selected_frame_indices"]:
            raise ArtifactContractError("GPU architecture repeat selected different frames")
        if (
            first["final_memory_tensor_sha256"]
            != repeat["final_memory_tensor_sha256"]
        ):
            raise ArtifactContractError("GPU architecture repeat changed memory bytes")
        stable_cache = []
        for component in ("vision", "memory", "sample"):
            first_after = first["compile_cache"][f"{component}_after"]
            if not (
                first_after
                == repeat["compile_cache"][f"{component}_before"]
                == repeat["compile_cache"][f"{component}_after"]
            ):
                raise ArtifactContractError(
                    "GPU architecture compilation cache was not stable on repeat"
                )
            stable_cache.append(first_after)
        observed_stable_caches.append(tuple(stable_cache))
    if case_count != len(expected) or len(cases) != len(expected):
        raise ArtifactContractError(
            "GPU architecture report does not contain exactly eight cases"
        )
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ArtifactContractError(
            "GPU architecture report does not contain the unique U/O/OC/R x H16/H64 matrix"
        )
    if len(set(observed_stable_caches)) != 1:
        raise ArtifactContractError(
            "GPU architecture compilation caches changed across arm/history cases"
        )
    expected_stable_cache = dict(
        zip(
            ("vision", "perceptual_memory", "sample_actions"),
            observed_stable_caches[0],
            strict=True,
        )
    )
    if report.get("stable_compile_cache") != expected_stable_cache:
        raise ArtifactContractError(
            "GPU architecture top-level compilation cache summary is inconsistent"
        )

    final_reset = report.get("final_reset_evidence")
    if final_reset != ARCHITECTURE_RESET_FIELDS:
        raise ArtifactContractError("GPU architecture report lacks reset evidence")

    expected_run_root = Path(run_root).resolve()
    for source, payload in (
        ("report", report),
        ("architecture submission", architecture_submission),
    ):
        recorded_root = payload.get("run_root")
        if not isinstance(recorded_root, str) or Path(recorded_root).resolve() != expected_run_root:
            raise ArtifactContractError(
                f"GPU architecture {source} run_root differs from the current run root"
            )

    recorded_job_id = architecture_submission.get("slurm_job_id")
    if not isinstance(recorded_job_id, str) or not recorded_job_id:
        raise ArtifactContractError(
            "GPU architecture submission lacks a recorded Slurm job ID"
        )
    if report.get("slurm_job_id") != recorded_job_id:
        raise ArtifactContractError(
            "GPU architecture report Slurm job ID differs from its submission"
        )


def is_retryable_infrastructure_exception(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if any(
        base.__module__.startswith("websockets")
        and base.__name__.startswith("ConnectionClosed")
        for base in type(exc).__mro__
    ):
        return True
    retryable_errno = {
        errno.EIO,
        errno.ENOSPC,
        errno.ESTALE,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
    }
    return isinstance(exc, OSError) and exc.errno in retryable_errno


def validate_prepared_smoke_root(
    run_root: str | Path,
    seed_table_path: str | Path,
    repo_root: str | Path,
    *,
    attempt_id: int,
) -> dict[str, Any]:
    """Reject direct/bypassed smoke execution and post-prepare code changes."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    protocol_dir = run_root / "protocol"
    manifest_path = protocol_dir / "launch_manifest.json"
    submission_name = (
        "submission_record.json"
        if attempt_id == 0
        else f"submission_record_attempt_{attempt_id:02d}.json"
    )
    submission_path = protocol_dir / submission_name
    architecture_submission_path = (
        protocol_dir / "architecture_submission_record.json"
    )
    architecture_path = run_root / "architecture_smoke" / "report.json"
    required = (
        manifest_path,
        submission_path,
        architecture_submission_path,
        architecture_path,
        protocol_dir / "protocol_snapshot.md",
        protocol_dir / "seed_table.json",
        protocol_dir / "formal_seed_audit_table.json",
        protocol_dir / "smoke_matrix.json",
    )
    for path in required:
        if not path.is_file():
            raise ArtifactContractError(f"Prepared smoke artifact is missing: {path}")
    if Path(seed_table_path).resolve() != (protocol_dir / "seed_table.json").resolve():
        raise ArtifactContractError("Evaluator seed table is not the prepared run-root table")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactContractError("Prepared smoke protocol version mismatch")
    digest_pairs = (
        ("protocol_sha256", protocol_dir / "protocol_snapshot.md"),
        ("seed_table_file_sha256", protocol_dir / "seed_table.json"),
        (
            "formal_seed_audit_file_sha256",
            protocol_dir / "formal_seed_audit_table.json",
        ),
        ("smoke_matrix_sha256", protocol_dir / "smoke_matrix.json"),
    )
    for field, path in digest_pairs:
        if manifest.get(field) != sha256_file(path):
            raise ArtifactContractError(f"Prepared smoke digest mismatch for {path.name}")
    seed_payload, seed_lookup = load_seed_table(
        protocol_dir / "seed_table.json",
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    if manifest.get("seed_table_scope") != seed_payload["scope"]:
        raise ArtifactContractError("Launch manifest seed-table scope mismatch")
    if manifest.get("seed_table_dataset") != seed_payload["dataset"]:
        raise ArtifactContractError("Launch manifest seed-table dataset mismatch")
    if manifest.get("seed_table_derivation") != seed_payload["derivation"]:
        raise ArtifactContractError("Launch manifest seed-table derivation mismatch")
    if manifest.get("seed_table_entries_sha256") != seed_payload["entries_sha256"]:
        raise ArtifactContractError("Launch manifest seed-table entry digest mismatch")
    expected_smoke_keys = {
        (task, 0, call_index)
        for task in FORMAL_TASKS
        for call_index in range(MAX_POLICY_CALLS)
    }
    expected_smoke_entry_count = len(expected_smoke_keys)
    if set(seed_lookup) != expected_smoke_keys:
        raise ArtifactContractError(
            "Smoke seed table is not the exact 16-task x episode-0 x 82-call matrix"
        )
    disjointness = manifest.get("seed_disjointness_audit")
    if disjointness != {
        "smoke_seed_count": expected_smoke_entry_count,
        "formal_seed_count": len(FORMAL_TASKS) * 50 * MAX_POLICY_CALLS,
        "intersection_count": 0,
    }:
        raise ArtifactContractError("Launch manifest lacks the exact seed-disjointness audit")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    try:
        run_root_relative = run_root.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactContractError("Smoke run root must be inside the repository") from exc
    runtime_artifact_root = Path("runs/keyframe_oracle_sampling")
    if run_root_relative.parent != runtime_artifact_root:
        raise ArtifactContractError(
            "Smoke run root must be one direct child of runs/keyframe_oracle_sampling"
        )
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            ".",
            f":(exclude){runtime_artifact_root.as_posix()}",
            ":(exclude)runs/test_time_scaling/checkpoints/"
            "perceptual-framesamp-modul/79999",
        ],
        cwd=repo_root,
        text=True,
    )
    if status:
        raise ArtifactContractError("Smoke evaluator requires a clean committed worktree")
    if manifest.get("repository", {}).get("commit_sha") != commit:
        raise ArtifactContractError("Live repository commit differs from prepared smoke commit")

    architecture = json.loads(architecture_path.read_text())
    architecture_submission = json.loads(architecture_submission_path.read_text())
    validate_architecture_pass_report(
        architecture,
        run_root=run_root,
        architecture_submission=architecture_submission,
    )
    architecture_commit = architecture.get("repository_commit_sha")
    if architecture_commit != commit:
        raise ArtifactContractError("GPU architecture smoke used a different commit")
    if (
        architecture.get("checkpoint_unpacked_metadata_sha256")
        != manifest.get("checkpoint_unpacked_metadata_sha256")
    ):
        raise ArtifactContractError("GPU architecture report used a different checkpoint")
    if (
        architecture.get("checkpoint_content_tree_algorithm")
        != manifest.get("checkpoint_content_tree_algorithm")
        or architecture.get("checkpoint_unpacked_content_tree_sha256")
        != manifest.get("checkpoint_unpacked_content_tree_sha256")
    ):
        raise ArtifactContractError(
            "GPU architecture report used different checkpoint file contents"
        )

    submission = json.loads(submission_path.read_text())
    if submission.get("repository_commit_sha") != commit:
        raise ArtifactContractError("Smoke submission used a different commit")
    if int(submission.get("attempt_id", -1)) != attempt_id:
        raise ArtifactContractError("Smoke submission attempt ID mismatch")
    if submission.get("architecture_report_sha256") != sha256_file(architecture_path):
        raise ArtifactContractError(
            "Architecture-smoke report changed after the smoke submission was recorded"
        )
    if (
        submission.get("checkpoint_content_tree_algorithm")
        != manifest.get("checkpoint_content_tree_algorithm")
        or submission.get("checkpoint_unpacked_content_tree_sha256")
        != manifest.get("checkpoint_unpacked_content_tree_sha256")
    ):
        raise ArtifactContractError(
            "Smoke submission used different checkpoint file contents"
        )
    active_array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not active_array_job or submission.get("slurm_array_job_id") != active_array_job:
        raise ArtifactContractError("Evaluator is not running inside the recorded Slurm array")
    active_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        active_row_id = int(active_array_task) if active_array_task is not None else -1
    except ValueError as exc:
        raise ArtifactContractError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    raw_row_ids = submission.get("row_ids")
    if not isinstance(raw_row_ids, list) or any(
        isinstance(row_id, bool) or not isinstance(row_id, int) for row_id in raw_row_ids
    ):
        raise ArtifactContractError("Smoke submission row_ids must be an integer list")
    if len(raw_row_ids) != len(set(raw_row_ids)):
        raise ArtifactContractError("Smoke submission row_ids contain duplicates")
    if int(submission.get("trajectory_count", -1)) != len(raw_row_ids):
        raise ArtifactContractError("Smoke submission trajectory_count/row_ids mismatch")
    if active_row_id not in raw_row_ids:
        raise ArtifactContractError(
            "Current SLURM_ARRAY_TASK_ID is not authorized by the smoke submission"
        )
    return manifest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonfinite(value: float) -> dict[str, str] | None:
    if math.isnan(value):
        return {"__nonfinite_float__": "NaN"}
    if math.isinf(value):
        return {"__nonfinite_float__": "+Infinity" if value > 0 else "-Infinity"}
    return None


def to_jsonable(value: Any) -> Any:
    """Normalize nested NumPy/JAX-like values without emitting invalid JSON."""
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    # JAX arrays expose tolist/item but should not be imported here.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return _nonfinite(value) or value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported structured-log value: {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Atomically publish bytes exactly once, never replacing a target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    pretty = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, pretty)


def append_jsonl_locked(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(record) + b"\n"
    with path.open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _build_seed_table(
    tasks: Sequence[str],
    episode_ids: Iterable[int],
    *,
    scope: str,
    dataset: str,
    derivation: str,
    derive_seed: Any,
    policy_call_count: int = MAX_POLICY_CALLS,
) -> dict[str, Any]:
    if policy_call_count != MAX_POLICY_CALLS:
        raise ValueError(f"Frozen seed tables require exactly {MAX_POLICY_CALLS} policy calls")
    if len(tasks) != len(set(tasks)):
        raise ValueError("Duplicate tasks are forbidden in a seed table")
    episodes = [int(value) for value in episode_ids]
    if len(episodes) != len(set(episodes)):
        raise ValueError("Duplicate episode IDs are forbidden in a seed table")
    if any(value < 0 for value in episodes):
        raise ValueError("Episode IDs must be non-negative")

    entries = []
    seen: set[tuple[str, int, int]] = set()
    for task in tasks:
        if not task:
            raise ValueError("Task names must be non-empty")
        for episode_id in episodes:
            for policy_call_index in range(MAX_POLICY_CALLS):
                key = (task, episode_id, policy_call_index)
                if key in seen:
                    raise ValueError(f"Duplicate seed key: {key}")
                seen.add(key)
                entries.append(
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "policy_call_index": policy_call_index,
                        "seed": derive_seed(task, episode_id, policy_call_index),
                    }
                )
    digest = sha256_payload(entries)
    return {
        "schema_version": SEED_TABLE_SCHEMA_VERSION,
        "scope": scope,
        "dataset": dataset,
        "derivation": derivation,
        "master_selector_seed": MASTER_SELECTOR_SEED,
        "selector": RANDOM_SELECTOR_LABEL,
        "policy_call_indices": [0, MAX_POLICY_CALLS - 1],
        "entry_count": len(entries),
        "entries": entries,
        "entries_sha256": digest,
    }


def build_seed_table(
    tasks: Sequence[str],
    episode_ids: Iterable[int],
    *,
    policy_call_count: int = MAX_POLICY_CALLS,
) -> dict[str, Any]:
    """Build the frozen formal Section 7.4 RandomSamp table.

    The historical name is retained so existing formal callers keep using the
    exact original derivation.  Smoke callers must use
    :func:`build_smoke_seed_table` explicitly.
    """
    return _build_seed_table(
        tasks,
        episode_ids,
        scope=FORMAL_SEED_SCOPE,
        dataset=FORMAL_SEED_DATASET,
        derivation=FORMAL_SEED_DERIVATION,
        derive_seed=derive_random_seed,
        policy_call_count=policy_call_count,
    )


def build_smoke_seed_table(
    tasks: Sequence[str],
    episode_ids: Iterable[int],
    *,
    policy_call_count: int = MAX_POLICY_CALLS,
) -> dict[str, Any]:
    """Build a development-only table disjoint in namespace from formal seeds."""
    return _build_seed_table(
        tasks,
        episode_ids,
        scope=SMOKE_SEED_SCOPE,
        dataset=SMOKE_SEED_DATASET,
        derivation=SMOKE_SEED_DERIVATION,
        derive_seed=derive_smoke_random_seed,
        policy_call_count=policy_call_count,
    )


def validate_seed_table(
    payload: Mapping[str, Any],
    *,
    expected_scope: str | None = None,
    expected_dataset: str | None = None,
) -> dict[tuple[str, int, int], int]:
    if int(payload.get("schema_version", -1)) != SEED_TABLE_SCHEMA_VERSION:
        raise ArtifactContractError(
            f"Seed table schema must be {SEED_TABLE_SCHEMA_VERSION}"
        )
    if int(payload.get("master_selector_seed", -1)) != MASTER_SELECTOR_SEED:
        raise ArtifactContractError("Seed table has the wrong master selector seed")
    if payload.get("selector") != RANDOM_SELECTOR_LABEL:
        raise ArtifactContractError("Seed table has the wrong selector")
    if payload.get("policy_call_indices") != [0, MAX_POLICY_CALLS - 1]:
        raise ArtifactContractError("Seed table has the wrong policy-call range")

    scope = payload.get("scope")
    dataset = payload.get("dataset")
    derivation = payload.get("derivation")
    contracts = {
        FORMAL_SEED_SCOPE: (
            FORMAL_SEED_DATASET,
            FORMAL_SEED_DERIVATION,
            derive_random_seed,
        ),
        SMOKE_SEED_SCOPE: (
            SMOKE_SEED_DATASET,
            SMOKE_SEED_DERIVATION,
            derive_smoke_random_seed,
        ),
    }
    try:
        required_dataset, required_derivation, derive_seed = contracts[str(scope)]
    except KeyError as exc:
        raise ArtifactContractError(f"Unknown seed-table scope: {scope!r}") from exc
    if dataset != required_dataset:
        raise ArtifactContractError(
            f"Seed-table scope {scope!r} requires dataset {required_dataset!r}"
        )
    if derivation != required_derivation:
        raise ArtifactContractError(
            f"Seed-table scope {scope!r} has the wrong derivation formula"
        )
    if expected_scope is not None and scope != expected_scope:
        raise ArtifactContractError(
            f"Seed-table scope {scope!r} does not match required {expected_scope!r}"
        )
    if expected_dataset is not None and dataset != expected_dataset:
        raise ArtifactContractError(
            f"Seed-table dataset {dataset!r} does not match required {expected_dataset!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ArtifactContractError("Seed table entries must be a list")
    if payload.get("entry_count") != len(entries):
        raise ArtifactContractError("Seed table entry count mismatch")
    if payload.get("entries_sha256") != sha256_payload(entries):
        raise ArtifactContractError("Seed table digest mismatch")
    lookup: dict[tuple[str, int, int], int] = {}
    seen_seeds: set[int] = set()
    per_episode: dict[tuple[str, int], set[int]] = {}
    for entry in entries:
        key = (
            str(entry["task"]),
            int(entry["episode_id"]),
            int(entry["policy_call_index"]),
        )
        if key in lookup:
            raise ArtifactContractError(f"Duplicate seed-table key: {key}")
        expected = derive_seed(*key)
        seed = int(entry["seed"])
        if seed != expected:
            raise ArtifactContractError(f"Seed mismatch for {key}: {seed} != {expected}")
        if seed in seen_seeds:
            raise ArtifactContractError(f"Duplicate derived seed value: {seed}")
        seen_seeds.add(seed)
        lookup[key] = seed
        per_episode.setdefault(key[:2], set()).add(key[2])
    expected_calls = set(range(MAX_POLICY_CALLS))
    for episode_key, calls in per_episode.items():
        if calls != expected_calls:
            raise ArtifactContractError(f"Incomplete seed calls for {episode_key}")
    return lookup


def validate_smoke_formal_seed_disjointness(
    smoke_payload: Mapping[str, Any],
    formal_payload: Mapping[str, Any],
    *,
    tasks: Sequence[str] = FORMAL_TASKS,
) -> dict[str, int]:
    """Hard-check the complete preregistered smoke and formal seed universes."""
    if tuple(tasks) != FORMAL_TASKS:
        raise ArtifactContractError("Seed-disjointness audit requires the exact 16 formal tasks")
    smoke = validate_seed_table(
        smoke_payload,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal = validate_seed_table(
        formal_payload,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    expected_smoke_keys = {
        (task, 0, call_index)
        for task in tasks
        for call_index in range(MAX_POLICY_CALLS)
    }
    expected_formal_keys = {
        (task, episode_id, call_index)
        for task in tasks
        for episode_id in range(50)
        for call_index in range(MAX_POLICY_CALLS)
    }
    if set(smoke) != expected_smoke_keys:
        raise ArtifactContractError(
            "Development-smoke seed table is not the exact 16 x 1 x 82 key matrix"
        )
    if set(formal) != expected_formal_keys:
        raise ArtifactContractError(
            "Formal seed audit table is not the exact 16 x 50 x 82 key matrix"
        )
    smoke_seeds = set(smoke.values())
    formal_seeds = set(formal.values())
    if len(smoke_seeds) != 16 * 1 * MAX_POLICY_CALLS:
        raise ArtifactContractError("Development-smoke seeds are not globally unique")
    if len(formal_seeds) != 16 * 50 * MAX_POLICY_CALLS:
        raise ArtifactContractError("Formal seeds are not globally unique")
    intersection = smoke_seeds & formal_seeds
    if intersection:
        raise ArtifactContractError(
            f"Development-smoke and formal RandomSamp seeds overlap ({len(intersection)})"
        )
    return {
        "smoke_seed_count": len(smoke_seeds),
        "formal_seed_count": len(formal_seeds),
        "intersection_count": 0,
    }


def load_seed_table(
    path: str | Path,
    *,
    expected_scope: str | None = None,
    expected_dataset: str | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], int]]:
    payload = json.loads(Path(path).read_text())
    return payload, validate_seed_table(
        payload,
        expected_scope=expected_scope,
        expected_dataset=expected_dataset,
    )


@dataclasses.dataclass(frozen=True, order=True)
class ScientificKey:
    task: str
    episode_id: int
    arm: str
    trajectory_kind: str = "formal"

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be non-empty")
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        object.__setattr__(self, "arm", parse_arm(self.arm).value)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class EpisodeAttemptWriter:
    """Own one append-only attempt directory for one trajectory."""

    def __init__(self, attempt_dir: str | Path, key: ScientificKey, attempt_id: int):
        self.attempt_dir = Path(attempt_dir)
        self.key = key
        self.attempt_id = int(attempt_id)
        if self.attempt_id < 0 or self.attempt_id > 2:
            raise ValueError("Protocol allows initial attempt 0 and at most retries 1 and 2")
        self.manifest_path = self.attempt_dir / "episode_manifest.json"
        self.initial_conditions_path = self.attempt_dir / "initial_condition_hashes.json"
        self.trace_path = self.attempt_dir / "selector_trace.jsonl"
        self.result_path = self.attempt_dir / "episode_result.json"

    def create(self, manifest: Mapping[str, Any]) -> None:
        if self.attempt_dir.exists():
            raise FileExistsError(f"Refusing to reuse attempt directory: {self.attempt_dir}")
        self.attempt_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            self.manifest_path,
            {
                **dict(manifest),
                "scientific_key": self.key.as_dict(),
                "attempt_id": self.attempt_id,
                "created_utc": utc_now(),
            },
        )

    def validate_resume(self) -> str:
        if not self.manifest_path.exists():
            raise ArtifactContractError(f"Attempt has no manifest: {self.attempt_dir}")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("scientific_key") != self.key.as_dict():
            raise ArtifactContractError(f"Attempt scientific key mismatch: {self.attempt_dir}")
        if int(manifest.get("attempt_id", -1)) != self.attempt_id:
            raise ArtifactContractError(f"Attempt ID mismatch: {self.attempt_dir}")
        if self.result_path.exists():
            result = json.loads(self.result_path.read_text())
            if result.get("scientific_key") != self.key.as_dict():
                raise ArtifactContractError(f"Result scientific key mismatch: {self.result_path}")
            expected_trace_hash = result.get("selector_trace_sha256")
            actual_trace_hash = sha256_file(self.trace_path) if self.trace_path.exists() else None
            if expected_trace_hash != actual_trace_hash:
                raise ArtifactContractError(f"Selector trace digest mismatch: {self.trace_path}")
            if result.get("episode_manifest_sha256") != sha256_file(self.manifest_path):
                raise ArtifactContractError(f"Episode manifest digest mismatch: {self.manifest_path}")
            if not self.initial_conditions_path.exists():
                raise ArtifactContractError(
                    f"Completed attempt lacks initial-condition hashes: {self.attempt_dir}"
                )
            if result.get("initial_condition_hashes_sha256") != sha256_file(
                self.initial_conditions_path
            ):
                raise ArtifactContractError(
                    f"Initial-condition digest mismatch: {self.initial_conditions_path}"
                )
            return "complete"
        return "incomplete"

    def record_initial_conditions(self, hashes: Mapping[str, str]) -> None:
        if not hashes or any(not value for value in hashes.values()):
            raise ArtifactContractError("Initial-condition hashes must all be present")
        atomic_write_json(self.initial_conditions_path, hashes)

    def append_trace(self, record: Mapping[str, Any]) -> None:
        if self.result_path.exists():
            raise ArtifactContractError("Completed selector traces are immutable")
        if "policy_call_index" in record and self.trace_path.exists():
            existing_calls = {
                int(json.loads(line)["policy_call_index"])
                for line in self.trace_path.read_text().splitlines()
                if line.strip() and "policy_call_index" in json.loads(line)
            }
            call_index = int(record["policy_call_index"])
            if call_index in existing_calls:
                raise ArtifactContractError(
                    f"Duplicate selector-trace policy_call_index: {call_index}"
                )
        append_jsonl_locked(self.trace_path, record)

    def finalize(self, result: Mapping[str, Any]) -> None:
        if not self.manifest_path.exists():
            raise ArtifactContractError("Cannot finalize an attempt without a manifest")
        if not self.initial_conditions_path.exists():
            raise ArtifactContractError("Cannot finalize without initial-condition hashes")
        if self.result_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {self.result_path}")
        terminal_reason = result.get("terminal_reason")
        if terminal_reason not in {"success", "fail", "timeout", "error"}:
            raise ArtifactContractError(
                f"Invalid official terminal_reason for scientific result: {terminal_reason!r}"
            )
        if terminal_reason == "error" and (
            not result.get("benchmark_error_message")
            or not result.get("benchmark_exception_type")
        ):
            raise ArtifactContractError(
                "A scientific error outcome requires FailAwareWrapper error evidence"
            )
        trace_hash = sha256_file(self.trace_path) if self.trace_path.exists() else None
        atomic_write_json(
            self.result_path,
            {
                **dict(result),
                "scientific_key": self.key.as_dict(),
                "attempt_id": self.attempt_id,
                "episode_manifest_sha256": sha256_file(self.manifest_path),
                "initial_condition_hashes_sha256": sha256_file(
                    self.initial_conditions_path
                ),
                "selector_trace_sha256": trace_hash,
                "completed_utc": utc_now(),
            },
        )


class RunArtifactStore:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.failures_path = self.run_root / "failures" / "failure_ledger.jsonl"

    def attempt_dir(self, key: ScientificKey, attempt_id: int) -> Path:
        base = (
            self.run_root
            / "trajectories"
            / key.task
            / f"episode_{key.episode_id:02d}"
            / key.arm
        )
        if key.trajectory_kind != "formal":
            base = base / key.trajectory_kind
        return base / f"attempt_{attempt_id:02d}"

    def scan_completed_keys(self) -> dict[ScientificKey, Path]:
        completed: dict[ScientificKey, Path] = {}
        for result_path in self.run_root.glob("trajectories/**/episode_result.json"):
            result = json.loads(result_path.read_text())
            raw_key = result.get("scientific_key", {})
            key = ScientificKey(**raw_key)
            if key in completed:
                raise ArtifactContractError(
                    f"Duplicate completed scientific key {key}: {completed[key]} and {result_path}"
                )
            completed[key] = result_path
        return completed

    def new_attempt(
        self,
        key: ScientificKey,
        attempt_id: int,
        manifest: Mapping[str, Any],
    ) -> EpisodeAttemptWriter:
        if key in self.scan_completed_keys():
            raise ArtifactContractError(f"Scientific key already has a completed result: {key}")
        if attempt_id > 0:
            if not self.failures_path.exists():
                raise ArtifactContractError("Retry has no infrastructure failure ledger")
            matching = [
                record
                for record in read_jsonl(self.failures_path)
                if str(record.get("task")) == key.task
                and int(record.get("episode_id", -1)) == key.episode_id
                and parse_arm(record.get("arm", "")).value == key.arm
                and str(record.get("trajectory_kind", "formal"))
                == key.trajectory_kind
                and int(record.get("attempt_id", -1)) == attempt_id - 1
            ]
            if len(matching) != 1 or matching[0].get("retry_allowed") is not True:
                raise ArtifactContractError(
                    "Retry requires exactly one retry-allowed infrastructure failure "
                    "for the immediately preceding attempt"
                )
            previous_dir = self.attempt_dir(key, attempt_id - 1)
            if not previous_dir.is_dir():
                raise ArtifactContractError("Retry cannot skip a missing preceding attempt")
        writer = EpisodeAttemptWriter(self.attempt_dir(key, attempt_id), key, attempt_id)
        writer.create(manifest)
        return writer

    def record_failure(self, record: Mapping[str, Any]) -> None:
        append_jsonl_locked(self.failures_path, {**dict(record), "recorded_utc": utc_now()})


def expected_keys(
    tasks: Sequence[str],
    episode_ids: Iterable[int],
    arms: Sequence[str] = ALL_ARMS,
    *,
    trajectory_kind: str = "formal",
) -> set[ScientificKey]:
    return {
        ScientificKey(task, int(episode_id), arm, trajectory_kind)
        for task in tasks
        for episode_id in episode_ids
        for arm in arms
    }


def completeness_report(
    expected: set[ScientificKey],
    completed: Mapping[ScientificKey, Path],
) -> dict[str, Any]:
    observed = set(completed)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {
        "expected_count": len(expected),
        "completed_count": len(expected & observed),
        "complete": not missing and not unexpected,
        "missing": [key.as_dict() for key in missing],
        "unexpected": [key.as_dict() for key in unexpected],
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def selector_latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(record["selector_latency_ms"]) for record in records])
    if values.size == 0:
        raise ArtifactContractError("Cannot summarize an empty selector trace")
    if not np.isfinite(values).all():
        raise ArtifactContractError("Selector latency trace contains non-finite values")
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def audit_attempt(writer: EpisodeAttemptWriter) -> dict[str, Any]:
    status = writer.validate_resume()
    traces = read_jsonl(writer.trace_path) if writer.trace_path.exists() else []
    if status == "complete" and not traces:
        raise ArtifactContractError("A completed scientific attempt has no selector trace")
    calls = [int(record["policy_call_index"]) for record in traces]
    if calls != list(range(len(calls))):
        raise ArtifactContractError(
            f"Selector trace calls must be contiguous from zero, got {calls}"
        )
    prior_history_length = 0
    for record in traces:
        history_length = int(record["history_length"])
        step_idx = int(record["current_history_index"])
        if history_length != step_idx + 1 or history_length < prior_history_length:
            raise ArtifactContractError("Selector trace history timeline is inconsistent")
        selected = validate_selector_output(record["selected_frame_indices"], step_idx)
        selected_hash = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if record["selected_indices_sha256"] != selected_hash:
            raise ArtifactContractError("Selected-index trace digest mismatch")
        if int(record["valid_frame_count"]) != len(selected):
            raise ArtifactContractError("Selector trace valid-frame count mismatch")
        if int(record["padding_frame_count"]) != 32 - len(selected):
            raise ArtifactContractError("Selector trace padding count mismatch")
        if int(record["valid_memory_token_count"]) != 16 * len(selected):
            raise ArtifactContractError("Selector trace valid-token count mismatch")
        prior_history_length = history_length
    return {
        "status": status,
        "policy_call_count": len(traces),
        "selector_latency": selector_latency_summary(traces) if traces else None,
    }


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX_DIGITS for character in value)
    ):
        raise ArtifactContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ArtifactContractError(f"{field} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ArtifactContractError(f"{field} must be a finite non-negative number")
    return normalized


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: str,
) -> None:
    mismatches = {
        field: {"expected": expected_value, "observed": payload.get(field)}
        for field, expected_value in expected.items()
        if field not in payload or payload.get(field) != expected_value
    }
    if mismatches:
        raise ArtifactContractError(
            f"{source} differs from the frozen smoke contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def audit_smoke_attempt(
    writer: EpisodeAttemptWriter,
    *,
    expected_key: ScientificKey,
    expected_row: Mapping[str, Any],
    launch_manifest: Mapping[str, Any],
    smoke_seed_payload: Mapping[str, Any],
    smoke_seed_lookup: Mapping[tuple[str, int, int], int],
) -> dict[str, Any]:
    """Fail-closed scientific audit for one completed smoke trajectory.

    ``audit_attempt`` intentionally remains a generic artifact-integrity check.
    This stricter layer binds every trace decision back to the frozen smoke row,
    seed table, selector definition, and model/runtime contract.
    """
    if writer.key != expected_key:
        raise ArtifactContractError("Attempt writer is bound to the wrong scientific key")
    expected_row_binding = {
        "task": expected_key.task,
        "episode_id": expected_key.episode_id,
        "arm": expected_key.arm,
        "trajectory_kind": expected_key.trajectory_kind,
    }
    _require_exact_fields(
        expected_row,
        expected_row_binding,
        source="Frozen smoke row",
    )
    if type(expected_row.get("episode_id")) is not int:
        raise ArtifactContractError("Frozen smoke episode_id must be an integer")
    if expected_row.get("dataset") != SMOKE_DATASET:
        raise ArtifactContractError("Frozen smoke row must use the val dataset")
    max_steps = expected_row.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise ArtifactContractError("Frozen smoke max_steps must be an integer")

    seed_lookup_from_payload = validate_seed_table(
        smoke_seed_payload,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    if dict(smoke_seed_lookup) != seed_lookup_from_payload:
        raise ArtifactContractError("Smoke seed lookup differs from the validated table")
    seed_digest = _require_sha256(
        smoke_seed_payload.get("entries_sha256"),
        field="Smoke seed-table entries_sha256",
    )
    _require_exact_fields(
        launch_manifest,
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": SMOKE_DATASET,
            "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
            "checkpoint_path": SMOKE_CHECKPOINT_PATH,
            "seed_table_scope": SMOKE_SEED_SCOPE,
            "seed_table_dataset": SMOKE_SEED_DATASET,
            "seed_table_derivation": SMOKE_SEED_DERIVATION,
            "seed_table_entries_sha256": seed_digest,
        },
        source="Launch manifest",
    )
    for field in (
        "checkpoint_archive_sha256_actual",
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        _require_sha256(launch_manifest.get(field), field=f"Launch manifest {field}")

    generic_report = audit_attempt(writer)
    manifest = json.loads(writer.manifest_path.read_text())
    result = json.loads(writer.result_path.read_text())
    initial_conditions = json.loads(writer.initial_conditions_path.read_text())
    if set(initial_conditions) != set(SMOKE_INITIAL_CONDITION_HASH_FIELDS):
        raise ArtifactContractError(
            "Smoke attempt must record the exact five initial-condition hashes"
        )
    for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS:
        _require_sha256(
            initial_conditions.get(field),
            field=f"Initial-condition {field}",
        )
    fixed_attempt_fields = {
        "dataset": SMOKE_DATASET,
        "max_steps": int(max_steps),
        "executed_action_horizon": SMOKE_EXECUTED_ACTION_HORIZON,
        "evaluation_policy_seed": SMOKE_EVALUATION_POLICY_SEED,
        "checkpoint_id": SMOKE_CHECKPOINT_ID,
    }
    _require_exact_fields(manifest, fixed_attempt_fields, source="Episode manifest")
    _require_exact_fields(result, fixed_attempt_fields, source="Episode result")
    for source, payload in (("Episode manifest", manifest), ("Episode result", result)):
        for field in (
            "max_steps",
            "executed_action_horizon",
            "evaluation_policy_seed",
            "checkpoint_id",
        ):
            if type(payload.get(field)) is not int:
                raise ArtifactContractError(f"{source} {field} must be an integer")
    _require_exact_fields(
        manifest,
        {
            "protocol_version": PROTOCOL_VERSION,
            "seed_table_sha256": seed_digest,
        },
        source="Episode manifest",
    )
    _require_exact_fields(
        result,
        {
            "task": expected_key.task,
            "episode_id": expected_key.episode_id,
            "selector_arm": expected_key.arm,
        },
        source="Episode result",
    )
    if type(result.get("episode_id")) is not int:
        raise ArtifactContractError("Episode result episode_id must be an integer")
    if result.get("terminal_reason") not in {"success", "fail", "timeout"}:
        raise ArtifactContractError("Smoke result must have a scientific terminal outcome")
    if type(result.get("success")) is not bool:
        raise ArtifactContractError("Smoke result success must be a boolean")
    if result["success"] != (result["terminal_reason"] == "success"):
        raise ArtifactContractError("Smoke result success disagrees with terminal_reason")
    if type(result.get("timeout")) is not bool:
        raise ArtifactContractError("Smoke result timeout must be a boolean")
    if result["timeout"] != (result["terminal_reason"] == "timeout"):
        raise ArtifactContractError("Smoke result timeout disagrees with terminal_reason")

    steps = result.get("steps")
    if (
        type(steps) is not int
        or steps < 1
        or steps > int(max_steps)
    ):
        raise ArtifactContractError("Episode result has an invalid smoke step count")
    expected_policy_calls = math.ceil(steps / SMOKE_EXECUTED_ACTION_HORIZON)
    traces = read_jsonl(writer.trace_path)
    if len(traces) != expected_policy_calls:
        raise ArtifactContractError(
            "Selector trace count must equal ceil(result.steps / 16): "
            f"{len(traces)} != {expected_policy_calls}"
        )

    result_latency_fields = (
        "policy_latency_ms",
        "policy_model_latency_ms",
        "history_lengths_at_policy_calls",
    )
    for field in result_latency_fields:
        values = result.get(field)
        if not isinstance(values, list) or len(values) != expected_policy_calls:
            raise ArtifactContractError(
                f"Episode result {field} must contain one value per policy call"
            )

    required_trace_hashes = (
        "seed_table_sha256",
        "selected_indices_sha256",
        "mask_sha256",
        "image_tensor_sha256",
        "position_tensor_sha256",
        "state_tensor_sha256",
        "prepared_memory_components_sha256",
        "prepared_memory_input_sha256",
        "final_memory_tensor_sha256",
    )
    required_trace_latencies = (
        "boundary_lookup_latency_ms",
        "selector_decision_latency_ms",
        "selector_bookkeeping_latency_ms",
        "selector_latency_ms",
        "model_latency_ms",
        "end_to_end_request_latency_ms",
    )
    initial_history_length: int | None = None
    for call_index, trace in enumerate(traces):
        _require_exact_fields(
            trace,
            {
                "schema_version": 1,
                "task": expected_key.task,
                "episode_id": expected_key.episode_id,
                "selector_name": expected_key.arm,
                "seed_table_sha256": seed_digest,
                "seed_table_scope": SMOKE_SEED_SCOPE,
                "seed_table_dataset": SMOKE_SEED_DATASET,
                "policy_call_index": call_index,
                "environment_step": call_index * SMOKE_EXECUTED_ACTION_HORIZON,
            },
            source=f"Selector trace call {call_index}",
        )
        for field in ("episode_id", "policy_call_index", "environment_step"):
            if type(trace.get(field)) is not int:
                raise ArtifactContractError(f"Trace {field} must be an integer")

        current_history_index = trace.get("current_history_index")
        if (
            type(current_history_index) is not int
            or current_history_index < 0
        ):
            raise ArtifactContractError("Trace current_history_index must be non-negative")
        history_length = trace.get("history_length")
        if (
            type(history_length) is not int
            or history_length != current_history_index + 1
        ):
            raise ArtifactContractError(
                "Trace history_length must equal current_history_index + 1"
            )
        if initial_history_length is None:
            initial_history_length = history_length
        if history_length != initial_history_length + trace["environment_step"]:
            raise ArtifactContractError(
                "Trace history cadence does not match the 16-step action horizon"
            )
        raw_boundaries = trace.get("visible_boundary_indices")
        if not isinstance(raw_boundaries, list):
            raise ArtifactContractError("Trace visible_boundary_indices must be a list")
        if any(
            type(value) is not int
            for value in raw_boundaries
        ):
            raise ArtifactContractError("Trace visible boundaries must be integer indices")
        visible_boundaries = [int(value) for value in raw_boundaries]
        if (
            visible_boundaries != sorted(set(visible_boundaries))
            or any(value < 0 or value > current_history_index for value in visible_boundaries)
            or not visible_boundaries
            or visible_boundaries[0] != 0
        ):
            raise ArtifactContractError(
                "Trace visible boundaries must be unique, sorted, causal, and include frame 0"
            )

        if "selector_seed" not in trace:
            raise ArtifactContractError("Trace must explicitly record selector_seed")
        expected_seed = None
        if expected_key.arm == SelectorArm.RANDOM.value:
            seed_key = (expected_key.task, expected_key.episode_id, call_index)
            try:
                expected_seed = int(smoke_seed_lookup[seed_key])
            except KeyError as exc:
                raise ArtifactContractError(
                    f"Smoke seed table lacks preregistered call {seed_key}"
                ) from exc
        if trace["selector_seed"] != expected_seed:
            raise ArtifactContractError(
                f"Selector trace call {call_index} has the wrong preregistered seed"
            )

        expected_selected = select_indices(
            expected_key.arm,
            current_history_index,
            boundary_indices=visible_boundaries,
            random_seed=expected_seed,
        )
        if trace.get("selected_frame_indices") != expected_selected:
            raise ArtifactContractError(
                f"Selector trace call {call_index} does not match arm {expected_key.arm}"
            )

        expected_valid_tokens = 16 * len(expected_selected)
        if trace.get("mask_shape") != [512] or trace.get("mask_dtype") != "bool":
            raise ArtifactContractError("Trace mask contract must be bool[512]")
        if (
            trace.get("mask_valid_prefix_all_true") is not True
            or trace.get("mask_padding_all_false") is not True
            or trace.get("valid_memory_token_count") != expected_valid_tokens
        ):
            raise ArtifactContractError(
                "Trace mask does not encode one valid prefix followed by false padding"
            )
        component_shapes = trace.get("prepared_memory_component_shapes")
        if component_shapes != [list(shape) for shape in SMOKE_PREPARED_COMPONENT_SHAPES]:
            raise ArtifactContractError(
                "Prepared memory component shapes differ from the frozen checkpoint contract"
            )
        component_dtypes = (
            trace.get("image_tensor_dtype"),
            trace.get("position_tensor_dtype"),
            trace.get("state_tensor_dtype"),
            trace.get("mask_dtype"),
        )
        if component_dtypes != SMOKE_PREPARED_COMPONENT_DTYPES:
            raise ArtifactContractError(
                "Prepared memory component dtypes differ from the frozen checkpoint contract"
            )
        if trace.get("prepared_memory_input_shape") != [512, 2824]:
            raise ArtifactContractError(
                "Prepared memory input shape must be [512, 2824]"
            )

        for field in required_trace_hashes:
            digest = _require_sha256(trace.get(field), field=f"Trace {field}")
            if field == "seed_table_sha256" and digest != seed_digest:
                raise ArtifactContractError("Trace seed-table digest mismatch")
        latencies = {
            field: _require_nonnegative_finite_number(
                trace.get(field), field=f"Trace {field}"
            )
            for field in required_trace_latencies
        }
        component_latency = (
            latencies["boundary_lookup_latency_ms"]
            + latencies["selector_decision_latency_ms"]
            + latencies["selector_bookkeeping_latency_ms"]
        )
        if not math.isclose(
            latencies["selector_latency_ms"],
            component_latency,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ArtifactContractError("Selector latency does not equal its components")

        memory_shape = trace.get("final_memory_tensor_shape")
        if (
            memory_shape != list(SMOKE_FINAL_MEMORY_SHAPE)
            or not isinstance(memory_shape, list)
            or any(type(value) is not int for value in memory_shape)
        ):
            raise ArtifactContractError("Final memory tensor shape must be [1, 512, 1024]")
        if trace.get("final_memory_tensor_is_floating") is not True:
            raise ArtifactContractError("Final memory tensor must have a floating dtype")
        dtype = trace.get("final_memory_tensor_dtype")
        if dtype != SMOKE_FINAL_MEMORY_DTYPE:
            raise ArtifactContractError(
                "Final memory tensor dtype differs from the frozen checkpoint contract"
            )
        if trace.get("final_memory_tensor_finite") is not True:
            raise ArtifactContractError("Final memory tensor must contain only finite values")

        expected_end_to_end = _require_nonnegative_finite_number(
            result["policy_latency_ms"][call_index],
            field="Episode result policy_latency_ms",
        )
        expected_model = _require_nonnegative_finite_number(
            result["policy_model_latency_ms"][call_index],
            field="Episode result policy_model_latency_ms",
        )
        if latencies["end_to_end_request_latency_ms"] != expected_end_to_end:
            raise ArtifactContractError("Trace/request latency differs from episode result")
        if latencies["model_latency_ms"] != expected_model:
            raise ArtifactContractError("Trace/model latency differs from episode result")
        if result["history_lengths_at_policy_calls"][call_index] != trace.get(
            "history_length"
        ):
            raise ArtifactContractError("Trace history length differs from episode result")

    return {
        **generic_report,
        "strict_smoke_contract": True,
        "expected_policy_call_count": expected_policy_calls,
        "selector_arm": expected_key.arm,
    }


def audit_initial_condition_fairness(
    manifests: Sequence[Mapping[str, Any]],
    *,
    required_arms: Sequence[str] = ALL_ARMS,
) -> dict[str, Any]:
    """Verify byte-identical reset hashes inside each four-arm paired block."""
    required = {parse_arm(arm).value for arm in required_arms}
    groups: dict[tuple[str, int, str], dict[str, Mapping[str, str]]] = {}
    for manifest in manifests:
        raw_key = manifest["scientific_key"]
        block = (
            str(raw_key["task"]),
            int(raw_key["episode_id"]),
            str(raw_key.get("trajectory_kind", "formal")),
        )
        arm = parse_arm(raw_key["arm"]).value
        if arm in groups.setdefault(block, {}):
            raise ArtifactContractError(f"Duplicate arm manifest in paired block {block}: {arm}")
        hashes = manifest.get("initial_condition_hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            raise ArtifactContractError(f"Missing initial-condition hashes for {block}/{arm}")
        groups[block][arm] = hashes

    audited = 0
    for block, arm_hashes in groups.items():
        if set(arm_hashes) != required:
            raise ArtifactContractError(
                f"Paired block {block} has arms {sorted(arm_hashes)}, expected {sorted(required)}"
            )
        distinct = {canonical_json_bytes(hashes) for hashes in arm_hashes.values()}
        if len(distinct) != 1:
            raise ArtifactContractError(f"Initial-condition hash mismatch across paired block {block}")
        audited += 1
    return {"paired_blocks": audited, "fair": True}


def audit_paired_manifest_invariants(
    manifests: Sequence[Mapping[str, Any]],
    *,
    required_arms: Sequence[str] = ALL_ARMS,
) -> dict[str, Any]:
    """Verify non-treatment configuration is identical in every four-arm block."""
    required = {parse_arm(arm).value for arm in required_arms}
    invariant_fields = (
        "dataset",
        "max_steps",
        "executed_action_horizon",
        "evaluation_policy_seed",
        "checkpoint_id",
        "seed_table_sha256",
        "resolved_environment_seed",
        "resolved_difficulty_hint",
        "difficulty",
    )
    groups: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = {}
    for manifest in manifests:
        raw_key = manifest["scientific_key"]
        block = (
            str(raw_key["task"]),
            int(raw_key["episode_id"]),
            str(raw_key.get("trajectory_kind", "formal")),
        )
        arm = parse_arm(raw_key["arm"]).value
        if arm in groups.setdefault(block, {}):
            raise ArtifactContractError(f"Duplicate arm manifest in paired block {block}: {arm}")
        groups[block][arm] = manifest

    for block, arm_manifests in groups.items():
        if set(arm_manifests) != required:
            raise ArtifactContractError(
                f"Paired block {block} has arms {sorted(arm_manifests)}, "
                f"expected {sorted(required)}"
            )
        signatures = {
            canonical_json_bytes(
                {field: manifest.get(field) for field in invariant_fields}
            )
            for manifest in arm_manifests.values()
        }
        if len(signatures) != 1:
            raise ArtifactContractError(
                f"Non-treatment manifest mismatch across paired block {block}"
            )
    return {"paired_blocks": len(groups), "invariants_match": True}
