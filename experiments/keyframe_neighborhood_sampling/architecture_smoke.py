#!/usr/bin/env python3
"""Fresh real-checkpoint GPU architecture gate for OC3/OC5 only."""

# Heavy policy/JAX imports stay inside the live path so --dry-run is CPU-only.
# ruff: noqa: PLC0415, SLF001

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from experiments.keyframe_neighborhood_sampling.direct_provenance import audit_direct_completion
from experiments.keyframe_neighborhood_sampling.direct_provenance import require_runtime_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import runner_backend
from experiments.keyframe_neighborhood_sampling.direct_provenance import runtime_direct_provenance
from experiments.keyframe_neighborhood_sampling.direct_provenance import validate_direct_runner
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.prepare_formal import repository_state
from experiments.keyframe_neighborhood_sampling.prepare_formal import require_clean_repository
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_oracle_sampling import architecture_smoke as _base
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import released_prepared_component_dtypes
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import EXPECTED_CHECKPOINT_ARCHIVE_SHA256
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from mme_vla_suite.shared.keyframe_oracle_sampling import select_indices

REPO = Path(__file__).resolve().parents[2]
HISTORY_LENGTHS = (16, 64)
CASE_COUNT = len(EXTENSION_ARMS) * len(HISTORY_LENGTHS)
REPORT_REQUIRED_FIELDS = (
    "protocol_version",
    "protocol_family",
    "formal_started",
    "arms",
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


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_architecture_pass_report(
    report: dict[str, Any],
    *,
    run_root: Path,
    architecture_submission: dict[str, Any],
    require_direct_completion: bool = True,
) -> None:
    """Validate the extension PASS artifact without accepting parent evidence."""
    backend = runner_backend(report)
    required = [field for field in REPORT_REQUIRED_FIELDS if field != "slurm_job_id" or backend == "slurm"]
    missing = [field for field in required if field not in report]
    if missing:
        raise RuntimeError(f"Extension architecture report lacks fields: {missing}")
    expected_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "passed": True,
        "run_root": str(run_root.resolve()),
        "case_count": CASE_COUNT,
    }
    mismatches = {
        field: {"expected": expected, "observed": report.get(field)}
        for field, expected in expected_identity.items()
        if report.get(field) != expected
    }
    if mismatches:
        raise RuntimeError("Extension architecture report identity mismatch: " + json.dumps(mismatches, sort_keys=True))
    if report.get("formal_test_outcomes_opened") is not False:
        raise RuntimeError("Architecture smoke may not open formal test outcomes")
    commit = report.get("repository_commit_sha")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("Extension architecture report has an invalid commit SHA")
    for field in (
        "checkpoint_unpacked_metadata_sha256",
        "checkpoint_unpacked_content_tree_sha256",
    ):
        if not _is_lower_sha256(report.get(field)):
            raise RuntimeError(f"Extension architecture report has invalid {field}")
    if report.get("checkpoint_content_tree_algorithm") != _base.CHECKPOINT_CONTENT_TREE_ALGORITHM:
        raise RuntimeError("Extension architecture report has the wrong checkpoint tree algorithm")

    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise RuntimeError("Extension architecture report must contain exactly four cases")
    observed = {(case.get("arm"), case.get("history_length")) for case in cases}
    expected = {(arm, length) for arm in EXTENSION_ARMS for length in HISTORY_LENGTHS}
    if observed != expected or any(case.get("passed") is not True for case in cases):
        raise RuntimeError("Extension architecture cases are incomplete or not all PASS")
    if (
        not isinstance(report.get("final_reset_evidence"), dict)
        or report["final_reset_evidence"].get("passed") is not True
    ):
        raise RuntimeError("Extension architecture final reset did not pass")

    submission_expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "repository_commit_sha": commit,
        "run_root": str(run_root.resolve()),
        "checkpoint_unpacked_metadata_sha256": report["checkpoint_unpacked_metadata_sha256"],
        "checkpoint_content_tree_algorithm": report["checkpoint_content_tree_algorithm"],
        "checkpoint_unpacked_content_tree_sha256": report["checkpoint_unpacked_content_tree_sha256"],
    }
    if backend == "slurm":
        submission_expected["slurm_job_id"] = report["slurm_job_id"]
    if runner_backend(architecture_submission) != backend:
        raise RuntimeError("Architecture report/submission runner backend mismatch")
    if any(
        architecture_submission.get(field) != expected_value for field, expected_value in submission_expected.items()
    ):
        raise RuntimeError("Extension architecture report/submission binding mismatch")
    if backend == "direct":
        if (architecture_submission.get("runtime_profile", {}).get("policy_lifetime") == "resident"
                and report.get("resident_cross_arm_reset", {}).get("passed") is not True):
            raise RuntimeError("Resident policy requires the interleaved reset/repeat GPU check")
        validate_direct_runner(
            report.get("runner", {}), run_root, stage="architecture_smoke", attempt_id=0, row_id=None,
            submission=architecture_submission,
            submission_path=run_root / "protocol/architecture_submission_record.json",
        )
        if require_direct_completion:
            audit_direct_completion(report["runner"], run_root, required_roles={"architecture"})


def _dry_case(arm: str, history_length: int) -> dict[str, Any]:
    case = _base._dry_contract_case(arm, history_length)
    case["selected_indices_match_extension_selector"] = True
    return case


def dry_run_contract() -> dict[str, Any]:
    """Exercise the four-case selector/report shape without a checkpoint or GPU."""
    cases = [_dry_case(arm, length) for arm in EXTENSION_ARMS for length in HISTORY_LENGTHS]
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "passed": True,
        "repository_commit_sha": "0" * 40,
        "checkpoint_unpacked_metadata_sha256": "0" * 64,
        "checkpoint_content_tree_algorithm": _base.CHECKPOINT_CONTENT_TREE_ALGORITHM,
        "checkpoint_unpacked_content_tree_sha256": "0" * 64,
        "run_root": str((REPO / "runs/keyframe_neighborhood_sampling/dry-run-contract").resolve()),
        "slurm_job_id": "dry-run-job",
        "case_count": CASE_COUNT,
        "cases": cases,
        "final_reset_evidence": _base._dry_contract_reset(),
        "formal_test_outcomes_opened": False,
    }
    submission = {
        field: report[field]
        for field in (
            "protocol_version",
            "protocol_family",
            "formal_started",
            "arms",
            "repository_commit_sha",
            "run_root",
            "slurm_job_id",
            "checkpoint_unpacked_metadata_sha256",
            "checkpoint_content_tree_algorithm",
            "checkpoint_unpacked_content_tree_sha256",
        )
    }
    validate_architecture_pass_report(
        report,
        run_root=Path(report["run_root"]),
        architecture_submission=submission,
    )
    return {
        "valid": True,
        "report_contract_valid": True,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "arms": list(EXTENSION_ARMS),
        "case_count": CASE_COUNT,
        "cases": [{"arm": arm, "history_length": length} for arm in EXTENSION_ARMS for length in HISTORY_LENGTHS],
        "checkpoint": str(CHECKPOINT_RELATIVE),
        "runs_gpu_inference": False,
        "submits_jobs": False,
    }


def _validate_live_bundle(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_dir = run_root / "protocol"
    manifest_path = protocol_dir / "launch_manifest.json"
    submission_path = protocol_dir / "architecture_submission_record.json"
    required = {
        "protocol": protocol_dir / "protocol_snapshot.md",
        "seed": protocol_dir / "seed_table.json",
        "formal_seed": protocol_dir / "formal_seed_audit_table.json",
        "matrix": protocol_dir / "smoke_matrix.json",
    }
    for path in (manifest_path, submission_path, *required.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Missing extension architecture prerequisite: {path}")
    manifest = json.loads(manifest_path.read_text())
    submission = json.loads(submission_path.read_text())
    expected_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "run_kind": "development_smoke",
        "dataset": "val",
        "trajectory_count": SMOKE_TRAJECTORY_COUNT,
        "arms": list(EXTENSION_ARMS),
        "formal_launch_authorized": False,
    }
    if any(manifest.get(field) != expected for field, expected in expected_manifest.items()):
        raise RuntimeError("Prepared root is not the exact fresh OC3/OC5 development smoke")
    digest_fields = {
        "protocol_sha256": required["protocol"],
        "seed_table_file_sha256": required["seed"],
        "formal_seed_audit_file_sha256": required["formal_seed"],
        "smoke_matrix_sha256": required["matrix"],
    }
    if any(manifest.get(field) != sha256_file(path) for field, path in digest_fields.items()):
        raise RuntimeError("Prepared extension architecture bundle digest mismatch")
    matrix = load_smoke_matrix(required["matrix"])
    if manifest.get("matrix") != matrix["rows"]:
        raise RuntimeError("Prepared extension smoke matrix was changed")
    state = repository_state()
    require_clean_repository(state)
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared root and live extension checkout use different commits")
    validate_environment_contract(manifest)

    backend = require_runtime_backend(submission)
    if runner_backend(manifest) != backend:
        raise RuntimeError("Architecture manifest/submission runner backend mismatch")
    active_job = os.environ.get("SLURM_JOB_ID")
    expected_submission = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root.resolve()),
        "launch_manifest_sha256": sha256_file(manifest_path),
    }
    if backend == "slurm":
        expected_submission["slurm_job_id"] = active_job
    if (backend == "slurm" and not active_job) or any(
        submission.get(field) != expected for field, expected in expected_submission.items()
    ):
        raise RuntimeError("Architecture gate is outside its recorded extension Slurm job")
    if backend == "direct":
        runtime_direct_provenance(
            run_root, stage="architecture_smoke", attempt_id=0, row_id=None,
            submission=submission, submission_path=submission_path,
        )
    return manifest, submission


def _run_architecture(report: dict[str, Any], run_root: Path) -> None:
    import jax

    from mme_vla_suite.policies import policy_config
    import mme_vla_suite.training.config as training_config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"Extension architecture smoke requires a GPU, got {jax.devices()}")
    manifest, submission = _validate_live_bundle(run_root)
    execution_provenance = (
        {
            "runner_backend": "direct",
            "runner": runtime_direct_provenance(
                run_root, stage="architecture_smoke", attempt_id=0, row_id=None,
                submission=submission, submission_path=run_root / "protocol/architecture_submission_record.json",
            ),
        }
        if runner_backend(submission) == "direct"
        else {"slurm_job_id": os.environ["SLURM_JOB_ID"]}
    )
    report.update(execution_provenance)
    checkpoint_dir = REPO / CHECKPOINT_RELATIVE
    checkpoint = checkpoint_identity(checkpoint_dir)
    content_tree = checkpoint_content_tree_identity(checkpoint_dir)
    expected_checkpoint = {
        "checkpoint_archive_sha256_expected": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
        "checkpoint_archive_sha256_actual": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
    }
    if any(manifest.get(field) != value for field, value in expected_checkpoint.items()):
        raise RuntimeError("Live checkpoint differs from the prepared extension smoke")
    if any(
        submission.get(field) != value
        for field, value in expected_checkpoint.items()
        if field.startswith("checkpoint_")
    ):
        raise RuntimeError("Architecture submission used a different checkpoint")

    seeds, seed_contract = _base._load_seed_configuration(run_root)
    policy = policy_config.create_trained_policy(
        training_config.get_config("mme_vla_suite"), checkpoint_dir, seed=_base.POLICY_SEED
    )
    capture = _base._capture_memory_prepare(policy)
    cases = report.setdefault("cases", [])
    seen_dtypes: set[tuple[str, ...]] = set()
    cache_after = {"vision": 0, "memory": 0, "sample": 0}
    for arm in EXTENSION_ARMS:
        for history_length in HISTORY_LENGTHS:
            first = _base._run_once(
                policy,
                capture,
                arm=arm,
                history_length=history_length,
                seeds=seeds,
                seed_table_contract=seed_contract,
            )
            repeat = _base._run_once(
                policy,
                capture,
                arm=arm,
                history_length=history_length,
                seeds=seeds,
                seed_table_contract=seed_contract,
            )
            stable, specialization, next_cache, dtype_contract = _base._compile_cache_contract(
                first,
                repeat,
                previous_after=cache_after,
                seen_component_dtypes=seen_dtypes,
            )
            expected_selected = select_indices(
                arm,
                history_length - 1,
                boundary_indices=first["visible_boundary_indices"],
            )
            case = {
                "arm": arm,
                "history_length": history_length,
                "first": first,
                "repeat": repeat,
                "same_selected_indices": first["selected_frame_indices"] == repeat["selected_frame_indices"],
                "selected_indices_match_extension_selector": first["selected_frame_indices"] == expected_selected,
                "same_memory_tensor_digest": first["final_memory_tensor_sha256"]
                == repeat["final_memory_tensor_sha256"],
                "same_live_process_action_digest_audit": first["action_sha256"] == repeat["action_sha256"],
                "released_shape_match": first["component_shapes"]
                == [list(shape) for shape in _base.EXPECTED_COMPONENT_SHAPES]
                and repeat["component_shapes"] == [list(shape) for shape in _base.EXPECTED_COMPONENT_SHAPES],
                "released_dtype_match": first["component_dtypes"]
                == list(released_prepared_component_dtypes(first["valid_frame_count"]))
                and repeat["component_dtypes"] == list(released_prepared_component_dtypes(repeat["valid_frame_count"])),
                "released_action_shape_match": first["action_shape"] == list(_base.EXPECTED_ACTION_SHAPE)
                and repeat["action_shape"] == list(_base.EXPECTED_ACTION_SHAPE),
                "released_action_dtype_match": first["action_dtype"] == _base.EXPECTED_ACTION_DTYPE
                and repeat["action_dtype"] == _base.EXPECTED_ACTION_DTYPE,
                "compile_cache_stable_after_first_inference": stable,
                "compile_cache_matches_released_dtype_specializations": specialization,
            }
            case["passed"] = all(
                value
                for key, value in case.items()
                if key not in {"arm", "history_length", "first", "repeat", "passed"}
            )
            cases.append(case)
            if not case["passed"]:
                raise RuntimeError(f"Extension architecture case failed: {arm}/{history_length}")
            cache_after = next_cache
            seen_dtypes.add(dtype_contract)

    if submission.get("runtime_profile", {}).get("policy_lifetime") == "resident":
        # Revisit A only after exercising every other arm/shape B. This checks
        # non-adjacent reuse, not a comparison of development success rates.
        original = cases[0]["repeat"]
        revisited = _base._run_once(
            policy, capture, arm=cases[0]["arm"], history_length=cases[0]["history_length"],
            seeds=seeds, seed_table_contract=seed_contract,
        )
        comparisons = {
            field: original[field] == revisited[field]
            for field in ("selected_frame_indices", "final_memory_tensor_sha256", "action_sha256")
        }
        stable = all(revisited["compile_cache"][f"{component}_before"] == revisited["compile_cache"][f"{component}_after"]
                     for component in ("vision", "memory", "sample"))
        report["resident_cross_arm_reset"] = {
            "pattern": "A -> all other arm/shape cases -> A", "comparisons": comparisons,
            "no_recompile": stable, "revisited": revisited, "passed": all(comparisons.values()) and stable,
        }
        if not report["resident_cross_arm_reset"]["passed"]:
            raise RuntimeError("Resident cross-arm reset changed actions/memory or recompiled")
    policy.reset()
    final_reset = _base._reset_snapshot(policy)
    if not final_reset["passed"]:
        raise RuntimeError("Extension architecture final policy reset failed")
    report.update(
        {
            "repository_commit_sha": manifest["repository"]["commit_sha"],
            "repo_dirty": False,
            "checkpoint_relative_path": str(CHECKPOINT_RELATIVE),
            "checkpoint_archive_sha256_expected": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
            "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
            "checkpoint_content_tree_algorithm": content_tree["algorithm"],
            "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
            **execution_provenance,
            "device_count": jax.device_count(),
            "devices": [str(device) for device in jax.devices()],
            "case_count": len(cases),
            "stable_compile_cache": {
                "vision": cache_after["vision"],
                "perceptual_memory": cache_after["memory"],
                "sample_actions": cache_after["sample"],
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
        print(json.dumps(dry_run_contract(), indent=2, sort_keys=True))
        return
    if args.run_root is None:
        parser.error("--run-root is required unless --dry-run is used")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Prepared extension smoke root does not exist: {run_root}")
    architecture_dir = run_root / "architecture_smoke"
    if architecture_dir.exists():
        raise FileExistsError(f"Refusing to reuse extension architecture output: {architecture_dir}")
    architecture_dir.mkdir()
    report_path = architecture_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": _base._utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "formal_started": False,
        "arms": list(EXTENSION_ARMS),
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
        submission = json.loads((run_root / "protocol/architecture_submission_record.json").read_text())
        validate_architecture_pass_report(
            report, run_root=run_root, architecture_submission=submission, require_direct_completion=False
        )
    except BaseException as exc:
        failure = exc
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    report["finished_utc"] = _base._utc_now()
    _base._write_once_json(report_path, report)
    print(report_path)
    if failure is not None:
        raise failure
    if not report["passed"] or len(report["cases"]) != CASE_COUNT:
        raise RuntimeError(f"Extension architecture report is incomplete: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
