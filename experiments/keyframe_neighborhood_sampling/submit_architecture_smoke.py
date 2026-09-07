#!/usr/bin/env python3
"""Submit the fresh OC3/OC5 real-checkpoint architecture gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_ARMS
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.prepare_formal import repository_state
from experiments.keyframe_neighborhood_sampling.prepare_formal import require_clean_repository
from experiments.keyframe_neighborhood_sampling.smoke_matrix import SMOKE_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.smoke_matrix import load_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import PROTOCOL_VERSION
from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import validate_smoke_formal_seed_disjointness
from experiments.keyframe_oracle_sampling.environment_contract import validate_environment_contract
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import EXPECTED_CHECKPOINT_ARCHIVE_SHA256
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_content_tree_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import FORMAL_SEED_SCOPE
from mme_vla_suite.shared.keyframe_oracle_sampling import MAX_POLICY_CALLS
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_DATASET
from mme_vla_suite.shared.keyframe_oracle_sampling import SMOKE_SEED_SCOPE

SBATCH_PATH = Path(__file__).with_name("run_architecture_smoke.sbatch")


def build_submission(run_root: Path) -> tuple[list[str], dict]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Extension smoke root does not exist: {run_root}")
    protocol_dir = run_root / "protocol"
    manifest_path = protocol_dir / "launch_manifest.json"
    matrix_path = protocol_dir / "smoke_matrix.json"
    record_path = protocol_dir / "architecture_submission_record.json"
    report_dir = run_root / "architecture_smoke"
    for path in (
        manifest_path,
        matrix_path,
        protocol_dir / "seed_table.json",
        protocol_dir / "formal_seed_audit_table.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing extension architecture prerequisite: {path}")
    if record_path.exists() or report_dir.exists():
        raise FileExistsError("Refusing duplicate or reused extension architecture gate")

    state = repository_state()
    require_clean_repository(state)
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "run_kind": "development_smoke",
        "dataset": "val",
        "trajectory_count": SMOKE_TRAJECTORY_COUNT,
        "arms": list(EXTENSION_ARMS),
        "formal_launch_authorized": False,
        "smoke_matrix_sha256": sha256_file(matrix_path),
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise RuntimeError("Prepared root is not the exact fresh OC3/OC5 smoke")
    if manifest.get("matrix") != load_smoke_matrix(matrix_path)["rows"]:
        raise RuntimeError("Prepared extension smoke matrix differs from its manifest")
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared extension smoke and live checkout use different commits")
    validate_environment_contract(manifest)

    smoke_seed_path = protocol_dir / "seed_table.json"
    formal_seed_path = protocol_dir / "formal_seed_audit_table.json"
    smoke_seed, smoke_lookup = load_seed_table(
        smoke_seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_seed, formal_lookup = load_seed_table(
        formal_seed_path,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    if set(smoke_lookup) != {(task, 0, call) for task in FORMAL_TASKS for call in range(MAX_POLICY_CALLS)} or set(
        formal_lookup
    ) != {(task, episode, call) for task in FORMAL_TASKS for episode in range(50) for call in range(MAX_POLICY_CALLS)}:
        raise RuntimeError("Extension architecture gate seed universes are incomplete")
    disjointness = validate_smoke_formal_seed_disjointness(smoke_seed, formal_seed)
    seed_fields = {
        "seed_table_file_sha256": sha256_file(smoke_seed_path),
        "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
        "seed_table_scope": smoke_seed["scope"],
        "seed_table_dataset": smoke_seed["dataset"],
        "seed_table_derivation": smoke_seed["derivation"],
        "seed_table_entries_sha256": smoke_seed["entries_sha256"],
        "seed_disjointness_audit": disjointness,
    }
    if any(manifest.get(field) != value for field, value in seed_fields.items()):
        raise RuntimeError("Extension architecture seed provenance mismatch")

    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    content_tree = checkpoint_content_tree_identity(REPO / CHECKPOINT_RELATIVE)
    checkpoint_fields = {
        "checkpoint_archive_sha256_expected": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
        "checkpoint_archive_sha256_actual": EXPECTED_CHECKPOINT_ARCHIVE_SHA256,
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": content_tree["algorithm"],
        "checkpoint_unpacked_content_tree_sha256": content_tree["content_tree_sha256"],
    }
    if any(manifest.get(field) != value for field, value in checkpoint_fields.items()):
        raise RuntimeError("Prepared extension checkpoint differs from live frozen bytes")

    log_dir = run_root / "slurm"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={REPO}",
        f"--export=KEYFRAME_NEIGHBORHOOD_REPO_ROOT={REPO}",
        f"--output={log_dir}/architecture-%j.out",
        f"--error={log_dir}/architecture-%j.err",
        str(SBATCH_PATH),
        str(run_root),
    ]
    record = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        "submitted_utc": utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "arms": list(EXTENSION_ARMS),
        "formal_started": False,
        "formal_launch_authorized": False,
        "launch_manifest_sha256": sha256_file(manifest_path),
        **checkpoint_fields,
        "command": command,
    }
    return command, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-authorized-extension-smoke-v1", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_authorized_extension_smoke_v1:
        parser.error(
            "live submission requires --confirm-authorized-extension-smoke-v1; "
            "use --dry-run for non-submitting validation"
        )
    command, record = build_submission(args.run_root)
    if args.dry_run:
        print(json.dumps({**record, "submits_jobs": False}, indent=2, sort_keys=True))
        return

    manifest = json.loads((args.run_root / "protocol/launch_manifest.json").read_text())
    if manifest.get("runner_backend") == "direct":
        parser.error("This root requires submit_direct, not a Slurm submission")

    (args.run_root / "slurm").mkdir(parents=True, exist_ok=True)
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    record["slurm_job_id"] = job_id
    atomic_write_json(args.run_root / "protocol/architecture_submission_record.json", record)
    print(job_id)


if __name__ == "__main__":
    main()
