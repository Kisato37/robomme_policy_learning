#!/usr/bin/env python3
"""Submit the GPU architecture gate and bind it to one prepared run root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.keyframe_oracle_sampling.artifacts import atomic_write_json
from experiments.keyframe_oracle_sampling.artifacts import load_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from experiments.keyframe_oracle_sampling.artifacts import (
    validate_smoke_formal_seed_disjointness,
)
from experiments.keyframe_oracle_sampling.environment_contract import (
    validate_environment_contract,
)
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_RELATIVE
from experiments.keyframe_oracle_sampling.prepare_smoke import CHECKPOINT_CONTENT_TREE_ALGORITHM
from experiments.keyframe_oracle_sampling.prepare_smoke import EXPECTED_CHECKPOINT_ARCHIVE_SHA256
from experiments.keyframe_oracle_sampling.prepare_smoke import REPO
from experiments.keyframe_oracle_sampling.prepare_smoke import checkpoint_identity
from experiments.keyframe_oracle_sampling.prepare_smoke import repository_state
from experiments.keyframe_oracle_sampling.prepare_smoke import require_clean_repository
from mme_vla_suite.shared.keyframe_oracle_sampling import (
    FORMAL_SEED_DATASET,
    FORMAL_SEED_SCOPE,
    SMOKE_SEED_DATASET,
    SMOKE_SEED_SCOPE,
)

SBATCH_PATH = Path(__file__).with_name("run_architecture_smoke.sbatch")


def build_submission(run_root: Path) -> tuple[list[str], dict]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Smoke run root does not exist: {run_root}")
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    seed_path = run_root / "protocol" / "seed_table.json"
    formal_seed_audit_path = run_root / "protocol" / "formal_seed_audit_table.json"
    record_path = run_root / "protocol" / "architecture_submission_record.json"
    report_dir = run_root / "architecture_smoke"
    for path in (manifest_path, seed_path, formal_seed_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing architecture prerequisite: {path}")
    if record_path.exists():
        raise FileExistsError(f"Refusing duplicate architecture submission: {record_path}")
    if report_dir.exists():
        raise FileExistsError(f"Refusing to reuse architecture output: {report_dir}")

    state = repository_state()
    require_clean_repository(state)
    manifest = json.loads(manifest_path.read_text())
    validate_environment_contract(manifest)
    seed_payload, _ = load_seed_table(
        seed_path,
        expected_scope=SMOKE_SEED_SCOPE,
        expected_dataset=SMOKE_SEED_DATASET,
    )
    formal_seed_payload, _ = load_seed_table(
        formal_seed_audit_path,
        expected_scope=FORMAL_SEED_SCOPE,
        expected_dataset=FORMAL_SEED_DATASET,
    )
    validate_smoke_formal_seed_disjointness(seed_payload, formal_seed_payload)
    if (
        manifest.get("seed_table_file_sha256") != sha256_file(seed_path)
        or manifest.get("formal_seed_audit_file_sha256")
        != sha256_file(formal_seed_audit_path)
    ):
        raise RuntimeError("Prepared seed-table file digest mismatch")
    if (
        manifest.get("seed_table_scope") != seed_payload["scope"]
        or manifest.get("seed_table_dataset") != seed_payload["dataset"]
        or manifest.get("seed_table_derivation") != seed_payload["derivation"]
        or manifest.get("seed_table_entries_sha256") != seed_payload["entries_sha256"]
    ):
        raise RuntimeError("Prepared manifest and smoke seed-table contract differ")
    if manifest.get("repository", {}).get("commit_sha") != state["commit_sha"]:
        raise RuntimeError("Prepared run root and current checkout use different commits")
    if manifest.get("checkpoint_path") != str(CHECKPOINT_RELATIVE):
        raise RuntimeError("Prepared run root does not use the frozen checkpoint path")
    if (
        manifest.get("checkpoint_archive_sha256_expected")
        != EXPECTED_CHECKPOINT_ARCHIVE_SHA256
    ):
        raise RuntimeError("Prepared run root has the wrong checkpoint provenance hash")
    if (
        manifest.get("checkpoint_archive_sha256_actual")
        != EXPECTED_CHECKPOINT_ARCHIVE_SHA256
    ):
        raise RuntimeError("Prepared run root did not hash the actual checkpoint archive")
    checkpoint = checkpoint_identity(REPO / CHECKPOINT_RELATIVE)
    if (
        manifest.get("checkpoint_unpacked_metadata_sha256")
        != checkpoint["metadata_sha256"]
    ):
        raise RuntimeError("Prepared and live checkpoint identities differ")
    content_tree_algorithm = manifest.get("checkpoint_content_tree_algorithm")
    content_tree_sha256 = manifest.get("checkpoint_unpacked_content_tree_sha256")
    if (
        content_tree_algorithm != CHECKPOINT_CONTENT_TREE_ALGORITHM
        or not isinstance(content_tree_sha256, str)
        or len(content_tree_sha256) != 64
        or any(value not in "0123456789abcdef" for value in content_tree_sha256)
    ):
        raise RuntimeError("Prepared run root lacks a valid checkpoint content-tree identity")

    log_dir = run_root / "slurm"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={REPO}",
        f"--export=KEYFRAME_REPO_ROOT={REPO}",
        f"--output={log_dir}/architecture-%j.out",
        f"--error={log_dir}/architecture-%j.err",
        str(SBATCH_PATH),
        str(run_root),
    ]
    record = {
        "schema_version": 1,
        "submitted_utc": utc_now(),
        "repository_commit_sha": state["commit_sha"],
        "run_root": str(run_root),
        "checkpoint_unpacked_metadata_sha256": checkpoint["metadata_sha256"],
        "checkpoint_content_tree_algorithm": content_tree_algorithm,
        "checkpoint_unpacked_content_tree_sha256": content_tree_sha256,
        "command": command,
    }
    return command, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    command, record = build_submission(run_root)
    if args.dry_run:
        print(json.dumps({**record, "submits_jobs": False}, indent=2, sort_keys=True))
        return

    (run_root / "slurm").mkdir(parents=True, exist_ok=True)
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    record["slurm_job_id"] = job_id
    atomic_write_json(
        run_root / "protocol" / "architecture_submission_record.json",
        record,
    )
    print(job_id)


if __name__ == "__main__":
    main()
