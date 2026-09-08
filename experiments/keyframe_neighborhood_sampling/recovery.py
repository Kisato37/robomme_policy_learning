"""Incident-specific, immutable cross-run recovery; no scientific outcome filtering."""
# ruff: noqa: PLC0415, SLF001
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.keyframe_neighborhood_sampling import aggregate_formal as audit
from experiments.keyframe_neighborhood_sampling import direct_provenance as direct
from experiments.keyframe_neighborhood_sampling.formal_matrix import FORMAL_TRAJECTORY_COUNT
from experiments.keyframe_neighborhood_sampling.record_launcher_failure import validate_extension_failure_record
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError, EpisodeAttemptWriter, RunArtifactStore
from experiments.keyframe_oracle_sampling.artifacts import canonical_json_bytes, read_jsonl, sha256_file, utc_now

PARENT_COMMIT = "8cd8376c56f983525be601bfbf5386736e4754c9"
PARENT_RUN = "20260908T003900Z_8cd8376_lighthouse_formal_v1"
INCIDENT_ROWS = {1352, 1354, 1447}
PLAN_NAME = "recovery_plan.json"
ALLOWLIST = {
    "submit_direct.py", "resident_policy.py", "direct_provenance.py", "submit_formal.py",
    "formal_artifacts.py", "recovery.py", "aggregate_recovery.py", "RECOVERY_2026-09-08.md",
}
SAME_INPUTS = (
    "protocol_sha256", "formal_matrix_sha256", "seed_table_file_sha256", "seed_table_entries_sha256",
    "environment_lock_sha256", "environment_locks", "checkpoint_path",
    "checkpoint_unpacked_metadata_sha256", "checkpoint_content_tree_algorithm",
    "checkpoint_unpacked_content_tree_sha256", "frozen_analysis_source_sha256",
    "frozen_aggregator_source_sha256", "dataset", "trajectory_count", "max_steps",
    "executed_action_horizon", "evaluation_policy_seed", "matrix", "reference_per_episode_sha256",
    "reference_summary_sha256", "reference_completeness_sha256", "runner_backend",
)


def load(path: Path) -> dict:
    return audit._load_json_object(path)


def digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def source_contract(parent: dict, child: dict, repo: Path) -> dict:
    if parent["repository"]["commit_sha"] != PARENT_COMMIT:
        raise ArtifactContractError("Recovery is authorized only for the reviewed parent commit")
    changed_inputs = [k for k in SAME_INPUTS if parent.get(k) != child.get(k)]
    if changed_inputs:
        raise ArtifactContractError(f"Recovery changed frozen scientific/environment inputs: {changed_inputs}")
    commit = child["repository"]["commit_sha"]
    subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_COMMIT, commit], cwd=repo, check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only", PARENT_COMMIT, commit], cwd=repo, text=True).splitlines()
    unexpected = [p for p in changed if not p.startswith("tests/") and
                  not (p.startswith("experiments/keyframe_neighborhood_sampling/") and Path(p).name in ALLOWLIST)]
    if unexpected:
        raise ArtifactContractError(f"Non-repair source changes require separate review: {unexpected}")
    audit._repository_provenance(repo, formal_commit=commit, launch=child)
    for field in ("frozen_analysis_source", "frozen_aggregator_source"):
        raw = subprocess.check_output(["git", "show", f"{PARENT_COMMIT}:{parent[field + '_relative']}"], cwd=repo)
        if hashlib.sha256(raw).hexdigest() != parent[field + "_sha256"]:
            raise ArtifactContractError("Parent frozen source differs from its recorded commit")
    return {"parent_commit": PARENT_COMMIT, "recovery_commit": commit,
            "changed_paths": changed, "scientific_inputs_identical": True,
            "frozen_analysis_and_aggregator_unchanged": True}


def inventory(root: Path) -> dict[str, str]:
    """Seal audit evidence, including failures/logs; never mutate or copy old data."""
    files = []
    for directory in ("protocol", "trajectories", "direct", "failures"):
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise ArtifactContractError(f"Recovery evidence cannot be a symlink: {path}")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".txt", ".md", ".out", ".err"}:
                files.append(path)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(files)}


def envelope(path: Path) -> dict:
    return {"backend": "direct", "dispatch_path": str(path),
            "dispatch_sha256": sha256_file(path), "dispatch": load(path)}


def audit_closed_sessions(root: Path) -> None:
    for path in sorted((root / "direct/sessions").glob("*/dispatch.json")):
        direct.audit_direct_completion(envelope(path), root, required_roles={"policy"})


def audited_results(root: Path, repo: Path, *, authorized_rows: list[int], progress: bool = False):
    """Reuse strict production validators, returning original unmodified records."""
    launch, matrix, seeds, _, provenance = audit.validate_extension_protocol_bundle(root, repo)
    _, authorizations = audit.validate_submission_attempt(root, attempt_id=0,
                                                         expected_row_ids=authorized_rows, launch=launch)
    store = RunArtifactStore(root)
    completed = store.scan_completed_keys()
    expected = audit._expected_formal_rows(matrix)
    failures = read_jsonl(store.failures_path) if store.failures_path.exists() else []
    attempts = audit.discover_attempts(root, store)
    by_row = {}
    for index, (key, result_path) in enumerate(sorted(completed.items(), key=lambda item: expected[item[0]]["row_id"])):
        if key not in expected or expected[key]["row_id"] not in authorized_rows:
            raise ArtifactContractError("Result outside authorized recovery/parent rows")
        result = load(result_path)
        attempt = result["attempt_id"]
        if type(attempt) is not int or attempt not in {0, 1, 2}:
            raise ArtifactContractError("Invalid result attempt")
        row = expected[key]
        writer = EpisodeAttemptWriter(store.attempt_dir(key, attempt), key, attempt)
        if result_path.resolve() != writer.result_path.resolve() or (key, attempt) not in attempts:
            raise ArtifactContractError("Result is not a canonical discovered attempt")
        if attempt:
            retry_rows = sorted(f["formal_matrix_row_id"] for f in failures if f["attempt_id"] == attempt - 1)
            _, retry_authorizations = audit.validate_submission_attempt(root, attempt_id=attempt,
                                                                        expected_row_ids=retry_rows, launch=launch)
            authorization = retry_authorizations[row["row_id"]]
        else:
            authorization = authorizations[row["row_id"]]
        audit._validate_attempt_submission_binding(load(writer.manifest_path), row_id=row["row_id"],
                                                   authorization=authorization, run_root=root)
        report, latencies, manifest, result = audit.audit_extension_attempt(
            writer, expected_key=key, expected_row=row, launch=launch, seed_payload=seeds)
        by_row[row["row_id"]] = {
            "key": key, "writer": writer, "report": report, "latencies": latencies,
            "manifest": manifest, "result": result, "initial": load(writer.initial_conditions_path),
        }
        if progress and (index + 1) % 50 == 0:
            print(f"STRICT_AUDIT {root.name}: {index + 1}/{len(completed)}", flush=True)
    return by_row, {"launch": launch, "provenance": provenance, "attempts": attempts, "failures": failures}


def audit_incident(root: Path, context: dict, completed: set[int]) -> dict:
    """Verify the exact reviewed incident; no generic waiver of missing receipts."""
    paths = sorted((root / "direct").glob("attempt_*/row_*/dispatch.json"))
    dispatched = {}
    for path in paths:
        dispatch = load(path)
        row = dispatch["row_id"]
        if row in dispatched or dispatch["attempt_id"] != 0 or dispatch["stage"] != "formal":
            raise ArtifactContractError("Parent contains unexpected duplicate/retry dispatch")
        dispatched[row] = path
    if set(dispatched) - completed != INCIDENT_ROWS or len(completed) != 1396:
        raise ArtifactContractError("Parent census differs from the reviewed 1396 + 3 incident")
    failures = {f["formal_matrix_row_id"]: f for f in context["failures"]}
    if len(context["failures"]) != 2 or set(failures) != {1352, 1447}:
        raise ArtifactContractError("Parent failure ledger differs from reviewed incident")
    for row, failure in failures.items():
        validate_extension_failure_record(root, failure, expected_row_id=row, require_direct_completion=False)
    oom = failures[1447]
    if not (oom["error_type"] == "AcceleratorError" and "CUDA error: out of memory" in oom["error"]
            and oom["classification"] == "hard_stop" and oom["retry_allowed"] is False
            and oom["environment_setup_completed"] is False and oom["scientific_actions_started"] is False):
        raise ArtifactContractError("Reviewed pre-action initialization OOM evidence differs")
    interrupted = failures[1352]
    if not (interrupted["classification"] == "infrastructure" and interrupted["retry_allowed"] is True
            and interrupted["evaluator_exit_status"] == 143):
        raise ArtifactContractError("Reviewed signal interruption evidence differs")
    # There must be exactly one parent attempt per completed row and two ledger
    # failures; row 1354 never reached evaluator/attempt creation.
    if len(context["attempts"]) != 1398 or any(a != 0 for _, a in context["attempts"]):
        raise ArtifactContractError("Unexpected parent attempt inventory")
    evidence = []
    for row in sorted(INCIDENT_ROWS):
        path = dispatched[row]
        env = envelope(path)
        manifest = failures[row] if row in failures else {"runner_backend": "direct", "runner": env}
        dispatch = direct.validate_direct_attempt(manifest, root, attempt_id=0, row_id=row, trajectory_kind="formal")
        starts = sorted(path.parent.glob("*_start.json"))
        roles = {p.stem.removesuffix("_start") for p in starts}
        expected_roles = {"preflight", "policy"} if row == 1354 else {"preflight", "policy", "evaluator", "reconcile"}
        if roles != expected_roles:
            raise ArtifactContractError("Reviewed interrupted process inventory differs")
        for role in roles:
            _, exited = direct._validated_role_evidence(dispatch, path, role)
            if exited["wall_clock_limit_reached"]:
                raise ArtifactContractError("A deadline failure is not this reviewed incident")
        if row != 1352:
            if (path.parent / "resident_reset.json").exists():
                raise ArtifactContractError("Reviewed pre-reset interruption unexpectedly has reset evidence")
            error = load(path.parent / "controller_error.json")
            expected_error = "InterruptedError" if row == 1354 else "FileNotFoundError"
            if error["error_type"] != expected_error:
                raise ArtifactContractError("Reviewed interrupted controller error differs")
        else:
            direct.audit_direct_completion(env, root, required_roles=expected_roles)
        evidence.append({"row_id": row, "dispatch_sha256": sha256_file(path),
                         "original_failure": failures.get(row), "global_retry_offset": 1})
    return {"dispatched_count": len(dispatched), "interrupted_rows": evidence,
            "undispatched_rows": sorted(set(range(FORMAL_TRAJECTORY_COUNT)) - set(dispatched)),
            "manual_review_override": "RECOVERY_2026-09-08.md; user explicitly authorized initialization-OOM recovery"}


def audit_parent(root: Path, repo: Path, *, progress: bool = False):
    if root.name != PARENT_RUN or (root / "aggregate").exists():
        raise ArtifactContractError("Not the reviewed interrupted parent run")
    before = inventory(root)
    audit_closed_sessions(root)
    results, context = audited_results(root, repo, authorized_rows=list(range(FORMAL_TRAJECTORY_COUNT)), progress=progress)
    incident = audit_incident(root, context, set(results))
    after = inventory(root)
    if before != after:
        raise ArtifactContractError("Parent evidence changed during read-only audit")
    report = {"passed": True, "strict_result_count": len(results), "retained_row_ids": sorted(results),
              "missing_row_ids": sorted(set(range(FORMAL_TRAJECTORY_COUNT)) - set(results)),
              "inventory": after, "inventory_sha256": digest(after), "incident": incident}
    return results, context, report


def validate_recovery_plan(root: Path, *, verify_parent: bool = False) -> dict:
    path = root / "protocol" / PLAN_NAME
    plan = load(path)
    launch = load(root / "protocol/launch_manifest.json")
    if (plan.get("schema") != "keyframe-neighborhood-recovery-v1" or plan.get("run_root") != str(root.resolve())
            or plan.get("launch_manifest_sha256") != sha256_file(root / "protocol/launch_manifest.json")
            or plan.get("repository_commit_sha") != launch["repository"]["commit_sha"]
            or plan.get("authorized") is not True):
        raise ArtifactContractError("Recovery plan lacks exact launch/source/authorization binding")
    parent = Path(plan["parent_root"])
    report = plan["parent_audit"]
    retained, missing = report["retained_row_ids"], report["missing_row_ids"]
    if (parent.name != PARENT_RUN or parent.resolve() == root.resolve() or report.get("passed") is not True
            or len(retained) != 1396 or len(missing) != 204
            or any(type(i) is not int for i in retained + missing)
            or sorted(retained + missing) != list(range(FORMAL_TRAJECTORY_COUNT))
            or plan["row_ids"] != missing or report["inventory_sha256"] != digest(report["inventory"])
            or plan["global_retry_offsets"] != {str(i): int(i in INCIDENT_ROWS) for i in missing}):
        raise ArtifactContractError("Recovery plan is not the exact reviewed missing complement")
    if verify_parent and inventory(parent) != report["inventory"]:
        raise ArtifactContractError("Sealed parent changed; recovery is not authorized")
    return plan


def initial_rows(root: Path, *, verify_parent: bool = False) -> list[int]:
    if not (root / "protocol" / PLAN_NAME).exists():
        return list(range(FORMAL_TRAJECTORY_COUNT))
    return validate_recovery_plan(root, verify_parent=verify_parent)["row_ids"]


def validate_retry_budget(root: Path, rows: list[int], attempt: int) -> None:
    if (root / "protocol" / PLAN_NAME).exists():
        plan = validate_recovery_plan(root)
        for row in rows:
            if row not in plan["row_ids"] or attempt + plan["global_retry_offsets"][str(row)] > 2:
                raise ArtifactContractError("Recovery row exceeds its original global retry allowance")


def bind_recovery(root: Path, parent: Path, repo: Path) -> Path:
    root, parent, repo = root.resolve(), parent.resolve(), repo.resolve()
    if (root / "protocol" / PLAN_NAME).exists() or (root / "trajectories").exists():
        raise ArtifactContractError("Bind recovery only once, before any execution")
    if list((root / "protocol").glob("submission_plan*.json")):
        raise ArtifactContractError("Cannot change a submitted run into recovery")
    child, *_ = audit.validate_extension_protocol_bundle(root, repo)
    _, context, report = audit_parent(parent, repo, progress=True)
    source = source_contract(context["launch"], child, repo)
    plan = {"schema": "keyframe-neighborhood-recovery-v1", "authorized": True,
            "recorded_utc": utc_now(), "run_root": str(root), "parent_root": str(parent),
            "repository_commit_sha": child["repository"]["commit_sha"],
            "launch_manifest_sha256": sha256_file(root / "protocol/launch_manifest.json"),
            "repair_source": source, "parent_audit": report, "row_ids": report["missing_row_ids"],
            "global_retry_offsets": {str(i): int(i in INCIDENT_ROWS) for i in report["missing_row_ids"]}}
    path = root / "protocol" / PLAN_NAME
    write_once_record(path, plan)
    validate_recovery_plan(root)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorize-reviewed-recovery", action="store_true")
    args = parser.parse_args()
    if not args.authorize_reviewed_recovery:
        parser.error("Explicit reviewed recovery authorization is required")
    path = bind_recovery(args.run_root, args.parent_root, args.repo_root)
    print(json.dumps({"recovery_plan": str(path), "retained": 1396, "remaining": 204}), flush=True)


if __name__ == "__main__":
    main()
