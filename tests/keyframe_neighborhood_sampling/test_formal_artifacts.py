# ruff: noqa: SLF001
# Tests intentionally exercise the evaluator's private routing seam.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from experiments.keyframe_neighborhood_sampling import formal_artifacts
from experiments.keyframe_neighborhood_sampling.formal_artifacts import EXTENSION_PROTOCOL_VERSION
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_formal_prepared_root
from experiments.keyframe_neighborhood_sampling.formal_artifacts import validate_prepared_formal_root
from experiments.keyframe_neighborhood_sampling.formal_matrix import EXTENSION_PROTOCOL_FAMILY
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import FORMAL_TASKS
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import build_seed_table
from experiments.keyframe_oracle_sampling.artifacts import build_smoke_seed_table
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.formal_matrix import build_formal_matrix as build_original_formal_matrix

REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _patch_clean_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_output(command, **_kwargs):
        return "commit\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(formal_artifacts.subprocess, "check_output", fake_check_output)


def _prepared_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    _patch_clean_repository(monkeypatch)
    repo_root = tmp_path
    run_root = repo_root / "runs" / "keyframe_neighborhood_sampling" / "extension-run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)

    protocol_snapshot = protocol / "protocol_snapshot.md"
    seed_path = protocol / "seed_table.json"
    development_seed_path = protocol / "development_seed_audit_table.json"
    matrix_path = protocol / "formal_matrix.json"
    architecture_path = protocol / "architecture_pass_report.json"
    architecture_submission_path = protocol / "architecture_submission_record.json"
    development_smoke_path = protocol / "development_smoke_audit.json"
    analysis_source = repo_root / formal_artifacts.ANALYSIS_SOURCE_RELATIVE
    aggregator_source = repo_root / formal_artifacts.AGGREGATOR_SOURCE_RELATIVE

    protocol_snapshot.write_text("frozen extension protocol")
    analysis_source.parent.mkdir(parents=True, exist_ok=True)
    analysis_source.write_text("# frozen analysis source\n")
    aggregator_source.write_text("# frozen aggregator source\n")
    seed = build_seed_table(FORMAL_TASKS, range(50))
    _write_json(seed_path, seed)
    _write_json(development_seed_path, {"scope": "extension-development-audit"})
    matrix = build_formal_matrix()
    _write_json(matrix_path, matrix)
    checkpoint = {
        "checkpoint_unpacked_metadata_sha256": "c" * 64,
        "checkpoint_content_tree_algorithm": "sha256-canonical-file-content-tree-v1",
        "checkpoint_unpacked_content_tree_sha256": "d" * 64,
    }
    identity = {
        "protocol_version": EXTENSION_PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
    }
    _write_json(
        architecture_path,
        {
            **identity,
            **checkpoint,
            "passed": True,
            "repository_commit_sha": "commit",
        },
    )
    _write_json(
        architecture_submission_path,
        {**identity, **checkpoint, "repository_commit_sha": "commit"},
    )
    _write_json(
        development_smoke_path,
        {
            **identity,
            "passed": True,
            "formal_started": False,
            "repository_commit_sha": "commit",
        },
    )
    manifest = {
        **identity,
        **checkpoint,
        "run_kind": "formal",
        "dataset": "test",
        "trajectory_count": 1600,
        "formal_launch_authorized": True,
        "selector_seed_table_role": "randomsamp_policy_call_rng_audit_only",
        "repository": {"commit_sha": "commit"},
        "protocol_sha256": sha256_file(protocol_snapshot),
        "seed_table_file_sha256": sha256_file(seed_path),
        "development_seed_audit_file_sha256": sha256_file(development_seed_path),
        "formal_matrix_sha256": sha256_file(matrix_path),
        "architecture_report_sha256": sha256_file(architecture_path),
        "architecture_submission_record_sha256": sha256_file(architecture_submission_path),
        "development_smoke_audit_sha256": sha256_file(development_smoke_path),
        "frozen_analysis_source_relative": formal_artifacts.ANALYSIS_SOURCE_RELATIVE.as_posix(),
        "frozen_analysis_source_sha256": sha256_file(analysis_source),
        "frozen_aggregator_source_relative": formal_artifacts.AGGREGATOR_SOURCE_RELATIVE.as_posix(),
        "frozen_aggregator_source_sha256": sha256_file(aggregator_source),
        "reference_per_episode_sha256": formal_artifacts.REFERENCE_PER_EPISODE_SHA256,
        "reference_summary_sha256": formal_artifacts.REFERENCE_SUMMARY_SHA256,
        "reference_completeness_sha256": formal_artifacts.REFERENCE_COMPLETENESS_SHA256,
        "reference_cross_run_verification": {
            "pairing_key": ["task", "episode_id"],
            "not_directly_verifiable_from_published_reference": [
                "environment_seed",
                "difficulty",
                "raw_initial_condition_hashes",
            ],
        },
        "seed_table_scope": seed["scope"],
        "seed_table_dataset": seed["dataset"],
        "seed_table_derivation": seed["derivation"],
        "seed_table_entries_sha256": seed["entries_sha256"],
        "matrix": matrix["rows"],
    }
    _write_json(protocol / "launch_manifest.json", manifest)
    return run_root, seed_path, manifest


def test_extension_prepared_root_requires_exact_matrix_seed_universe_and_root(tmp_path, monkeypatch):
    run_root, seed_path, manifest = _prepared_root(tmp_path, monkeypatch)
    assert validate_formal_prepared_root(run_root, seed_path, tmp_path) == manifest


def test_extension_prepared_root_rejects_changed_frozen_analysis_source(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_root(tmp_path, monkeypatch)
    (tmp_path / formal_artifacts.ANALYSIS_SOURCE_RELATIVE).write_text("# changed after preparation\n")
    with pytest.raises(ArtifactContractError, match="frozen analysis source digest mismatch"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)


def test_extension_prepared_root_rejects_ambiguous_selector_seed_role(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_root(tmp_path, monkeypatch)
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selector_seed_table_role"] = "environment_seed_evidence"
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="RandomSamp policy-call RNG evidence"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)


def test_extension_prepared_root_rejects_changed_oc_completeness_binding(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_root(tmp_path, monkeypatch)
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reference_completeness_sha256"] = "0" * 64
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="OC reference digest binding mismatch"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)


def test_extension_prepared_root_rejects_nonexact_formal_commit(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_root(tmp_path, monkeypatch)
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["repository"]["commit_sha"] = "ancestor-commit"
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="Live repository commit differs"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)


def test_extension_prepared_root_rejects_original_family_and_matrix(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_root(tmp_path, monkeypatch)
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_family"] = "keyframe_oracle_sampling_v1"
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="protocol family mismatch"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)

    manifest["protocol_family"] = EXTENSION_PROTOCOL_FAMILY
    matrix_path = run_root / "protocol" / "formal_matrix.json"
    original_matrix = build_original_formal_matrix()
    _write_json_replace(matrix_path, original_matrix)
    manifest["formal_matrix_sha256"] = sha256_file(matrix_path)
    manifest["matrix"] = original_matrix["rows"]
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="frozen 16 x 50 x 2"):
        validate_formal_prepared_root(run_root, seed_path, tmp_path)


def _write_json_replace(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def test_extension_submission_authorization_is_bound_to_exact_1600_row_plan(tmp_path, monkeypatch):
    run_root, seed_path, manifest = _prepared_root(tmp_path, monkeypatch)
    shards = []
    for shard_id, start in enumerate((0, 1000)):
        rows = list(range(start, min(start + 1000, 1600)))
        local = list(range(len(rows)))
        shards.append(
            {
                "shard_id": shard_id,
                "shard_count": 2,
                "array_task_ids": local,
                "row_ids": rows,
                "array": f"0-{local[-1]}%1",
                "command": ["sbatch", str(shard_id)],
            }
        )
    identity = {
        "protocol_version": EXTENSION_PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
    }
    plan_path = run_root / "protocol" / "submission_plan.json"
    _write_json(
        plan_path,
        {
            **identity,
            "attempt_id": 0,
            "trajectory_count": 1600,
            "shard_count": 2,
            "max_rows_per_array": 1000,
            "shards": shards,
        },
    )
    active = shards[0]
    submission_path = run_root / "protocol" / "submission_record_shard_00.json"
    _write_json(
        submission_path,
        {
            **identity,
            **active,
            "attempt_id": 0,
            "trajectory_count": len(active["row_ids"]),
            "repository_commit_sha": "commit",
            "formal_launch_authorized": True,
            "launch_manifest_sha256": sha256_file(run_root / "protocol" / "launch_manifest.json"),
            "formal_matrix_sha256": manifest["formal_matrix_sha256"],
            "submission_plan_sha256": sha256_file(plan_path),
            "slurm_array_job_id": "extension-123",
        },
    )
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "extension-123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    assert (
        validate_prepared_formal_root(
            run_root,
            seed_path,
            tmp_path,
            attempt_id=0,
            authorization_digest=sha256_file(submission_path),
        )
        == manifest
    )
    with pytest.raises(ArtifactContractError, match="authorization digest"):
        validate_prepared_formal_root(
            run_root,
            seed_path,
            tmp_path,
            attempt_id=0,
            authorization_digest="0" * 64,
        )


def _prepared_smoke_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    _patch_clean_repository(monkeypatch)
    run_root = tmp_path / "runs" / "keyframe_neighborhood_sampling" / "smoke-run"
    protocol = run_root / "protocol"
    protocol.mkdir(parents=True)
    protocol_snapshot = protocol / "protocol_snapshot.md"
    seed_path = protocol / "seed_table.json"
    formal_seed_path = protocol / "formal_seed_audit_table.json"
    matrix_path = protocol / "smoke_matrix.json"
    protocol_snapshot.write_text("frozen extension smoke protocol")
    seed = build_smoke_seed_table(FORMAL_TASKS, [0])
    formal_seed = build_seed_table(FORMAL_TASKS, range(50))
    matrix = build_smoke_matrix()
    _write_json(seed_path, seed)
    _write_json(formal_seed_path, formal_seed)
    _write_json(matrix_path, matrix)
    identity = {
        "protocol_version": EXTENSION_PROTOCOL_VERSION,
        "protocol_family": EXTENSION_PROTOCOL_FAMILY,
    }
    disjointness = {
        "smoke_seed_count": 16 * 82,
        "formal_seed_count": 16 * 50 * 82,
        "intersection_count": 0,
    }
    manifest = {
        **identity,
        "run_kind": "development_smoke",
        "dataset": "val",
        "trajectory_count": 48,
        "formal_launch_authorized": False,
        "repository": {"commit_sha": "commit"},
        "protocol_sha256": sha256_file(protocol_snapshot),
        "seed_table_file_sha256": sha256_file(seed_path),
        "formal_seed_audit_file_sha256": sha256_file(formal_seed_path),
        "smoke_matrix_sha256": sha256_file(matrix_path),
        "seed_table_scope": seed["scope"],
        "seed_table_dataset": seed["dataset"],
        "seed_table_derivation": seed["derivation"],
        "seed_table_entries_sha256": seed["entries_sha256"],
        "seed_disjointness_audit": disjointness,
        "matrix": matrix["rows"],
    }
    manifest_path = protocol / "launch_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        protocol / "submission_record.json",
        {
            **identity,
            "repository_commit_sha": "commit",
            "attempt_id": 0,
            "smoke_launch_authorized": True,
            "formal_launch_authorized": False,
            "launch_manifest_sha256": sha256_file(manifest_path),
            "smoke_matrix_sha256": manifest["smoke_matrix_sha256"],
            "slurm_array_job_id": "smoke-123",
            "trajectory_count": 48,
            "row_ids": list(range(48)),
        },
    )
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "smoke-123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    return run_root, seed_path, manifest


def test_extension_smoke_prepared_root_binds_48_rows_val_seeds_and_submission(tmp_path, monkeypatch):
    run_root, seed_path, manifest = _prepared_smoke_root(tmp_path, monkeypatch)
    assert formal_artifacts.validate_prepared_smoke_root(run_root, seed_path, tmp_path, attempt_id=0) == manifest


def test_extension_smoke_rejects_original_family_or_old_matrix(tmp_path, monkeypatch):
    run_root, seed_path, _ = _prepared_smoke_root(tmp_path, monkeypatch)
    manifest_path = run_root / "protocol" / "launch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_family"] = "keyframe_oracle_sampling_v1"
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="protocol family mismatch"):
        formal_artifacts.validate_prepared_smoke_root(run_root, seed_path, tmp_path, attempt_id=0)

    manifest["protocol_family"] = EXTENSION_PROTOCOL_FAMILY
    matrix_path = run_root / "protocol" / "smoke_matrix.json"
    original_matrix_path = REPO / "experiments/keyframe_oracle_sampling/SMOKE_MATRIX.json"
    matrix_path.write_bytes(original_matrix_path.read_bytes())
    manifest["smoke_matrix_sha256"] = sha256_file(matrix_path)
    manifest["matrix"] = json.loads(matrix_path.read_text()).get("rows")
    _write_json_replace(manifest_path, manifest)
    with pytest.raises(ArtifactContractError, match="extension smoke matrix"):
        formal_artifacts.validate_prepared_smoke_root(run_root, seed_path, tmp_path, attempt_id=0)


def _load_eval_module(monkeypatch: pytest.MonkeyPatch):
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.websocket_client_policy = types.SimpleNamespace(MMEVLAWebsocketClientPolicy=object)
    utils = types.ModuleType("utils")
    utils.pack_buffer = lambda *args, **kwargs: None
    utils.check_args = lambda _args: None
    utils.TASK_NAME_LIST = []
    utils.TASK_WITH_VIDEO_DEMO = set()
    utils.SUBGOAL_TYPES = set()
    utils.EpisodeState = type("EpisodeState", (), {})
    utils.RolloutRecorder = type("RolloutRecorder", (), {})
    env_runner = types.ModuleType("env_runner")
    env_runner.EnvRunner = type("EnvRunner", (), {})
    subgoal = types.ModuleType("subgoal_predictor")
    subgoal.build_subgoal_predictor = lambda *args, **kwargs: None
    subgoal.SubgoalPredictorBase = type("SubgoalPredictorBase", (), {})
    records = types.ModuleType("evaluation_records")
    records.EpisodeResultWriter = type("EpisodeResultWriter", (), {})
    for name, module in (
        ("openpi_client", openpi_client),
        ("utils", utils),
        ("env_runner", env_runner),
        ("subgoal_predictor", subgoal),
        ("evaluation_records", records),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_extension_eval_routing_test", REPO / "examples/robomme/eval.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_evaluator_routes_old_and_extension_formal_contracts_without_crossing(
    monkeypatch,
):
    evaluator = _load_eval_module(monkeypatch)
    old_prepared, old_row = evaluator._keyframe_formal_validators("OC")
    new_prepared, new_row = evaluator._keyframe_formal_validators("OC3")
    assert old_prepared is evaluator.validate_prepared_formal_root
    assert old_row is evaluator.validate_formal_runtime_row_binding
    assert new_prepared is evaluator.validate_extension_prepared_formal_root
    assert new_row is evaluator.validate_extension_formal_runtime_row_binding
    assert evaluator._keyframe_formal_validators("OC5") == (new_prepared, new_row)


def test_evaluator_routes_extension_smoke_without_falling_back_to_old_contract(
    monkeypatch,
):
    evaluator = _load_eval_module(monkeypatch)
    old_prepared, old_row = evaluator._keyframe_smoke_validators("R")
    new_prepared, new_row = evaluator._keyframe_smoke_validators("OC5")
    assert old_prepared is evaluator.validate_prepared_smoke_root
    assert old_row is evaluator.validate_runtime_row_binding
    assert new_prepared is evaluator.validate_extension_prepared_smoke_root
    assert new_row is evaluator.validate_extension_smoke_runtime_row_binding
    assert evaluator._keyframe_smoke_validators("OC3") == (new_prepared, new_row)


def test_attempt_manifest_takes_protocol_identity_from_validated_manifest(monkeypatch):
    evaluator_module = _load_eval_module(monkeypatch)
    args = types.SimpleNamespace(
        keyframe_selector_arm="OC3",
        max_steps=1300,
        obs_horizon=16,
        model_seed=7,
        model_ckpt_id=79999,
        port=8011,
        keyframe_trajectory_kind="formal",
    )
    evaluator = types.SimpleNamespace(
        _validated_run_manifest={
            "protocol_version": EXTENSION_PROTOCOL_VERSION,
            "protocol_family": EXTENSION_PROTOCOL_FAMILY,
        },
        _seed_table_payload={"entries_sha256": "a" * 64},
    )
    env = types.SimpleNamespace(
        dataset="test",
        resolved_environment_seed=1,
        resolved_difficulty_hint="hard",
        difficulty="hard",
    )
    attempt = evaluator_module._keyframe_attempt_manifest(args, evaluator, env, environment_setup_completed=True)
    assert attempt["protocol_version"] == EXTENSION_PROTOCOL_VERSION
    assert attempt["protocol_family"] == EXTENSION_PROTOCOL_FAMILY

    args.keyframe_selector_arm = "OC"
    evaluator._validated_run_manifest = {"protocol_version": "v1.0"}
    original_attempt = evaluator_module._keyframe_attempt_manifest(
        args, evaluator, env, environment_setup_completed=True
    )
    assert original_attempt["protocol_version"] == "v1.0"
    assert "protocol_family" not in original_attempt
