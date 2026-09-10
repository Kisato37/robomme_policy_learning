from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.keyframe_oracle_sampling.artifacts import SMOKE_INITIAL_CONDITION_HASH_FIELDS, sha256_file
from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion.artifacts import ExpansionArtifactError, ExpansionRunStore

_spec = importlib.util.spec_from_file_location("expansion_artifact_trace_fixture", Path(__file__).with_name("test_trace_validation.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def provenance():
    return {"code_commit": "a" * 40, "benchmark_commit": "b" * 40,
            "protocol_sha256": "c" * 64, "environment_manifest_sha256": "d" * 64,
            "checkpoint_archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
            "host": "synthetic-cpu-host", "hardware": {"device": "fixture-no-gpu"},
            "command": ["cpu-fixture", "not-a-launch"]}


def create_store(tmp_path, stage="smoke"):
    return ExpansionRunStore.create(tmp_path / "uniform_keyframe_expansion" / "synthetic-run",
                                    stage=stage, run_manifest=provenance())


def raw_initial(writer, *, count=64):
    archive = io.BytesIO()
    np.savez_compressed(archive, front=np.zeros((count, 2, 2, 3), np.uint8),
                        wrist=np.zeros((count, 2, 2, 3), np.uint8), robot_state=np.zeros((count, 8), np.float32),
                        current_task_index=np.asarray([int(index >= 17) + int(index >= 35) for index in range(count)], np.int64))
    writer.write_attachment("initial_observations.npz", archive.getvalue())
    writer.write_attachment("initial_task_state.json", b'{"type":"dict","items":[]}')
    writer.write_attachment("initial_task_instruction.json", b'"synthetic task"')


def initial(writer, *, count=64, seed=5):
    raw_initial(writer, count=count)
    hashes = {field: hashlib.sha256(field.encode()).hexdigest() for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS}
    env = {"resolved_environment_seed": seed, "difficulty": "synthetic", "dataset": writer.row["dataset"]}
    reset = {"policy_seed": 7, "memory_cleared": True, "policy_rng_reset": True,
             "reset_prefix_frame_count": count, "reset_prefix_stage_count": count,
             "reset_prefix_frames_sha256": hashes["front_observations_sha256"], "reset_prefix_stages_sha256": "b" * 64}
    writer.record_initial_conditions(hashes, environment_provenance=env, reset_evidence=reset)
    return hashes, env, reset


def trace(writer, call=0, *, count=64):
    return _fixtures.synthetic_trace(writer.row["arm"], step=count - 1 + call * 16,
                                      boundaries=(0, 17, 35), call=call, task=writer.row["task"],
                                      split=writer.row["dataset"], episode=writer.row["episode_id"])


def result(reason="success", *, steps=1, calls=1, official=True):
    return {"terminal_reason": reason, "success": reason == "success", "environment_steps": steps,
            "policy_call_count": calls, "official_terminal": official, "reset_verified": True,
            "terminal_metadata": {"official_stop_flag": official, "source": "synthetic"}}


def finish(writer, reason="success", *, steps=1, calls=1):
    initial(writer)
    for call in range(calls):
        writer.append_trace(trace(writer, call))
    payload = result(reason, steps=steps, calls=calls)
    if reason == "error":
        payload.update(benchmark_error_message="synthetic benchmark failure", benchmark_exception_type="FixtureBenchmarkError")
    return writer.finalize_scientific(payload)


def test_run_preparation_is_new_write_once_scoped_and_configuration_bound(tmp_path):
    store = create_store(tmp_path)
    assert store.stage == "smoke"
    assert store.manifest["creation_scope"] == "artifact_preparation_not_launch_authorization"
    assert store.completeness()["expected_count"] == 48
    assert store.completeness()["completed_count"] == 0
    assert ExpansionRunStore.open(store.run_root).manifest == store.manifest
    with pytest.raises(FileExistsError):
        create_store(tmp_path)
    with pytest.raises(ExpansionArtifactError):
        ExpansionRunStore.create(tmp_path / "old-family" / "run", stage="smoke", run_manifest=provenance())
    snapshot = store.run_root / "protocol/inference_contract.json"
    payload = json.loads(snapshot.read_text())
    payload["frame_capacity"] = 32
    snapshot.write_text(json.dumps(payload))
    with pytest.raises(ExpansionArtifactError):
        ExpansionRunStore.open(store.run_root)


@pytest.mark.parametrize("key,value", [("checkpoint_archive_sha256", "e" * 64), ("code_commit", "main"),
                                      ("host", ""), ("hardware", {}), ("command", "shell string")])
def test_bad_run_provenance_rejected_before_root_creation(tmp_path, key, value):
    metadata = {**provenance(), key: value}
    root = tmp_path / "uniform_keyframe_expansion" / "bad"
    with pytest.raises(ExpansionArtifactError):
        ExpansionRunStore.create(root, stage="smoke", run_manifest=metadata)
    assert not root.exists()


def test_attempt_reservation_precedes_setup_and_cannot_be_reused_or_skipped(tmp_path):
    store = create_store(tmp_path)
    row = c.build_smoke_matrix()["rows"][0]
    writer = store.new_attempt(row, 0)
    assert writer.manifest_path.is_file()
    assert not writer.initial_conditions_path.exists()
    assert store.audit_attempt(row, 0)["status"] == "incomplete"
    for attempt in (0, 1, 2, 3, True):
        with pytest.raises((ExpansionArtifactError, c.ExpansionContractError, FileExistsError)):
            store.new_attempt(row, attempt)
    assert not store.audit_attempt(row, 0)["mid_episode_resume_allowed"]
    bad = deepcopy(row)
    bad["arm"] = "OC"
    with pytest.raises(c.ExpansionContractError):
        store.new_attempt(bad, 0)


def test_two_concurrent_reservations_cannot_claim_the_same_attempt(tmp_path):
    store = create_store(tmp_path)
    row = c.build_smoke_matrix()["rows"][0]
    def reserve():
        try:
            store.new_attempt(row, 0)
            return "created"
        except (ExpansionArtifactError, FileExistsError):
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: reserve(), range(2))) == ["blocked", "created"]


def test_initial_raw_evidence_required_and_bound_before_trace(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    with pytest.raises(ExpansionArtifactError):
        writer.append_trace(trace(writer))
    hashes, env, reset = initial(writer)
    payload = json.loads(writer.initial_conditions_path.read_text())
    assert len(payload["raw_initial_attachment_sha256"]) == 3
    with pytest.raises(FileExistsError):
        writer.record_initial_conditions(hashes, environment_provenance=env, reset_evidence=reset)
    writer.append_trace(trace(writer))
    assert store.audit_attempt(writer.row, 0)["status"] == "incomplete"
    (writer.attempt_dir / "attachments/initial_task_instruction.json").write_bytes(b'"changed"')
    with pytest.raises(ExpansionArtifactError):
        store.audit_attempt(writer.row, 0)


@pytest.mark.parametrize("mutation", ["missing_hash", "wrong_seed", "alignment", "false_reset", "raw_count"])
def test_invalid_initial_evidence_not_published(tmp_path, mutation):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    raw_initial(writer, count=64)
    hashes = {field: "a" * 64 for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS}
    env = {"resolved_environment_seed": 4, "difficulty": "synthetic", "dataset": "val"}
    reset = {"policy_seed": 7, "memory_cleared": True, "policy_rng_reset": True,
             "reset_prefix_frame_count": 64, "reset_prefix_stage_count": 64,
             "reset_prefix_frames_sha256": "a" * 64, "reset_prefix_stages_sha256": "b" * 64}
    if mutation == "missing_hash":
        hashes.pop(next(iter(hashes)))
    elif mutation == "wrong_seed":
        reset["policy_seed"] = 8
    elif mutation == "alignment":
        reset["reset_prefix_stage_count"] = 63
    elif mutation == "false_reset":
        reset["policy_rng_reset"] = False
    else:
        reset["reset_prefix_frame_count"] = reset["reset_prefix_stage_count"] = 65
    with pytest.raises(ExpansionArtifactError):
        writer.record_initial_conditions(hashes, environment_provenance=env, reset_evidence=reset)
    assert not writer.initial_conditions_path.exists()


@pytest.mark.parametrize("reason", ["success", "fail", "timeout", "error"])
def test_scientific_outcomes_all_close_cell_without_retry(tmp_path, reason):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    payload = finish(writer, reason)
    audit = store.audit_attempt(writer.row, 0)
    assert audit["status"] == "complete" and audit["retry_allowed"] is False
    assert audit["smoke_readiness_pass"] is (reason != "error")
    assert payload["success"] is (reason == "success")
    assert writer.row["row_id"] in store.completed_rows()
    for action in (lambda: store.new_attempt(writer.row, 1),
                   lambda: writer.append_trace(trace(writer, 1)),
                   lambda: writer.record_failure("infrastructure", "transport", "fake retry", {"phase": "fixture"}),
                   lambda: writer.finalize_scientific(result())):
        with pytest.raises(ExpansionArtifactError):
            action()


def test_infrastructure_before_setup_has_maximum_two_fresh_attempts(tmp_path):
    store = create_store(tmp_path)
    row = c.build_smoke_matrix()["rows"][0]
    for attempt in range(3):
        writer = store.new_attempt(row, attempt)
        failure = writer.record_failure("infrastructure", "transport", "synthetic connection lost", {"phase": "client_connect"})
        assert failure["retry_allowed"] is (attempt < 2)
        assert store.audit_attempt(row, attempt)["status"] == "infrastructure_failure"
        assert not writer.initial_conditions_path.exists()
    with pytest.raises(ExpansionArtifactError):
        store.new_attempt(row, 3)
    assert store.completeness()["completed_count"] == 0


def test_hard_stop_is_never_automatically_retryable(tmp_path):
    store = create_store(tmp_path)
    row = c.build_smoke_matrix()["rows"][0]
    writer = store.new_attempt(row, 0)
    writer.record_failure("hard_stop", "capacity_overflow", "synthetic invariant", {"capacity": 48, "requested": 49})
    assert store.audit_attempt(row, 0)["retry_allowed"] is False
    with pytest.raises(ExpansionArtifactError):
        store.new_attempt(row, 1)
    with pytest.raises(ExpansionArtifactError, match="Run-wide"):
        store.new_attempt(c.build_smoke_matrix()["rows"][1], 0)


def test_live_trace_append_does_not_redecode_big_initial_archive_but_finalize_rechecks(tmp_path, monkeypatch):
    import experiments.uniform_keyframe_expansion.artifacts as module
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    original_load = module.np.load
    calls = []
    def counted_load(*args, **kwargs):
        calls.append(str(args[0]))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(module.np, "load", counted_load)
    initial(writer)
    assert len(calls) == 1
    writer.append_trace(trace(writer))
    writer.append_trace(trace(writer, 1))
    assert len(calls) == 1
    writer.finalize_scientific(result(steps=17, calls=2))
    assert len(calls) == 2
    store.audit_attempt(writer.row, 0)
    assert len(calls) == 3


def test_live_initial_cache_cannot_hide_raw_tampering_at_finalization(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    initial(writer)
    writer.append_trace(trace(writer))
    (writer.attempt_dir / "attachments/initial_task_instruction.json").write_bytes(b'"changed"')
    writer.append_trace(trace(writer, 1))  # No enormous raw-data reread each call.
    with pytest.raises(ExpansionArtifactError, match="raw initial"):
        writer.finalize_scientific(result(steps=17, calls=2))
    assert not writer.result_path.exists()


def test_first_trace_boundaries_must_match_actual_archived_reset_stages(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    initial(writer)
    wrong = _fixtures.synthetic_trace(writer.row["arm"], step=63, boundaries=(0,),
                                       task=writer.row["task"], split=writer.row["dataset"], episode=0)
    with pytest.raises(ExpansionArtifactError, match="raw reset"):
        writer.append_trace(wrong)


def test_retry_checks_actual_initial_seed_hash_and_history_not_just_config(tmp_path):
    store = create_store(tmp_path)
    row = c.build_smoke_matrix()["rows"][0]
    first = store.new_attempt(row, 0)
    initial(first)
    first.record_failure("infrastructure", "transport", "lost", {"phase": "infer"})
    second = store.new_attempt(row, 1)
    with pytest.raises(ExpansionArtifactError, match="retry environment"):
        initial(second, seed=6)
    assert not second.initial_conditions_path.exists()


def test_wrong_trace_context_seed_missing_duplicate_call_and_suffix_rejected(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][1], 0)
    initial(writer)
    bad = trace(writer)
    bad["selector_seed"] += 1
    with pytest.raises(ValueError):
        writer.append_trace(bad)
    assert list((writer.attempt_dir / "traces").iterdir()) == []
    writer.append_trace(trace(writer))
    with pytest.raises(ValueError):
        writer.append_trace(trace(writer))
    with pytest.raises(ExpansionArtifactError, match="count/step"):
        writer.finalize_scientific(result(steps=17, calls=2))
    assert not writer.result_path.exists()


def test_missing_last_trace_after_finalization_is_detected(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    finish(writer, steps=17, calls=2)
    (writer.attempt_dir / "traces/call_001.json").unlink()
    with pytest.raises(ExpansionArtifactError, match="census"):
        store.audit_attempt(writer.row, 0)


def test_tampered_result_or_ledger_cannot_change_outcome_or_authorize_retry(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    finish(writer)
    modified = json.loads(writer.result_path.read_text())
    modified["success"], modified["terminal_reason"] = False, "fail"
    writer.result_path.write_text(json.dumps(modified))
    with pytest.raises(ExpansionArtifactError, match="checksum"):
        store.audit_attempt(writer.row, 0)


def test_short_limit_and_early_official_timeout_rules(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    initial(writer)
    writer.append_trace(trace(writer))
    bad = result("timeout", steps=1, calls=1)
    bad["terminal_metadata"]["official_stop_flag"] = False
    with pytest.raises(ExpansionArtifactError, match="Early timeout"):
        writer.finalize_scientific(bad)
    writer.finalize_scientific(result("timeout", steps=1, calls=1))
    short = store.new_attempt(c.build_smoke_matrix()["rows"][1], 0)
    initial(short)
    for call in range(4):
        short.append_trace(trace(short, call))
    short.finalize_scientific(result("short_limit", steps=64, calls=4, official=False))
    assert store.audit_attempt(short.row, 0)["status"] == "complete"


def test_attachment_and_video_publication_are_write_once_and_path_scoped(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    digest = writer.write_attachment("notes.json", b"{}")
    assert digest == hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(FileExistsError):
        writer.write_attachment("notes.json", b"changed")
    for name in ("../escape", "/absolute", "sub/file", "\\escape", ".hidden", ".."):
        with pytest.raises(ExpansionArtifactError):
            writer.write_attachment(name, b"bad")
    staged = writer.attempt_dir / ".video_staging/rollout.mp4"
    staged.write_bytes(b"synthetic-mp4-content")
    video_hash = writer.publish_video(staged)
    assert not staged.exists()
    assert sha256_file(writer.attempt_dir / "rollout.mp4") == video_hash
    staged.write_bytes(b"new-video")
    with pytest.raises(FileExistsError):
        writer.publish_video(staged)
    assert staged.read_bytes() == b"new-video"


def test_complete_result_covers_raw_recording_and_attachment_bytes(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    initial(writer)
    writer.append_trace(trace(writer))
    staged = writer.attempt_dir / ".video_staging/rollout.mp4"
    staged.write_bytes(b"fixture-recording")
    sha = writer.publish_video(staged)
    payload = {**result(), "video_filename": "rollout.mp4", "video_sha256": sha}
    final = writer.finalize_scientific(payload)
    assert final["artifact_sha256"]["rollout.mp4"] == sha
    assert "attachments/initial_observations.npz" in final["artifact_sha256"]
    assert store.audit_attempt(writer.row, 0)["status"] == "complete"
    with pytest.raises(ExpansionArtifactError):
        writer.write_attachment("late.json", b"{}")


def test_failed_attempt_ledger_missing_or_rebound_is_not_retry_authority(tmp_path):
    store = create_store(tmp_path)
    writer = store.new_attempt(c.build_smoke_matrix()["rows"][0], 0)
    writer.record_failure("infrastructure", "node", "lost node", {"node_id": "synthetic"})
    ledger = store.run_root / "failures/failure_ledger.jsonl"
    ledger.write_text(ledger.read_text() + ledger.read_text())
    with pytest.raises(ExpansionArtifactError, match="exactly one"):
        store.new_attempt(writer.row, 1)
