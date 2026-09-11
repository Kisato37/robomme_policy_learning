"""CPU tests: full-census mapping fixtures plus real-store rejection checks.

The in-memory store below stubs *already audited* closures only to exercise all
2,400 normalized records/triples cheaply. It is not evidence that a synthetic
completion is an actual GPU result. Real-store tests separately verify that
ingestion delegates corrupted/missing/duplicate evidence to the artifact audit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.keyframe_oracle_sampling.artifacts import SMOKE_INITIAL_CONDITION_HASH_FIELDS
from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion import outcome_ingestion as oi
from experiments.uniform_keyframe_expansion.analysis import validate_outcome_records

_spec = importlib.util.spec_from_file_location("expansion_ingestion_artifact_fixtures", Path(__file__).with_name("test_artifacts.py"))
_artifact_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_artifact_fixtures)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class AuditedCensusFixture:
    """Small virtual byte files, not a replacement for production store audits."""

    def __init__(self, root):
        self.run_root = root / "uniform_keyframe_expansion" / "complete-synthetic"
        self.stage = "formal"
        self.rows = c.build_formal_matrix()["rows"]
        self.files, self.completed, self.audited = {}, {}, []
        self.audit_status = "complete"
        self.snapshots = {name: digest(self.put_json(name, {"synthetic": name})) for name in (
            "protocol/matrix.json", "protocol/seed_manifest.json", "protocol/inference_contract.json")}
        self.manifest = {
            "stage": "formal", "run_id": self.run_root.name,
            "snapshot_sha256": self.snapshots, "provenance": _artifact_fixtures.provenance(),
        }
        run_sha = digest(self.put_json("run_manifest.json", self.manifest))
        self.files[self.run_root / "completions/completion_ledger.jsonl"] = b"synthetic-audited-ledger\n"
        for row in self.rows:
            directory = self.attempt_dir(row, 0)
            episode = {"row": row, "run_manifest_sha256": run_sha,
                       "selector_config": {"arm": row["arm"], "intentionally_arm_specific_seed": row["row_id"]}}
            episode_sha = digest(self.put_json(directory / "episode_manifest.json", episode))
            hashes = {field: digest(f"{row['task']}/{row['episode_id']}/{field}".encode())
                      for field in SMOKE_INITIAL_CONDITION_HASH_FIELDS}
            initial = {
                "hashes": hashes,
                "environment_provenance": {"dataset": "test", "resolved_environment_seed": row["episode_id"] + 500001,
                                           "difficulty": "medium", "resolved_difficulty_hint": None},
                "reset_evidence": {
                    "policy_seed": 7, "memory_cleared": True, "policy_rng_reset": True,
                    "reset_prefix_frame_count": 64, "reset_prefix_stage_count": 64,
                    "reset_prefix_frames_sha256": hashes["front_observations_sha256"],
                    "reset_prefix_stages_sha256": "c" * 64,
                },
                "reset_prefix_boundary_indices": [0, 17, 35],
                "episode_manifest_sha256": episode_sha,
            }
            initial_sha = digest(self.put_json(directory / "initial_condition_hashes.json", initial))
            reason = ("success", "fail", "timeout", "error")[row["row_id"] % 4]
            result = {
                "protocol_family": c.PROTOCOL_FAMILY, "row": row, "attempt_id": 0,
                "terminal_reason": reason, "success": reason == "success", "official_terminal": True,
                "reset_verified": True, "environment_steps": 16 + row["row_id"],
                "policy_call_count": row["row_id"] % 82,
                "artifact_sha256": {"episode_manifest.json": episode_sha, "initial_condition_hashes.json": initial_sha},
            }
            path = directory / "episode_result.json"
            self.put_json(path, result)
            self.completed[row["row_id"]] = path

    def put_json(self, path, value):
        path = Path(path)
        if not path.is_absolute():
            path = self.run_root / path
        raw = c.canonical_json(value).encode()
        self.files[path] = raw
        return raw

    def json(self, path):
        return json.loads(self.files[Path(path)])

    def attempt_dir(self, row, attempt_id):
        return self.run_root / "trajectories" / f"row_{row['row_id']:04d}" / f"attempt_{attempt_id:02d}"

    def completed_rows(self):
        return self.completed

    def audit_attempt(self, row, attempt_id):
        self.audited.append((row["row_id"], attempt_id))
        return {"status": self.audit_status, "retry_allowed": False,
                "result": self.json(self.completed[row["row_id"]])}

    def change_initial(self, row_id, change):
        result_path = self.completed[row_id]
        path = result_path.parent / "initial_condition_hashes.json"
        value = self.json(path)
        change(value)
        initial_sha = digest(self.put_json(path, value))
        result = self.json(result_path)
        result["artifact_sha256"]["initial_condition_hashes.json"] = initial_sha
        self.put_json(result_path, result)

    def change_result(self, row_id, change):
        path = self.completed[row_id]
        value = self.json(path)
        change(value)
        self.put_json(path, value)


@pytest.fixture
def census(tmp_path, monkeypatch):
    fixture = AuditedCensusFixture(tmp_path)
    monkeypatch.setattr(oi.ExpansionRunStore, "open", lambda root: fixture)
    monkeypatch.setattr(oi, "_read_bytes", lambda path: fixture.files[Path(path)])
    return fixture


def test_full_three_arm_census_yields_bound_sources_and_only_initial_pairing(census):
    before = deepcopy(census.files)
    output = oi.ingest_formal_run(census.run_root)
    assert len(output["records"]) == len(output["sources"]) == output["study_cell_count"] == 2400
    assert {row["arm"] for row in output["records"]} == {"U", "UK48", "UN48"}
    assert {row["terminal_status"] for row in output["records"]} == {"success", "fail", "timeout", "error"}
    assert all(row["success"] is (row["terminal_status"] == "success") for row in output["records"])
    assert output["pairing_audit"]["paired_block_count"] == 800
    assert output["pairing_audit"]["all_three_arm_blocks_matched"] is True
    assert output["formal_statistics_executed"] is False
    assert "baseline_attestation" not in output
    assert census.audited == [(row["row_id"], 0) for row in census.rows]
    assert census.files == before  # no source rewriting or derived-file writes
    for record, source in zip(output["records"], output["sources"]):
        assert (record["task"], record["episode_id"], record["arm"]) == (source["task"], source["episode_id"], source["arm"])
        for path, sha in (("result_path", "result_sha256"), ("initial_conditions_path", "initial_conditions_sha256"),
                          ("episode_manifest_path", "episode_manifest_sha256")):
            assert digest(census.files[Path(source[path])]) == source[sha]
    assert output["normalized_records_sha256"] == c.canonical_sha256(output["records"])
    assert output["sources_sha256"] == c.canonical_sha256(output["sources"])
    assert output["pairing_audit_sha256"] == c.canonical_sha256(output["pairing_audit"])
    # Schema compatibility only. These synthetic rows do not constitute an
    # attestation and do not invoke any statistical or formal result function.
    assert validate_outcome_records(output["records"]).cell_count == 2400


@pytest.mark.parametrize("field", SMOKE_INITIAL_CONDITION_HASH_FIELDS)
def test_each_initial_hash_mismatch_rejects_the_entire_census(census, field):
    def mutate(initial):
        initial["hashes"][field] = "f" * 64
        if field == "front_observations_sha256":
            initial["reset_evidence"]["reset_prefix_frames_sha256"] = "f" * 64
    census.change_initial(1, mutate)
    with pytest.raises(oi.ExpansionIngestionError, match="U/UK48 BinFill episode 0 initial_condition_hashes"):
        oi.ingest_formal_run(census.run_root)


@pytest.mark.parametrize("field,value", [("resolved_environment_seed", 999), ("difficulty", "hard"),
                                         ("resolved_difficulty_hint", "hard")])
def test_actual_environment_pairing_mismatch_rejected(census, field, value):
    census.change_initial(1, lambda initial: initial["environment_provenance"].__setitem__(field, value))
    with pytest.raises(oi.ExpansionIngestionError, match="environment_provenance mismatch"):
        oi.ingest_formal_run(census.run_root)


def test_absent_hint_is_not_silently_equated_to_null(census):
    census.change_initial(1, lambda initial: initial["environment_provenance"].pop("resolved_difficulty_hint"))
    with pytest.raises(oi.ExpansionIngestionError, match="environment_provenance mismatch"):
        oi.ingest_formal_run(census.run_root)


@pytest.mark.parametrize("mutation", ["prefix_count", "prefix_stage_digest", "prefix_boundaries"])
def test_reset_prefix_pairing_mismatch_rejected(census, mutation):
    def change(initial):
        if mutation == "prefix_count":
            initial["reset_evidence"]["reset_prefix_frame_count"] = 65
            initial["reset_evidence"]["reset_prefix_stage_count"] = 65
        elif mutation == "prefix_stage_digest":
            initial["reset_evidence"]["reset_prefix_stages_sha256"] = "d" * 64
        else:
            initial["reset_prefix_boundary_indices"] = [0, 18, 35]
    census.change_initial(1, change)
    with pytest.raises(oi.ExpansionIngestionError, match="U/UK48 BinFill episode 0 reset_"):
        oi.ingest_formal_run(census.run_root)


@pytest.mark.parametrize("field,value", [("policy_seed", 8), ("memory_cleared", False), ("policy_rng_reset", False)])
def test_invalid_policy_reset_rejected_even_if_both_arms_agree(census, field, value):
    for row_id in (0, 1):
        census.change_initial(row_id, lambda initial: initial["reset_evidence"].__setitem__(field, value))
    with pytest.raises(oi.ExpansionIngestionError, match="initial reset"):
        oi.ingest_formal_run(census.run_root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_source", "wrong_path", "wrong_row", "noncomplete"])
def test_exact_census_source_and_completion_identity_required(census, mutation):
    if mutation == "missing":
        census.completed.pop(2399)
    elif mutation == "extra":
        census.completed[2400] = census.completed[0]
    elif mutation == "duplicate_source":
        census.completed[1] = census.completed[0]
    elif mutation == "wrong_path":
        source = census.completed[0]
        wrong = source.with_name("unbound_result.json")
        census.files[wrong] = census.files[source]
        census.completed[0] = wrong
    elif mutation == "wrong_row":
        census.change_result(0, lambda result: result["row"].__setitem__("episode_id", 49))
    else:
        census.audit_status = "infrastructure_failure"
    with pytest.raises(oi.ExpansionIngestionError):
        oi.ingest_formal_run(census.run_root)


@pytest.mark.parametrize("mutation", ["short", "infrastructure", "invariant", "unofficial", "wrong_success"])
def test_non_scientific_or_contradictory_result_rejected(census, mutation):
    def change(result):
        if mutation == "short":
            result["terminal_reason"] = "short_limit"
        elif mutation == "infrastructure":
            result["infrastructure_failure"] = True
        elif mutation == "invariant":
            result["protocol_invariant_failure"] = True
        elif mutation == "unofficial":
            result["official_terminal"] = False
        else:
            result["success"] = False  # row 0 has reason success
    census.change_result(0, change)
    with pytest.raises(oi.ExpansionIngestionError):
        oi.ingest_formal_run(census.run_root)


def test_unbound_initial_bytes_rejected(census):
    path = census.completed[0].parent / "initial_condition_hashes.json"
    census.files[path] += b" "  # decoded fields identical; original checksum is not
    with pytest.raises(oi.ExpansionIngestionError, match="initial_condition_hashes.json checksum"):
        oi.ingest_formal_run(census.run_root)


def test_ledger_change_during_ingestion_rejected(census, monkeypatch):
    original = census.audit_attempt
    def concurrent_change(row, attempt):
        if row["row_id"] == 1599:
            census.files[census.run_root / "completions/completion_ledger.jsonl"] += b"changed\n"
        return original(row, attempt)
    monkeypatch.setattr(census, "audit_attempt", concurrent_change)
    with pytest.raises(oi.ExpansionIngestionError, match="completion ledger changed"):
        oi.ingest_formal_run(census.run_root)


def test_real_smoke_store_is_never_a_formal_source(tmp_path):
    store = _artifact_fixtures.create_store(tmp_path)
    with pytest.raises(oi.ExpansionIngestionError, match="Smoke/development"):
        oi.ingest_formal_run(store.run_root)


def test_real_formal_empty_store_is_rejected_without_source_writes(tmp_path):
    store = _artifact_fixtures.create_store(tmp_path, stage="formal")
    before = {path: path.read_bytes() for path in store.run_root.rglob("*") if path.is_file()}
    with pytest.raises(oi.ExpansionIngestionError, match="2400-cell"):
        oi.ingest_formal_run(store.run_root)
    assert before == {path: path.read_bytes() for path in store.run_root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("corruption", ["trace", "resealed_invalid_trace", "duplicate_ledger", "initial_attachment", "result_hash"])
def test_real_store_corruption_is_rejected_before_census_admission(tmp_path, corruption):
    store = _artifact_fixtures.create_store(tmp_path, stage="formal")
    writer = store.new_attempt(c.build_formal_matrix()["rows"][0], 0)
    _artifact_fixtures.finish(writer)
    if corruption in ("trace", "resealed_invalid_trace"):
        path = writer.attempt_dir / "traces/call_000.json"
        payload = json.loads(path.read_text())
        payload["selected_frame_indices"] = []
        path.write_text(json.dumps(payload))
        if corruption == "resealed_invalid_trace":
            # Adversarial CPU fixture: valid checksums cannot make an invalid
            # selector trace scientifically admissible. No real result is edited.
            writer.trace_path.write_text(c.canonical_json(payload) + "\n")
            closure = json.loads(writer.result_path.read_text())
            for name in ("traces/call_000.json", "selector_trace.jsonl"):
                closure["artifact_sha256"][name] = digest((writer.attempt_dir / name).read_bytes())
            writer.result_path.write_text(c.canonical_json(closure))
            ledger_path = store.run_root / "completions/completion_ledger.jsonl"
            ledger = json.loads(ledger_path.read_text())
            ledger["result_sha256"] = digest(writer.result_path.read_bytes())
            ledger_path.write_text(c.canonical_json(ledger) + "\n")
    elif corruption == "duplicate_ledger":
        path = store.run_root / "completions/completion_ledger.jsonl"
        path.write_bytes(path.read_bytes() * 2)
    elif corruption == "initial_attachment":
        (writer.attempt_dir / "attachments/initial_task_instruction.json").write_bytes(b'"changed task"')
    else:
        writer.result_path.write_bytes(writer.result_path.read_bytes() + b" ")
    with pytest.raises(ValueError) as error:
        oi.ingest_formal_run(store.run_root)
    if corruption == "resealed_invalid_trace":
        assert "selected_frame_indices" in str(error.value)
