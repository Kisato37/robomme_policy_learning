"""Synthetic CPU byte archives only; never connects to the Athena source."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.uniform_keyframe_expansion import baseline_export as be


def seal(bundle):
    bundle["bundle_sha256"] = be.sha(be.canonical({k: v for k, v in bundle.items() if k != "bundle_sha256"}))
    return bundle


@pytest.fixture(scope="module")
def source_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("old-u-byte-fixture") / be.RUN_ID
    root.mkdir()
    seed_map = []
    failure_line = None
    for task_index, task in enumerate(be.TASKS):
        for episode in range(50):
            key = be.scientific_key(task, episode)
            attempt = 1 if (task_index, episode) == (0, 0) else 0
            relative = be.attempt_relative(task, episode, attempt)
            directory = root / relative
            directory.mkdir(parents=True)
            manifest = {**be.FIXED, "scientific_key": key, "attempt_id": attempt, "protocol_version": "v1.0",
                        "environment_setup_completed": True, "seed_table_sha256": be.SEED_TABLE_ENTRIES_SHA256,
                        "resolved_environment_seed": 500001 + task_index * 10000 + episode * 100,
                        "difficulty": "easy" if episode < 25 else "medium", "resolved_difficulty_hint": "fixture"}
            seed_map.append({"task": task, "episode_id": episode,
                             "resolved_environment_seed": manifest["resolved_environment_seed"], "difficulty": manifest["difficulty"]})
            # Noncanonical spacing/newline verifies that transport preserves
            # original bytes, not a parse/dump approximation.
            manifest_raw = json.dumps(manifest, indent=2).encode() + b"\n\n"
            initial = {name: be.sha(f"{task}/{episode}/{name}".encode()) for name in be.INITIAL_FIELDS}
            initial_raw = json.dumps(initial, indent=1).encode() + b"\n"
            first = {"task": task, "episode_id": episode, "selector_name": "U", "policy_call_index": 0,
                     "environment_step": 0, "history_length": 4, "current_history_index": 3,
                     "seed_table_sha256": be.SEED_TABLE_ENTRIES_SHA256, "seed_table_dataset": "test"}
            second = {**first, "policy_call_index": 1, "environment_step": 16,
                      "history_length": 20, "current_history_index": 19}
            trace_raw = be.canonical(first) + b"\n\n" + be.canonical(second) + b"\n"
            reason = ("success", "fail", "timeout", "error")[(task_index + episode) % 4]
            result = {**be.FIXED, "scientific_key": key, "attempt_id": attempt,
                      "task": task, "episode_id": episode, "selector_arm": "U",
                      "terminal_reason": reason, "success": reason == "success", "steps": 17,
                      "history_lengths_at_policy_calls": [4, 20],
                      "episode_manifest_sha256": be.sha(manifest_raw),
                      "initial_condition_hashes_sha256": be.sha(initial_raw), "selector_trace_sha256": be.sha(trace_raw)}
            if reason == "error":
                result.update(benchmark_error_message="synthetic official error", benchmark_exception_type="FixtureError")
            for name, raw in (("episode_manifest.json", manifest_raw), ("initial_condition_hashes.json", initial_raw),
                              ("selector_trace.jsonl", trace_raw), ("episode_result.json", be.canonical(result) + b"\n")):
                (directory / name).write_bytes(raw)
            if attempt:
                prior = root / be.attempt_relative(task, episode, 0)
                prior.mkdir()
                prior_raw = be.canonical({"scientific_key": key, "attempt_id": 0, "environment_setup_completed": False})
                (prior / "episode_manifest.json").write_bytes(prior_raw)
                failure_line = be.canonical({**key, "attempt_id": 0, "classification": "infrastructure",
                                             "retry_allowed": True, "episode_manifest_sha256": be.sha(prior_raw)}) + b"\n"
    (root / "failures").mkdir()
    (root / "failures/failure_ledger.jsonl").write_bytes(failure_line)
    (root / "protocol").mkdir()
    protocol_raw = b"# Original protocol fixture\n"
    (root / "protocol/protocol_snapshot.md").write_bytes(protocol_raw)
    launch = {"run_id": be.RUN_ID, "run_kind": "formal", "dataset": "test", "evaluation_policy_seed": 7,
              "max_steps": 1300, "protocol_sha256": be.sha(protocol_raw),
              "seed_table_entries_sha256": be.SEED_TABLE_ENTRIES_SHA256,
              "python_environments": {"fixture": "not-real-evidence"}}
    (root / "protocol/launch_manifest.json").write_bytes(be.canonical(launch))
    mapping_sha = be.sha(be.canonical(seed_map))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(be, "SOURCE_ROOT", str(root))
        patch.setattr(be, "EXPECTED_SEED_MAPPING_SHA256", mapping_sha)
        bundle = be.collect_baseline()
    return root, mapping_sha, bundle


@pytest.fixture
def fixture(source_fixture, monkeypatch):
    root, mapping_sha, bundle = source_fixture
    monkeypatch.setattr(be, "SOURCE_ROOT", str(root))
    monkeypatch.setattr(be, "EXPECTED_SEED_MAPPING_SHA256", mapping_sha)
    return root, deepcopy(bundle)


def change_json(bundle, filename, change, *, rebind=False, index=0):
    entry = bundle["entries"][index]
    value = be.parse(be.decode_blob(entry["files"][filename], filename), filename)
    change(value)
    entry["files"][filename] = be.blob(be.canonical(value))
    if rebind and filename != "episode_result.json":
        result = be.parse(be.decode_blob(entry["files"]["episode_result.json"], "result"), "result")
        key = {"episode_manifest.json": "episode_manifest_sha256", "initial_condition_hashes.json": "initial_condition_hashes_sha256"}[filename]
        result[key] = entry["files"][filename]["sha256"]
        entry["files"]["episode_result.json"] = be.blob(be.canonical(result))
    seal(bundle)


def test_full_collection_preserves_all_800_original_values_and_bytes(fixture):
    root, bundle = fixture
    report, files = be.validate_bundle(bundle)
    assert report["scientific_count"] == report["initial_hash_value_records"] == 800
    assert len(report["records"]) == 800
    assert set(report["outcome_counts"]) == {"success", "fail", "timeout", "error"}
    assert report["baseline_status"] == "unresolved" and report["launch_authorized"] is False
    assert "baseline_attestation" not in report
    assert report["retry_evidence"]["count"] == 1
    assert report["retry_evidence"]["historical_manifest_digest_present_count"] == 1
    assert "protocol/environment_manifest.json" in report["missing_optional_documents"]
    assert report["documents"]["protocol/launch_manifest.json"]["status"] == "present"
    for entry, record in zip(bundle["entries"], report["records"]):
        assert entry["scientific_key"] == record["scientific_key"]
        for filename in be.CORE_FILES:
            relative = entry["relative_directory"] + "/" + filename
            assert files[relative] == (root / relative).read_bytes()
        assert len(record["initial_hashes"]) == 5
        assert all(isinstance(value, str) and len(value) == 64 for value in record["initial_hashes"].values())
        assert entry["trace"]["policy_call_count"] == 2
    assert not any(path.endswith((".mp4", ".h5", ".safetensors")) for path in files)
    assert not any(path.endswith("/selector_trace.jsonl") for path in files)
    assert any("trace_first_lines/" in path for path in files)


def test_collect_read_only_and_ignores_other_arms_and_large_video(fixture):
    root, _ = fixture
    # Scoped fixture side files must not leak into the export. These are test
    # data, not another user's active job or real rollout.
    other = root / "trajectories/BinFill/episode_00/OC"
    other.mkdir(exist_ok=True)
    marker = other / "do-not-export.secret"
    marker.write_bytes(b"unrelated-fixture")
    video = root / "trajectories/BinFill/episode_00/U/attempt_01/rollout.mp4"
    video.write_bytes(b"not-a-real-video")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    bundle = be.collect_baseline()
    after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after
    assert b"unrelated-fixture" not in be.canonical(bundle)
    assert b"not-a-real-video" not in be.canonical(bundle)


@pytest.mark.parametrize("mutation", ["source_root", "other_arm", "missing", "duplicate", "path_traversal", "extra_file", "missing_initial"])
def test_source_scope_and_exact_800_census_fail_closed(fixture, mutation):
    _, bundle = fixture
    if mutation == "source_root":
        bundle["source_root"] = "/a/different/run"
    elif mutation == "other_arm":
        bundle["entries"][0]["scientific_key"]["arm"] = "OC"
    elif mutation == "missing":
        bundle["entries"].pop()
    elif mutation == "duplicate":
        bundle["entries"][-1] = deepcopy(bundle["entries"][0])
    elif mutation == "path_traversal":
        bundle["entries"][0]["relative_directory"] = "../../escape"
    elif mutation == "extra_file":
        bundle["entries"][0]["files"]["rollout.mp4"] = be.blob(b"notallowed")
    else:
        bundle["entries"][0]["files"].pop("initial_condition_hashes.json")
    seal(bundle)
    with pytest.raises(be.BaselineExportError):
        be.validate_bundle(bundle)


@pytest.mark.parametrize("mutation", ["base64", "byte_digest", "old_manifest_chain", "initial_boolean", "seed_mapping", "policy_seed", "trace_digest", "trace_count", "trace_first", "terminal"])
def test_independent_local_checks_do_not_trust_remote_success_flags(fixture, mutation):
    _, bundle = fixture
    entry = bundle["entries"][0]
    if mutation == "base64":
        entry["files"]["episode_manifest.json"]["base64"] = "!"
    elif mutation == "byte_digest":
        entry["files"]["episode_manifest.json"]["sha256"] = "f" * 64
    elif mutation == "old_manifest_chain":
        change_json(bundle, "episode_manifest.json", lambda value: value.__setitem__("extra", "changed"))
    elif mutation == "initial_boolean":
        change_json(bundle, "initial_condition_hashes.json", lambda value: value.__setitem__(be.INITIAL_FIELDS[0], True), rebind=True)
    elif mutation == "seed_mapping":
        change_json(bundle, "episode_manifest.json", lambda value: value.__setitem__("resolved_environment_seed", 999999), rebind=True)
    elif mutation == "policy_seed":
        change_json(bundle, "episode_manifest.json", lambda value: value.__setitem__("evaluation_policy_seed", 8), rebind=True)
    elif mutation == "trace_digest":
        entry["trace"]["sha256"] = "f" * 64
    elif mutation == "trace_count":
        entry["trace"]["policy_call_count"] = 1
    elif mutation == "trace_first":
        first = be.parse(be.decode_blob(entry["trace"]["first_nonempty_line"]["bytes"], "line"), "line")
        first["policy_call_index"] = 1
        entry["trace"]["first_nonempty_line"]["bytes"] = be.blob(be.canonical(first))
    else:
        change_json(bundle, "episode_result.json", lambda value: value.__setitem__("terminal_reason", "infrastructure"))
    seal(bundle)
    with pytest.raises(be.BaselineExportError):
        be.validate_bundle(bundle)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hard_stop", "manifest_mismatch"])
def test_retry_evidence_is_required_and_immutable(fixture, mutation):
    _, bundle = fixture
    ledger = bundle["failure_ledger"]
    if mutation == "missing":
        ledger["selected_U_lines"] = []
    elif mutation == "duplicate":
        ledger["selected_U_lines"] *= 2
    else:
        value = be.parse(be.decode_blob(ledger["selected_U_lines"][0]["bytes"], "line"), "line")
        value["classification" if mutation == "hard_stop" else "episode_manifest_sha256"] = "hard_stop" if mutation == "hard_stop" else "f" * 64
        ledger["selected_U_lines"][0]["bytes"] = be.blob(be.canonical(value))
    seal(bundle)
    with pytest.raises(be.BaselineExportError):
        be.validate_bundle(bundle)


def test_legacy_minimal_retry_hash_absence_is_reported_not_fabricated(fixture):
    _, bundle = fixture
    line = bundle["failure_ledger"]["selected_U_lines"][0]
    value = be.parse(be.decode_blob(line["bytes"], "line"), "line")
    value.pop("episode_manifest_sha256")
    line["bytes"] = be.blob(be.canonical(value))
    report, _ = be.validate_bundle(seal(bundle))
    assert report["retry_evidence"]["legacy_historical_manifest_digest_missing_count"] == 1
    assert report["retry_evidence"]["historical_manifest_digest_present_count"] == 0
    assert report["baseline_status"] == "unresolved"


def test_archived_launch_must_bind_original_protocol_bytes(fixture):
    _, bundle = fixture
    bundle["documents"]["protocol/protocol_snapshot.md"]["bytes"] = be.blob(b"changed protocol")
    with pytest.raises(be.BaselineExportError, match="archived launch binding"):
        be.validate_bundle(seal(bundle))


def test_write_once_import_preserves_bytes_and_never_reuses_directory(fixture, tmp_path):
    _, bundle = fixture
    output = tmp_path / "new-raw-U-import"
    report = be.import_bundle(bundle, output)
    assert report["baseline_status"] == "unresolved"
    assert (output / "IMPORT_REPORT.json").is_file()
    for relative, metadata in report["imported_files"].items():
        raw = (output / relative).read_bytes()
        assert be.sha(raw) == metadata["sha256"] and len(raw) == metadata["size_bytes"]
    before = (output / "IMPORT_REPORT.json").read_bytes()
    with pytest.raises(FileExistsError):
        be.import_bundle(bundle, output)
    assert (output / "IMPORT_REPORT.json").read_bytes() == before


def test_bad_import_has_no_partial_destination(fixture, tmp_path):
    _, bundle = fixture
    bundle["entries"].pop()
    output = tmp_path / "must-not-exist"
    with pytest.raises(be.BaselineExportError):
        be.import_bundle(seal(bundle), output)
    assert not output.exists()


def test_source_and_symlink_destinations_rejected(fixture, tmp_path):
    root, bundle = fixture
    with pytest.raises(be.BaselineExportError, match="source run"):
        be.import_bundle(bundle, root / "not-allowed")
    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(be.BaselineExportError, match="symlink"):
        be.import_bundle(bundle, alias / "new")
    with pytest.raises(be.BaselineExportError, match="exact authorized"):
        be.collect_baseline(tmp_path)


def test_standalone_command_needs_no_package_or_third_party_dependencies():
    path = Path(be.__file__)
    result = subprocess.run([sys.executable, "-I", "-B", str(path), "--help"], capture_output=True, text=True, check=True)
    assert "collect" in result.stdout and "validate" in result.stdout and "import" in result.stdout
