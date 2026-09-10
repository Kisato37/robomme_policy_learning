"""Scoped, standard-library-only export/import of the immutable Athena U run.

Run ``collect`` on the already authorized source host; it only reads and emits
JSON on stdout. Transport (including SSH) is deliberately outside this tool.
``validate`` performs local checks; ``import`` additionally writes original
bytes to a NEW directory. Neither operation signs a baseline attestation.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys


RUN_ID = "20260829T231425Z_7b594786_formal_v1"
SOURCE_ROOT = "/zpool-00/home/jp673/robomme_repro/robomme_policy_learning/runs/keyframe_oracle_sampling/" + RUN_ID
TASKS = ("BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask", "VideoUnmask",
         "VideoUnmaskSwap", "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton",
         "VideoPlaceOrder", "MoveCube", "InsertPeg", "PatternLock", "RouteStick")
INITIAL_FIELDS = ("front_observations_sha256", "wrist_observations_sha256", "robot_states_sha256",
                  "task_state_sha256", "task_instruction_sha256")
SEED_TABLE_ENTRIES_SHA256 = "296dbd435906a0affe4239472b0805894aa2e97c01c4f1f093212cc27cdef65b"
# Derived from the previously downloaded Athena U manifests, not an invented
# task-base + episode formula. This does not verify a future host's resolver.
EXPECTED_SEED_MAPPING_SHA256 = "ed5fef94211e17c6ef4b2efadd91e5446df780e931f6cdce2222a35ea4f8d483"
CORE_FILES = {"episode_result.json", "episode_manifest.json", "initial_condition_hashes.json"}
DOCUMENT_PATHS = (
    "protocol/launch_manifest.json", "protocol/protocol_snapshot.md", "protocol/protocol_sha256.txt",
    "protocol/seed_table.json", "protocol/formal_matrix.json", "protocol/architecture_pass_report.json",
    "protocol/development_smoke_audit.json", "protocol/environment_manifest.json", "environment_manifest.json",
)
FIXED = {"dataset": "test", "max_steps": 1300, "executed_action_horizon": 16,
         "evaluation_policy_seed": 7, "checkpoint_id": 79999}
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
KIND = "athena-old-U-raw-evidence-bundle-v1"


class BaselineExportError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise BaselineExportError(f"{label} mismatch")


def require_digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BaselineExportError(f"Invalid SHA-256: {label}")


def parse(raw, label):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise BaselineExportError(f"Duplicate JSON key in {label}: {key}")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        canonical(value)
        return value
    except (ValueError, TypeError, UnicodeError) as exc:
        raise BaselineExportError(f"Invalid JSON: {label}") from exc


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise BaselineExportError("Invalid source-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")) or str(path) != value:
        raise BaselineExportError("Source path traversal or normalization is forbidden")
    return value


def readable(path):
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise BaselineExportError(f"Missing or symlinked evidence: {path}")


def read_bytes(path):
    readable(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise BaselineExportError(f"Metadata file exceeds transfer cap: {path}")
    return raw


def blob(raw):
    return {"size_bytes": len(raw), "sha256": sha(raw), "base64": base64.b64encode(raw).decode("ascii")}


def decode_blob(value, label):
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256", "base64"}:
        raise BaselineExportError(f"Malformed byte envelope: {label}")
    size = value["size_bytes"]
    if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
        raise BaselineExportError(f"Invalid metadata byte count: {label}")
    require_digest(value["sha256"], label)
    encoded = value["base64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((size + 2) // 3):
        raise BaselineExportError(f"Invalid base64 size: {label}")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise BaselineExportError(f"Invalid base64: {label}") from exc
    equal(len(raw), size, label + " length")
    equal(sha(raw), value["sha256"], label + " checksum")
    return raw


def scientific_key(task, episode):
    return {"task": task, "episode_id": episode, "arm": "U", "trajectory_kind": "formal"}


def attempt_relative(task, episode, attempt):
    return f"trajectories/{task}/episode_{episode:02d}/U/attempt_{attempt:02d}"


def check_trace_record(record, key, call, initial_count):
    if not isinstance(record, dict):
        raise BaselineExportError("Trace must be an object")
    for field, expected in (("task", key["task"]), ("episode_id", key["episode_id"]), ("selector_name", "U"),
                            ("policy_call_index", call), ("environment_step", call * 16),
                            ("seed_table_sha256", SEED_TABLE_ENTRIES_SHA256), ("seed_table_dataset", "test")):
        equal(record.get(field), expected, "trace " + field)
    count = record.get("history_length")
    if type(count) is not int or count < 1:
        raise BaselineExportError("Invalid trace history length")
    equal(record.get("current_history_index"), count - 1, "trace current history index")
    equal(count, initial_count + 16 * call, "trace history cadence")


def collect_trace(path, key):
    readable(path)
    hasher, size, count, physical_lines = hashlib.sha256(), 0, 0, 0
    first = None
    initial_count = None
    with path.open("rb") as stream:
        for raw in stream:
            size += len(raw)
            if size > MAX_TRACE_BYTES or len(raw) > MAX_FILE_BYTES:
                raise BaselineExportError("Trace exceeds bounded read limit")
            physical_lines += 1
            hasher.update(raw)
            if not raw.strip():
                continue
            record = parse(raw, "selector trace")
            if first is None:
                first = {"physical_line_number": physical_lines, "bytes": blob(raw)}
                initial_count = record.get("history_length")
            check_trace_record(record, key, count, initial_count)
            count += 1
    if first is None:
        raise BaselineExportError("Completed U outcome has an empty trace")
    return {"sha256": hasher.hexdigest(), "size_bytes": size, "policy_call_count": count,
            "physical_line_count": physical_lines, "first_nonempty_line": first,
            "scope": "remote_full_byte_hash_and_structural_scan; local_first_line_only"}


def collect_failure_lines(root):
    path = root / "failures/failure_ledger.jsonl"
    if not path.exists():
        return {"status": "missing", "relative_path": "failures/failure_ledger.jsonl"}
    raw = read_bytes(path)
    lines = []
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        entry = parse(line, "failure ledger")
        if not isinstance(entry, dict):
            raise BaselineExportError("Invalid failure ledger row")
        if entry.get("arm") == "U":
            lines.append({"physical_line_number": number, "bytes": blob(line)})
    return {"status": "present", "relative_path": "failures/failure_ledger.jsonl", "sha256": sha(raw),
            "size_bytes": len(raw), "selected_U_lines": lines, "full_ledger_transferred": False}


def collect_baseline(source_root=None):
    """No network, subprocess, writes, unscoped globs, videos or checkpoints."""
    root = Path(SOURCE_ROOT if source_root is None else source_root)
    if str(root) != SOURCE_ROOT or any(part.is_symlink() for part in (root, *root.parents)):
        raise BaselineExportError("Only the exact authorized Athena baseline root is allowed")
    if not root.is_dir():
        raise BaselineExportError("Authorized baseline root is absent on this host")
    expected_dirs = {root / f"trajectories/{task}/episode_{episode:02d}/U" for task in TASKS for episode in range(50)}
    observed_dirs = set(root.glob("trajectories/*/episode_*/U"))
    if observed_dirs != expected_dirs:
        raise BaselineExportError("The source must contain exactly 800 expected U scientific directories")
    entries = []
    for task in TASKS:
        for episode in range(50):
            key = scientific_key(task, episode)
            directory = root / f"trajectories/{task}/episode_{episode:02d}/U"
            attempts = sorted(directory.iterdir())
            if ([p.name for p in attempts] != [f"attempt_{i:02d}" for i in range(len(attempts))]
                    or not 1 <= len(attempts) <= 3 or any(p.is_symlink() or not p.is_dir() for p in attempts)):
                raise BaselineExportError(f"Invalid U attempt census: {directory}")
            complete = [path for path in attempts if (path / "episode_result.json").exists()]
            if len(complete) != 1 or complete[0] != attempts[-1]:
                raise BaselineExportError(f"Missing, duplicate, or retried scientific U completion: {directory}")
            path, attempt = complete[0], len(attempts) - 1
            files = {name: blob(read_bytes(path / name)) for name in sorted(CORE_FILES)}
            prior = []
            for index, old in enumerate(attempts[:-1]):
                old_files = {"episode_manifest.json": blob(read_bytes(old / "episode_manifest.json"))}
                if (old / "initial_condition_hashes.json").exists():
                    old_files["initial_condition_hashes.json"] = blob(read_bytes(old / "initial_condition_hashes.json"))
                prior.append({"attempt_id": index, "files": old_files})
            entries.append({"scientific_key": key, "attempt_id": attempt,
                            "relative_directory": attempt_relative(task, episode, attempt), "files": files,
                            "trace": collect_trace(path / "selector_trace.jsonl", key), "prior_attempts": prior})
    documents = {}
    for name in DOCUMENT_PATHS:
        path = root / name
        documents[name] = {"status": "present", "bytes": blob(read_bytes(path))} if path.exists() else {"status": "missing"}
    result = {"schema_version": 1, "kind": KIND, "source_root": SOURCE_ROOT, "source_run_id": RUN_ID,
              "arm": "U", "expected_scientific_count": 800, "entries": entries,
              "documents": documents, "failure_ledger": collect_failure_lines(root)}
    result["bundle_sha256"] = sha(canonical(result))
    validate_bundle(result)  # Refuse an invalid export rather than report partial success.
    if len(canonical(result)) > MAX_BUNDLE_BYTES:
        raise BaselineExportError("Bundle exceeds bounded metadata transfer size")
    return result


def validate_bundle(bundle):
    """Locally recompute raw-file checksums; do not fabricate missing evidence."""
    if not isinstance(bundle, dict):
        raise BaselineExportError("Bundle must be an object")
    equal(bundle.get("kind"), KIND, "bundle kind")
    equal(bundle.get("schema_version"), 1, "bundle schema")
    equal(bundle.get("source_root"), SOURCE_ROOT, "source root")
    equal(bundle.get("source_run_id"), RUN_ID, "source run")
    equal(bundle.get("arm"), "U", "source arm")
    equal(bundle.get("expected_scientific_count"), 800, "source census")
    equal(bundle.get("bundle_sha256"), sha(canonical({k: v for k, v in bundle.items() if k != "bundle_sha256"})), "bundle digest")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != 800:
        raise BaselineExportError("Exactly 800 U outcomes are required; no partial mode")
    decoded, observations, seed_map, traces, prior_ids = {}, [], [], [], set()
    for index, entry in enumerate(entries):
        task, episode = TASKS[index // 50], index % 50
        key = scientific_key(task, episode)
        equal(entry.get("scientific_key"), key, "canonical U scientific key/order")
        attempt = entry.get("attempt_id")
        if type(attempt) is not int or not 0 <= attempt <= 2:
            raise BaselineExportError("Invalid U attempt id")
        relative = attempt_relative(task, episode, attempt)
        equal(entry.get("relative_directory"), relative, "U attempt path")
        if not isinstance(entry.get("files"), dict) or set(entry["files"]) != CORE_FILES:
            raise BaselineExportError("The three original JSON files must all be present")
        payloads = {}
        for name, envelope in entry["files"].items():
            raw = decode_blob(envelope, relative + "/" + name)
            decoded[relative + "/" + name] = raw
            payloads[name] = parse(raw, name)
        result, manifest, initial = (payloads[name] for name in ("episode_result.json", "episode_manifest.json", "initial_condition_hashes.json"))
        if any(not isinstance(value, dict) for value in (result, manifest, initial)):
            raise BaselineExportError("U original JSON files must contain objects")
        for name, value in (("result", result), ("manifest", manifest)):
            equal(value.get("scientific_key"), key, name + " scientific key")
            equal(value.get("attempt_id"), attempt, name + " attempt")
            for field, expected in FIXED.items():
                equal(value.get(field), expected, name + " " + field)
        equal(manifest.get("protocol_version"), "v1.0", "original protocol")
        equal(manifest.get("seed_table_sha256"), SEED_TABLE_ENTRIES_SHA256, "original seed-table identity")
        equal(manifest.get("environment_setup_completed"), True, "completed U environment setup")
        for field, expected in (("task", task), ("episode_id", episode), ("selector_arm", "U")):
            equal(result.get(field), expected, "original result " + field)
        for name, field in (("episode_manifest.json", "episode_manifest_sha256"),
                            ("initial_condition_hashes.json", "initial_condition_hashes_sha256")):
            equal(result.get(field), sha(decoded[relative + "/" + name]), "original result checksum chain " + name)
        if set(initial) != set(INITIAL_FIELDS):
            raise BaselineExportError("Exactly five original initial hash VALUES are required")
        for field, value in initial.items():
            require_digest(value, field)
        reason = result.get("terminal_reason")
        if reason not in ("success", "fail", "timeout", "error"):
            raise BaselineExportError("Non-scientific terminal result is forbidden")
        equal(result.get("success"), reason == "success", "U terminal outcome")
        if reason == "error" and (not result.get("benchmark_error_message") or not result.get("benchmark_exception_type")):
            raise BaselineExportError("Benchmark error lacks evidence")
        steps = result.get("steps")
        if type(steps) is not int or not 1 <= steps <= 1300:
            raise BaselineExportError("Invalid original trajectory step count")
        trace = entry.get("trace")
        if not isinstance(trace, dict):
            raise BaselineExportError("Missing trace hash/count/first-line evidence")
        require_digest(trace.get("sha256"), "full source trace")
        equal(result.get("selector_trace_sha256"), trace["sha256"], "original trace checksum chain (remote full hash)")
        equal(trace.get("policy_call_count"), math.ceil(steps / 16), "trace count/result steps")
        if type(trace.get("size_bytes")) is not int or not 1 <= trace["size_bytes"] <= MAX_TRACE_BYTES:
            raise BaselineExportError("Invalid source trace size")
        first = trace.get("first_nonempty_line")
        if not isinstance(first, dict) or type(first.get("physical_line_number")) is not int or first["physical_line_number"] < 1:
            raise BaselineExportError("Missing first original trace line")
        first_raw = decode_blob(first.get("bytes"), "first original trace line")
        first_value = parse(first_raw, "first original trace line")
        if not isinstance(first_value, dict):
            raise BaselineExportError("First trace must be a mapping")
        check_trace_record(first_value, key, 0, first_value.get("history_length"))
        decoded["trace_first_lines/" + relative + "/selector_trace.first.jsonl"] = first_raw
        histories = result.get("history_lengths_at_policy_calls")
        if not isinstance(histories, list) or len(histories) != trace["policy_call_count"]:
            raise BaselineExportError("Original result lacks aligned per-call history lengths")
        equal(histories, [first_value["history_length"] + 16 * call for call in range(len(histories))], "result history cadence")
        seed, difficulty = manifest.get("resolved_environment_seed"), manifest.get("difficulty")
        if type(seed) is not int or seed < 0 or type(difficulty) not in (str, int) or difficulty == "":
            raise BaselineExportError("Missing actual environment seed/difficulty")
        seed_map.append({"task": task, "episode_id": episode, "resolved_environment_seed": seed, "difficulty": difficulty})
        observations.append({"scientific_key": key, "attempt_id": attempt, "success": result["success"],
                             "terminal_reason": reason, "source_relative_directory": relative,
                             "source_result_sha256": entry["files"]["episode_result.json"]["sha256"],
                             "initial_hashes": initial, "resolved_environment_seed": seed, "difficulty": difficulty,
                             "resolved_difficulty_hint": manifest.get("resolved_difficulty_hint"),
                             "evaluation_policy_seed": 7, "first_history_length": first_value["history_length"]})
        traces.append({"relative_path": relative + "/selector_trace.jsonl", **trace})
        prior = entry.get("prior_attempts")
        if not isinstance(prior, list) or len(prior) != attempt:
            raise BaselineExportError("Prior retry evidence is incomplete")
        for old_attempt, old in enumerate(prior):
            equal(old.get("attempt_id"), old_attempt, "prior attempt sequence")
            old_files = old.get("files")
            if (not isinstance(old_files, dict) or "episode_manifest.json" not in old_files
                    or set(old_files) - {"episode_manifest.json", "initial_condition_hashes.json"}):
                raise BaselineExportError("Invalid prior-attempt metadata")
            old_relative = attempt_relative(task, episode, old_attempt)
            for name, envelope in old_files.items():
                raw = decode_blob(envelope, old_relative + "/" + name)
                decoded[old_relative + "/" + name] = raw
            old_manifest = parse(decoded[old_relative + "/episode_manifest.json"], "prior manifest")
            equal(old_manifest.get("scientific_key"), key, "prior scientific key")
            equal(old_manifest.get("attempt_id"), old_attempt, "prior attempt")
            prior_ids.add((task, episode, old_attempt))
    mapping_sha = sha(canonical(seed_map))
    equal(mapping_sha, EXPECTED_SEED_MAPPING_SHA256, "actual U seed/difficulty map versus previously exported Athena U")
    retry = validate_retry_lines(bundle.get("failure_ledger"), prior_ids, decoded)
    documents, missing = validate_documents(bundle.get("documents"), decoded)
    report = {
        "kind": "raw-U-evidence-import-audit-not-baseline-attestation", "source_root": SOURCE_ROOT,
        "source_run_id": RUN_ID, "source_bundle_sha256": bundle["bundle_sha256"],
        "scientific_count": 800, "initial_hash_value_records": 800,
        "actual_seed_mapping_sha256": mapping_sha, "matches_prior_Athena_seed_mapping": True,
        "outcome_counts": dict(Counter(record["terminal_reason"] for record in observations)),
        "records": observations, "trace_evidence": traces, "retry_evidence": retry,
        "documents": documents, "missing_optional_documents": missing,
        "baseline_status": "unresolved", "launch_authorized": False,
        "limitations": [
            "Full trace bytes are scanned/hashed on the source host; only the first original line is locally rehashed.",
            "Original image/state arrays, exact task text, and per-frame stage values are not reconstructed from hashes.",
            "No weights or videos are transferred, and no checkpoint is loaded.",
            "Archived launch/environment statements still require review against the chosen new runtime.",
            "U versus UK48/UN48 input alignment and formal baseline attestation are not established.",
        ],
    }
    return report, decoded


def validate_retry_lines(evidence, expected, decoded):
    if not isinstance(evidence, dict) or evidence.get("relative_path") != "failures/failure_ledger.jsonl":
        raise BaselineExportError("Missing failure-ledger inventory")
    if evidence.get("status") == "missing":
        if expected:
            raise BaselineExportError("Retried U outcomes require original infrastructure failure evidence")
        return {"status": "missing", "required_retry_count": 0}
    if evidence.get("status") != "present" or evidence.get("full_ledger_transferred") is not False:
        raise BaselineExportError("Invalid scoped failure-ledger evidence")
    require_digest(evidence.get("sha256"), "source failure ledger")
    lines = evidence.get("selected_U_lines")
    if not isinstance(lines, list):
        raise BaselineExportError("Missing original U failure lines")
    found, raw_lines, bound_count = set(), [], 0
    for line in lines:
        raw = decode_blob(line.get("bytes"), "original U failure line")
        value = parse(raw, "U failure line")
        if not isinstance(value, dict) or value.get("arm") != "U" or value.get("trajectory_kind", "formal") != "formal":
            raise BaselineExportError("Unscoped failure-ledger entry")
        key = (value.get("task"), value.get("episode_id"), value.get("attempt_id"))
        if key not in expected or key in found:
            raise BaselineExportError("Unexpected/duplicate U failure evidence")
        equal(value.get("retry_allowed"), True, "prior infrastructure retry permission")
        if value.get("classification") != "infrastructure":
            raise BaselineExportError("Hard-stop evidence does not authorize a retry")
        relative = attempt_relative(*key) + "/episode_manifest.json"
        # The original aggregator explicitly accepts its evaluator_minimal
        # ledger schema without a manifest digest. Preserve that missing fact;
        # do not invent the historical checksum or relax a digest that exists.
        if value.get("episode_manifest_sha256") is not None:
            equal(value["episode_manifest_sha256"], sha(decoded[relative]), "failure/prior-manifest binding")
            bound_count += 1
        found.add(key)
        raw_lines.append(raw)
    equal(sorted(found), sorted(expected), "complete prior failure ledger evidence")
    decoded["retry_evidence/failure_ledger.selected_U_lines.jsonl"] = b"".join(raw_lines)
    return {"status": "locally_rehashed_selected_U_lines", "count": len(found),
            "historical_manifest_digest_present_count": bound_count,
            "legacy_historical_manifest_digest_missing_count": len(found) - bound_count,
            "source_full_ledger_sha256": evidence["sha256"], "source_full_ledger_locally_rehashed": False}


def validate_documents(documents, decoded):
    if not isinstance(documents, dict) or set(documents) != set(DOCUMENT_PATHS):
        raise BaselineExportError("Document inventory must equal the scoped path allowlist")
    inventory, missing = {}, []
    for name, entry in documents.items():
        safe_relative(name)
        if entry == {"status": "missing"}:
            missing.append(name)
            inventory[name] = {"status": "missing"}
            continue
        if not isinstance(entry, dict) or set(entry) != {"status", "bytes"} or entry["status"] != "present":
            raise BaselineExportError("Invalid document inventory entry")
        raw = decode_blob(entry["bytes"], name)
        decoded[name] = raw
        inventory[name] = {"status": "present", "sha256": sha(raw), "size_bytes": len(raw)}
    launch_raw = decoded.get("protocol/launch_manifest.json")
    if launch_raw is not None:
        launch = parse(launch_raw, "launch manifest")
        equal(launch.get("run_id"), RUN_ID, "archived launch run id")
        equal(launch.get("run_kind"), "formal", "archived launch scope")
        for field, expected in (("dataset", "test"), ("evaluation_policy_seed", 7), ("max_steps", 1300)):
            equal(launch.get(field), expected, "archived launch " + field)
        for field, name in (("protocol_sha256", "protocol/protocol_snapshot.md"),
                            ("seed_table_file_sha256", "protocol/seed_table.json"),
                            ("formal_matrix_sha256", "protocol/formal_matrix.json")):
            if name in decoded:
                equal(launch.get(field), sha(decoded[name]), "archived launch binding " + name)
        equal(launch.get("seed_table_entries_sha256"), SEED_TABLE_ENTRIES_SHA256, "archived launch seed entries")
    return inventory, sorted(missing)


def load_bundle(path):
    path = Path(path)
    readable(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(raw) > MAX_BUNDLE_BYTES:
        raise BaselineExportError("Bundle exceeds transfer cap")
    return parse(raw, "bundle")


def import_bundle(bundle, destination):
    """Validate completely before creating an exclusive, never-overwritten root."""
    report, decoded = validate_bundle(bundle)
    root = Path(destination).absolute()
    if ".." in root.parts or any(part.is_symlink() for part in (root, *root.parents)):
        raise BaselineExportError("Import destination cannot traverse a symlink or '..'")
    if root == Path(SOURCE_ROOT) or Path(SOURCE_ROOT) in root.parents:
        raise BaselineExportError("Never write into the source run")
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    # IMPORT_REPORT.json is written last. An interrupted import remains visible
    # as incomplete and may never be overwritten/reused by a later import.
    for relative, raw in sorted(decoded.items()):
        safe_relative(relative)
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        equal(sha(target.read_bytes()), sha(raw), "imported bytes " + relative)
        files["source/" + relative] = {"sha256": sha(raw), "size_bytes": len(raw)}
    report["imported_files"] = files
    with (root / "IMPORT_REPORT.json").open("xb") as stream:
        stream.write(canonical(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("collect", help="Read only the fixed Athena run; JSON bundle on stdout; no SSH")
    for name in ("validate", "import"):
        sub = commands.add_parser(name)
        sub.add_argument("--bundle", required=True, type=Path)
        if name == "import":
            sub.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            sys.stdout.buffer.write(canonical(collect_baseline()) + b"\n")
        else:
            bundle = load_bundle(args.bundle)
            report = import_bundle(bundle, args.output) if args.command == "import" else validate_bundle(bundle)[0]
            print(json.dumps({key: report[key] for key in ("kind", "scientific_count", "initial_hash_value_records",
                                                          "actual_seed_mapping_sha256", "missing_optional_documents", "baseline_status")}, indent=2))
    except (BaselineExportError, OSError) as exc:
        parser.exit(2, f"Baseline evidence operation failed: {exc}\n")


if __name__ == "__main__":
    main()
