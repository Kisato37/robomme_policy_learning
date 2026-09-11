"""Synthetic CPU files only; no production checkpoint, authorization or GPU.

The fixed 12 GB parent metadata gate is replaced ONLY inside this test fixture
so the evidence graph can be exercised using tiny ordinary files.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

import pytest

from experiments.uniform_keyframe_expansion import contract as c
from experiments.uniform_keyframe_expansion import launch_contract as lc
from experiments.uniform_keyframe_expansion.artifacts import ExpansionRunStore
from tests.uniform_keyframe_expansion.test_artifacts import finish, initial, result, trace

GPU = "GPU-11111111-2222-3333-4444-555555555555"


class Fixture:
    def __init__(self, path, monkeypatch, *, formal=False):
        self.path = path
        self.serial = 0
        self.raw = self.write("source.txt", b"CPU fixture evidence, not a real run or user approval")
        self.lock = self.write("uv.lock", b"fixture lock")
        self.protocol = self.write("protocol.md", f"# Fixture\n**Protocol version:** {'v1.1' if formal else 'v0.9'}\n".encode())
        self.evidence = {"protocol": self.protocol}
        for role, revision in (("policy", "a" * 40), ("benchmark", "b" * 40)):
            root = path / role
            root.mkdir()
            module = root / "module.py"
            module.write_text("# CPU fixture source\n")
            self.evidence[f"{role}_source"] = self.write(f"{role}.json", {
                "schema_version": 1, "kind": "source_checkout", "root": str(root),
                "revision": revision, "branch": "exp/fixture" if role == "policy" else "HEAD", "clean": True,
                "files": [{"path": "module.py", "sha256": lc.file_reference(module)["sha256"]}], "sources": [self.raw]})
        profile = {"python_executable": str(Path(sys.executable).absolute()), "python_version": "fixture",
                   "packages": {"numpy": "fixture"}, "process_environment": {"CUDA_VISIBLE_DEVICES": GPU},
                   "command_prefix": [], "lockfiles": [self.lock]}
        (path / "host_locks").mkdir()
        self.environment = {"schema_version": 1, "kind": "runtime_environment", "host": "cpu-fixture", "lock_directory": str(path / "host_locks"),
                            "roles": {"policy": deepcopy(profile), "simulator": deepcopy(profile)},
                            "hardware": {"gpu_uuid": GPU, "gpu_name": "not-a-real-gpu", "minimum_free_memory_mib": 1},
                            "sources": [self.raw]}
        self.evidence["environment"] = self.write("environment.json", self.environment)
        checkpoint_dir = path / "model" / "79999"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "params").mkdir()
        weights = checkpoint_dir / "params" / "weights"
        weights.write_bytes(b"tiny synthetic model data")
        archive = self.write("model.zip", b"tiny synthetic archive")
        history = self.write("model/history_config.txt", b"released fixture config")
        monkeypatch.setattr(c, "CHECKPOINT_ARCHIVE_SHA256", archive["sha256"])
        entries = [{"path": "params/weights", "size": weights.stat().st_size, "sha256": lc.file_reference(weights)["sha256"]}]
        manifest = {"algorithm": lc.CHECKPOINT_CONTENT_TREE_ALGORITHM, "files": entries}
        metadata = {"metadata_sha256": "d" * 64, "file_count": 1, "total_bytes": weights.stat().st_size}
        monkeypatch.setattr(lc, "checkpoint_identity", lambda _: metadata)
        tree = {"algorithm": lc.CHECKPOINT_CONTENT_TREE_ALGORITHM, "content_tree_sha256": c.canonical_sha256(manifest),
                "file_count": 1, "total_bytes": weights.stat().st_size}
        self.evidence["checkpoint"] = self.write("checkpoint.json", {
            "schema_version": 1, "kind": "checkpoint_source", "checkpoint_dir": str(checkpoint_dir),
            "archive": archive, "history_config": history, "metadata": metadata, "content_tree": tree,
            "content_manifest": self.write("content_manifest.json", manifest), "sources": [self.raw]})
        self.provenance = {"code_commit": "a" * 40, "benchmark_commit": "b" * 40,
                           "protocol_sha256": self.protocol["sha256"],
                           "environment_manifest_sha256": self.evidence["environment"]["sha256"],
                           "checkpoint_archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
                           "host": "cpu-fixture", "hardware": deepcopy(self.environment["hardware"]), "command": ["fixture"]}
        self.store = ExpansionRunStore.create(path / "uniform_keyframe_expansion" / "main",
                                             stage="formal" if formal else "smoke", run_manifest=self.provenance)
        self.binding = {"policy_commit": "a" * 40, "benchmark_commit": "b" * 40,
                        "environment_sha256": self.evidence["environment"]["sha256"],
                        "checkpoint_archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
                        "checkpoint_content_tree_sha256": tree["content_tree_sha256"],
                        "inference_contract_sha256": c.canonical_sha256(c.frozen_inference_contract())}

    def write(self, name, payload):
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else c.canonical_json(payload).encode())
        return lc.file_reference(path)

    def change(self, key, field, value):
        payload = json.loads(Path(self.evidence[key]["path"]).read_text())
        payload[field] = value
        self.serial += 1
        self.evidence[key] = self.write(f"changed-{self.serial}.json", payload)

    def gate(self, kind, **extra):
        return {"schema_version": 1, "protocol_family": c.PROTOCOL_FAMILY, "kind": kind,
                "status": "PASS", "binding": self.binding, "sources": [self.raw], **extra}

    def cpu(self):
        junit = self.write("junit.xml", b'<testsuite><testcase name="real_fixture"/></testsuite>')
        self.evidence["cpu_gate"] = self.write("cpu_gate.json", self.gate("cpu_gate", junit=junit,
                                                passed_test_count=1, sources=[self.raw, junit]))

    def architecture(self):
        from tests.uniform_keyframe_expansion.architecture_fixtures import create_architecture_fixture
        fixture = create_architecture_fixture(self)
        self.evidence["architecture_gate"] = fixture["gate"]
        return fixture

    def smoke(self, *, count=64, error_at=None, unequal_at=None):
        self.architecture()
        store = ExpansionRunStore.create(self.path / "uniform_keyframe_expansion" / "source-smoke",
                                         stage="smoke", run_manifest=self.provenance)
        from mme_vla_suite.shared.uniform_keyframe_config import RELEASED_HISTORY_CONFIG, expanded_history_mapping
        from mme_vla_suite.shared.uniform_keyframe_config import payload_digest
        _, expected = expanded_history_mapping(RELEASED_HISTORY_CONFIG)
        selected_rows = c.build_smoke_matrix()["rows"][:count]
        execution_records = []
        owner_by_row = {}
        for ordinal, variant in enumerate((c.U_POLICY_VARIANT, c.EXPANDED_POLICY_VARIANT)):
            row_ids = [row["row_id"] for row in selected_rows
                       if c.policy_variant_for_arm(row["arm"]) == variant]
            if not row_ids:
                continue
            request = lc.build_execution_request(
                run_root=store.run_root, stage="end_to_end_smoke", row_ids=row_ids,
                execution_id=str(uuid.uuid4()), gpu_uuid=GPU, evidence=self.evidence)
            approval = self.write(f"source_approval_{ordinal}.json", {
                "schema_version": 1, "kind": "user_authorization",
                "protocol_family": c.PROTOCOL_FAMILY, "stage": "end_to_end_smoke", "approved": True,
                "request_sha256": request["request_sha256"], "instruction": self.raw,
                "recorded_utc": "2026-09-10T00:00:00Z", "recorded_by": "not-real-CPU-fixture"})
            source_plan = lc.validate_execution_plan(lc.build_execution_plan(request, approval))
            identity = source_plan.execution_identity
            for row_id in row_ids:
                owner_by_row[row_id] = identity
            is_u = variant == c.U_POLICY_VARIANT
            execution_records.append({
                "plan": self.write(f"source_execution_{ordinal}.json", source_plan.payload),
                "controller_result": self.write(f"source_controller_{ordinal}.json", {
                    "execution_identity": identity, "owned_process_cleanup_confirmed": True}),
                "shard_result": self.write(f"source_shard_{ordinal}.json", {
                    "execution_identity": identity, "row_ids": row_ids}),
                "policy_bootstrap": self.write(f"source_bootstrap_{ordinal}.json", {
                    "policy_execution_identity": identity,
                    "checkpoint_content_verified_in_this_process": True, "model_gpu_uuid": GPU,
                    "model_process_pid": 99 + ordinal,
                    "checkpoint_content_evidence": {"archive_sha256": c.CHECKPOINT_ARCHIVE_SHA256,
                                                     "content_tree": source_plan.checkpoint["content_tree"]}}),
                "policy_ready": self.write(f"source_ready_{ordinal}.json", {
                    "wire_schema": 1, "experiment_family": c.PROTOCOL_FAMILY,
                    "policy_variant": variant,
                    "effective_memory_budget": 512 if is_u else 768,
                    "evaluation_policy_seed": 7, "resident_policy": True,
                    "strict_weight_tree_load": True,
                    "effective_history_config_sha256": (payload_digest(RELEASED_HISTORY_CONFIG)
                                                          if is_u else expected["effective_history_config_sha256"]),
                    "source_history_config_sha256": expected["source_history_config_sha256"],
                    "transport_keepalive_timeout_seconds": 600,
                    "direct_execution": identity, "model_process_pid": 99 + ordinal}),
            })
        results = []
        for row in selected_rows:
            writer = store.new_attempt(row, 0, {"policy_execution_identity": owner_by_row[row["row_id"]]})
            if row["row_id"] == unequal_at:
                initial(writer, seed=999)
                writer.append_trace(trace(writer))
                writer.finalize_scientific(result())
            else:
                finish(writer, "error" if row["row_id"] == error_at else "fail")
            results.append({"row_id": row["row_id"], "result": lc.file_reference(writer.result_path),
                            "manifest": lc.file_reference(writer.manifest_path), "initial": lc.file_reference(writer.initial_conditions_path)})
        manifest = lc.file_reference(store.run_root / "run_manifest.json")
        self.evidence["end_to_end_gate"] = self.write("end_to_end_gate.json", self.gate(
            "end_to_end_gate", run_manifest=manifest, completed_count=count, sources=[manifest],
            results=results, execution_records=execution_records))
        return store

    def formal(self, **kwargs):
        self.smoke(**kwargs)
        self.evidence["exposure"] = self.write("exposure.json", {
            "schema_version": 1, "kind": "prior_exposure", "protocol_family": c.PROTOCOL_FAMILY,
            "present_formal_outcomes_inspected": False, "sources": [self.raw],
            "records": [{"task": task, "episode_id": ep, "dataset": "test"} for task in c.FORMAL_TASKS for ep in range(50)]})
        self.freeze()

    def freeze(self):
        self.evidence["freeze"] = self.write("freeze.json", {"schema_version": 1, "kind": "protocol_freeze",
            "protocol_version": "v1.1", "binding": self.binding, "sources": [self.raw],
            **{f"{key}_sha256": self.evidence[key]["sha256"] for key in ("protocol", "end_to_end_gate", "architecture_gate", "exposure")}})

    def request(self, stage="cpu_prepare", *, rows=None):
        if rows is None:
            rows = [] if stage in ("cpu_prepare", "architecture_smoke") else [1, 2]
        return lc.build_execution_request(run_root=self.store.run_root, stage=stage, row_ids=rows,
                                         execution_id=str(uuid.uuid4()), gpu_uuid=GPU, evidence=self.evidence)

    def plan(self, stage="cpu_prepare", *, rows=None):
        request = self.request(stage, rows=rows)
        authorization = None if stage == "cpu_prepare" else self.write("approval.json", {
            "schema_version": 1, "kind": "user_authorization", "protocol_family": c.PROTOCOL_FAMILY,
            "stage": stage, "approved": True, "request_sha256": request["request_sha256"],
            "instruction": self.raw, "recorded_utc": "2026-09-10T00:00:00Z", "recorded_by": "CPU test fixture only"})
        return lc.build_execution_plan(request, authorization)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def test_cpu_preparation_is_readonly_and_cannot_authorize_runtime(fixture):
    before = sorted(str(path) for path in fixture.path.rglob("*"))
    payload = fixture.plan()
    plan = lc.validate_execution_plan(payload)
    assert sorted(str(path) for path in fixture.path.rglob("*")) == before
    assert plan.stage == "cpu_prepare" and plan.rows == []
    with pytest.raises(lc.ExpansionLaunchError, match="runtime"):
        plan.require_runtime_stage()
    with pytest.raises(lc.ExpansionLaunchError):
        lc.ValidatedExecutionPlan(payload, {})
    assert plan.live_checkpoint_verification_required is True


def test_architecture_authority_cannot_start_episodes_and_payload_is_defensive(fixture):
    fixture.cpu()
    plan = lc.validate_execution_plan(fixture.plan("architecture_smoke"))
    assert plan.execution_identity["dispatch_sha256"] == plan.payload["request"]["request_sha256"]
    assert plan.require_runtime_stage("architecture_smoke") is plan
    assert plan.revalidate().payload == plan.payload
    copy = plan.payload
    copy["request"]["stage"] = "formal"
    assert plan.stage == "architecture_smoke"
    with pytest.raises(lc.ExpansionLaunchError):
        plan.require_runtime_stage("end_to_end_smoke")
    with pytest.raises(lc.ExpansionLaunchError):
        fixture.request("architecture_smoke", rows=[0])


@pytest.mark.parametrize("stage", ["architecture_smoke", "end_to_end_smoke", "formal"])
def test_missing_gates_fail_even_if_an_old_user_approval_exists(fixture, stage):
    with pytest.raises(lc.ExpansionLaunchError):
        fixture.request(stage)


@pytest.mark.parametrize("mutation", ["missing", "wrong_stage", "wrong_request", "false", "empty_instruction"])
def test_authorization_needs_its_own_stage_request_and_source(fixture, mutation):
    fixture.cpu()
    payload = fixture.plan("architecture_smoke")
    if mutation == "missing":
        payload["authorization"] = None
    else:
        approval = json.loads(Path(payload["authorization"]["path"]).read_text())
        if mutation == "wrong_stage": approval["stage"] = "formal"
        if mutation == "wrong_request": approval["request_sha256"] = "a" * 64
        if mutation == "false": approval["approved"] = False
        if mutation == "empty_instruction": approval["instruction"] = fixture.write("empty.txt", b"")
        payload["authorization"] = fixture.write("bad_approval.json", approval)
    with pytest.raises(lc.ExpansionLaunchError):
        lc.validate_execution_plan(payload)


@pytest.mark.parametrize("mutation", ["row", "stage", "execution_id", "gpu", "extra"])
def test_request_digest_binds_all_runtime_scope(fixture, mutation):
    fixture.cpu()
    payload = fixture.plan("architecture_smoke")
    if mutation == "row": payload["request"]["rows"] = c.build_smoke_matrix()["rows"][:1]
    if mutation == "stage": payload["request"]["stage"] = "formal"
    if mutation == "execution_id": payload["request"]["execution_id"] = str(uuid.uuid4())
    if mutation == "gpu": payload["request"]["gpu_uuid"] = "0"
    if mutation == "extra": payload["request"]["run_anything"] = True
    with pytest.raises(lc.ExpansionLaunchError):
        lc.validate_execution_plan(payload)


def test_source_changed_after_validation_is_detected(fixture):
    payload = fixture.plan()
    source = lc.validate_execution_plan(payload).policy_root / "module.py"
    source.write_text("# same code claim but modified bytes")
    with pytest.raises(lc.ExpansionLaunchError):
        lc.validate_execution_plan(payload)


def test_large_checkpoint_files_are_not_read_by_per_row_contract(fixture, monkeypatch):
    real = lc.sha256_file
    def guarded(path):
        if Path(path).name in ("weights", "model.zip"):
            raise AssertionError("Per-row validation must not read large archive/parameter bytes")
        return real(path)
    monkeypatch.setattr(lc, "sha256_file", guarded)
    plan = lc.validate_execution_plan(fixture.plan())
    assert plan.revalidate().live_checkpoint_verification_required


@pytest.mark.parametrize("mutation", ["ordinal", "missing_threshold", "source_dirty", "wrong_revision", "wrong_archive", "content_digest", "wrapper_shell"])
def test_declared_source_and_runtime_constraints_fail_closed(fixture, mutation):
    if mutation == "ordinal": fixture.environment["roles"]["policy"]["process_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    if mutation == "missing_threshold": fixture.environment["hardware"].pop("minimum_free_memory_mib")
    if mutation == "wrapper_shell": fixture.environment["roles"]["policy"]["command_prefix"] = ["/bin/bash", "-c", "echo no"]
    if mutation in ("ordinal", "missing_threshold", "wrapper_shell"):
        # Test environment validator directly, otherwise stored manifest digest
        # correctly catches the changed environment before profile validation.
        ref = fixture.write("bad_environment.json", fixture.environment)
        with pytest.raises(lc.ExpansionLaunchError): lc._environment(ref, "cpu-fixture", GPU)
        return
    if mutation == "source_dirty": fixture.change("policy_source", "clean", False)
    if mutation == "wrong_revision": fixture.change("policy_source", "revision", "c" * 40)
    if mutation == "wrong_archive": fixture.change("checkpoint", "archive", {"path": fixture.raw["path"], "sha256": "f" * 64})
    if mutation == "content_digest": fixture.change("checkpoint", "content_tree", {})
    with pytest.raises(lc.ExpansionLaunchError): fixture.request()


def test_end_to_end_accepts_bound_architecture_with_nonzero_positional_difference(fixture):
    fixture.architecture()
    plan = lc.validate_execution_plan(fixture.plan("end_to_end_smoke", rows=[1, 2]))
    assert [row["row_id"] for row in plan.rows] == [1, 2]
    assert plan.store_root == fixture.store.run_root
    assert plan.checkpoint_dir.name == "79999"
    with pytest.raises(lc.ExpansionLaunchError, match="cannot mix"):
        fixture.plan("end_to_end_smoke", rows=[0, 1])


@pytest.mark.parametrize("mutation", ["bare_pass", "missing_measurements", "missing_long", "nonfinite", "training", "different_runtime"])
def test_architecture_string_pass_cannot_bypass_underlying_evidence(fixture, mutation):
    fixture.architecture()
    gate = json.loads(Path(fixture.evidence["architecture_gate"]["path"]).read_text())
    if mutation == "bare_pass": gate = {"status": "PASS"}
    elif mutation == "missing_measurements": gate["measurements"]["sha256"] = "a" * 64
    elif mutation == "different_runtime": gate["binding"]["policy_commit"] = "f" * 40
    else:
        raw = json.loads(Path(gate["measurements"]["path"]).read_text())
        if mutation == "missing_long": raw["cases"] = raw["cases"][:1]
        if mutation == "nonfinite": raw["cases"][0]["cold_latency_ms"] = -1
        if mutation == "training": raw["strict_load"]["random_initialized"] = ["new parameter"]
        gate["measurements"] = fixture.write("bad_measurements.json", raw)
        gate["sources"] = [gate["measurements"]]
    fixture.evidence["architecture_gate"] = fixture.write("bad_architecture.json", gate)
    with pytest.raises(lc.ExpansionLaunchError): fixture.request("end_to_end_smoke")


def test_cpu_pass_cannot_hide_failed_junit(fixture):
    fixture.cpu()
    gate = json.loads(Path(fixture.evidence["cpu_gate"]["path"]).read_text())
    junit = fixture.write("failed.xml", b'<testsuite><testcase><failure/></testcase></testsuite>')
    gate.update(junit=junit, sources=[junit])
    fixture.evidence["cpu_gate"] = fixture.write("failed_cpu.json", gate)
    with pytest.raises(lc.ExpansionLaunchError): fixture.request("architecture_smoke")


def test_formal_requires_full_real_store_audit_and_fresh_same_run_u(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch, formal=True)
    fixture.formal()
    plan = lc.validate_execution_plan(fixture.plan("formal", rows=[1, 2]))
    assert plan.stage == "formal" and len(plan.rows) == 2
    assert plan.require_runtime_stage("formal") is plan
    assert lc.deep_validate_runtime_evidence(plan)["source_smoke_raw_episodes_audited"] == 64
    # Scope remains a shard of the frozen 2400 matrix, not a reduced study.
    assert fixture.store.completeness()["expected_count"] == 2400
    raw = next((fixture.path / "uniform_keyframe_expansion/source-smoke/trajectories").rglob("initial_observations.npz"))
    raw.write_bytes(b"changed underlying archive, not the small manifest")
    assert plan.revalidate().stage == "formal"  # Deliberately not a deep raw-file audit.
    with pytest.raises(ValueError):
        lc.deep_validate_runtime_evidence(plan)


@pytest.mark.parametrize("mutation", ["incomplete_smoke", "benchmark_error", "unpaired_smoke", "v10", "exposure"])
def test_formal_gates_reject_semantic_and_source_shortcuts(tmp_path, monkeypatch, mutation):
    fixture = Fixture(tmp_path, monkeypatch, formal=True)
    kwargs = {"count": 63} if mutation == "incomplete_smoke" else {"error_at": 0} if mutation == "benchmark_error" else {"unequal_at": 1} if mutation == "unpaired_smoke" else {}
    fixture.formal(**kwargs)
    if mutation == "v10":
        # Even if one could rewrite a manifest hash, formal execution must not
        # reuse a pre-transport-amendment protocol or its superseded gates.
        Path(fixture.protocol["path"]).write_text("**Protocol version:** v1.0\n")
        fixture.evidence["protocol"] = lc.file_reference(fixture.protocol["path"])
    if mutation == "exposure":
        exposure = json.loads(Path(fixture.evidence["exposure"]["path"]).read_text())
        exposure["records"].pop()
        fixture.evidence["exposure"] = fixture.write("bad_exposure.json", exposure)
        fixture.freeze()
    with pytest.raises((lc.ExpansionLaunchError, ValueError)):
        fixture.request("formal")


def test_symlink_evidence_rejected(fixture):
    link = fixture.path / "alias"
    link.symlink_to(fixture.raw["path"])
    with pytest.raises(lc.ExpansionLaunchError): lc.file_reference(link)


@pytest.mark.parametrize("rows", [[0, 0], [True], [-1], [64], [0, 1]])
def test_invalid_shard_rows_never_change_the_census(fixture, rows):
    fixture.architecture()
    with pytest.raises((lc.ExpansionLaunchError, ValueError)):
        fixture.request("end_to_end_smoke", rows=rows)
