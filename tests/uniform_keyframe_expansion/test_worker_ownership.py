"""CPU synthetic Linux identity tables: no processes started, signaled or killed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.uniform_keyframe_expansion import worker_ownership as w


BOOT = "11111111-1111-4111-8111-111111111111"


class Plan:
    stage = "end_to_end_smoke"
    rows = [{"row_id": 0}]
    execution_identity = {"execution_id": BOOT, "dispatch_sha256": "a" * 64}
    policy_root = Path("/fixture/policy")

    def require_runtime_stage(self):
        if self.stage == "cpu_prepare":
            raise ValueError("No runtime authority")


@pytest.fixture
def case(tmp_path, monkeypatch):
    plan = Plan()
    plan.store_root = tmp_path
    monkeypatch.setitem(sys.modules, "experiments.uniform_keyframe_expansion.launch_contract",
                        SimpleNamespace(ValidatedExecutionPlan=Plan))
    monkeypatch.setattr(w.sys, "platform", "linux")
    monkeypatch.setattr(w.os, "getuid", lambda: 501)
    monkeypatch.setattr(w.os, "getpid", lambda: 300)
    monkeypatch.setattr(w.time, "monotonic", lambda: 100)
    monkeypatch.setattr(w, "_boot_id", lambda: BOOT)
    def identity(pid, parent, group):
        return dict(boot_id=BOOT, pid=pid, parent_pid=parent, process_group=group,
                    session=group, start_ticks=pid * 100, uid=501)
    processes = {100: identity(100, 50, 50), 200: identity(200, 100, 200),
                 250: identity(250, 200, 200), 300: identity(300, 250, 200)}
    record = {"execution_identity": deepcopy(plan.execution_identity), "role": "simulator",
              "process_identity": deepcopy(processes[200]), "controller_identity": deepcopy(processes[100]),
              "cwd": str(plan.policy_root), "deadline_monotonic": 200}
    monkeypatch.setattr(w, "_record", lambda path: (deepcopy(record), "c" * 64))
    reads = []
    def read(pid, boot):
        reads.append(pid)
        if pid not in processes:
            raise ProcessLookupError("Fixture process not live")
        return deepcopy(processes[pid])
    monkeypatch.setattr(w, "_read_own_process", read)
    return plan, record, processes, reads


def test_controlled_shell_ancestry_is_accepted_and_read_only(case):
    plan, record, processes, reads = case
    before = deepcopy((record, processes))
    result = w.verify_worker_ownership(plan, "simulator", row_id=0)
    assert result["ancestor_pid_chain"] == [300, 250, 200]
    assert result["worker_ownership_verified"] is True
    assert result["start_record_path"].endswith("row_0000_start.json")
    assert set(reads) == {100, 200, 250, 300}
    assert (record, processes) == before


def test_direct_policy_child_and_policy_artifact_name(case):
    plan, record, processes, _ = case
    record["role"] = "policy"
    processes[300]["parent_pid"] = 200
    result = w.verify_worker_ownership(plan, "policy")
    assert result["ancestor_pid_chain"] == [300, 200]
    assert result["start_record_path"].endswith("policy_start.json")


def test_architecture_uses_separate_owned_start_record_without_trajectory_authority(case):
    plan, record, _, _ = case
    plan.stage, plan.rows, record["role"] = "architecture_smoke", [], "policy"
    result = w.verify_worker_ownership(plan, "architecture")
    assert result["role"] == "architecture" and result["row_id"] is None
    assert result["start_record_path"].endswith("architecture_start.json")
    for role in ("policy", "simulator"):
        with pytest.raises(w.WorkerOwnershipError):
            w.verify_worker_ownership(plan, role)


@pytest.mark.parametrize("fault", ["trajectory_stage", "nonempty_rows", "row_argument"])
def test_architecture_cannot_borrow_other_stage_or_row_authority(case, monkeypatch, fault):
    plan, _, _, _ = case
    plan.stage, plan.rows = "architecture_smoke", []
    if fault == "trajectory_stage": plan.stage = "formal"
    elif fault == "nonempty_rows": plan.rows = [{"row_id": 0}]
    monkeypatch.setattr(w, "_record", lambda _: pytest.fail("Must reject before process evidence"))
    with pytest.raises(w.WorkerOwnershipError):
        w.verify_worker_ownership(plan, "architecture", row_id=0 if fault == "row_argument" else None)


@pytest.mark.parametrize("fault", ["bare_plan", "cpu", "architecture", "nonlinux", "wrong_row", "policy_row"])
def test_rejects_before_reading_start_evidence(case, monkeypatch, fault):
    plan, _, _, _ = case
    role, row = "simulator", 0
    if fault == "bare_plan": plan = {"approved": True}
    elif fault == "cpu": plan.stage = "cpu_prepare"
    elif fault == "architecture": plan.stage = "architecture_smoke"
    elif fault == "nonlinux": monkeypatch.setattr(w.sys, "platform", "darwin")
    elif fault == "wrong_row": row = 1
    else: role = "policy"
    monkeypatch.setattr(w, "_record", lambda _: pytest.fail("No start evidence should be read"))
    with pytest.raises((w.WorkerOwnershipError, ValueError)):
        w.verify_worker_ownership(plan, role, row_id=row)


@pytest.mark.parametrize("fault", ["execution", "role", "missing_controller", "cwd", "expired", "nan",
                                   "boot", "uid", "boolean_pid", "leader_group", "leader_parent"])
def test_forged_or_stale_start_records_rejected(case, fault):
    plan, record, _, _ = case
    if fault == "execution": record["execution_identity"]["dispatch_sha256"] = "b" * 64
    elif fault == "role": record["role"] = "policy"
    elif fault == "missing_controller": record.pop("controller_identity")
    elif fault == "cwd": record["cwd"] = "/other/checkout"
    elif fault == "expired": record["deadline_monotonic"] = 100
    elif fault == "nan": record["deadline_monotonic"] = float("nan")
    elif fault == "boot": record["process_identity"]["boot_id"] = "another-boot"
    elif fault == "uid": record["controller_identity"]["uid"] = 999
    elif fault == "boolean_pid": record["controller_identity"]["pid"] = True
    elif fault == "leader_group": record["process_identity"]["process_group"] = 199
    else: record["process_identity"]["parent_pid"] = 101
    with pytest.raises(w.WorkerOwnershipError):
        w.verify_worker_ownership(plan, "simulator", row_id=0)


@pytest.mark.parametrize("fault", ["controller_dead", "controller_reused", "watchdog_reused",
                                   "escaped", "wrong_session", "other_user", "reparented", "cycle"])
def test_live_process_identity_ancestry_failures(case, fault):
    plan, _, processes, _ = case
    if fault == "controller_dead": processes.pop(100)
    elif fault == "controller_reused": processes[100]["start_ticks"] += 1
    elif fault == "watchdog_reused": processes[200]["start_ticks"] += 1
    elif fault == "escaped": processes[300]["process_group"] = 999
    elif fault == "wrong_session": processes[250]["session"] = 999
    elif fault == "other_user": processes[250]["uid"] = 999
    elif fault == "reparented": processes[300]["parent_pid"] = 100
    else: processes[250]["parent_pid"] = 300
    with pytest.raises((w.WorkerOwnershipError, ProcessLookupError)):
        w.verify_worker_ownership(plan, "simulator", row_id=0)


def test_parent_exit_between_initial_and_final_check_is_not_hidden(case, monkeypatch):
    plan, _, processes, _ = case
    calls = []
    def read(pid, boot):
        calls.append(pid)
        if pid == 100 and calls.count(100) > 1:
            raise ProcessLookupError("controller exited during check")
        return deepcopy(processes[pid])
    monkeypatch.setattr(w, "_read_own_process", read)
    with pytest.raises(ProcessLookupError):
        w.verify_worker_ownership(plan, "simulator", row_id=0)


def test_start_file_reader_rejects_symlinks_and_handles_ordinary_json(tmp_path):
    path = tmp_path / "start.json"
    path.write_text(json.dumps({"fixture": True}))
    value, checksum = w._record(path)
    assert value == {"fixture": True} and len(checksum) == 64
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(w.WorkerOwnershipError):
        w._record(alias)


@pytest.mark.parametrize("state", ["S", "R", "Z", "X"])
def test_linux_stat_parser_handles_parentheses_and_rejects_dead_processes(tmp_path, monkeypatch, state):
    monkeypatch.setattr(w, "_PROC_ROOT", tmp_path)
    directory = tmp_path / "300"
    directory.mkdir()
    fields = [state, "250", "200", "200", *("0" for _ in range(15)), "30000"]
    (directory / "stat").write_text("300 (worker (name) with spaces)) " + " ".join(fields))
    if state in {"Z", "X"}:
        with pytest.raises(w.WorkerOwnershipError, match="exited"):
            w._read_own_process(300, BOOT)
    else:
        identity = w._read_own_process(300, BOOT)
        assert identity["parent_pid"] == 250 and identity["start_ticks"] == 30000
        assert identity["process_group"] == identity["session"] == 200
