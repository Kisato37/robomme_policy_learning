"""CPU contracts for persistent lifetime, isolated reset, transport and provenance."""
# ruff: noqa: SLF001, PLC0415
import asyncio
from contextlib import suppress
from dataclasses import replace
import json
import os
import sys
import threading
from types import SimpleNamespace
import uuid

from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from experiments.keyframe_neighborhood_sampling import direct_provenance as provenance
from experiments.keyframe_neighborhood_sampling import resident_policy as resident
from experiments.keyframe_neighborhood_sampling import submit_direct as runner
from experiments.keyframe_neighborhood_sampling.formal_matrix import build_formal_matrix
from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import process_exit_record
from experiments.keyframe_neighborhood_sampling.runner_contract import process_start_record
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_neighborhood_sampling.smoke_matrix import build_smoke_matrix
from experiments.keyframe_oracle_sampling.artifacts import ArtifactContractError
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now
from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer

GPU = "GPU-11111111-1111-1111-1111-111111111111"
ROW = {"task": "BinFill", "episode_id": 0, "arm": "OC3"}


@pytest.mark.parametrize("fault", ["reset", "task", "episode_id", "arm"])
def test_reset_must_match_exact_row(fault):
    request = {"reset": True, "keyframe_selector_config": dict(ROW)}
    if fault == "reset":
        request["reset"] = False
    else:
        request["keyframe_selector_config"][fault] = "wrong"
    with pytest.raises(ValueError, match="reset"):
        resident.validate_reset_request(request, ROW)


@pytest.mark.parametrize("field", list(resident.RESET_STATE))
def test_each_reset_invariant_is_required(field):
    state = {**resident.RESET_STATE, field: "wrong"}
    with pytest.raises(ValueError, match="reset state"):
        resident.validate_reset_response({"reset_finished": True, "resident_reset": {"state": state, "model_process_pid": 11}})


class TinyPolicy:
    """Deterministic stateful CPU stand-in; never scientific GPU evidence."""
    def __init__(self):
        self.reset_count = 0
        self.history = []

    def reset(self):
        self.reset_count += 1
        self.history = []

    def reset_evidence(self):
        assert not self.history
        return dict(resident.RESET_STATE)

    def configure_keyframe_selector(self, config):
        self.arm = config["arm"]

    def add_buffer(self, obs):
        self.history.extend(obs["values"])

    def infer(self, obs):
        return {"actions": [sum(self.history), self.arm]}


def test_real_server_one_model_48_resets_and_no_cross_episode_history():
    async def exercise():
        policy = TinyPolicy()
        handler = WebsocketPolicyServer(policy, execution_id=str(uuid.uuid4()), dispatch_sha256="a" * 64,
                                        exclusive_clients=True)
        packer = msgpack_numpy.Packer()
        actions = []
        async with serve(handler._handler, "127.0.0.1", 0) as server:
            uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            for index in range(48):
                async with connect(uri) as client:
                    info = msgpack_numpy.unpackb(await client.recv())
                    assert info["resident_policy"] is True
                    assert info["model_process_pid"] == os.getpid()
                    await client.send(packer.pack({"reset": True, "keyframe_selector_config": {**ROW, "arm": "OC3" if index % 2 == 0 else "OC5"}}))
                    resident.validate_reset_response(msgpack_numpy.unpackb(await client.recv()))
                    await client.send(packer.pack({"add_buffer": True, "values": [1, 2]}))
                    await client.recv()
                    await client.send(packer.pack({"observation": index}))
                    actions.append(msgpack_numpy.unpackb(await client.recv())["actions"])
                await asyncio.sleep(0)  # let the connection finalizer release its lease
            assert actions[::2] == [[3, "OC3"]] * 24
            assert actions[1::2] == [[3, "OC5"]] * 24
            assert policy.reset_count == 48
    asyncio.run(exercise())


@pytest.mark.parametrize("fault", ["missing_reset", "second_reset", "concurrent_client"])
def test_resident_server_rejects_cross_episode_mutation(fault):
    async def exercise():
        handler = WebsocketPolicyServer(TinyPolicy(), execution_id=str(uuid.uuid4()), dispatch_sha256="a" * 64,
                                        exclusive_clients=True)
        packer = msgpack_numpy.Packer()
        request = packer.pack({"reset": True, "keyframe_selector_config": ROW})
        async with serve(handler._handler, "127.0.0.1", 0) as server:
            uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with connect(uri) as one:
                await one.recv()
                if fault == "missing_reset":
                    await one.send(packer.pack({"observation": 1}))
                    assert "must begin" in await one.recv()
                else:
                    await one.send(request)
                    await one.recv()
                    if fault == "second_reset":
                        await one.send(request)
                        assert "reset twice" in await one.recv()
                    else:
                        async with connect(uri) as two:
                            await two.recv()
                            await two.send(request)
                            assert "Another trajectory" in await two.recv()
    asyncio.run(exercise())


def test_controller_uses_one_resident_queue_for_all_48(monkeypatch, tmp_path):
    matrix = build_smoke_matrix()
    (tmp_path / "protocol").mkdir()
    write_once_record(tmp_path / "protocol/smoke_matrix.json", matrix)
    calls = []
    monkeypatch.setattr(resident, "run_resident_rows", lambda *args: calls.append(args))
    record = {"max_concurrent": 1, "row_ids": list(range(48)), "runtime_profile": {"policy_lifetime": "resident"}}
    runner.run_controller(tmp_path, "development_smoke", [(tmp_path / "submission.json", record)], [[GPU, GPU]])
    assert len(calls) == 1
    assert [r["row_id"] for r in calls[0][3]] == list(range(48))


@pytest.mark.parametrize("stage", ["development_smoke", "formal"])
@pytest.mark.parametrize("failed_row_offset", [None, 0, 10])
def test_resident_rows_acquire_once_load_once_and_cleanup_once(monkeypatch, tmp_path, stage, failed_row_offset):
    rows = build_smoke_matrix()["rows"] if stage == "development_smoke" else build_formal_matrix()["rows"][1000:1600]
    counts = {"admission": 0, "lease": 0, "load": 0, "cleanup": 0}
    class Context:
        pass_fds = (9,)
        port = 22222
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Lease(Context):
        def __enter__(self):
            counts["lease"] += 1
            return self
    class Execution:
        def __init__(self, dispatch, *args, session_row_count):
            self.dispatch = dispatch
            self.digest = "c" * 64
            assert session_row_count == len(rows)
        def start(self, role, cmd, fds):
            counts["load"] += 1
            assert "--exclusive-clients" in cmd
        def complete(self):
            counts["cleanup"] += 1
    def admission(*args):
        counts["admission"] += 1
        return {}
    from experiments.keyframe_neighborhood_sampling import gpu_telemetry
    monkeypatch.setattr(runner, "GpuLease", Lease)
    monkeypatch.setattr(runner, "PortReservation", Context)
    monkeypatch.setattr(runner, "Execution", Execution)
    monkeypatch.setattr(gpu_telemetry, "GpuTelemetry", Context)
    monkeypatch.setattr(runner, "check_gpu_admission", admission)
    monkeypatch.setattr(runner, "readiness", lambda _: None)
    visited = []
    def execute(*a, **k):
        visited.append((a[3]["row_id"], k["resident"]))
        if failed_row_offset is not None and a[3]["row_id"] == rows[failed_row_offset]["row_id"]:
            raise runner.RowInfrastructureError("fixture proved infrastructure failure")
    monkeypatch.setattr(runner, "execute_one", execute)
    root = tmp_path / "runs/keyframe_neighborhood_sampling/batch"
    root.mkdir(parents=True)
    path = root / "submission.json"
    write_once_record(path, {"runtime_profile": {"lock_directory": str(tmp_path)}, "attempt_id": 0,
                            "repository_commit_sha": "a" * 40, "launch_manifest_sha256": "b" * 64,
                            "smoke_matrix_sha256": "c" * 64, "formal_matrix_sha256": "d" * 64,
                            "row_ids": [r["row_id"] for r in rows], "shard_id": 1 if stage == "formal" else None})
    # Linux process identity is deliberately stubbed in this CPU orchestration fixture.
    real_read = type(root).read_text
    def read_text(path, *a, **k):
        return "55555555-5555-4555-8555-555555555555" if str(path) == "/proc/sys/kernel/random/boot_id" else real_read(path, *a, **k)
    monkeypatch.setattr(type(root), "read_text", read_text)
    stop = threading.Event()
    if failed_row_offset is None:
        resident.run_resident_rows(root, stage, path, rows, (GPU, GPU), stop)
    else:
        with pytest.raises(runner.RowInfrastructureError, match="recorded failures"):
            resident.run_resident_rows(root, stage, path, rows, (GPU, GPU), stop)
        assert not stop.is_set()
    assert counts == {"admission": 2, "lease": 1, "load": 1, "cleanup": 1}
    assert [row for row, _ in visited] == [row["row_id"] for row in rows]
    assert len({id(session) for _, session in visited}) == 1


def test_actual_policy_reset_clears_state_but_keeps_model_and_compiled_functions():
    import jax

    from mme_vla_suite.policies.policy import MME_VLA_Policy
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    model = policy._model = object()
    compiled = policy._sample_actions = object()
    policy._seed = 7
    policy.mem_buffer = object()
    policy._prepare_mem_buffer = lambda: setattr(policy, "mem_buffer", SimpleNamespace(_history_feats={}, _history_metadata={}))
    for _ in range(3):
        policy.step_idx = 900
        policy.exec_start_idx = 15
        policy._rng = jax.random.key(123)
        policy._selector_rng = object()
        policy._pending_selector_trace = {"stale": True}
        policy._keyframe_selector_config = dict(ROW)
        policy._selector_call_index = 81
        policy.reset()
        assert policy.reset_evidence() == resident.RESET_STATE
        assert policy._model is model
        assert policy._sample_actions is compiled


@pytest.mark.skipif(sys.platform != "linux", reason="Linux inherited-listener validation")
@pytest.mark.parametrize(("stage", "indices"), [("development_smoke", (0, 1)), ("formal", (999, 1000, 1599))])
def test_real_proxies_share_one_model_and_record_distinct_resets(tmp_path, stage, indices):
    async def exercise():
        root = tmp_path / "runs/keyframe_neighborhood_sampling/proxies"
        (root / "protocol").mkdir(parents=True)
        matrix = build_smoke_matrix() if stage == "development_smoke" else build_formal_matrix()
        name = "smoke_matrix.json" if stage == "development_smoke" else "formal_matrix.json"
        matrix_hash = write_once_record(root / "protocol" / name, matrix)
        policy = TinyPolicy()
        with runner.PortReservation() as model_port:
            session = DirectDispatch(
                execution_id=str(uuid.uuid4()), run_root=str(root), repository_commit_sha="a" * 40,
                stage="policy_session", launch_manifest_sha256="b" * 64, submission_plan_sha256="c" * 64,
                matrix_sha256=matrix_hash, attempt_id=0, row_id=None, shard_id=None, gpu_uuids=(GPU, GPU),
                host_name="fixture", host_boot_id="55555555-5555-4555-8555-555555555555",
                policy_port=model_port.port, gpu_layout="colocated",
            )
            session_path = provenance.dispatch_path(root, stage="policy_session", attempt_id=0, row_id=None, execution_id=session.execution_id)
            session_path.parent.mkdir(parents=True)
            session_hash = write_once_record(session_path, session.as_record())
            server = WebsocketPolicyServer(policy, listen_fd=model_port.pass_fds[0], execution_id=session.execution_id,
                                            dispatch_sha256=session_hash, exclusive_clients=True)
            server_task = asyncio.create_task(server.run())
            receipts = []
            try:
                for index in indices:
                    with runner.PortReservation() as row_port:
                        dispatch = replace(session, execution_id=str(uuid.uuid4()), stage=stage, row_id=index,
                                           shard_id=int(index >= 1000) if stage == "formal" else None, policy_port=row_port.port)
                        row_path = provenance.dispatch_path(root, stage=dispatch.stage, attempt_id=0, row_id=index)
                        row_path.parent.mkdir(parents=True)
                        row_hash = write_once_record(row_path, dispatch.as_record())
                        binding_path = row_path.parent / "policy_session.json"
                        write_once_record(binding_path, {
                            "row_execution_id": dispatch.execution_id, "row_dispatch_sha256": row_hash,
                            "session": {"backend": "direct", "dispatch_path": str(session_path),
                                        "dispatch_sha256": session_hash, "dispatch": session.as_record()},
                        })
                        task = asyncio.create_task(resident.serve_proxy(SimpleNamespace(binding=binding_path, listen_fd=row_port.pass_fds[0])))
                        try:
                            uri = f"ws://127.0.0.1:{row_port.port}"
                            # A readiness probe must not consume the only allowed trajectory.
                            async with connect(uri, open_timeout=3) as ready:
                                metadata = msgpack_numpy.unpackb(await ready.recv())
                                assert metadata["direct_execution"]["execution_id"] == dispatch.execution_id
                            async with connect(uri, open_timeout=3) as client:
                                await client.recv()
                                config = {key: matrix["rows"][index][key] for key in ("task", "episode_id", "arm")}
                                for request in ({"reset": True, "keyframe_selector_config": config},
                                                {"add_buffer": True, "values": [5]}, {"observation": 1}):
                                    await client.send(msgpack_numpy.packb(request))
                                    reply = msgpack_numpy.unpackb(await client.recv())
                                assert reply["actions"] == [5, config["arm"]]
                            receipts.append(json.loads((row_path.parent / "resident_reset.json").read_bytes()))
                        finally:
                            task.cancel()
                            with suppress(asyncio.CancelledError):
                                await task
                assert policy.reset_count == len(indices)
                assert receipts[0]["model_process_pid"] == receipts[1]["model_process_pid"] == os.getpid()
                assert receipts[0]["row_execution_id"] != receipts[1]["row_execution_id"]
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task
    asyncio.run(asyncio.wait_for(exercise(), timeout=20))


@pytest.fixture(params=[None, 0, 999, 1000, 1599])
def resident_evidence(tmp_path, request):
    stage = "development_smoke" if request.param is None else "formal"
    row_id = request.param or 0
    shard_id = int(row_id >= 1000) if stage == "formal" else None
    root = tmp_path / "runs/keyframe_neighborhood_sampling/evidence"
    (root / "protocol").mkdir(parents=True)
    matrix = build_smoke_matrix() if stage == "development_smoke" else build_formal_matrix()
    matrix_hash = write_once_record(root / "protocol" / ("smoke_matrix.json" if stage == "development_smoke" else "formal_matrix.json"), matrix)
    config = {key: matrix["rows"][row_id][key] for key in ("task", "episode_id", "arm")}
    submission_name = "submission_record.json" if stage == "development_smoke" else f"submission_record_shard_{shard_id:02d}.json"
    write_once_record(root / "protocol" / submission_name, {"runtime_profile": {"policy_lifetime": "resident"}})
    session = DirectDispatch(
        execution_id=str(uuid.uuid4()), run_root=str(root), repository_commit_sha="a" * 40,
        stage="policy_session", launch_manifest_sha256="b" * 64, submission_plan_sha256="c" * 64,
        matrix_sha256=matrix_hash, attempt_id=0, row_id=None, shard_id=None, gpu_uuids=(GPU, GPU),
        host_name="fixture", host_boot_id="55555555-5555-4555-8555-555555555555", policy_port=22222, gpu_layout="colocated",
    )
    row = replace(session, execution_id=str(uuid.uuid4()), stage=stage, row_id=row_id, shard_id=shard_id, policy_port=22223)
    def envelope(dispatch):
        path = provenance.dispatch_path(root, stage=dispatch.stage, attempt_id=0, row_id=dispatch.row_id, execution_id=dispatch.execution_id)
        path.parent.mkdir(parents=True)
        return {"backend": "direct", "dispatch_path": str(path), "dispatch_sha256": write_once_record(path, dispatch.as_record()), "dispatch": dispatch.as_record()}
    session_env = envelope(session)
    row_env = envelope(row)
    from pathlib import Path
    session_path = Path(session_env["dispatch_path"])
    row_path = Path(row_env["dispatch_path"])
    def start(dispatch, directory, role, pid):
        data = process_start_record(dispatch, process_identity={
            "boot_id": session.host_boot_id, "pid": pid, "parent_pid": 10, "process_group": pid,
            "session": pid, "start_ticks": pid, "uid": os.getuid(),
        }, command=["fixture", role], working_directory="/fixture")
        return write_once_record(directory / f"{role}_start.json", data)
    model_start = start(session, session_path.parent, "policy", 99)
    binding = {"schema": "resident-policy-binding-v1", "row_execution_id": row.execution_id,
               "row_dispatch_sha256": row_env["dispatch_sha256"], "session": session_env,
               "policy_start_sha256": model_start}
    binding_hash = write_once_record(row_path.parent / "policy_session.json", binding)
    write_once_record(session_path.parent / "row_plan.json", {"stage": stage, "row_ids": [row_id], "submission_shard_id": shard_id})
    reset_hash = write_once_record(row_path.parent / "resident_reset.json", {
        "schema": "resident-reset-v1", "row_execution_id": row.execution_id,
        "session_execution_id": session.execution_id, "binding_sha256": binding_hash,
        "state": dict(resident.RESET_STATE), "model_process_pid": 101,
        "selector_config": config, "recorded_utc": utc_now(),
    })
    completion = {"resident_policy": {"binding_sha256": binding_hash, "reset_sha256": reset_hash}, "finished_utc": utc_now()}
    exit_hash = write_once_record(session_path.parent / "policy_exit.json", process_exit_record(
        session, started_record_sha256=model_start, returncode=-15, wall_clock_limit_reached=False))
    write_once_record(session_path.parent / "completion.json", {
        "backend": "direct", "execution_id": session.execution_id, "dispatch_sha256": session_env["dispatch_sha256"],
        "cleanup_confirmed": True, "roles": {"policy": {"start_sha256": model_start, "exit_sha256": exit_hash}},
        "finished_utc": utc_now(),
    })
    return root, row_env, completion, row_path.parent, session_path.parent


def test_real_resident_provenance_audit_accepts_linked_lifecycle(resident_evidence):
    root, envelope, completion, *_ = resident_evidence
    provenance.audit_resident_binding(envelope, root, completion)


@pytest.mark.parametrize("fault", ["binding_hash", "reset_hash", "reset_state", "session_cleanup", "wrong_selector"])
def test_real_resident_audit_rejects_broken_evidence(resident_evidence, fault):
    root, envelope, completion, row_dir, session_dir = resident_evidence
    if fault == "binding_hash":
        completion["resident_policy"]["binding_sha256"] = "0" * 64
    elif fault == "reset_hash":
        completion["resident_policy"]["reset_sha256"] = "0" * 64
    else:
        path = session_dir / "completion.json" if fault == "session_cleanup" else row_dir / "resident_reset.json"
        data = json.loads(path.read_bytes())
        if fault == "session_cleanup":
            data["cleanup_confirmed"] = False
        elif fault == "wrong_selector":
            data["selector_config"]["arm"] = "wrong"
        else:
            data["state"]["history_empty"] = False
        from experiments.keyframe_neighborhood_sampling.runner_contract import canonical_bytes
        path.write_bytes(canonical_bytes(data))  # deliberate synthetic corruption, never a real artifact
        if fault != "session_cleanup":
            completion["resident_policy"]["reset_sha256"] = sha256_file(path)
    with pytest.raises((ArtifactContractError, ValueError)):
        provenance.audit_resident_binding(envelope, root, completion)


def test_queue_deadline_preserves_each_trajectory_limit():
    assert runner.ROW_SECONDS == 43200
    assert runner.resident_session_seconds(500) == 240 + 500 * (43200 + 120 + 40)
    for invalid in (None, True, 0, -1, 1001):
        with pytest.raises(ValueError, match="queue"):
            runner.resident_session_seconds(invalid)


@pytest.mark.parametrize("fault", [None, "cross_arm", "per_row", "mixed", "failed"])
def test_formal_resident_gate_requires_matching_smoke_lifetime(tmp_path, fault):
    (tmp_path / "protocol").mkdir()
    architecture = {"resident_cross_arm_reset": {"passed": fault != "cross_arm"}}
    audit = {"passed": fault != "failed", "policy_lifetimes": ["resident"]}
    if fault == "per_row":
        audit["policy_lifetimes"] = ["per_row"]
    if fault == "mixed":
        audit["policy_lifetimes"] = ["per_row", "resident"]
    write_once_record(tmp_path / "protocol/architecture_pass_report.json", architecture)
    write_once_record(tmp_path / "protocol/development_smoke_audit.json", audit)
    if fault:
        with pytest.raises(ValueError, match="Resident formal"):
            resident.require_resident_smoke(tmp_path)
    else:
        resident.require_resident_smoke(tmp_path)


def test_resident_formal_never_reads_development_matrix(resident_evidence):
    _root, envelope, *_ = resident_evidence
    dispatch = DirectDispatch.from_record(envelope["dispatch"])
    row = resident.load_resident_row(dispatch)
    assert row["dataset"] == ("test" if dispatch.stage == "formal" else "val")
    with pytest.raises(ValueError, match="digest"):
        resident.load_resident_row(replace(dispatch, matrix_sha256="0" * 64))
