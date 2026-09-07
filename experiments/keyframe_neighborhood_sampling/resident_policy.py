"""Resident model session plus lightweight, isolated per-trajectory transport.

The per-row ``policy`` process is an explicitly recorded transport proxy, not a
second model. Only the session loads weights. Requests/responses are forwarded
byte-for-byte; the first request must reset and configure the frozen trajectory.
"""

# Lazy dependencies keep the proxy free of model initialization.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import threading
import uuid

from experiments.keyframe_neighborhood_sampling.runner_contract import DirectDispatch
from experiments.keyframe_neighborhood_sampling.runner_contract import write_once_record
from experiments.keyframe_oracle_sampling.artifacts import sha256_file
from experiments.keyframe_oracle_sampling.artifacts import utc_now

RESET_STATE = {
    "seed": 7, "history_empty": True, "boundary_metadata_empty": True, "step_idx": -1, "exec_start_idx": 0,
    "selector_call_index": 0, "selector_unconfigured": True,
    "selector_rng_empty": True, "pending_trace_empty": True, "rng_matches_seed": True,
}


def validate_reset_request(obs: dict, row: dict) -> None:
    config = obs.get("keyframe_selector_config")
    if obs.get("reset") is not True or not isinstance(config, dict):
        raise ValueError("First resident request must reset and configure its trajectory")
    expected = {key: row[key] for key in ("task", "episode_id", "arm")}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Resident reset selector differs from its submitted row")


def validate_reset_response(reply: dict) -> dict:
    evidence = reply.get("resident_reset", {})
    if reply.get("reset_finished") is not True or evidence.get("state") != RESET_STATE:
        raise ValueError("Resident model did not attest the complete frozen reset state")
    if type(evidence.get("model_process_pid")) is not int or evidence["model_process_pid"] <= 0:
        raise ValueError("Resident reset lacks actual model PID")
    return evidence


@dataclass
class ResidentSession:
    execution: object
    lease: object

    def binding(self, row_execution) -> dict:
        session = self.execution
        return {
            "schema": "resident-policy-binding-v1",
            "row_execution_id": row_execution.dispatch.execution_id,
            "row_dispatch_sha256": row_execution.digest,
            "session": {
                "backend": "direct", "dispatch_path": str(session.path),
                "dispatch_sha256": session.digest, "dispatch": session.dispatch.as_record(),
            },
            "policy_start_sha256": sha256_file(session.directory / "policy_start.json"),
        }


def run_resident_rows(run_root: Path, stage: str, submission_path: Path, rows: list[dict],
                      gpu_uuids: tuple[str, ...], stop: threading.Event) -> None:
    # Import at the orchestration boundary; proxy startup never imports JAX.
    from experiments.keyframe_neighborhood_sampling import submit_direct as runner
    from experiments.keyframe_neighborhood_sampling.gpu_telemetry import GpuTelemetry

    record = json.loads(submission_path.read_text())
    profile = record["runtime_profile"]
    if stage != "development_smoke":
        raise ValueError("Resident backend is smoke-only until its validation is reviewed")
    physical = tuple(dict.fromkeys(gpu_uuids))
    runner.check_gpu_admission(physical, profile)
    with runner.GpuLease(Path(profile["lock_directory"]), physical) as lease, runner.PortReservation() as port:
        admission = runner.check_gpu_admission(physical, profile)
        session_id = str(uuid.uuid4())
        directory = run_root / "direct/sessions" / session_id
        directory.mkdir(parents=True, exist_ok=False)
        dispatch = DirectDispatch(
            execution_id=session_id, run_root=str(run_root),
            repository_commit_sha=record["repository_commit_sha"], stage="policy_session",
            launch_manifest_sha256=record["launch_manifest_sha256"],
            submission_plan_sha256=sha256_file(submission_path), matrix_sha256=record["smoke_matrix_sha256"],
            attempt_id=record["attempt_id"], row_id=None, shard_id=None, gpu_uuids=gpu_uuids,
            host_name=socket.gethostname(), host_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            policy_port=port.port, gpu_layout="colocated",
        )
        execution = runner.Execution(dispatch, directory, profile, lease, stop)
        write_once_record(directory / "gpu_admission.json", admission)
        write_once_record(directory / "row_plan.json", {"stage": stage, "row_ids": [r["row_id"] for r in rows]})
        command = [
            str(runner.REPO / ".venv/bin/python"), "scripts/serve_policy.py", "--seed=7",
            f"--port={port.port}", f"--listen-fd={port.pass_fds[0]}",
            f"--execution-id={session_id}", f"--dispatch-sha256={execution.digest}", "--exclusive-clients",
            "policy:checkpoint", f"--policy.dir={runner.REPO / runner.CHECKPOINT_RELATIVE}",
            "--policy.config=mme_vla_suite",
        ]
        with GpuTelemetry(directory / "gpu_telemetry.jsonl", physical):
            try:
                execution.start("policy", command, port.pass_fds)
                readiness = runner.readiness(execution)
                if readiness is not None:
                    raise RuntimeError(f"Resident policy startup failed: {readiness}")
                resident = ResidentSession(execution, lease)
                for row in rows:
                    if stop.is_set():
                        raise InterruptedError("Resident batch stopped; remaining rows were not dispatched")
                    runner.execute_one(run_root, stage, submission_path, row, gpu_uuids, stop, resident=resident)
            except BaseException as error:
                write_once_record(directory / "controller_error.json", {
                    "error_type": type(error).__name__, "error": str(error), "recorded_utc": utc_now(),
                    "retry_authorized": False,
                })
                raise
            finally:
                execution.complete()


async def serve_proxy(args) -> None:
    from openpi_client import msgpack_numpy
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed

    from experiments.keyframe_neighborhood_sampling.direct_provenance import _load_runner
    from openpi.serving.websocket_policy_server import adopt_listening_socket

    directory = args.binding.parent
    binding = json.loads(args.binding.read_bytes())
    dispatch = DirectDispatch.from_record(json.loads((directory / "dispatch.json").read_bytes()))
    if binding["row_execution_id"] != dispatch.execution_id or binding["row_dispatch_sha256"] != sha256_file(directory / "dispatch.json"):
        raise ValueError("Resident binding does not name this row dispatch")
    session, _ = _load_runner(binding["session"], Path(dispatch.run_root))
    matrix = json.loads((Path(dispatch.run_root) / "protocol/smoke_matrix.json").read_bytes())
    row = matrix["rows"][dispatch.row_id]
    expected_session = {"execution_id": session.execution_id, "dispatch_sha256": binding["session"]["dispatch_sha256"]}
    metadata = {"direct_execution": {"execution_id": dispatch.execution_id, "dispatch_sha256": binding["row_dispatch_sha256"]}}
    packer = msgpack_numpy.Packer()
    claimed = False

    async def handler(client):
        nonlocal claimed
        await client.send(packer.pack(metadata))
        try:
            # Readiness connections send no data and must not claim an episode.
            first = await client.recv()
            if claimed:
                raise ValueError("Resident row has already accepted its only trajectory")
            claimed = True
            request = msgpack_numpy.unpackb(first)
            validate_reset_request(request, row)
            async with connect(f"ws://127.0.0.1:{session.policy_port}", compression=None, max_size=None,
                               ping_timeout=600, open_timeout=10, close_timeout=10) as upstream:
                info = msgpack_numpy.unpackb(await upstream.recv())
                if info.get("direct_execution") != expected_session or info.get("resident_policy") is not True:
                    raise ValueError("Resident upstream identity mismatch")
                await upstream.send(first)
                raw_reply = await upstream.recv()
                if isinstance(raw_reply, str):
                    raise RuntimeError(raw_reply)
                evidence = validate_reset_response(msgpack_numpy.unpackb(raw_reply))
                if evidence["model_process_pid"] != info.get("model_process_pid"):
                    raise ValueError("Reset and handshake model PIDs differ")
                write_once_record(directory / "resident_reset.json", {
                    "schema": "resident-reset-v1", "row_execution_id": dispatch.execution_id,
                    "session_execution_id": session.execution_id, "binding_sha256": sha256_file(args.binding),
                    "selector_config": request["keyframe_selector_config"], **evidence, "recorded_utc": utc_now(),
                })
                await client.send(raw_reply)
                async for message in client:
                    # Relay exactly the same msgpack bytes; no action transformation.
                    await upstream.send(message)
                    await client.send(await upstream.recv())
        except ConnectionClosed:
            return
        except Exception as error:
            try:
                await client.send(f"Resident policy proxy: {type(error).__name__}: {error}")
                await client.close(code=1011, reason="Resident policy contract failed")
            except ConnectionClosed:
                pass

    listener = adopt_listening_socket(args.listen_fd)
    try:
        async with serve(handler, sock=listener, compression=None, max_size=None, ping_timeout=600) as server:
            await server.serve_forever()
    finally:
        listener.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--listen-fd", type=int, required=True)
    asyncio.run(serve_proxy(parser.parse_args()))


if __name__ == "__main__":
    main()
