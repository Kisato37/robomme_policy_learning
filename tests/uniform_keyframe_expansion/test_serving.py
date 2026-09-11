"""CPU fake-policy websocket tests; no checkpoint, simulator, GPU or SSH."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
import queue
import threading
import time
import types
from unittest import mock

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.server import serve
from websockets.sync.client import connect

from experiments.uniform_keyframe_expansion.contract import (
    EXPANDED_POLICY_VARIANT, U_POLICY_VARIANT, build_selector_config,
    build_smoke_matrix,
)
from experiments.uniform_keyframe_expansion import serving
from mme_vla_suite.shared.uniform_keyframe_config import (
    RELEASED_HISTORY_CONFIG, expanded_history_mapping,
)
from mme_vla_suite.shared.uniform_keyframe_expansion import ExpansionInvariantError


IDENTITY = {"execution_id": "11111111-1111-4111-8111-111111111111", "dispatch_sha256": "a" * 64}


def config(arm="UK48"):
    rows = build_smoke_matrix()["rows"]
    return build_selector_config(next(row for row in rows if row["arm"] == arm))


class FakePolicy:
    def __init__(self):
        from omegaconf import OmegaConf
        self._seed = 7
        self.metadata = {"uniform_keyframe_expansion": expanded_history_mapping(RELEASED_HISTORY_CONFIG)[1]}
        self.config = OmegaConf.create(RELEASED_HISTORY_CONFIG)
        self.resets = 0
        self.frames = []
        self.selector_config = None
        self.failure = None
        self.bad_reset = False

    def reset(self):
        self.resets += 1
        self.frames = []
        self.selector_config = None

    def reset_evidence(self):
        state = deepcopy(serving.RESET_STATE)
        if self.bad_reset:
            state["history_empty"] = False
        return state

    def configure_uniform_keyframe_expansion(self, value):
        self.selector_config = deepcopy(value)

    def configure_keyframe_selector(self, value):
        self.selector_config = deepcopy(value)

    def add_buffer(self, value):
        self.frames.append(value)

    def infer(self, value):
        if self.failure:
            raise self.failure
        return {"actions": np.arange(160, dtype=np.float64).reshape(20, 8),
                "selector_trace": {"fake_fixture": True}, "infer_time_ms": 1.0}


@contextmanager
def loopback(policy=None, *, policy_variant=EXPANDED_POLICY_VARIANT):
    policy = policy or FakePolicy()
    server = serving.ExpansionPolicyServer(policy, execution_identity=IDENTITY,
                                           policy_variant=policy_variant)
    ready = queue.Queue()
    stop = threading.Event()

    async def worker():
        try:
            async with serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as listener:
                ready.put(listener.sockets[0].getsockname()[1])
                await asyncio.to_thread(stop.wait)
        except BaseException as error:
            ready.put(error)

    thread = threading.Thread(target=lambda: asyncio.run(worker()), daemon=True)
    thread.start()
    try:
        port = ready.get(timeout=5)
        if isinstance(port, BaseException):
            raise port
        yield policy, server, port
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), "Loopback fixture did not shut down"


def client(port, *, expected_policy_variant=EXPANDED_POLICY_VARIANT, **kwargs):
    return serving.ExpansionClient("127.0.0.1", port, expected_execution_identity=IDENTITY,
                                   expected_policy_variant=expected_policy_variant,
                                   connect_timeout=2, response_timeout=2, **kwargs)


def observation():
    return {"observation/image": np.zeros((2, 2, 3), np.uint8),
            "observation/wrist_image": np.ones((2, 2, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": "unchanged task", "keyframe_environment_step": 0}


def history():
    return {"add_buffer": True, "images": np.zeros((1, 1, 2, 2, 3), np.uint8),
            "state": np.zeros((1, 8), np.float32), "exec_start_idx": 0,
            "current_task_index": np.asarray([0], np.int64)}


def wait_released(server):
    deadline = time.monotonic() + 2
    while server._active_trajectory is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server._active_trajectory is None


def test_real_loopback_reset_append_infer_and_cross_arm_new_connection():
    with loopback() as (policy, server, port):
        assert server._keepalive_timeout == serving.TRANSPORT_KEEPALIVE_TIMEOUT_SECONDS
        for arm in ("UK48", "UN48"):
            with client(port) as remote:
                metadata = remote.get_server_metadata()
                assert metadata["effective_memory_budget"] == 768
                assert metadata["transport_keepalive_timeout_seconds"] == 600
                assert metadata["direct_execution"] == IDENTITY
                reply = remote.reset(config(arm))
                evidence = serving.validate_reset_response(reply, config(arm))
                assert evidence["state"] == serving.RESET_STATE
                assert evidence["model_process_pid"] == metadata["model_process_pid"]
                assert remote.add_buffer(history())["add_buffer_finished"] is True
                response = remote.infer(observation())
                np.testing.assert_array_equal(response["actions"], np.arange(160).reshape(20, 8))
                np.testing.assert_array_equal(policy.frames[0]["current_task_index"], [0])
                assert policy.selector_config["arm"] == arm
                metadata["effective_memory_budget"] = 512
                assert remote.get_server_metadata()["effective_memory_budget"] == 768
            wait_released(server)
        assert policy.resets == 2


def test_released_u_server_has_512_budget_and_translates_reset_without_expansion():
    with loopback(policy_variant=U_POLICY_VARIANT) as (policy, server, port):
        with client(port, expected_policy_variant=U_POLICY_VARIANT) as remote:
            metadata = remote.get_server_metadata()
            assert metadata["policy_variant"] == U_POLICY_VARIANT
            assert metadata["effective_memory_budget"] == 512
            remote.reset(config("U"))
            assert policy.selector_config["arm"] == "U"
            assert policy.selector_config["seed_table_dataset"] == "val"
            assert len(policy.selector_config["random_seeds"]) == 82
        wait_released(server)


def test_readiness_connection_never_claims_episode_and_other_client_cannot_steal():
    with loopback() as (policy, server, port):
        with client(port):
            assert server._active_trajectory is None
        with client(port) as first:
            first.reset(config())
            with client(port) as second:
                with pytest.raises(serving.ExpansionRemoteError, match="Another trajectory"):
                    second.reset(config("UN48"))
            assert server._hard_failure is None
            first.infer(observation())
        wait_released(server)
        assert policy.resets == 1


def test_server_first_message_and_second_reset_are_enforced_on_wire():
    with loopback() as (_, server, port):
        with connect(f"ws://127.0.0.1:{port}", compression=None) as raw:
            msgpack_numpy.unpackb(raw.recv(timeout=2))
            raw.send(msgpack_numpy.packb({"operation": "infer", "observation": observation()}))
            error = msgpack_numpy.unpackb(raw.recv(timeout=2))["error"]
            assert error["classification"] == "hard_stop"
            assert "First trajectory" in error["message"]
        assert server._hard_failure is None
        with connect(f"ws://127.0.0.1:{port}", compression=None) as raw:
            raw.recv(timeout=2)
            reset = msgpack_numpy.packb({"operation": "reset", "config": config()})
            raw.send(reset)
            assert msgpack_numpy.unpackb(raw.recv(timeout=2))["reset_finished"] is True
            raw.send(reset)
            assert "reset twice" in msgpack_numpy.unpackb(raw.recv(timeout=2))["error"]["message"]
        assert server._hard_failure is not None


def test_protocol_model_failure_preserves_structured_evidence_and_latches_session():
    policy = FakePolicy()
    policy.failure = ExpansionInvariantError("capacity_overflow", {"required_frame_count": 49})
    with loopback(policy) as (_, server, port):
        with client(port) as remote:
            remote.reset(config())
            with pytest.raises(serving.ExpansionRemoteError) as error:
                remote.infer(observation())
            assert error.value.classification == "hard_stop"
            assert error.value.error_type == "ExpansionInvariantError"
            assert error.value.evidence["required_frame_count"] == 49
            assert error.value.evidence["error_code"] == "capacity_overflow"
        wait_released(server)
        with client(port) as remote:
            with pytest.raises(serving.ExpansionRemoteError, match="latched"):
                remote.reset(config("UN48"))
        assert policy.resets == 1


def test_privileged_labels_cannot_enter_model_inference_inputs():
    with loopback() as (_, server, port):
        with client(port) as remote:
            remote.reset(config())
            obs = observation()
            obs["current_task_index"] = 7
            with pytest.raises(serving.ExpansionRemoteError, match="boundary labels"):
                remote.infer(obs)
        assert server._hard_failure is not None


def test_bad_reset_evidence_is_not_acknowledged():
    policy = FakePolicy()
    policy.bad_reset = True
    with loopback(policy) as (_, server, port):
        with client(port) as remote:
            with pytest.raises(serving.ExpansionRemoteError, match="reset state"):
                remote.reset(config())
        assert server._hard_failure is not None


def test_client_local_lifecycle_and_bad_config_do_not_send_partial_requests():
    with loopback() as (policy, _, port):
        with client(port) as remote:
            with pytest.raises(serving.ExpansionRemoteError, match="Reset/configure"):
                remote.infer(observation())
            with pytest.raises(serving.ExpansionRemoteError, match="Reset/configure"):
                remote.add_buffer(history())
            bad = config()
            bad["random_seeds"][0] += 1
            with pytest.raises(ValueError, match="seed table"):
                remote.reset(bad)
            assert policy.resets == 0
            remote.reset(config())
            with pytest.raises(serving.ExpansionRemoteError, match="reset twice"):
                remote.reset(config())
        remote.close()  # idempotent
        with pytest.raises(serving.ExpansionRemoteError, match="closed"):
            remote.infer(observation())


@pytest.mark.parametrize("field,value", [
    ("experiment_family", "keyframe_oracle_sampling"), ("wire_schema", True),
    ("policy_variant", U_POLICY_VARIANT),
    ("effective_memory_budget", 512), ("evaluation_policy_seed", 42),
    ("resident_policy", False), ("strict_weight_tree_load", False),
    ("transport_keepalive_timeout_seconds", 20),
    ("source_history_config_sha256", "b" * 64), ("effective_history_config_sha256", "b" * 64),
    ("model_process_pid", True), ("direct_execution", {**IDENTITY, "dispatch_sha256": "b" * 64}),
])
def test_metadata_contract_rejects_wrong_server(field, value):
    server = serving.ExpansionPolicyServer(FakePolicy(), execution_identity=IDENTITY)
    metadata = deepcopy(server._metadata)
    metadata[field] = value
    with pytest.raises(ValueError):
        serving.validate_server_metadata(metadata, IDENTITY)


def test_real_handshake_rejects_wrong_execution_before_any_model_reset():
    with loopback() as (policy, _, port):
        with pytest.raises(serving.ExpansionRemoteError, match="handshake rejected"):
            serving.ExpansionClient("127.0.0.1", port,
                                    expected_execution_identity={**IDENTITY, "dispatch_sha256": "b" * 64},
                                    connect_timeout=2)
        assert policy.resets == 0


def test_connection_failure_does_not_retry_and_response_timeout_remains_transport():
    with mock.patch.object(serving.websockets.sync.client, "connect", side_effect=ConnectionRefusedError) as dial:
        with pytest.raises(ConnectionRefusedError):
            client(12345)
        assert dial.call_count == 1
    server = serving.ExpansionPolicyServer(FakePolicy(), execution_identity=IDENTITY)
    fake_ws = mock.Mock()
    fake_ws.recv.side_effect = [msgpack_numpy.packb(server._metadata), TimeoutError("bounded receive")]
    with mock.patch.object(serving.websockets.sync.client, "connect", return_value=fake_ws) as dial:
        remote = client(12345)
        assert dial.call_args.kwargs["ping_timeout"] == serving.TRANSPORT_KEEPALIVE_TIMEOUT_SECONDS
        with pytest.raises(TimeoutError):
            remote.reset(config())
        assert dial.call_count == 1
        fake_ws.close.assert_called_once()


def test_loader_is_explicit_and_lazy_without_loading_actual_weights(monkeypatch, tmp_path):
    loader = mock.Mock(return_value="fake-policy")
    train_config = object()
    monkeypatch.setitem(__import__("sys").modules, "mme_vla_suite.policies.policy_config",
                        types.SimpleNamespace(create_trained_policy=loader))
    monkeypatch.setitem(__import__("sys").modules, "mme_vla_suite.training.config",
                        types.SimpleNamespace(get_config=lambda name: train_config if name == "mme_vla_suite" else None))
    path = tmp_path / "79999"
    assert serving.load_expansion_policy(path) == "fake-policy"
    loader.assert_called_once_with(train_config, path, seed=7,
                                   experimental_memory_expansion="uniform_keyframe_expansion-v1",
                                   strict_weight_tree_load=True)
    with pytest.raises(ValueError, match="79999"):
        serving.load_expansion_policy(tmp_path / "wrong")


def test_no_generic_policy_or_default_identity_can_enable_new_serving():
    policy = FakePolicy()
    policy.metadata = {}
    with pytest.raises(ValueError, match="configuration evidence"):
        serving.ExpansionPolicyServer(policy, execution_identity=IDENTITY)
    for invalid in (None, {}, {"execution_id": None, "dispatch_sha256": None}):
        with pytest.raises(ValueError):
            serving.ExpansionPolicyServer(FakePolicy(), execution_identity=invalid)
