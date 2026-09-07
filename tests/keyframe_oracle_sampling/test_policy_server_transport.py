"""CPU-only transport tests: no model, checkpoint, JAX, or simulator is loaded."""
# ruff: noqa: SLF001 - lifecycle assertions intentionally inspect transport internals.

import asyncio
from contextlib import suppress
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

from openpi_client import msgpack_numpy
import pytest
import tyro
from websockets.asyncio.client import connect
from websockets.sync.client import connect as sync_connect

from mme_vla_suite.serving import websocket_policy_server as mme_server
from openpi.serving import websocket_policy_server as transport
from scripts import serve_policy

EXECUTION_ID = "12345678-1234-4234-8234-123456789abc"
DISPATCH_SHA256 = "a" * 64
DIRECT_EXECUTION = {"execution_id": EXECUTION_ID, "dispatch_sha256": DISPATCH_SHA256}


@pytest.fixture
def reservation():
    if sys.platform != "linux":
        pytest.skip("strict inherited-listener adoption requires Linux SO_ACCEPTCONN support")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.metadata = {"policy": "cpu-test"}

    def infer(self, observation):
        self.calls.append(("infer", observation))
        return {"actions": [1]}

    def reset(self):
        self.calls.append(("reset",))

    def configure_keyframe_selector(self, config):
        self.calls.append(("configure", config))

    def add_buffer(self, observation):
        self.calls.append(("add_buffer", observation))


def direct_server(server_class, policy, listener, **kwargs):
    return server_class(
        policy,
        metadata=policy.metadata,
        listen_fd=listener.fileno(),
        execution_id=EXECUTION_ID,
        dispatch_sha256=DISPATCH_SHA256,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("listen_fd", "execution_id", "dispatch_sha256"),
    [
        (True, EXECUTION_ID, DISPATCH_SHA256),
        (2, EXECUTION_ID, DISPATCH_SHA256),
        (3, None, None),
        (None, EXECUTION_ID, None),
        (None, None, DISPATCH_SHA256),
        (None, "not-a-uuid", DISPATCH_SHA256),
        (None, EXECUTION_ID.upper(), DISPATCH_SHA256),
        (None, EXECUTION_ID, "A" * 64),
        (None, EXECUTION_ID, "a" * 63),
    ],
)
def test_invalid_direct_options_fail(listen_fd, execution_id, dispatch_sha256):
    with pytest.raises(ValueError, match=r"listen_fd|execution_id|dispatch_sha256|UUID"):
        transport.validate_direct_options(listen_fd, execution_id, dispatch_sha256)


@pytest.mark.parametrize("server_class", [transport.WebsocketPolicyServer, mme_server.WebsocketPolicyServer])
def test_metadata_is_backwards_compatible_and_direct_identity_does_not_mutate_policy(server_class):
    policy = FakePolicy()
    legacy = server_class(policy, metadata=policy.metadata)
    assert legacy._metadata == {"policy": "cpu-test"}
    direct = server_class(policy, metadata=policy.metadata, execution_id=EXECUTION_ID, dispatch_sha256=DISPATCH_SHA256)
    assert direct._metadata == {"policy": "cpu-test", "direct_execution": DIRECT_EXECUTION}
    assert policy.metadata == {"policy": "cpu-test"}
    with pytest.raises(ValueError, match="reserved"):
        server_class(
            policy,
            metadata={"direct_execution": {}},
            execution_id=EXECUTION_ID,
            dispatch_sha256=DISPATCH_SHA256,
        )


def test_adoption_duplicates_without_rebinding_and_caller_can_close_its_copy(reservation):
    address = reservation.getsockname()
    with transport.adopt_listening_socket(reservation.fileno()) as adopted:
        assert adopted.fileno() != reservation.fileno()
        assert not os.get_inheritable(adopted.fileno())
        assert adopted.getsockname() == address
        reservation.close()
        assert adopted.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        with socket.socket() as competitor, pytest.raises(OSError, match=r"[Aa]ddress already in use"):
            competitor.bind(address)


@pytest.mark.parametrize("invalid_kind", ["unbound", "bound", "wildcard", "datagram"])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux listener validation")
def test_invalid_socket_rejection_preserves_callers_descriptor(invalid_kind):
    kind = socket.SOCK_DGRAM if invalid_kind == "datagram" else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, kind) as listener:
        if invalid_kind != "unbound":
            listener.bind(("0.0.0.0" if invalid_kind == "wildcard" else "127.0.0.1", 0))
        if invalid_kind == "wildcard":
            listener.listen()
        original_fd = listener.fileno()
        with pytest.raises(ValueError, match="listen_fd"):
            transport.adopt_listening_socket(original_fd)
        assert listener.fileno() == original_fd
        listener.getsockname()


def test_non_socket_rejection_preserves_callers_descriptor(tmp_path):
    with (tmp_path / "ordinary-file").open("x+") as handle:
        with pytest.raises(OSError, match=r"[Ss]ocket"):
            transport.adopt_listening_socket(handle.fileno())
        os.fstat(handle.fileno())


@pytest.mark.parametrize("server_class", [transport.WebsocketPolicyServer, mme_server.WebsocketPolicyServer])
def test_real_inherited_socket_metadata_readiness_and_cancellation(server_class, reservation, monkeypatch):
    policy = FakePolicy()
    # These intentionally unusable values prove that inherited mode doesn't bind host/port.
    server = direct_server(server_class, policy, reservation, host="not-a-local-host.invalid", port=-1)
    adopted = []
    original_adopt = transport.adopt_listening_socket

    def track_adopt(fd):
        listener = original_adopt(fd)
        adopted.append(listener)
        return listener

    monkeypatch.setattr(transport, "adopt_listening_socket", track_adopt)
    address = reservation.getsockname()

    async def exercise():
        task = asyncio.create_task(server.run())
        try:
            async with connect(f"ws://127.0.0.1:{address[1]}", open_timeout=3, close_timeout=1) as websocket:
                metadata = msgpack_numpy.unpackb(await asyncio.wait_for(websocket.recv(), timeout=3))
                assert metadata == {"policy": "cpu-test", "direct_execution": DIRECT_EXECUTION}
                assert policy.calls == []
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)

    asyncio.run(exercise())
    assert len(adopted) == 1
    assert adopted[0].fileno() == -1
    assert reservation.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
    with socket.socket() as competitor, pytest.raises(OSError, match=r"[Aa]ddress already in use"):
        competitor.bind(address)


def test_mme_handlers_are_preserved_with_direct_transport(reservation):
    policy = FakePolicy()
    server = direct_server(mme_server.WebsocketPolicyServer, policy, reservation)
    selector = {"condition": "test-only"}
    reset = {"reset": True, "keyframe_selector_config": selector}
    buffer = {"add_buffer": True, "observation": 1}
    infer = {"observation": 2}

    async def exercise():
        task = asyncio.create_task(server.run())
        try:
            async with connect(
                f"ws://127.0.0.1:{reservation.getsockname()[1]}", open_timeout=3, close_timeout=1
            ) as websocket:
                await asyncio.wait_for(websocket.recv(), timeout=3)
                responses = []
                for request in (reset, buffer, infer):
                    await websocket.send(msgpack_numpy.Packer().pack(request))
                    responses.append(msgpack_numpy.unpackb(await asyncio.wait_for(websocket.recv(), timeout=3)))
                assert responses[0]["reset_finished"] is True
                assert responses[1]["add_buffer_finished"] is True
                assert responses[2] == {"actions": [1]}
                assert policy.calls == [("reset",), ("configure", selector), ("add_buffer", buffer), ("infer", infer)]
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)

    asyncio.run(exercise())


def test_real_child_adopts_passed_fd_and_retains_listener_after_parent_close(reservation):
    address = reservation.getsockname()
    child_source = """
import sys
from types import SimpleNamespace
from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer
assert not any(name in sys.modules for name in ('jax', 'torch', 'sapien', 'mani_skill'))
policy = SimpleNamespace(infer=lambda obs: {'actions': []})
WebsocketPolicyServer(
    policy,
    listen_fd=int(sys.argv[1]),
    execution_id=sys.argv[2],
    dispatch_sha256=sys.argv[3],
).serve_forever()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_source, str(reservation.fileno()), EXECUTION_ID, DISPATCH_SHA256],
        pass_fds=(reservation.fileno(),),
    )
    try:
        with sync_connect(f"ws://127.0.0.1:{address[1]}", open_timeout=5, close_timeout=1) as websocket:
            assert msgpack_numpy.unpackb(websocket.recv(timeout=3)) == {"direct_execution": DIRECT_EXECUTION}
        reservation.close()
        # The child's copy now owns the listener independently of the parent.
        with sync_connect(f"ws://127.0.0.1:{address[1]}", open_timeout=3, close_timeout=1) as websocket:
            assert msgpack_numpy.unpackb(websocket.recv(timeout=3)) == {"direct_execution": DIRECT_EXECUTION}
        assert child.poll() is None
    finally:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)


def test_startup_failure_closes_only_adopted_copy(reservation, monkeypatch):
    captured = {}

    def fail_startup(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("test startup failure")

    monkeypatch.setattr(transport._server, "serve", fail_startup)
    with pytest.raises(RuntimeError, match="test startup failure"):
        asyncio.run(direct_server(mme_server.WebsocketPolicyServer, FakePolicy(), reservation).run())
    assert captured["sock"].fileno() == -1
    assert "host" not in captured
    assert "port" not in captured
    assert reservation.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)


def test_legacy_run_preserves_bind_and_health_options(monkeypatch):
    captured = {}

    def capture_startup(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before binding")

    monkeypatch.setattr(transport._server, "serve", capture_startup)
    with pytest.raises(RuntimeError, match="stop before binding"):
        asyncio.run(mme_server.WebsocketPolicyServer(FakePolicy(), host="localhost", port=8123).run())
    assert captured == {
        "host": "localhost",
        "port": 8123,
        "compression": None,
        "max_size": None,
        "process_request": transport._health_check,
    }
    connection = SimpleNamespace(respond=lambda status, body: (status, body))
    assert transport._health_check(connection, SimpleNamespace(path="/healthz")) == (200, "OK\n")
    assert transport._health_check(connection, SimpleNamespace(path="/")) is None


def test_cli_flags_parse_and_reach_actual_mme_server(reservation, monkeypatch):
    args = tyro.cli(
        serve_policy.Args,
        args=[
            "--listen-fd",
            str(reservation.fileno()),
            "--execution-id",
            EXECUTION_ID,
            "--dispatch-sha256",
            DISPATCH_SHA256,
            "--port",
            "9001",
            "policy:checkpoint",
            "--policy.config",
            "test",
            "--policy.dir",
            "/not-loaded",
        ],
    )
    policy = FakePolicy()
    captured = {}
    monkeypatch.setattr(serve_policy, "create_policy", lambda actual_args: policy)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda host: "127.0.0.1")

    def record_run(server):
        captured["server"] = server

    monkeypatch.setattr(mme_server.WebsocketPolicyServer, "serve_forever", record_run)
    serve_policy.main(args)
    assert isinstance(captured["server"], mme_server.WebsocketPolicyServer)
    assert captured["server"]._listen_fd == reservation.fileno()
    assert captured["server"]._metadata["direct_execution"] == DIRECT_EXECUTION
    assert reservation.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)


def test_invalid_descriptor_fails_before_policy_loading(monkeypatch):
    def forbidden_load(args):
        pytest.fail("invalid descriptor must fail before loading policy")

    monkeypatch.setattr(serve_policy, "create_policy", forbidden_load)
    with socket.socket() as unbound, pytest.raises(ValueError, match="listening"):
        serve_policy.main(
            serve_policy.Args(listen_fd=unbound.fileno(), execution_id=EXECUTION_ID, dispatch_sha256=DISPATCH_SHA256)
        )


def test_cli_direct_flags_parse_without_loading_models():
    args = tyro.cli(
        serve_policy.Args,
        args=[
            "--listen-fd",
            "9",
            "--execution-id",
            EXECUTION_ID,
            "--dispatch-sha256",
            DISPATCH_SHA256,
            "policy:checkpoint",
            "--policy.config",
            "test",
            "--policy.dir",
            "/not-loaded",
        ],
    )
    assert args.listen_fd == 9
    assert args.execution_id == EXECUTION_ID
    assert args.dispatch_sha256 == DISPATCH_SHA256
    assert args.policy.config == "test"


def test_socket_validation_failure_closes_duplicate_not_original(monkeypatch):
    closed = []
    monkeypatch.setattr(transport.os, "dup", lambda fd: 99)
    monkeypatch.setattr(transport.os, "close", closed.append)

    def not_a_socket(*, fileno):
        assert fileno == 99
        raise OSError("not a socket")

    monkeypatch.setattr(transport.socket, "socket", not_a_socket)
    with pytest.raises(OSError, match="not a socket"):
        transport.adopt_listening_socket(9)
    assert closed == [99]
