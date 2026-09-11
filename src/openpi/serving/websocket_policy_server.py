import asyncio
import http
import ipaddress
import logging
import math
import os
import re
import socket
import time
import traceback
import uuid

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def validate_direct_options(
    listen_fd: int | None, execution_id: str | None, dispatch_sha256: str | None
) -> dict[str, str] | None:
    """Validate optional direct-runner identity without loading a policy or touching a socket.

    This identity binds a readiness response to an execution; it isn't authentication.
    A passed socket must always carry an identity. Legacy host/port startup needs neither.
    """
    if listen_fd is not None and (type(listen_fd) is not int or listen_fd < 3):
        raise ValueError("listen_fd must be an inherited descriptor >= 3")
    if (execution_id is None) != (dispatch_sha256 is None):
        raise ValueError("execution_id and dispatch_sha256 must be provided together")
    if execution_id is None:
        if listen_fd is not None:
            raise ValueError("listen_fd requires execution_id and dispatch_sha256")
        return None
    if not isinstance(execution_id, str) or str(uuid.UUID(execution_id)) != execution_id:
        raise ValueError("execution_id must be a canonical lowercase UUID")
    if not isinstance(dispatch_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", dispatch_sha256) is None:
        raise ValueError("dispatch_sha256 must be 64 lowercase hexadecimal characters")
    return {"execution_id": execution_id, "dispatch_sha256": dispatch_sha256}


def adopt_listening_socket(listen_fd: int) -> socket.socket:
    """Duplicate a reserved loopback TCP listener, without binding or taking the caller's FD.

    The returned socket is owned by the server. Closing it doesn't close the caller's
    reservation. Neither caller nor server should call shutdown() on the shared listener.
    Requires working SO_ACCEPTCONN support (Linux); unsupported platforms fail closed.
    """
    if type(listen_fd) is not int or listen_fd < 3:
        raise ValueError("listen_fd must be an inherited descriptor >= 3")
    duplicate_fd = os.dup(listen_fd)
    try:
        listener = socket.socket(fileno=duplicate_fd)
    except BaseException:
        os.close(duplicate_fd)
        raise
    try:
        if listener.family not in (socket.AF_INET, socket.AF_INET6):
            raise ValueError("listen_fd must be an IPv4 or IPv6 TCP socket")
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise ValueError("listen_fd must be a TCP stream socket")
        if listener.proto not in (0, socket.IPPROTO_TCP):
            raise ValueError("listen_fd must use TCP")
        try:
            is_listening = listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        except OSError as error:
            raise ValueError("cannot verify listen_fd is already listening on this platform") from error
        if not is_listening:
            raise ValueError("listen_fd must already be listening")
        if not ipaddress.ip_address(listener.getsockname()[0]).is_loopback:
            raise ValueError("listen_fd must be bound to a loopback address")
        return listener
    except BaseException:
        listener.close()
        raise


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        listen_fd: int | None = None,
        execution_id: str | None = None,
        dispatch_sha256: str | None = None,
        keepalive_timeout: float | None = None,
    ) -> None:
        if keepalive_timeout is not None and (
            type(keepalive_timeout) not in (int, float)
            or not math.isfinite(keepalive_timeout)
            or keepalive_timeout <= 0
        ):
            raise ValueError("keepalive_timeout must be a positive finite duration or None")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._listen_fd = listen_fd
        self._keepalive_timeout = keepalive_timeout
        direct_execution = validate_direct_options(listen_fd, execution_id, dispatch_sha256)
        if direct_execution is not None:
            if "direct_execution" in self._metadata:
                raise ValueError("policy metadata contains the reserved direct_execution key")
            self._metadata = {**self._metadata, "direct_execution": direct_execution}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # asyncio takes ownership of sock. Pass a duplicate and retain it through
        # startup failure / cancellation so every exit closes only our reference.
        listener = None if self._listen_fd is None else adopt_listening_socket(self._listen_fd)
        try:
            bind_options = {"host": self._host, "port": self._port} if listener is None else {"sock": listener}
            keepalive_options = (
                {} if self._keepalive_timeout is None else {"ping_timeout": self._keepalive_timeout}
            )
            async with _server.serve(
                self._handler,
                compression=None,
                max_size=None,
                process_request=_health_check,
                **keepalive_options,
                **bind_options,
            ) as server:
                await server.serve_forever()
        finally:
            if listener is not None:
                listener.close()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
