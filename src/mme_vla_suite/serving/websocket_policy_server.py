import http
import logging
import os
import time
import traceback

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.serving.websocket_policy_server import WebsocketPolicyServer as _TransportServer

logger = logging.getLogger(__name__)


class WebsocketPolicyServer(_TransportServer):
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(self, *args, exclusive_clients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._exclusive_clients = exclusive_clients
        self._active_trajectory = None
        if exclusive_clients:
            if "direct_execution" not in self._metadata:
                raise ValueError("Resident mode requires explicit direct execution identity")
            self._metadata = {**self._metadata, "resident_policy": True, "model_process_pid": os.getpid()}

    async def _handler(self, websocket: _server.ServerConnection):
        try:
            await self._handle_trajectory(websocket)
        finally:
            if self._active_trajectory is websocket:
                self._active_trajectory = None

    async def _handle_trajectory(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        initialized = False

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if self._exclusive_clients:
                    if self._active_trajectory not in (None, websocket):
                        raise RuntimeError("Another trajectory owns the resident policy")
                    if not initialized and (obs.get("reset") is not True or "keyframe_selector_config" not in obs):
                        raise RuntimeError("Resident trajectory must begin with an explicit configured reset")
                    if initialized and obs.get("reset"):
                        raise RuntimeError("A resident trajectory cannot reset twice on one connection")
                    self._active_trajectory = websocket

                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset()
                    evidence = self._policy.reset_evidence() if self._exclusive_clients else None
                    selector_config = obs.get("keyframe_selector_config")
                    if selector_config is not None:
                        self._policy.configure_keyframe_selector(selector_config)
                    tend = time.monotonic() - tstart
                    initialized = True
                    reply = {"reset_finished": True, "reset_time_ms": tend * 1000}
                    if evidence is not None:
                        reply["resident_reset"] = {"state": evidence, "model_process_pid": os.getpid()}
                    await websocket.send(packer.pack(reply))
                elif obs.get("add_buffer", False):
                    tstart = time.monotonic()
                    self._policy.add_buffer(obs)
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack({"add_buffer_finished": True, "add_buffer_time_ms": tend * 1000}))
                else:
                    outputs = self._policy.infer(obs)
                    await websocket.send(packer.pack(outputs))

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
