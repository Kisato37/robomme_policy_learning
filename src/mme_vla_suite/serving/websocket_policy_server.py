import http
import logging
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

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset()
                    selector_config = obs.get("keyframe_selector_config")
                    if selector_config is not None:
                        self._policy.configure_keyframe_selector(selector_config)
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack({"reset_finished": True, "reset_time_ms": tend * 1000}))
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
