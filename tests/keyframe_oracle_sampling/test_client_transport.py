from __future__ import annotations

import pytest

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def test_all_websocket_operations_preserve_server_text_tracebacks():
    for operation in ("server handshake", "inference", "reset", "add_buffer"):
        with pytest.raises(RuntimeError, match=operation):
            WebsocketClientPolicy._unpack_response("remote traceback", operation)


def test_websocket_response_helper_still_decodes_binary_payloads():
    payload = {"ok": True, "count": 3}
    packed = msgpack_numpy.Packer().pack(payload)
    assert WebsocketClientPolicy._unpack_response(packed, "test") == payload
