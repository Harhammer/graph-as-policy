"""Blocking client for GaP's msgpack bridge (``gap/envs/msgpack_bridge.py``).

GaP's ``FrankaRealEnv`` runs the TCP *server*; this module is the client side that
``robots_realtime`` would otherwise be. One exchange = one framed observation out, the
latest action back. Frames are a 4-byte big-endian length plus a msgpack payload,
numpy arrays via ``msgpack_numpy`` — identical to both ends of the upstream bridge,
including ``raw=True`` decoding (keys arrive as bytes).
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from typing import Any

import msgpack
import msgpack_numpy

log = logging.getLogger(__name__)

_HEADER = struct.Struct("!I")


def encode_msg(obj: dict) -> bytes:
    """Pack one message exactly as GaP's ``encode_msg`` does."""
    return msgpack.packb(obj, use_bin_type=True, default=msgpack_numpy.encode)


def decode_msg(raw: bytes) -> dict:
    """Unpack one message exactly as GaP's ``decode_msg`` does (bytes keys)."""
    return msgpack.unpackb(raw, raw=True, object_hook=msgpack_numpy.decode)


def field(d: dict | None, key: str) -> Any:
    """Look up ``key`` in a ``raw=True`` map, tolerating either str or bytes keys."""
    if d is None:
        return None
    if key in d:
        return d[key]
    return d.get(key.encode())


class WireClient:
    """Framed request/response client for GaP's ``MsgpackNumpyServer``.

    Args:
        host: GaP server address (``FrankaRealEnv`` binds 127.0.0.1 by default).
        port: GaP server port (``gap.connector.real(port=...)``, default 9000).
        timeout_s: Socket timeout for one exchange.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, timeout_s: float = 2.0) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def connect(self, wait_s: float = 60.0) -> None:
        """Connect, retrying until GaP's server is listening or ``wait_s`` runs out."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ConnectionError(
                        f"GaP msgpack server at {self._host}:{self._port} not reachable after "
                        f"{wait_s:.0f}s — start `gap run ... --real franka --no-rr-autostart` "
                        "first"
                    ) from exc
                time.sleep(0.5)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            log.info("connected to GaP at %s:%d", self._host, self._port)
            return

    def exchange(self, observation: dict) -> dict:
        """Send one observation; return GaP's latest action (``{}`` until it has one)."""
        if self._sock is None:
            raise ConnectionError("WireClient is not connected")
        payload = encode_msg(observation)
        self._sock.sendall(_HEADER.pack(len(payload)) + payload)
        (length,) = _HEADER.unpack(self._recv_exactly(_HEADER.size))
        return decode_msg(self._recv_exactly(length))

    def _recv_exactly(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("GaP closed the connection")
            buf += chunk
        return bytes(buf)

    def close(self) -> None:
        """Close the socket (idempotent)."""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
