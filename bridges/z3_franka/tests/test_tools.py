"""The GaP stand-in holds the arm: the real bridge, driven by it, never commands motion."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
from test_bridge import FakeCamera, FakeFranky, FakeGripper

from z3_gap_bridge.bridge import Bridge, BridgeConfig
from z3_gap_bridge.tools import FakeServer, fake_server
from z3_gap_bridge.wire import WireClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_fake_server_keeps_the_bridge_gates_shut(tmp_path):
    port = _free_port()
    frame = tmp_path / "frame.npz"
    server = threading.Thread(
        target=fake_server, args=(FakeServer(port=port, save_frame=str(frame)),), daemon=True
    )
    server.start()
    franky = FakeFranky()
    bridge = Bridge(
        BridgeConfig(execute=True, camera_hz=1e6, gap_port=port),
        franky,
        FakeCamera(),
        WireClient("127.0.0.1", port),
        np.eye(4),
        FakeGripper(franky.calls),
    )
    bridge.connect()
    for _ in range(20):
        bridge.tick()
    bridge.close()
    server.join(timeout=2)
    assert not franky.moves(), "a seed-only GaP must never move the arm or gripper"
    with np.load(frame) as saved:
        assert saved["pose_mat"].shape == (4, 4)
        assert saved["depth_m"].dtype == np.float32


def test_test_move_streams_a_small_rotation_and_back():
    from test_bridge import FakeCamera, FakeFranky, FakeGripper

    port = _free_port()
    server = threading.Thread(
        target=fake_server,
        args=(FakeServer(port=port, test_move_rad=0.1, test_joint=6, test_hold_s=0.3),),
        daemon=True,
    )
    server.start()
    franky = FakeFranky()
    q0 = franky.q.copy()
    bridge = Bridge(
        BridgeConfig(execute=True, camera_hz=1e6, gap_port=port),
        franky,
        FakeCamera(),
        WireClient("127.0.0.1", port),
        np.eye(4),
        FakeGripper(franky.calls),
    )
    bridge.connect()
    t_end = time.monotonic() + 2.0
    while time.monotonic() < t_end:
        bridge.tick()
        time.sleep(0.005)
    bridge.close()
    steps = np.array([c[1] for c in franky.calls if c[0] == "move_joints"])
    assert steps[:, 6].max() == pytest.approx(q0[6] + 0.1, abs=1e-6)
    np.testing.assert_allclose(steps[-1], q0, atol=1e-6)  # and back
    np.testing.assert_allclose(steps[:, :6], np.tile(q0[:6], (len(steps), 1)), atol=1e-9)
