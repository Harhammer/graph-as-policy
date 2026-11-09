"""Positive control: our client against GaP's OWN server and observation parser.

Loads ``gap/envs/msgpack_bridge.py`` from this checkout by path (it needs only msgpack),
so a change upstream to the framing breaks this test rather than the first robot run.
The second test runs ``FrankaRealEnv._convert_observation`` itself on what our bridge
sends — skipped unless GaP's env is importable (run it from GaP's venv to include it).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import z3_gap_bridge
from z3_gap_bridge.wire import WireClient

GAP_ROOT = Path(z3_gap_bridge.__file__).resolve().parents[4]
MSGPACK_BRIDGE = GAP_ROOT / "gap" / "envs" / "msgpack_bridge.py"


def _load_gap_msgpack_bridge():
    if not MSGPACK_BRIDGE.exists():
        pytest.skip(f"not inside a GaP checkout ({MSGPACK_BRIDGE} missing)")
    spec = importlib.util.spec_from_file_location("gap_msgpack_bridge", MSGPACK_BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation() -> dict:
    return {
        "left": {"joint_pos": np.arange(8, dtype=np.float32)},
        "camera_top": {
            "images": {"rgb": np.full((48, 64, 3), 7, np.uint8)},
            "depth_data": np.full((48, 64), 0.25, np.float32),
            "intrinsics": {"left": {"intrinsics_matrix": np.eye(3, dtype=np.float32)}},
            "pose_mat": np.eye(4, dtype=np.float32),
        },
    }


@pytest.fixture
def gap_server():
    mb = _load_gap_msgpack_bridge()
    server = mb.MsgpackNumpyServer(host="127.0.0.1", port=0)
    mb.start_server_in_background(server)
    deadline = time.monotonic() + 5
    while server._asyncio_server is None:
        assert time.monotonic() < deadline, "GaP server did not start"
        time.sleep(0.01)
    server.port = server._asyncio_server.sockets[0].getsockname()[1]
    yield server
    server.stop()


def test_round_trip_through_gap_server(gap_server):
    gap_server.latest_action = {
        "timestamp": 1.0,
        "left": {"joint_pos": [0.1] * 7, "gripper": 0.5},
    }
    client = WireClient("127.0.0.1", gap_server.port)
    client.connect(wait_s=5)
    try:
        action = client.exchange(_observation())
    finally:
        client.close()
    assert action[b"left"][b"joint_pos"] == pytest.approx([0.1] * 7)
    assert action[b"left"][b"gripper"] == pytest.approx(0.5)
    received = gap_server.latest_observation
    np.testing.assert_array_equal(received[b"left"][b"joint_pos"], np.arange(8, dtype=np.float32))
    cam = received[b"camera_top"]
    assert cam[b"images"][b"rgb"].shape == (48, 64, 3)
    assert cam[b"depth_data"].dtype == np.float32


def test_gap_franka_env_parses_our_observation(gap_server):
    pytest.importorskip("viser")
    pytest.importorskip("robot_descriptions")
    sys.path.insert(0, str(GAP_ROOT))
    try:
        from gap.envs.franka_real_env import FrankaRealEnv
    except ImportError as exc:
        pytest.skip(f"GaP env not importable here: {exc}")
    finally:
        sys.path.remove(str(GAP_ROOT))

    client = WireClient("127.0.0.1", gap_server.port)
    client.connect(wait_s=5)
    try:
        client.exchange(_observation())
    finally:
        client.close()
    env = FrankaRealEnv.__new__(FrankaRealEnv)  # parse only: no server, no threads
    env.camera_names = ["robot0_robotview"]
    env._obs = {"seen": True}
    env._gripper_fraction = 1.0
    env._urdf = None
    env._compute_fk = lambda q: (np.zeros(3), np.array([1.0, 0, 0, 0]))
    obs = env._convert_observation(gap_server.latest_observation)
    assert obs["robot_joint_pos_0"][7] == pytest.approx(7.0)
    cam = obs["robot0_robotview"]
    assert cam["images"]["rgb"].shape == (48, 64, 3)
    assert cam["images"]["depth"].shape == (48, 64)
    assert cam["intrinsics"].shape == (3, 3)
    np.testing.assert_allclose(cam["pose"], [0, 0, 0, 1, 0, 0, 0], atol=1e-6)
