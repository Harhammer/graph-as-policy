"""FrankaRealEnv's FK must put the Robotiq tip where the IK backend aims it.

Both read the same ``tcp_offset=(0, 0, -0.157)``, stored in gap's NEGATED link-to-TCP
convention. The IK side (``CuRoboBackend._tcp_to_link``) had it right; the env's FK added
the offset and reported the tip 0.157 m behind the flange — ``get_ee_pose`` ~0.31 m high
on a top-down pose (z3 fork, found on the real arm 2026-11-06).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("viser")
pytest.importorskip("robot_descriptions")

from gap.connector.ik import CuRoboBackend  # noqa: E402
from gap.envs.franka_real_env import FrankaRealEnv  # noqa: E402

# Joints GaP's IK chose for a top-down approach on the real arm (2026-11-06).
Q_TOP_DOWN = np.array([0.2315, -0.5534, -0.5017, -2.3481, -0.2890, 1.8467, -0.0302])


@pytest.fixture(scope="module")
def env():
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    e = FrankaRealEnv.__new__(FrankaRealEnv)  # FK only: no server, no threads
    e._urdf = load_robot_description("panda_description")
    return e


def _hand(env, q):
    env._compute_fk(q)  # leaves the URDF at q
    return env._urdf.get_transform("panda_hand", "panda_link0")


def test_tip_is_in_front_of_the_hand(env):
    pos, _ = env._compute_fk(Q_TOP_DOWN)
    hand = _hand(env, Q_TOP_DOWN)
    np.testing.assert_allclose(pos, hand[:3, 3] + 0.157 * hand[:3, 2], atol=1e-6)
    assert pos[2] < hand[2, 3], "top-down: the fingertips are BELOW the hand"


@pytest.mark.parametrize("seed", range(5))
def test_fk_inverts_the_ik_tcp_conversion(env, seed):
    rng = np.random.default_rng(seed)
    q = Q_TOP_DOWN + rng.uniform(-0.4, 0.4, 7)
    pos, wxyz = env._compute_fk(q)
    backend = CuRoboBackend(
        arm_dof=7,
        tcp_offset=np.array(FrankaRealEnv._TCP_OFFSET),
        tcp_rotation=Rotation.from_euler("z", FrankaRealEnv._TCP_ROTATION_Z),
    )
    tool = {
        "position": dict(zip("xyz", map(float, pos), strict=True)),
        "rotation": dict(zip("wxyz", map(float, wxyz), strict=True)),
    }
    link = backend._tcp_to_link(tool)
    hand = _hand(env, q)
    np.testing.assert_allclose([link["position"][k] for k in "xyz"], hand[:3, 3], atol=1e-6)
    r = link["rotation"]
    got = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    assert math.degrees(Rotation.from_matrix(got.T @ hand[:3, :3]).magnitude()) < 1e-4


def test_default_tcp_is_the_stock_robotiq(monkeypatch):
    from gap.envs.franka_real_env import franka_tcp

    monkeypatch.delenv("GAP_FRANKA_TCP_FLANGE_M", raising=False)
    monkeypatch.delenv("GAP_FRANKA_TCP_ROT_Z_DEG", raising=False)
    offset, rot = franka_tcp()
    np.testing.assert_allclose(offset, FrankaRealEnv._TCP_OFFSET, atol=1e-12)
    assert rot == pytest.approx(FrankaRealEnv._TCP_ROTATION_Z)


def test_configured_tcp_moves_fk_and_ik_together(env, monkeypatch):
    """Custom fingertips: tool point 0.205 m out and 4 mm sideways in the flange frame."""
    from gap.envs.franka_real_env import franka_tcp

    monkeypatch.setenv("GAP_FRANKA_TCP_FLANGE_M", "0.004,0,0.205")
    offset, rot = franka_tcp()
    custom = FrankaRealEnv.__new__(FrankaRealEnv)
    custom._urdf = env._urdf
    custom._TCP_OFFSET, custom._TCP_ROTATION_Z = offset, rot
    pos, wxyz = custom._compute_fk(Q_TOP_DOWN)
    flange = custom._urdf.get_transform("panda_link8", "panda_link0")
    np.testing.assert_allclose(pos, flange[:3, :3] @ [0.004, 0, 0.205] + flange[:3, 3], atol=1e-6)
    backend = CuRoboBackend(
        arm_dof=7, tcp_offset=offset, tcp_rotation=Rotation.from_euler("z", rot)
    )
    tool = {
        "position": dict(zip("xyz", map(float, pos), strict=True)),
        "rotation": dict(zip("wxyz", map(float, wxyz), strict=True)),
    }
    link = backend._tcp_to_link(tool)
    hand = _hand(custom, Q_TOP_DOWN)
    np.testing.assert_allclose([link["position"][k] for k in "xyz"], hand[:3, 3], atol=1e-6)
