# Copied from oktonex packages/robot/src/okto_robot/panda_fk.py @ c83e4ed9 (DH chain and
# tool-tip helpers only; the Wuji mount geometry below them is not needed here).
"""Analytic forward kinematics for the Franka Panda (flange frame).

Used by the bridge's workspace-floor guard and for the wrist-camera pose — computing
the tool-tip height of every candidate target without a service round trip.
Correctness is NOT assumed: ``Bridge.connect()`` validates this FK against the
robot's own model (franky ``Model.pose``) at the current pose and refuses to
connect on mismatch (fail-closed — the floor guard and camera pose depend on it).

Parameters: official Franka Craig-convention (modified) DH table.
"""

from __future__ import annotations

import numpy as np

# (a_{i-1} [m], d_i [m], alpha_{i-1} [rad]) per joint; last row = fixed flange offset.
_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
    (0.0, 0.107, 0.0),
)


def _dh_step(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """One modified-DH link transform: RotX(alpha) TransX(a) RotZ(theta) TransZ(d)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def flange_pose(q) -> np.ndarray:
    """4x4 base->flange transform for joint positions q (7 floats, rad)."""
    thetas = (*q, 0.0)
    pose = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, thetas, strict=True):
        pose = pose @ _dh_step(a, d, alpha, theta)
    return pose


def link_poses(q) -> list[np.ndarray]:
    """Base->frame_i transforms for i = 1..7 plus the flange (8 matrices, 4x4).

    Frame origins are the joint-frame origins of the modified-DH chain — the polyline
    through them is the kinematic skeleton drawn by ``okto-capture-replay``.
    """
    thetas = (*q, 0.0)
    pose = np.eye(4)
    poses = []
    for (a, d, alpha), theta in zip(_DH, thetas, strict=True):
        pose = pose @ _dh_step(a, d, alpha, theta)
        poses.append(pose.copy())
    return poses


def tool_tip_position(q, tool_length_m: float) -> np.ndarray:
    """Tool-tip xyz [m, base frame]: flange origin + tool_length along flange +z."""
    pose = flange_pose(q)
    return pose[:3, 3] + pose[:3, 2] * tool_length_m


def tool_tip_z(q, tool_length_m: float) -> float:
    """Tool-tip height above the robot base plane [m]."""
    return float(tool_tip_position(q, tool_length_m)[2])
