"""Pivot calibration against a known tool point and table point."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from z3_gap_bridge.tcp import solve_pivot

TOOL = np.array([0.004, -0.002, 0.205])  # custom fingertips, flange frame
POINT = np.array([0.45, 0.06, 0.017])  # on the table (z from the 2026-11-06 hand-eye)


def _touch(rotvec_deg, noise_m=0.0, rng=None):
    """A flange pose whose tool point rests on POINT."""
    r = (
        Rotation.from_euler("x", 180, degrees=True) * Rotation.from_rotvec(np.radians(rotvec_deg))
    ).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = r
    pose[:3, 3] = POINT - r @ TOOL
    if noise_m:
        pose[:3, 3] += rng.normal(scale=noise_m, size=3)
    return pose


def test_recovers_tool_and_point_from_tilted_touches():
    rng = np.random.default_rng(1)
    tilts = [
        [0, 0, 0],
        [30, 0, 0],
        [-30, 0, 0],
        [0, 30, 0],
        [0, -30, 0],
        [20, 20, 40],
        [-20, 25, -60],
    ]
    result = solve_pivot([_touch(t, 0.0003, rng) for t in tilts])
    np.testing.assert_allclose(result["tool_in_flange_m"], TOOL, atol=1e-3)
    np.testing.assert_allclose(result["point_in_base_m"], POINT, atol=1e-3)
    assert result["rms_mm"] < 1.0
    assert result["tilt_deg"][1] > 20


def test_rotation_about_the_tool_axis_only_is_flagged():
    """Spinning about flange z leaves the tool's z unobservable: the tilt check must say so."""
    result = solve_pivot([_touch([0, 0, a]) for a in (-60, -20, 20, 60, 100, 140)])
    assert result["tilt_deg"][1] < 1


def test_a_slipped_touch_shows_in_the_residual():
    tilts = [[0, 0, 0], [30, 0, 0], [-30, 0, 0], [0, 30, 0], [0, -30, 0], [20, 20, 40]]
    poses = [_touch(t) for t in tilts]
    poses[3][:3, 3] += [0.008, 0, 0]  # the tip slid 8 mm on one touch
    assert solve_pivot(poses)["rms_mm"] > 1.5
