"""Ball-in-cone solver on a simulated arm with known joint offsets."""

from __future__ import annotations

import json

import numpy as np
import pytest

from z3_gap_bridge import kincal
from z3_gap_bridge.panda_fk import flange_pose

TRUE_DQ = np.radians([0.0, 1.7, 0.1, -0.2, 1.0, 0.5, 0.0])  # the camera fit's kind of answer
BALL = np.array([0.004, -0.002, 0.215])  # ball centre in the flange frame
CONES = {"A": np.array([0.45, -0.10, 0.03]), "B": np.array([0.35, 0.15, 0.03])}


def _config_reaching(cone, rng):
    """A random joint configuration whose (true) ball lands on the cone: solve IK on the
    true model from a random seed, with random wrist orientation."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    seed = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.8, 0.8]) + rng.uniform(-0.8, 0.8, 7)
    tilt = R.from_euler("xyz", [180 + rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-150, 150)], degrees=True)
    def f(q):
        F = flange_pose(q + TRUE_DQ)
        pos = (F @ np.array([*BALL, 1.0]))[:3] - cone
        rot = (R.from_matrix(F[:3, :3]).inv() * tilt).as_rotvec() * 0.1
        return np.r_[pos, rot]
    r = least_squares(f, seed, xtol=1e-12, ftol=1e-12)
    assert np.linalg.norm(r.fun[:3]) < 1e-6
    return r.x


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(3)
    qs, cones = [], []
    for label, cone in CONES.items():
        for _ in range(16):
            qs.append(_config_reaching(cone, rng) + rng.normal(0, 2e-5, 7))  # encoder noise
            cones.append(label)
    return np.asarray(qs), cones


def test_recovers_the_offsets_and_the_ball(dataset):
    q, cones = dataset
    res = kincal.solve(q, cones)
    got = np.degrees(res["offsets_rad"])
    want = np.degrees(TRUE_DQ[list(res["joints"])])
    assert np.abs(got - want).max() < 0.03, (got, want)
    assert np.linalg.norm(res["ball_offset_m"] - BALL) < 3e-4
    assert np.sqrt(np.mean(res["nominal_residual_mm"] ** 2)) > 3.0, "the nominal model is visibly wrong"
    assert res["residual_mm"].max() < 0.1
    assert np.degrees(res["offsets_se_rad"]).max() < 0.05


def test_report_and_files(dataset, tmp_path):
    q, cones = dataset
    f = tmp_path / "cap.json"
    f.write_text(json.dumps({"samples": [
        {"q": list(qi), "cone": c, "repeat": i % 5 == 1 and i > 0} for i, (qi, c) in enumerate(zip(q, cones))
    ]}))
    assert kincal.solve_files(kincal.SolveConfig(files=(f,))) == 0
    fit = json.loads((tmp_path / "cap.fit.json").read_text())
    assert fit["joints_1indexed"] == [2, 3, 4, 5, 6]
    assert abs(fit["joint_offsets_deg"][0] - 1.7) < 0.03
    assert fit["residual_rms_mm"] < 0.1 < fit["nominal_residual_rms_mm"]


def test_refuses_unidentifiable_joints(dataset):
    q, cones = dataset
    with pytest.raises(ValueError, match="not identifiable"):
        kincal.solve(q, cones, joints=(0, 1))


def test_socket_fits_the_ball():
    """The socket's inner surface is the ball's sphere: every inner profile point is on it."""
    prof = kincal.socket_stem_profile(20.0)
    z_top = 30.0 + 0.4 * 20.0 + 2.0
    z_c = z_top + 2.0
    inner = [(r, z) for r, z in prof if z <= z_top and r < 9.0 and z > 30.0]
    assert len(inner) > 20
    for r, z in inner:
        assert abs(np.hypot(r, z - z_c) - 10.0) < 1e-6
