"""Droop test scripts: the droop/torque maths, one measure node end to end, the report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gap.testing.fakes import FakeContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import droop_report  # noqa: E402
import measure_droop  # noqa: E402


def _pose(xyz, rot=None):
    r = Rotation.from_euler("x", 180, degrees=True) if rot is None else rot
    x, y, z, w = r.as_quat()
    return {"position": dict(zip("xyz", map(float, xyz), strict=True)),
            "rotation": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}}


def test_pure_sag_and_tilt():
    cmd = _pose([0.43, 0.027, 0.111])
    # 2 mm down, tilted 0.5 deg about base y: the fingertips (60 mm further along the tool
    # axis, which points down) swing 60 mm * sin(0.5 deg) = 0.52 mm sideways
    tilt = Rotation.from_euler("y", 0.5, degrees=True)
    meas = _pose([0.43, 0.027, 0.109], tilt * Rotation.from_euler("x", 180, degrees=True))
    q_cmd = np.zeros(7)
    q = np.array([0, 0.01, 0, 0, 0, -0.002, 0])
    k = [300, 200, 300, 300, 50, 100, 50]
    d = measure_droop.compute_droop(cmd, meas, q_cmd, q, k)
    assert d["droop_mm"] == pytest.approx(2.0, abs=1e-6)
    assert d["droop_deg"] == pytest.approx(0.5, abs=1e-6)
    tip = np.array(d["tip_dp_mm"])
    assert tip[2] == pytest.approx(-2.0, abs=0.01)
    assert abs(tip[0]) == pytest.approx(60 * np.sin(np.radians(0.5)), abs=0.01)
    # spring torque = k * (commanded - measured)
    assert d["torque_nm"][1] == pytest.approx(-2.0)
    assert d["torque_nm"][5] == pytest.approx(0.2)


def test_measure_node_writes_a_record(tmp_path):
    cmd = _pose([0.43, 0.027, 0.111])
    meas = _pose([0.43, 0.028, 0.108])
    q_cmd = [0.0, -0.3, 0.0, -2.4, 0.0, 2.1, 0.8]
    q = [0.0, -0.29, 0.0, -2.4, 0.0, 2.1, 0.8, 0.04]  # + gripper
    ctx = FakeContext({
        "robot.get_ee_pose": {"pose": meas},
        "robot.get_observation": {"arms": [{"joint_state": {"positions": q}}]},
        "robot.solve_ik": {"joint_config": {"positions": q_cmd}},
    })
    session = droop_report.begin("t", str(tmp_path))
    out = measure_droop.run(ctx, pose=cmd, label="pre_insert", phase="soft", rep=0, session=session,
                            k_theta=[300, 200, 300, 300, 50, 100, 50], still_s=0.05, timeout_s=1.0)
    assert out["droop_mm"] == pytest.approx(np.hypot(1.0, 3.0), abs=1e-6)
    assert ctx.calls_to("robot.solve_ik")[0].kwargs["pose"] == cmd
    recs = droop_report.load(session)
    assert len(recs) == 1 and recs[0]["label"] == "pre_insert"
    assert recs[0]["torque_nm"][1] == pytest.approx(-2.0)


def test_report_groups_by_pose_and_phase(tmp_path):
    session = droop_report.begin("", str(tmp_path))
    cmd = _pose([0.43, 0.027, 0.111])
    for rep, dz in enumerate([0.002, 0.003]):
        ctx = FakeContext({
            "robot.get_ee_pose": {"pose": _pose([0.43, 0.027, 0.111 - dz])},
            "robot.get_observation": {"arms": [{"joint_state": {"positions": [0.0] * 7}}]},
            "robot.solve_ik": {"joint_config": {"positions": [0.0] * 7}},
        })
        measure_droop.run(ctx, pose=cmd, label="pre_insert", phase="soft", rep=rep, session=session,
                          k_theta=[1.0] * 7, still_s=0.05, timeout_s=1.0)
    s = droop_report.summarise(droop_report.load(session))[("pre_insert", "soft")]
    assert s["n"] == 2
    assert s["droop_mm"][0] == pytest.approx(2.5, abs=1e-6)
    assert s["droop_mm"][1] == pytest.approx(np.std([2.0, 3.0], ddof=1), abs=1e-6)
    assert "pre_insert" in droop_report.table(droop_report.load(session))
