"""The bridge-side jerk-limited reference."""

from __future__ import annotations

import numpy as np

from z3_gap_bridge.trajectory import JointReference

DT = 0.02
V, A, J = np.full(7, 0.7), np.full(7, 0.9), np.full(7, 4.5)


def _run(ref, ticks):
    out = []
    for _ in range(ticks):
        if not ref.active:
            break
        out.append(ref.step())
    return out


def test_respects_limits_and_arrives():
    ref = JointReference(DT, V, A, J)
    ref.reset(np.zeros(7))
    goal = np.array([0.8, -0.3, 0.1, 0.0, 0.2, -0.5, 0.4])
    ref.set_target(goal)
    traj = _run(ref, 1000)
    q = np.array([p for p, _ in traj])
    dq = np.array([v for _, v in traj])
    assert not ref.active
    np.testing.assert_allclose(q[-1], goal, atol=1e-9)
    assert np.abs(dq).max() <= 0.7 + 1e-9
    assert np.abs(np.diff(np.vstack([np.zeros(7), q]), axis=0)).max() <= 0.7 * DT + 1e-9


def test_phase_sync_keeps_a_straight_joint_space_line():
    ref = JointReference(DT, V, A, J)
    ref.reset(np.zeros(7))
    goal = np.array([0.8, 0.4, 0.0, 0.0, 0.0, 0.0, -0.2])
    ref.set_target(goal)
    q = np.array([p for p, _ in _run(ref, 1000)])
    s = q @ goal / (goal @ goal)  # progress along the line
    off_line = np.linalg.norm(q - np.outer(s, goal), axis=1).max()
    assert off_line < 1e-6


def test_retargeting_mid_motion_is_continuous():
    ref = JointReference(DT, V, A, J)
    ref.reset(np.zeros(7))
    ref.set_target(np.full(7, 0.6))
    before = _run(ref, 30)
    ref.set_target(np.full(7, -0.2))  # reverse while moving
    after = _run(ref, 1000)
    v = np.array([v for _, v in before + after])
    acc = np.abs(np.diff(v, axis=0)).max() / DT
    assert acc <= 0.9 * 1.05, acc  # no velocity jump at the switch
    np.testing.assert_allclose(after[-1][0], np.full(7, -0.2), atol=1e-9)
