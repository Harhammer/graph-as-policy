"""Hand-eye calibration: undistortion, board fit and solver, each against a known answer."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from z3_gap_bridge.handeye import HandEyeConfig, board_spread, detect_board, make_board, solve
from z3_gap_bridge.realsense_client import RealSenseFrame, undistort_pixels

# The D405 colour stream's real intrinsics (read from 409122274238 on 2026-11-06).
K = np.array([[433.91, 0, 421.24], [0, 432.79, 241.33], [0, 0, 1.0]])
COEFFS = np.array([-0.05302, 0.06166, -8e-05, 0.00047, -0.02042])


def test_undistort_matches_librealsense():
    """The vendor's own deprojection is the reference; the model's name misleads."""
    rs = pytest.importorskip("pyrealsense2")
    intr = rs.intrinsics()
    intr.width, intr.height = 848, 480
    intr.fx, intr.fy, intr.ppx, intr.ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intr.model = rs.distortion.inverse_brown_conrady
    intr.coeffs = COEFFS.tolist()
    pixels = np.array([[0, 0], [847, 479], [421, 241], [100, 400], [700, 50]], dtype=float)
    ours = undistort_pixels(pixels, K, COEFFS, "inverse_brown_conrady")
    for (u, v), mine in zip(pixels, ours, strict=True):
        x, y, z = rs.rs2_deproject_pixel_to_point(intr, [u, v], 1.0)
        np.testing.assert_allclose(
            mine, [x / z * K[0, 0] + K[0, 2], y / z * K[1, 1] + K[1, 2]], atol=0.02
        )
    # Measured: the extreme corner pixel moves 4.7 px — small, but not zero.
    assert 3 < np.abs(ours[0] - pixels[0]).max() < 8


def _synthetic_frame(cfg_board, square_px=50, margin=40, depth_m=None):
    """The board rendered frontally; with f = 600 px it sits at Z = f * square_m / square_px."""
    import cv2

    board = make_board(cfg_board)
    w = cfg_board.squares_x * square_px + 2 * margin
    h = cfg_board.squares_y * square_px + 2 * margin
    gray = board.generateImage((w, h), marginSize=margin, borderBits=1)
    k = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]])
    z = 600.0 * cfg_board.square_m / square_px
    depth = np.full((h, w), z if depth_m is None else depth_m, np.float32)
    frame = RealSenseFrame(
        rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        depth_m=depth,
        intrinsics=k,
        frame_number=0,
        hw_timestamp_ms=0.0,
        timestamp_domain="",
        host_time_s=0.0,
    )
    return frame, z


def test_board_fit_recovers_the_known_distance():
    cfg = HandEyeConfig()
    frame, z = _synthetic_frame(cfg)
    det = detect_board(frame, make_board(cfg), cfg)
    assert det["ok"], det.get("reason")
    assert det["corners"] == (cfg.squares_x - 1) * (cfg.squares_y - 1)
    assert det["camera_T_board"][2, 3] == pytest.approx(z, rel=0.01)
    assert det["reproj_px"] < 0.5 and det["marker_fit_px"] < 1.0


def test_wrong_layout_is_rejected_even_though_corners_fit():
    frame, _ = _synthetic_frame(HandEyeConfig(legacy_pattern=True))
    wrong = HandEyeConfig(legacy_pattern=False)
    det = detect_board(frame, make_board(wrong), wrong)
    assert not det["ok"]
    assert "does not match" in det["reason"] or "corners" in det["reason"]


def test_wrong_square_size_is_caught_by_depth():
    cfg = HandEyeConfig()
    frame, z = _synthetic_frame(cfg, depth_m=None)
    misprinted = HandEyeConfig(square_m=0.021 * 1.1, marker_m=0.0155 * 1.1)
    det = detect_board(frame, make_board(misprinted), misprinted)
    assert not det["ok"] and "depth says" in det["reason"]


def _random_pose(rng, center, spread_m, tilt_deg):
    t = np.eye(4)
    t[:3, :3] = (
        Rotation.from_euler("x", 180, degrees=True)
        * Rotation.from_rotvec(rng.normal(size=3) * np.radians(tilt_deg) / 2)
    ).as_matrix()
    t[:3, 3] = center + rng.uniform(-spread_m, spread_m, 3)
    return t


def test_solver_recovers_a_known_transform_and_spread_exposes_a_wrong_one():
    rng = np.random.default_rng(0)
    x_true = np.eye(4)
    x_true[:3, :3] = Rotation.from_euler("xyz", [3, -14, 46], degrees=True).as_matrix()
    x_true[:3, 3] = [0.06, -0.04, 0.03]
    base_T_board = np.eye(4)
    base_T_board[:3, 3] = [0.5, 0.0, -0.04]
    flanges, boards = [], []
    for _ in range(15):
        b = _random_pose(rng, np.array([0.45, 0.0, 0.40]), 0.12, 35)
        c = np.linalg.inv(x_true) @ np.linalg.inv(b) @ base_T_board
        c[:3, 3] += rng.normal(scale=0.0003, size=3)  # 0.3 mm of board-fit noise
        flanges.append(b)
        boards.append(c)
    result = solve(flanges, boards)
    best = result["methods"][result["best"]]
    assert np.abs(best["flange_T_camera"][:3, 3] - x_true[:3, 3]).max() < 1e-3
    assert best["trans_rms_mm"] < 1.0
    np.testing.assert_allclose(best["board_xyz_m"], base_T_board[:3, 3], atol=1e-3)
    x_wrong = x_true.copy()
    x_wrong[:3, 3] += [0.01, 0, 0]  # 1 cm off
    assert board_spread(flanges, boards, x_wrong)["trans_rms_mm"] > 3.0


def test_tilt_coverage_catches_last_joint_only_rotation():
    from z3_gap_bridge.handeye import tilt_coverage_deg

    def pose(rotvec_deg):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
        return t

    only_z = [pose([0, 0, a]) for a in (-80, -40, 0, 40, 80)]
    cov = tilt_coverage_deg(only_z)
    assert cov[0] > 40 and cov[2] < 1, cov
    mixed = only_z + [pose([35, 0, 0]), pose([-35, 0, 0]), pose([0, 35, 0]), pose([0, -35, 0])]
    assert tilt_coverage_deg(mixed)[2] > 15


def test_refine_recovers_joint_offsets_and_scale():
    import cv2

    """Synthetic sessions on a simulated arm whose joints 2/5/6 read 1.7/1.0/0.5 deg off."""
    from z3_gap_bridge.handeye import refine
    from z3_gap_bridge.panda_fk import flange_pose

    rng = np.random.default_rng(5)
    x_true = np.eye(4)
    x_true[:3, :3] = Rotation.from_euler("xyz", [15, -2, 90], degrees=True).as_matrix()
    x_true[:3, 3] = [-0.065, -0.005, 0.061]
    dq_true = np.radians([0, 1.7, 0, 0, 1.0, 0.5, 0])
    scale_true = 1.01
    k = np.array([[434.0, 0, 421.0], [0, 433.0, 241.0], [0, 0, 1]])
    cfg = HandEyeConfig()
    board = make_board(cfg)
    obj_all = np.concatenate([np.asarray(c).reshape(-1, 3) for c in board.getChessboardCorners().reshape(1, -1, 3)])
    samples, flanges, boards = [], [], []
    for session, board_xy in enumerate(([0.47, 0.05], [0.40, -0.10])):
        base_T_board = np.eye(4)
        base_T_board[:3, 3] = [*board_xy, 0.017]
        for _ in range(12):
            # a wrist pose looking down at the board from 25-45 cm, tilted up to 35 deg
            cam = _random_pose(rng, np.array([*board_xy, 0.35]), 0.08, 35)
            cam[:3, 3] += [0.1, 0.07, 0.0]
            flange_true = cam @ np.linalg.inv(x_true)
            # joints that the true arm needs for that flange, reported with the offsets removed
            from scipy.optimize import least_squares
            def f(q):
                F = flange_pose(q + dq_true)
                return np.r_[(F[:3, 3] - flange_true[:3, 3]) * 10,
                             Rotation.from_matrix(F[:3, :3].T @ flange_true[:3, :3]).as_rotvec()]
            q = least_squares(f, np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.9, 0.8]) + rng.normal(0, 0.3, 7),
                              xtol=1e-12, ftol=1e-12).x
            if np.linalg.norm(f(q)) > 1e-6:
                continue
            pts = (np.linalg.inv(cam) @ base_T_board @ np.c_[obj_all * scale_true, np.ones(len(obj_all))].T)[:3]
            uv = np.stack([k[0, 0] * pts[0] / pts[2] + k[0, 2], k[1, 1] * pts[1] / pts[2] + k[1, 2]], 1)
            uv += rng.normal(0, 0.3, uv.shape)
            ok, rv, tv = cv2.solvePnP(obj_all, uv, k, None, flags=cv2.SOLVEPNP_SQPNP)
            ctb = np.eye(4); ctb[:3, :3] = cv2.Rodrigues(rv)[0]; ctb[:3, 3] = tv.ravel()
            samples.append({"q": q, "session": session, "K": k, "obj": obj_all, "img": uv, "camera_T_board": ctb})
            flanges.append(flange_pose(q))
            boards.append(ctb)
    assert len(samples) >= 20
    closed = solve(flanges, boards)
    x0 = closed["methods"][closed["best"]]["flange_T_camera"]
    assert closed["methods"][closed["best"]]["trans_rms_mm"] > 2.0, "the nominal chain is visibly wrong"
    out = refine(samples, x0, (1, 4, 5), True, np.zeros(7), leave_one_out=False)
    got = np.degrees(out["joint_offsets_rad"])
    assert np.abs(got - np.degrees(dq_true)).max() < 0.05, got
    assert abs(out["board_scale"] - scale_true) < 0.002
    assert np.abs(out["flange_T_camera"][:3, 3] - x_true[:3, 3]).max() < 1e-3
    assert np.sqrt(np.mean(out["board_error_mm"] ** 2)) < 0.5
