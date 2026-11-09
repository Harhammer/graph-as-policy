"""gearbox_vision on real frames (z3 cell, 2026-11-09, corrected kinematics) and the
detect_gearbox script with a fake robot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gearbox_vision import GearboxNotFound, detect_gearbox, draw_gearbox_popup  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
SEATED_XY = (0.4310, 0.0270)
AXIS = np.array([0.4281, 0.0277])  # the close frame's result, checked by eye on the overlay


def _load(name):
    d = np.load(DATA / name)
    return d["rgb"], d["depth"], d["K"], d["T_wc"]


def test_close_view_finds_holes_and_ring_agreeing():
    rgb, depth, K, T = _load("gearbox_close.npz")
    det = detect_gearbox(rgb, depth, K, T, SEATED_XY, 0.034)
    assert np.hypot(*(det.center_xy - AXIS)) < 0.0003
    assert det.square_error_m < 0.0005
    assert det.ring_offset_m < 0.001
    assert det.z_top == pytest.approx(0.0339, abs=0.002)
    assert abs(det.rotation_deg) < 5


def test_far_view_agrees_within_its_resolution():
    """From 35 cm (0.8 mm/px, depth 4 mm off) the axis still lands within ~2 mm."""
    rgb, depth, K, T = _load("gearbox_far.npz")
    det = detect_gearbox(rgb, depth, K, T, SEATED_XY, 0.034)
    assert np.hypot(*(det.center_xy - AXIS)) < 0.0025


def test_search_started_a_few_mm_off_still_finds_it():
    rgb, depth, K, T = _load("gearbox_close.npz")
    det = detect_gearbox(rgb, depth, K, T, (SEATED_XY[0] + 0.004, SEATED_XY[1] - 0.004), 0.034)
    assert np.hypot(*(det.center_xy - AXIS)) < 0.0003


def test_nothing_there_raises():
    rgb, depth, K, T = _load("gearbox_close.npz")
    with pytest.raises(GearboxNotFound):
        detect_gearbox(rgb, depth, K, T, (0.36, -0.15), 0.034)


def test_popup_found_and_not_found():
    rgb, depth, K, T = _load("gearbox_close.npz")
    det = detect_gearbox(rgb, depth, K, T, SEATED_XY, 0.034)
    assert draw_gearbox_popup(rgb, K, T, det, SEATED_XY).ndim == 3
    assert draw_gearbox_popup(rgb, K, T, None, SEATED_XY, z_top=0.034, error="x").ndim == 3


def _script():
    spec = importlib.util.spec_from_file_location(
        "detect_gearbox_script", Path(__file__).resolve().parents[1] / "scripts" / "detect_gearbox.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(name):
    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    rgb, depth, K, T = _load(name)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    obs = {"cameras": [{"rgb": rgb, "depth": depth, "intrinsics": K,
                        "pose": make_pose(T[:3, 3], [q[3], q[0], q[1], q[2]])}],
           "arms": [{"joint_state": {"positions": np.zeros(8)}}]}

    class Ctx:
        def tool(self, tool_name, **kw):
            assert tool_name == "robot.get_observation"
            return obs
    return Ctx()


COLUMN_Z = {"above_gearbox": 0.1359, "pre_insert": 0.1133, "touching_gearbox": 0.1017, "in_gearbox": 0.0981}


def test_script_puts_the_column_on_the_axis(tmp_path):
    out = _script().run(_ctx("gearbox_close.npz"), nominal_xy=list(SEATED_XY), column_z=COLUMN_Z,
                        save_dir=str(tmp_path), popup=False)
    for name, z in COLUMN_Z.items():
        p = out[name]["position"]
        assert (p["x"], p["y"]) == pytest.approx(tuple(AXIS), abs=3e-4)
        assert p["z"] == z
        assert out[name]["rotation"] == {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    assert 2.5 < out["offset_mm"] < 3.5
    assert len(list(tmp_path.glob("*_gearbox.png"))) == 1 and len(list(tmp_path.glob("*.npz"))) == 1


def test_script_refuses_a_far_detection(tmp_path):
    with pytest.raises(RuntimeError, match="from the taught column"):
        _script().run(_ctx("gearbox_close.npz"), nominal_xy=list(SEATED_XY), column_z=COLUMN_Z,
                      max_offset_m=0.002, save_dir=str(tmp_path), popup=False)


def test_a_frame_taken_while_the_arm_moved_is_refused(tmp_path):
    """16:58 run: the frame came from higher up than its pose said -> top 7.7 mm low."""
    with pytest.raises(RuntimeError, match="does not match the frame|not on its fixture"):
        _script().run(_ctx("gearbox_while_moving.npz"), nominal_xy=list(SEATED_XY), column_z=COLUMN_Z,
                      save_dir=str(tmp_path), popup=False)


def test_observe_still_waits_for_the_arm_to_stop():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from capture_frame import observe_still

    calls = {"n": 0}

    class MovingThenStill:
        def tool(self, name, **kw):
            calls["n"] += 1
            q = np.zeros(8)
            q[0] = min(calls["n"], 6) * 0.01  # moves for 6 reads, then stays
            return {"arms": [{"joint_state": {"positions": q}}], "cameras": [], "n": calls["n"]}

    obs = observe_still(MovingThenStill(), still_s=0.2)
    assert obs["n"] >= 6 + 4, "returned only after the joints stopped changing for 0.2 s"
