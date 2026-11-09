"""Sun gear detection, grasp and insertion planning on real frames (z3, 2026-11-09)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sun_vision import SunNotFound, detect_sun  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def _load(name):
    d = np.load(DATA / name)
    return d["rgb"], d["depth"], d["K"], d["T_wc"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
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
            return obs
    return Ctx()


def test_finds_the_standing_sun_even_from_far():
    rgb, depth, K, T = _load("sun_far.npz")
    det = detect_sun(rgb, depth, K, T, (0.318, -0.063))
    assert np.hypot(*(det.center_xy - [0.3172, -0.0642])) < 0.001
    assert det.radius_m == pytest.approx(0.0088, rel=0.08)
    assert det.height_m == pytest.approx(0.0175, abs=0.003)
    assert det.bore_offset_m is not None and det.bore_offset_m < 0.0015


def test_nothing_standing_there_raises():
    rgb, depth, K, T = _load("sun_far.npz")
    with pytest.raises(SunNotFound):
        detect_sun(rgb, depth, K, T, (0.40, -0.17))


def test_grasp_holds_the_top_and_clears_the_table(tmp_path):
    out = _script("detect_sun").run(_ctx("sun_far.npz"), nominal_xy=[0.318, -0.063], grip_m=0.004,
                                    save_dir=str(tmp_path), popup=False)
    tips = out["grasp_pose"]["position"]["z"] - 0.060
    assert tips == pytest.approx(out["top_z"] - 0.004, abs=1e-9)
    assert out["above_pose"]["position"]["z"] > out["grasp_pose"]["position"]["z"]
    assert out["grasp_pose"]["rotation"] == {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def test_plan_heights():
    out = _script("plan_sun_insert").run(None, center_xy=[0.43, 0.027], housing_top_z=0.033,
                                         sun_height_m=0.0212, grip_m=0.004, clearance_m=0.002,
                                         start_gap_m=0.008, press_gap_m=0.008)
    tip = lambda pose: pose["position"]["z"] - 0.060  # noqa: E731
    top = 0.033  # flush with the housing
    assert out["rotor_top_z"] == pytest.approx(top)
    # held move: the sun's bottom starts 8 mm above the rotor top, fingertips end 2 mm above it
    assert tip(out["pre"]) + 0.004 - 0.0212 == pytest.approx(top + 0.008)
    assert tip(out["pre"]) - out["insert_depth_m"] == pytest.approx(top + 0.002)
    # press: closed fingertips 8 mm above the released sun's top, down to flush
    assert tip(out["press_start"]) == pytest.approx(top + 0.002 + 0.004 + 0.008)
    assert tip(out["press_start"]) - out["press_depth_m"] == pytest.approx(top)
    assert tip(out["release_lift"]) > top + 0.002 + 0.004
    # a stall counts once the sun's bottom is >= 4 mm into the rotor; the press accepts any stall
    sun_bottom_at_band = tip(out["pre"]) - out["insert_depth_m"] + out["insert_stall_ok_m"] + 0.004 - 0.0212
    assert sun_bottom_at_band == pytest.approx(top - 0.004)
    assert out["press_stall_ok_m"] == pytest.approx(out["press_depth_m"])
    for k in ("above_rotor", "pre", "release_lift", "press_start"):
        assert (out[k]["position"]["x"], out[k]["position"]["y"]) == (0.43, 0.027)


def test_plan_rejects_an_implausible_sun_height():
    with pytest.raises(RuntimeError, match="implausible"):
        _script("plan_sun_insert").run(None, center_xy=[0.43, 0.027], housing_top_z=0.033, sun_height_m=0.004)


def test_gearbox_script_reports_the_housing_top(tmp_path):
    out = _script("detect_gearbox").run(
        _ctx("rotor_seated_in_gearbox.npz"), nominal_xy=[0.4310, 0.0270],
        column_z={"above_gearbox": 0.13, "pre_insert": 0.11, "touching_gearbox": 0.1, "in_gearbox": 0.096},
        save_dir=str(tmp_path), popup=False)
    assert out["housing_top_z"] == pytest.approx(0.033, abs=0.0015)


EXCLUDE = [[0.4257, -0.0967, 0.03], [0.4310, 0.0270, 0.04]]  # build.py SUN_SEARCH_EXCLUDE


@pytest.mark.parametrize("frame, where", [
    ("sun_far.npz", (0.3172, -0.0642)),  # the start-pose view of 16:23
    ("start_view_older.npz", (0.2566, 0.0014)),  # 15:48: the sun near the image's edge
])
def test_wide_search_finds_the_sun_and_looks_over_it(frame, where, tmp_path):
    out = _script("find_sun").run(_ctx(frame), prefer_xy=[0.317, -0.064], exclude=EXCLUDE,
                                  save_dir=str(tmp_path), popup=False)
    assert np.hypot(*(np.array(out["center_xy"]) - where)) < 0.003
    p = out["look_pose"]["position"]
    assert (p["x"], p["y"], p["z"]) == pytest.approx((out["center_xy"][0] + 0.015, out["center_xy"][1], 0.105))


def test_wide_search_ignores_the_empty_rotor_fixture(tmp_path):
    """With the rotor in the gearbox its empty fixture looks sun-like from afar."""
    from sun_vision import find_sun_candidates

    rgb, depth, K, T = _load("empty_rotor_fixture.npz")
    assert find_sun_candidates(depth, K, T, rgb=rgb), "without the exclusion it is a candidate"
    with pytest.raises(RuntimeError, match="not found"):
        _script("find_sun").run(_ctx("empty_rotor_fixture.npz"), exclude=EXCLUDE, save_dir=str(tmp_path), popup=False)


@pytest.mark.parametrize("frame, nominal", [("sun_look_1720.npz", (0.2743, -0.0961)),
                                            ("sun_look_1632.npz", (0.317, -0.064))])
def test_centre_ignores_the_side_wall_facing_the_camera(frame, nominal):
    """From the look pose the camera also sees the sun's side wall (teeth) on its near side;
    a free circle fit grew to r 9.2 and moved 0.8 mm toward the camera (-x). The known-radius
    fit moves it back by that much, away from the camera."""
    from sun_vision import SunConfig

    rgb, depth, K, T = _load(frame)
    tr = {}
    det = detect_sun(rgb, depth, K, T, nominal, cfg=SunConfig(fixed_radius_fit=True), trace=tr)
    shift = (det.center_xy - tr["free_fit_center"]) * 1e3
    assert 0.5 < shift[0] < 1.2 and abs(shift[1]) < 0.4
    assert not det.warnings
    # the default (what the graph uses) is the free fit, which the working runs had
    det = detect_sun(rgb, depth, K, T, nominal)
    np.testing.assert_allclose(det.center_xy, tr["free_fit_center"], atol=1e-9)
