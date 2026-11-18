"""Screw holder: the block, its four holes and the M4x18 standing in them, and the plan that
takes each screw out and puts it back (z3, 2026-11-16 15:11).

The frame was read from the D405 with the arm in user stop: no joints, so its T_wc is a frame
on the fitted table plane (z = 0 there, x = image up, y = image left), as in test_screws.py.
The holder's row runs along that y: hole 0 is the leftmost in the image.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from holder_vision import HolderGeometry, detect_holder  # noqa: E402
from rotor_vision import project  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
GEOM = HolderGeometry()


def _load():
    d = np.load(DATA / "holder_1511.npz")
    return d["rgb"], d["depth"], d["K"], d["T_wc"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(rgb, depth, K, T):
    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    obs = {"cameras": [{"rgb": rgb, "depth": depth, "intrinsics": K,
                        "pose": make_pose(T[:3, 3], [q[3], q[0], q[1], q[2]])}],
           "arms": [{"joint_state": {"positions": np.zeros(8)}}]}

    class Ctx:
        def tool(self, tool_name, **kw):
            return obs
    return Ctx()


@pytest.fixture(scope="module")
def scene():
    return detect_holder(*_load())


def test_the_block_and_its_row(scene):
    assert scene.size_m[0] == pytest.approx(GEOM.length_m, abs=0.008)
    assert scene.size_m[1] == pytest.approx(GEOM.width_m, abs=0.008)
    # the row runs along image left -> right, i.e. -y here: hole 0 -> 3
    assert abs(scene.axis_deg + 90) < 5
    assert scene.holes[0].xy[1] > scene.holes[3].xy[1]
    assert not scene.warnings


def test_four_screws_standing_in_the_holes(scene):
    assert [h.occupied for h in scene.holes] == [True] * 4
    for h in scene.holes:  # the heads' tops: 27 mm in the CAD
        assert h.top_m == pytest.approx(GEOM.head_top_m(), abs=0.0025)
    gaps = [np.hypot(*(b.xy - a.xy)) for a, b in zip(scene.holes, scene.holes[1:], strict=False)]
    assert gaps == pytest.approx([GEOM.pitch_m] * 3, abs=1e-6)


def test_the_holes_are_on_the_heads(scene):
    """Checked by hand on the image (sockets at these pixels): hole centres projected at the
    heads' top within 1.5 px (~0.7 mm)."""
    _, _, K, T = _load()
    uv, _ = project(T, K, np.c_[[h.xy for h in scene.holes], np.full(4, GEOM.head_top_m())])
    sockets = np.array([[386, 251], [419, 252], [452, 253], [484, 254]])
    assert np.abs(uv - sockets).max() < 3.0, np.round(uv, 1)


def test_an_empty_hole_is_seen_and_stops_the_plan(tmp_path):
    """Hole 2's head flattened to the block's top in the depth."""
    rgb, depth, K, T = _load()
    sc = detect_holder(rgb, depth, K, T)
    depth = depth.copy()
    (u, v), = project(T, K, np.r_[sc.holes[2].xy, GEOM.head_top_m()][None])[0]
    cam_top = np.median(depth[int(v) + 18 : int(v) + 24, int(u) - 6 : int(u) + 7])  # the block, below the row
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    depth[np.hypot(xx - u, yy - v) < 11] = cam_top
    sc2 = detect_holder(rgb, depth, K, T)
    assert [h.occupied for h in sc2.holes] == [True, True, False, True]
    mod = _script("detect_holder")
    with pytest.raises(RuntimeError, match="no screw in hole 2"):
        mod.run(_ctx(rgb, depth, K, T), save_dir=str(tmp_path), popup=False)
    out = mod.run(_ctx(rgb, depth, K, T), require_all=False, save_dir=str(tmp_path), popup=False)
    assert out["n_screws"] == 3


@pytest.mark.parametrize("axis, yaw", [(0, 0), (-92.6, 87.4), (92.6, -87.4), (180, 0), (45, 45), (-135, 45),
                                       (90, 90), (-90, 90)])
def test_the_jaws_close_across_the_row(axis, yaw):
    assert _script("detect_holder").grasp_yaw(axis) == pytest.approx(yaw)


def test_detect_holder_plans_out_and_back_in_on_the_same_spot(tmp_path):
    out = _script("detect_holder").run(_ctx(*_load()), save_dir=str(tmp_path), popup=False)
    assert out["n_screws"] == 4 and len(out["holes"]) == 4
    t = 0.0046  # table_z
    for h in out["holes"]:
        a, g, r = (h[k]["position"] for k in ("above_pose", "grasp_pose", "release_pose"))
        assert a["x"] == g["x"] == r["x"] == pytest.approx(h["xy"][0])
        assert a["y"] == g["y"] == r["y"] == pytest.approx(h["xy"][1])
        # fingertips 4 mm over the block's top (17), let go 2 mm higher; TCP 60 mm above them
        assert g["z"] == pytest.approx(t + 0.017 + 0.004 + 0.060)
        assert r["z"] == pytest.approx(g["z"] + 0.002)
        assert a["z"] == pytest.approx(t + 0.05 + 0.060)
        assert h["grasp_pose"]["rotation"] == h["above_pose"]["rotation"] == h["release_pose"]["rotation"]
    assert abs(out["yaw_deg"]) <= 90
    assert list(tmp_path.glob("*_holder.png"))


def test_the_holder_graph_builds_and_validates(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("z3_build", HERE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    if build.HOLDER_LOOK not in build.EEF_POSES:  # not recorded yet
        monkeypatch.setitem(build.EEF_POSES, build.HOLDER_LOOK, {"xyz": [0.4, -0.08, 0.2], "rpy_deg": [180, 0, 0]})
    monkeypatch.setattr(build, "STEPS", build.holder_steps())
    build.build().save(tmp_path / "workflow.json")
    assert sum(s[0] == "holder_screw" for s in build.STEPS) == 4


def test_real_look_pose_frame_in_the_base_frame(tmp_path):
    """15:19 frame from look_holder with the real camera pose: the holder where it lay, the
    table seen 2.3 mm below the touched 4.6, the jaws across the row."""
    d = np.load(DATA / "holder_look_1519.npz")
    rgb, depth, K, T = d["rgb"], d["depth"], d["K"], d["T_wc"]
    sc = detect_holder(rgb, depth, K, T)
    assert sc.center_xy == pytest.approx([0.4267, -0.2253], abs=0.002)
    assert sc.z_table == pytest.approx(0.0023, abs=0.002)
    assert [h.occupied for h in sc.holes] == [True] * 4
    assert sc.holes[0].xy[1] == pytest.approx(-0.2021, abs=0.002)  # hole 0: the robot's left
    out = _script("detect_holder").run(_ctx(rgb, depth, K, T), save_dir=str(tmp_path), popup=False)
    assert out["yaw_deg"] == pytest.approx(86.8, abs=2.0)  # row -93 deg: jaws along x


def test_the_cream_holder_heads_from_colour():
    """15:40, the holder reprinted in cream and taped down (black tape along both sides): the
    heads stand out dark, the row comes from their centres; the depth's was 0.5 mm off."""
    d = np.load(DATA / "holder_cream_1540.npz")
    sc = detect_holder(d["rgb"], d["depth"], d["K"], d["T_wc"])
    assert sc.refined.startswith("colour (4 heads"), sc.refined
    assert [h.occupied for h in sc.holes] == [True] * 4
    for h, m in zip(sc.holes, sc.heads_seen, strict=True):
        assert np.hypot(*(h.xy - m)) < 0.0006
    assert sc.center_xy == pytest.approx([0.4239, -0.2300], abs=0.002)
    gaps = [np.hypot(*(b.xy - a.xy)) for a, b in zip(sc.holes, sc.holes[1:], strict=False)]
    assert gaps == pytest.approx([GEOM.pitch_m] * 3, abs=1e-6)


def test_the_black_holder_falls_back_to_the_depth(scene):
    assert scene.refined.startswith("depth (") and "contrast" in scene.refined


def test_the_taped_holder_from_the_look_pose(tmp_path):
    """15:45, the first run's look (it stopped there): the tape over the block's long edges
    slopes down to the table and made the blob 69 x 28 mm at half the block's height."""
    d = np.load(DATA / "holder_taped_1545.npz")
    rgb, depth, K, T = d["rgb"], d["depth"], d["K"], d["T_wc"]
    sc = detect_holder(rgb, depth, K, T)
    assert sc.size_m[0] == pytest.approx(GEOM.length_m, abs=0.005)
    assert sc.size_m[1] == pytest.approx(GEOM.width_m, abs=0.004)
    assert sc.refined.startswith("colour (4 heads"), sc.refined
    assert [h.occupied for h in sc.holes] == [True] * 4
    out = _script("detect_holder").run(_ctx(rgb, depth, K, T), save_dir=str(tmp_path), popup=False)
    assert out["n_screws"] == 4


def test_the_tool_correction_turns_with_the_wrist(tmp_path):
    """+1 mm along the tool's yaw-0 x: at the grasp yaw (~85 deg here) that is ~+1 mm along y."""
    d = np.load(DATA / "holder_taped_1545.npz")
    ctx = _ctx(d["rgb"], d["depth"], d["K"], d["T_wc"])
    mod = _script("detect_holder")
    a = mod.run(ctx, save_dir=str(tmp_path), popup=False)
    b = mod.run(ctx, tool_correction_xy=[0.001, 0.0], save_dir=str(tmp_path), popup=False)
    yaw = np.radians(b["yaw_deg"])
    for ha, hb in zip(a["holes"], b["holes"], strict=True):
        assert hb["xy"] == ha["xy"]  # the hole stays where it is seen
        ga, gb = ha["grasp_pose"]["position"], hb["grasp_pose"]["position"]
        assert gb["x"] - ga["x"] == pytest.approx(0.001 * np.cos(yaw), abs=1e-9)
        assert gb["y"] - ga["y"] == pytest.approx(0.001 * np.sin(yaw), abs=1e-9)
