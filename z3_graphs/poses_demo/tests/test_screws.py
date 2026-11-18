"""Screw kitting: detection, length classes, finger room and the pick/place plan on the real
frame of the two piles (z3, 2026-11-10 09:56).

The frame was read from the D405 with the arm in user stop: no joints, so its T_wc is a frame
on the fitted table plane (z = 0 there, x = image up, y = image left). Lengths, classes and
finger room do not depend on where that frame sits; positions below are in it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from screw_vision import ScrewGeometry, detect_screws  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
LONE_25 = (0.0474, 0.0003)  # head of the one long screw lying apart (left of the lower pile)
LONE_18 = (0.1003, -0.0561)  # head of the short one just right of the upper pile


def _build_grip(build):
    from screw_vision import GripperGeometry

    return GripperGeometry(**{k: build.SCREW[k] for k in ("finger_width_m", "finger_thick_m", "approach_open_m")})


def _load():
    d = np.load(DATA / "screws_two_piles.npz")
    return d["rgb"], d["depth"], d["K"], d["T_wc"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx():
    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    rgb, depth, K, T = _load()
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
    return detect_screws(*_load())


def _near(scene, xy, tol=0.003):
    hits = [s for s in scene.screws if np.hypot(*(s.head_xy - xy)) < tol]
    assert hits, f"no screw with its head near {xy}"
    return hits[0]


def test_table_is_the_plane_of_the_frame(scene):
    assert abs(scene.z_table) < 0.001


def test_the_lone_long_screw_is_an_m4x25_and_graspable(scene):
    s = _near(scene, LONE_25)
    assert s.label == "M4x25"
    assert s.length_m == pytest.approx(ScrewGeometry().total_m("M4x25"), abs=0.0015)
    assert s.graspable, s.reasons
    # its balance point is preferred, but there the finger strips take in 38 mm2 of the pile's
    # nearest screw: the head is next, and free
    assert s.grip_along_m == pytest.approx(0.002)
    assert scene.pick("M4x25") is s


def test_the_short_screw_beside_the_pile_is_taken_where_the_fingers_fit(scene):
    """Its middle and head have neighbours in the finger strips; the thin (1.5 mm) tips fit
    near the far end of its shank."""
    s = _near(scene, LONE_18)
    assert s.label == "M4x18"
    assert s.length_m == pytest.approx(ScrewGeometry().total_m("M4x18"), abs=0.0015)
    assert s.graspable and s.grip_rank > 1 and s.grip_along_m > 0.012


def test_lengths_tell_the_piles_apart(scene):
    """The upper pile is the short screws, the lower one the long: every screw found there
    that measures true to its class is of that pile's class."""
    upper = [s for s in scene.screws if s.head_xy[0] > 0.07 and s.head_xy[1] < -0.005 and s.head_xy[1] > -0.08]
    lower = [s for s in scene.screws if s.head_xy[0] < 0.06 and s.head_xy[1] > -0.05 and s.head_xy[1] < 0.01]
    true = lambda s: not any("measured" in r or "flat" in r for r in s.reasons)  # noqa: E731
    assert sum(true(s) for s in upper) >= 4
    assert all(s.label == "M4x18" for s in upper if true(s))
    assert all(s.label == "M4x25" for s in lower if true(s))


def test_nothing_is_taken_from_an_excluded_box():
    rgb, depth, K, T = _load()
    box = (0.03, -0.02, 0.07, 0.02)  # around the lone M4x25
    sc = detect_screws(rgb, depth, K, T, exclude=[box])
    assert all(not (box[0] <= s.head_xy[0] <= box[2] and box[1] <= s.head_xy[1] <= box[3]) for s in sc.screws)
    assert detect_screws(rgb, depth, K, T, exclude=[(-1, -1, 1, 1)]).screws == []


@pytest.mark.parametrize("axis", [-170.0, -95.0, -30.0, 0.0, 45.0, 90.0, 135.0, 179.0])
@pytest.mark.parametrize("kit", [90.0, -90.0, 30.0, -150.0])
def test_wrist_turn_puts_the_screw_along_the_kit(axis, kit):
    """Any screw, for a kit axis with |axis| in [20, 160] (both turns of the wrist exist)."""
    g, p = _script("detect_screw").choose_yaws(axis, kit)
    jaws = lambda yaw: 90.0 + yaw  # noqa: E731  (VISION jaw_axis_deg)
    across = lambda a, b: abs(math.cos(math.radians(a - b))) < 1e-9  # noqa: E731
    assert across(jaws(g), axis)  # the jaws close across the screw
    held = axis + (p - g)  # the held screw turns with the tool
    assert math.cos(math.radians(held - kit)) == pytest.approx(1.0)  # head -> tip as in the kit
    assert abs(g) <= 160 and abs(p) <= 160


def test_detect_screw_plans_the_pick_and_the_place(tmp_path):
    mod = _script("detect_screw")
    out = mod.run(_ctx(), label="M4x25", slot_head_xy=[0.42, 0.15], kit_axis_deg=90.0,
                  save_dir=str(tmp_path), popup=False)
    g, pl = out["grasp_pose"]["position"], out["place_pose"]["position"]
    assert out["grip_along_mm"] == pytest.approx(2.0)  # on the head (see above)
    assert np.hypot(g["x"] - LONE_25[0], g["y"] - LONE_25[1]) < 0.004
    assert pl["x"] == pytest.approx(0.42, abs=1e-6)
    assert pl["y"] == pytest.approx(0.15 + out["grip_along_mm"] * 1e-3, abs=1e-6)
    # fingertips 2.5 mm over the configured table on the head; TCP 60 mm above them
    assert g["z"] == pytest.approx(0.0046 + 0.0025 + 0.060, abs=1e-6)
    assert pl["z"] == pytest.approx(0.0046 + 0.004 + 0.060, abs=1e-6)  # let go 4 mm up
    assert out["lift_pose"]["position"]["z"] > g["z"] + 0.09
    assert list(tmp_path.glob("*_screws.png"))


def test_detect_screw_stops_when_none_of_that_length_can_be_taken(tmp_path):
    mod = _script("detect_screw")
    with pytest.raises(RuntimeError, match="spread the pile"):
        mod.run(_ctx(), label="M4x18", slot_head_xy=[0.42, 0.105], exclude=[[-1, -1, 1, 1]],
                save_dir=str(tmp_path), popup=False)


def test_the_screw_graph_builds_and_validates(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("z3_build", HERE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "STEPS", build.screw_steps())
    build.build().save(tmp_path / "workflow.json")  # the runtime's own validator
    steps = [s for s in build.STEPS if s[0] == "screw"]
    assert len(steps) == 2 * build.SCREW_COUNT


def test_a_kit_along_x_cannot_take_every_screw():
    """Kit axis 0: the place yaw is 0 or 180, and 180 is past joint 7 — only screws within
    +-160 deg of it can be turned onto it. Hence SCREW_KIT_AXIS_DEG 90."""
    with pytest.raises(RuntimeError, match="no wrist turn"):
        _script("detect_screw").choose_yaws(-170.0, 0.0)


def test_real_look_pose_frame_in_the_base_frame():
    """10:12 frame from look_screws with the real camera pose: the piles where they lie,
    the table 3.8 mm below the touched 4.6 (why detect_screw uses table_z), nothing taken
    from the gearbox, the rotor fixture or the kit."""
    spec = importlib.util.spec_from_file_location("z3_build", HERE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    d = np.load(DATA / "screws_look_1012.npz")
    sc = detect_screws(d["rgb"], d["depth"], d["K"], d["T_wc"], exclude=build.SCREW_EXCLUDE,
                       grip=_build_grip(build))
    assert sc.z_table == pytest.approx(0.0008, abs=0.0015)
    assert all(-0.26 < s.head_xy[1] < -0.16 for s in sc.screws)  # only the two piles
    true = [s for s in sc.screws if not any("measured" in r or "flat" in r for r in s.reasons)]
    assert all(s.label == "M4x18" for s in true if s.head_xy[0] > 0.335)  # far pile: short
    assert all(s.label == "M4x25" for s in true if s.head_xy[0] < 0.33)  # near pile: long
    # loosened piles: enough of both for a kit of 4 + 4 at a 12 mm finger gap
    assert sum(s.label == "M4x18" and s.graspable for s in sc.screws) >= 4
    assert sum(s.label == "M4x25" and s.graspable for s in sc.screws) >= 4


def test_two_frames_of_the_same_scene_agree_and_grip_in_the_middle():
    """10:12 and 10:16: the same loosened piles, arm still. Depth noise (up to 4.6 mm on
    bare wood) and the template's 5 deg steps once made screws flip between graspable and not;
    now both frames give the same screws, and most grips are at the balance point."""
    spec = importlib.util.spec_from_file_location("z3_build", HERE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    result = []
    for name in ("screws_look_1012.npz", "screws_look_1016.npz"):
        d = np.load(DATA / name)
        sc = detect_screws(d["rgb"], d["depth"], d["K"], d["T_wc"], exclude=build.SCREW_EXCLUDE,
                           grip=_build_grip(build))
        ok = [s for s in sc.screws if s.graspable]
        centred = [s for s in ok if s.grip_rank == 0]  # the balance point (GripperGeometry.grip_points)
        assert len(centred) >= len(ok) / 2
        result.append(sorted((s.label, round(s.head_xy[0], 3), round(s.head_xy[1], 3)) for s in ok))
    a, b = result
    assert len(a) == len(b)
    for (la, xa, ya), (lb, xb, yb) in zip(a, b, strict=True):
        assert la == lb and abs(xa - xb) <= 0.0015 and abs(ya - yb) <= 0.0015


def test_the_full_demo_graph_builds_and_validates(tmp_path, monkeypatch):
    """Gearbox first (all of STEPS, unchanged), then the kitting, without a second open/start."""
    spec = importlib.util.spec_from_file_location("z3_build", HERE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    gearbox = list(build.STEPS)
    full = build.full_steps()
    assert full[: len(gearbox)] == gearbox
    assert sum(s[0] == "screw" for s in full) == 2 * build.SCREW_COUNT
    assert full[len(gearbox)] == ("stiffness", "startup") and full[-1] == ("pose", "start")
    monkeypatch.setattr(build, "STEPS", full)
    build.build().save(tmp_path / "workflow.json")
