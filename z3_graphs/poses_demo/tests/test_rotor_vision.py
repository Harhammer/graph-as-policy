"""rotor_vision on rendered frames: a ray-cast rotor (white disk, 8 holes, alternating
black screws / fixture showing through) seen by a tilted camera like the z3 wrist D405
at above_gears. Run: .venv/bin/python -m pytest z3_graphs/poses_demo/tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rotor_vision import RotorGeometry, RotorNotFound, detect_rotor, draw_detection  # noqa: E402

GEOM = RotorGeometry()
NOMINAL = np.array([0.4306, -0.0979])
Z_TOP = 0.105
K = np.array([[430.0, 0, 424.0], [0, 430.0, 240.0], [0, 0, 1]])


def look_at(pos, target, roll_deg=0.0):
    """Camera-to-world (OpenCV: z forward, x right, y down)."""
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, -1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a = math.radians(roll_deg)
    x, y = math.cos(a) * x + math.sin(a) * y, -math.sin(a) * x + math.cos(a) * y
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


# roughly the measured setup: camera 6.5 cm behind the TCP axis, tilted ~15 deg
T_WC = look_at(np.array([0.365, -0.093, 0.244]), [0.4306, -0.1, 0.105], roll_deg=90.0)


def render(center, phase, screws_even, *, fixture=110.0, screw=25.0, rotor=235.0,
           z_top=Z_TOP, T_wc=T_WC, noise=2.0, depth_noise=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    h, w = 480, 848
    u, v = np.meshgrid(np.arange(w), np.arange(h))  # OpenCV: pixel centres at integers
    d_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_w = d_c @ T_wc[:3, :3].T
    o = T_wc[:3, 3]

    def hit(z):
        t = (z - o[2]) / d_w[..., 2]  # = camera-z depth (d_c has z = 1)
        return o[:2] + t[..., None] * d_w[..., :2], t

    xy, t_top = hit(z_top)
    _, t_out = hit(z_top - 0.004)
    _, t_hole = hit(z_top - 0.008)
    _, t_screw = hit(z_top - 0.0005)
    rel = xy - center
    in_disk = np.hypot(rel[..., 0], rel[..., 1]) <= GEOM.outer_d_m / 2
    gray = np.where(in_disk, rotor, fixture)
    depth = np.where(in_disk, t_top, t_out)
    rb, rh = GEOM.bolt_circle_d_m / 2, GEOM.hole_d_m / 2
    for k in range(8):
        a = phase + k * math.pi / 4
        hc = center + rb * np.array([math.cos(a), math.sin(a)])
        in_hole = np.hypot(*(xy - hc).transpose(2, 0, 1)) <= rh
        is_screw = (k % 2 == 0) == screws_even
        gray = np.where(in_hole, screw if is_screw else fixture, gray)
        depth = np.where(in_hole, t_screw if is_screw else t_hole, depth)
    import cv2

    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    gray = np.clip(gray + rng.normal(0, noise, gray.shape), 0, 255).astype(np.uint8)
    depth = (depth + rng.normal(0, depth_noise, depth.shape)).astype(np.float32)
    return np.repeat(gray[..., None], 3, axis=2), depth


def _on_empty_diameter(det, yaw_deg, jaw_axis_deg, phase, screws_even):
    """The jaw axis (jaw_axis + yaw) must pass through two opposite empty holes."""
    empty0 = math.degrees(phase) + (45.0 if screws_even else 0.0)
    off = (jaw_axis_deg + yaw_deg - empty0) % 90.0
    return min(off, 90.0 - off)


@pytest.mark.parametrize(
    "offset_mm, phase_deg, screws_even, seed",
    [((0, 0), 10.0, True, 0), ((3, -2), 33.0, False, 1), ((-4, 3), 2.0, True, 2),
     ((1.5, 4), 44.0, False, 3), ((-2, -4), 21.0, True, 4)],
)
def test_finds_centre_holes_and_grasp(offset_mm, phase_deg, screws_even, seed):
    center = NOMINAL + np.array(offset_mm) / 1e3
    phase = math.radians(phase_deg)
    rgb, depth = render(center, phase, screws_even, seed=seed)
    det = detect_rotor(rgb, depth, K, T_WC, NOMINAL, z_top_guess=Z_TOP + 0.004)
    assert np.hypot(*(det.center_xy - center)) < 0.0002, "centre within 0.2 mm"
    assert det.radius_m == pytest.approx(GEOM.outer_d_m / 2, rel=0.03)
    assert det.z_top == pytest.approx(Z_TOP, abs=0.001) and det.z_from_depth
    assert det.screws_even == screws_even
    hole_err = (math.degrees(det.hole_phase_rad - phase) + 22.5) % 45 - 22.5
    assert abs(hole_err) < 1.5
    for jaw_axis in (0.0, 90.0, 30.0):
        yaw = det.grasp_yaw_deg(jaw_axis)
        assert -45.0 < yaw <= 45.0
        assert _on_empty_diameter(det, yaw, jaw_axis, phase, screws_even) < 1.5


def test_colour_alone_decides_without_depth():
    phase = math.radians(17)
    rgb, _ = render(NOMINAL, phase, False)
    det = detect_rotor(rgb, None, K, T_WC, NOMINAL, z_top_guess=Z_TOP)
    assert det.screws_even is False and not det.z_from_depth


def test_depth_decides_when_the_fixture_is_as_dark_as_the_screws():
    rgb, depth = render(NOMINAL, math.radians(8), True, fixture=40.0, screw=30.0)
    det = detect_rotor(rgb, depth, K, T_WC, NOMINAL, z_top_guess=Z_TOP)
    assert det.screws_even is True
    assert det.cue_scores["depth"] > 2.0


def test_no_rotor_raises():
    rgb = np.full((480, 848, 3), 110, np.uint8)
    depth = np.full((480, 848), 0.15, np.float32)
    with pytest.raises(RotorNotFound):
        detect_rotor(rgb, depth, K, T_WC, NOMINAL, z_top_guess=Z_TOP)


def test_debug_picture_draws_without_error():
    rgb, depth = render(NOMINAL, 0.3, True)
    det = detect_rotor(rgb, depth, K, T_WC, NOMINAL, z_top_guess=Z_TOP)
    out = draw_detection(rgb, K, T_WC, det, jaw_axis_deg=0.0)
    assert out.shape == rgb.shape and (out != rgb).any()


def test_detect_script_returns_grasp_poses_and_saves_pictures(tmp_path):
    """scripts/detect_rotor.py with a fake ctx: plumbing, pose convention, files."""
    import importlib.util

    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    spec = importlib.util.spec_from_file_location(
        "detect_rotor_script", Path(__file__).resolve().parents[1] / "scripts" / "detect_rotor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    center, phase = NOMINAL + np.array([0.002, -0.003]), math.radians(12)
    rgb, depth = render(center, phase, screws_even=False)
    q = Rotation.from_matrix(T_WC[:3, :3]).as_quat()  # xyzw
    obs = {
        "cameras": [{"name": "robot0_robotview", "rgb": rgb, "depth": depth, "intrinsics": K,
                     "pose": make_pose(T_WC[:3, 3], [q[3], q[0], q[1], q[2]])}],
        "arms": [{"joint_state": {"positions": np.zeros(8)}}],
    }

    class Ctx:
        def tool(self, name, **kw):
            assert name == "robot.get_observation"
            return obs

    out = mod.run(Ctx(), nominal_xy=list(NOMINAL), z_top_guess=Z_TOP, grasp_z=0.0975,
                  above_z=0.1481, jaw_axis_deg=90.0, save_dir=str(tmp_path), popup=False)
    g = out["grasp_pose"]
    assert (g["position"]["x"], g["position"]["y"]) == pytest.approx(tuple(center), abs=2e-4)
    assert g["position"]["z"] == 0.0975 and out["above_pose"]["position"]["z"] == 0.1481
    r = g["rotation"]
    expected = Rotation.from_euler("xyz", [180, 0, out["yaw_deg"]], degrees=True).as_quat()
    got = np.array([r["x"], r["y"], r["z"], r["w"]])
    assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-9
    # odd holes have screws -> the jaw axis (90 + yaw) lies on an even (empty) hole diameter
    off = (90.0 + out["yaw_deg"] - math.degrees(phase)) % 90.0
    assert min(off, 90.0 - off) < 1.5
    assert len(list(tmp_path.glob("*.npz"))) == 1 and len(list(tmp_path.glob("*_detection.png"))) == 1
    assert len(list(tmp_path.glob("*_popup.png"))) == 1


def test_detect_script_refuses_a_far_detection(tmp_path):
    import importlib.util

    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    spec = importlib.util.spec_from_file_location(
        "detect_rotor_script2", Path(__file__).resolve().parents[1] / "scripts" / "detect_rotor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rgb, depth = render(NOMINAL + np.array([0.0, 0.009]), 0.2, True)
    q = Rotation.from_matrix(T_WC[:3, :3]).as_quat()
    obs = {"cameras": [{"rgb": rgb, "depth": depth, "intrinsics": K,
                        "pose": make_pose(T_WC[:3, 3], [q[3], q[0], q[1], q[2]])}],
           "arms": [{"joint_state": {"positions": np.zeros(8)}}]}

    class Ctx:
        def tool(self, name, **kw):
            return obs

    with pytest.raises(RuntimeError, match="from nominal"):
        mod.run(Ctx(), nominal_xy=list(NOMINAL), z_top_guess=Z_TOP, grasp_z=0.0975,
                above_z=0.1481, jaw_axis_deg=0.0, max_offset_m=0.008, save_dir=str(tmp_path),
                popup=False)


# -- real frames (z3 cell, 2026-11-09): rotor in its black fixture on the wooden table ---------

DATA = Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
    "name, screws_even",
    [("real_00.npz", False), ("real_03_hand.npz", True)],  # checked by eye on the frames
)
def test_real_frames(name, screws_even):
    d = np.load(DATA / name)
    det = detect_rotor(d["rgb"], d["depth"], d["K"], d["T_wc"], (0.4306, -0.0979), 0.039)
    # all 15 captured frames put the centre within ~1 mm of here (rotor in its fixture)
    assert np.hypot(*(det.center_xy - np.array([0.4333, -0.1018]))) < 0.0015
    assert det.z_top == pytest.approx(0.0385, abs=0.002)
    assert det.screws_even == screws_even
    assert abs(det.screw_score) > 4.0, "clear margin, not a coin flip"


def test_step_figure_found_and_not_found():
    from rotor_vision import draw_steps

    d = np.load(DATA / "real_00.npz")
    trace: dict = {}
    det = detect_rotor(d["rgb"], d["depth"], d["K"], d["T_wc"], (0.4306, -0.0979), 0.039,
                       trace=trace)
    for key in ("top", "mask", "outline_xy", "height_map", "profile", "hole_centroids", "bolt_fit"):
        assert key in trace, key
    img = draw_steps(d["rgb"], d["K"], d["T_wc"], det, trace, 90.0)
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] > 600

    # far from the rotor: fails in step 2, the figure still shows steps 1-2
    trace = {}
    with pytest.raises(RotorNotFound):
        detect_rotor(d["rgb"], d["depth"], d["K"], d["T_wc"], (0.36, -0.20), 0.039, trace=trace)
    img = draw_steps(d["rgb"], d["K"], d["T_wc"], None, trace, 90.0, error="test")
    assert img.ndim == 3 and "top" in trace


def test_popup_picture_found_and_not_found():
    from rotor_vision import draw_popup

    d = np.load(DATA / "real_03_hand.npz")
    det = detect_rotor(d["rgb"], d["depth"], d["K"], d["T_wc"], (0.4306, -0.0979), 0.039)
    img = draw_popup(d["rgb"], d["K"], d["T_wc"], det, (0.4306, -0.0979), 90.0)
    assert img.ndim == 3 and img.shape[0] > 300
    img = draw_popup(d["rgb"], d["K"], d["T_wc"], None, (0.4306, -0.0979), 90.0, z_top=0.039,
                     error="test")
    assert img.ndim == 3


def test_detect_script_applies_the_correction(tmp_path):
    """correction_xy shifts the grasp, not the camera-frame search: same rotor, same image."""
    import importlib.util

    from gap_core.types import make_pose
    from scipy.spatial.transform import Rotation

    spec = importlib.util.spec_from_file_location(
        "detect_rotor_script3", Path(__file__).resolve().parents[1] / "scripts" / "detect_rotor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    center = NOMINAL + np.array([0.003, -0.004])  # where the camera sees the rotor
    rgb, depth = render(center, 0.3, True)
    q = Rotation.from_matrix(T_WC[:3, :3]).as_quat()
    obs = {"cameras": [{"rgb": rgb, "depth": depth, "intrinsics": K,
                        "pose": make_pose(T_WC[:3, 3], [q[3], q[0], q[1], q[2]])}],
           "arms": [{"joint_state": {"positions": np.zeros(8)}}]}

    class Ctx:
        def tool(self, name, **kw):
            return obs

    corr = [-0.0026, 0.0038]
    taught = list(center + np.array(corr))  # the taught grasp is where the rotor really is
    out = mod.run(Ctx(), nominal_xy=taught, z_top_guess=Z_TOP, grasp_z=0.0975, above_z=0.1481,
                  jaw_axis_deg=90.0, save_dir=str(tmp_path), popup=False, correction_xy=corr)
    assert out["center_xy"] == pytest.approx(taught, abs=2e-4)
    assert out["offset_mm"] < 0.2


@pytest.mark.parametrize("name", ["real_mv_top.npz", "real_mv_px_tilted.npz"])
def test_real_multiview_frames_three_shade_background(name):
    """2026-11-09 multi-view frames: rotor centred, lighter wood — Otsu merged wood and rotor."""
    d = np.load(DATA / name)
    det = detect_rotor(d["rgb"], d["depth"], d["K"], d["T_wc"], (0.4306, -0.0979), 0.039)
    assert det.radius_m == pytest.approx(GEOM.outer_d_m / 2, rel=0.08)
    assert det.hole_contrast > 0.15 and abs(det.screw_score) > 3.0
