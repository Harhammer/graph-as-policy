"""Find a screw of one length class (screw_vision.py) and plan its pick and its place.

From a look pose that sees the piles: every screw in view, its length class, whether the
fingers fit around it; the best graspable one of ``label`` is taken. Returns the poses for
the whole pick-and-place, all vertical (fingers down), z from the table seen in the depth:

    above_pose -> grasp_pose (close) -> lift_pose -> slot_high -> slot_above -> place_pose
    (release) -> slot_above

The fingertips close ``head_tip_above_table_m`` above the table on the head (3.5 mm to its
equator) or ``shank_tip_above_table_m`` on the shank (2 mm), and let go
``place_tip_above_table_m`` above it in the kit. The table is ``table_z`` (touched with the
fingertips: the same model as the arm's), not the depth's: from the 30 cm look pose the depth
reads the table and the gearbox's top both 3.8 mm low (2026-11-10 10:12), from the closer
look poses it does not. The depth's table is only compared with it (``max_table_gap_m``). The screw is turned on the way
so that it lies in the kit along ``kit_axis_deg`` (head -> tip, base frame) with its head's
end on ``slot_head_xy``. Of the two jaw orientations that grasp it, the one with less wrist
turn is taken. The tool point is ``tip_below_tcp_m`` above the fingertips (GaP's TCP 0.157
vs the fingertips' 0.217).

Saves the frame and a picture (_screws.png: every screw, class colour, red cross = can't be
grasped, the chosen one's finger strips) into <save_dir>, pops it up, and raises (the graph
stops) when no screw of ``label`` can be grasped.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from gap_core.types import Se3Pose, pose_to_matrix

from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_frame import observe_still, save_frame  # noqa: E402
from detect_rotor import _pose, pop_up  # noqa: E402
from rotor_vision import _wrap  # noqa: E402
from screw_vision import (  # noqa: E402
    GripperGeometry,
    ScrewGeometry,
    ScrewsNotFound,
    detect_screws,
    draw_screws,
)


class Output(TypedDict):
    label: str
    head_xy: list[float]
    axis_deg: float
    length_mm: float
    grip_along_mm: float
    grasp_yaw_deg: float
    place_yaw_deg: float
    z_table: float
    above_pose: Se3Pose
    grasp_pose: Se3Pose
    lift_pose: Se3Pose
    slot_high: Se3Pose
    slot_above: Se3Pose
    place_pose: Se3Pose


def choose_yaws(axis_deg: float, kit_axis_deg: float, max_yaw_deg: float = 160.0) -> tuple[float, float]:
    """(grasp yaw, place yaw) [deg] of the tool. At yaw 0 the jaws close along base y
    (jaw_axis_deg 90, see build.py VISION), so yaw = the screw's axis (mod 180) closes them
    across it; turning the tool by (place - grasp) turns the held screw from ``axis_deg``
    to ``kit_axis_deg``. Of the two solutions (180 deg apart) the one with the smaller
    wrist turns within +-``max_yaw_deg`` (joint 7's range, with margin)."""
    options = []
    for k in (0, 1):
        g = _wrap(axis_deg + 180.0 * k, 360.0)
        p = _wrap(kit_axis_deg + 180.0 * k, 360.0)
        if abs(g) <= max_yaw_deg and abs(p) <= max_yaw_deg:
            options.append((abs(g) + abs(p), g, p))
    if not options:
        raise RuntimeError(f"no wrist turn within +-{max_yaw_deg:.0f} deg takes the screw from "
                           f"{axis_deg:.0f} to {kit_axis_deg:.0f} deg")
    _, g, p = min(options)
    return g, p


def run(
    ctx: NodeContext,
    label: str,
    slot_head_xy: list[float],
    kit_axis_deg: float = 90.0,
    exclude: list[list[float]] | None = None,
    region: list[float] | None = None,
    head_tip_above_table_m: float = 0.0025,
    shank_tip_above_table_m: float = 0.0015,
    table_z: float = 0.0046,
    max_table_gap_m: float = 0.008,
    place_tip_above_table_m: float = 0.004,
    approach_m: float = 0.03,
    carry_m: float = 0.10,
    tip_below_tcp_m: float = 0.060,
    finger_width_m: float = 0.007,
    finger_thick_m: float = 0.0015,
    approach_open_m: float = 0.013,
    allow_shank_grip: bool = True,
    prefer_head: bool = False,
    max_yaw_deg: float = 160.0,
    save_dir: str = "~/z3_rotor_images/screws",
    popup: bool = True,
) -> Output:
    geom = ScrewGeometry()
    if label not in dict(geom.lengths_m):
        raise ValueError(f"label {label!r}: one of {[n for n, _ in geom.lengths_m]}")
    grip = GripperGeometry(finger_width_m=finger_width_m, finger_thick_m=finger_thick_m,
                           approach_open_m=approach_open_m, allow_shank_grip=allow_shank_grip,
                           prefer_head=prefer_head)
    obs = observe_still(ctx)
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    if depth is None:
        raise RuntimeError("the screw detection needs depth")
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser(), label)
    picture = frame.with_name(frame.stem + "_screws.png")
    slot = np.asarray(slot_head_xy, float)

    def show(scene, chosen, lines=(), error=""):
        img = draw_screws(rgb, K, T_wc, scene, chosen, geom, grip, lines=lines, error=error)
        cv2.imwrite(str(picture), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if popup:
            pop_up(picture, f"screws: {label}" + (" - NOT FOUND" if error else ""), frame.parent)

    try:
        scene = detect_screws(rgb, depth, K, T_wc, region=tuple(region) if region else None,
                              exclude=[tuple(b) for b in (exclude or [])], geom=geom, grip=grip)
    except ScrewsNotFound as exc:
        show(None, None, error=str(exc))
        raise RuntimeError(f"screw detection failed: {exc} (picture: {picture})") from exc
    s = scene.pick(label)
    if s is None:
        n = sum(x.label == label for x in scene.screws)
        why = sorted({r for x in scene.screws if x.label == label for r in x.reasons})
        msg = (f"no {label} the gripper can take: {n} seen" + (f" ({'; '.join(why[:3])})" if why else "")
               + " — spread the pile a little")
        show(scene, None, error=msg)
        raise RuntimeError(f"{msg} (picture: {picture})")
    g_yaw, p_yaw = choose_yaws(s.axis_deg, kit_axis_deg, max_yaw_deg)
    on_head = s.grip_along_m < geom.head_len_m
    grasp_xy = s.point(s.grip_along_m)
    seen_z = scene.table_z_at(grasp_xy)
    if abs(seen_z - table_z) > max_table_gap_m:
        show(scene, s, error=f"the depth puts the table at z {seen_z * 1e3:.1f} mm, expected {table_z * 1e3:.1f}")
        raise RuntimeError(f"table seen at z {seen_z * 1e3:.1f} mm, {table_z * 1e3:.1f} expected: "
                           f"camera pose or depth off, not grasping (picture: {picture})")
    z_table = table_z
    tip_z = z_table + (head_tip_above_table_m if on_head else shank_tip_above_table_m)
    kit_axis = np.array([math.cos(math.radians(kit_axis_deg)), math.sin(math.radians(kit_axis_deg))])
    place_xy = slot + s.grip_along_m * kit_axis  # heads lined up on the slots
    tcp = lambda tip: tip + tip_below_tcp_m  # noqa: E731
    lines = [f"{label}: head ({s.head_xy[0]:.4f}, {s.head_xy[1]:.4f}), axis {s.axis_deg:+.0f} deg, "
             f"{s.length_m * 1e3:.1f} mm, grip {'head' if on_head else f'shank {s.grip_along_m * 1e3:.0f} mm'}",
             f"yaw {g_yaw:+.0f} -> {p_yaw:+.0f} deg, table z {z_table * 1e3:.1f} mm (seen {seen_z * 1e3:.1f}), "
             f"slot ({slot[0]:.3f}, {slot[1]:.3f})", *scene.warnings]
    show(scene, s, lines)
    print(f"[detect_screw] {lines[0]}; {lines[1]} -> {picture}")
    return {
        "label": label,
        "head_xy": [float(v) for v in s.head_xy],
        "axis_deg": float(s.axis_deg),
        "length_mm": s.length_m * 1e3,
        "grip_along_mm": s.grip_along_m * 1e3,
        "grasp_yaw_deg": g_yaw,
        "place_yaw_deg": p_yaw,
        "z_table": z_table,
        "above_pose": _pose(*grasp_xy, tcp(z_table + approach_m), g_yaw),
        "grasp_pose": _pose(*grasp_xy, tcp(tip_z), g_yaw),
        "lift_pose": _pose(*grasp_xy, tcp(z_table + carry_m), g_yaw),
        "slot_high": _pose(*place_xy, tcp(z_table + carry_m), p_yaw),
        "slot_above": _pose(*place_xy, tcp(z_table + approach_m), p_yaw),
        "place_pose": _pose(*place_xy, tcp(z_table + place_tip_above_table_m), p_yaw),
    }
