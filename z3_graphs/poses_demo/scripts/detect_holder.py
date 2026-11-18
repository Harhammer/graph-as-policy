"""Find the screw holder and its M4x18 screws (holder_vision.py) and plan, per hole, taking the
screw out and putting it back into the same hole.

From a look pose that sees the holder: the block, its four holes (CAD pattern, numbered from
the robot's left), which hold a screw. Per hole, all vertical (fingers down), the jaws closing
across the row (the 7 mm wide fingertips then lie along it: 8 mm to the next head):

    above_pose -> grasp_pose (close) -> above_pose (screw out) -> release_pose (open) -> above_pose

The screw goes back in exactly where it came out: the same x/y as the grasp, so the vision's
error only has to stay inside the open fingers (13 mm gap over a 7 mm head), not inside the
hole's 0.1 mm clearance. It is let go ``release_above_m`` higher than it was grasped and drops
the last bit onto the hole's floor.

Heights from the touched ``table_z`` and the CAD (block 17 mm, head 23-27 mm above the table);
the depth's table is only compared with it (``max_table_gap_m``). The fingertips close
``grip_tip_above_block_m`` above the block's top: on the last 2 mm of the free shank and the
whole head, which the Fin Ray tips wrap. ``tip_below_tcp_m``: GaP's TCP (0.157) vs the
fingertips (0.217).

``tool_correction_xy`` [m] is added to every hole's x/y, turned with the tool's yaw (it is
the fingertips' centre's offset from where the tool is sent, in the tool's yaw-0 frame = base
x/y at yaw 0): the grasps turn the wrist ~90 deg from the look, so an off-axis fingertip
centre would show there. Measure it from the peek frames (build.py HOLDER_PEEK).

``require_all``: every hole must hold a screw (the start, and the check at the end that all
went back), else the graph stops. Saves the frame and a picture (_holder.png) into
<save_dir> and pops it up.
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
from holder_vision import HolderGeometry, detect_holder, draw_holder  # noqa: E402
from screw_vision import ScrewsNotFound  # noqa: E402


class HolePlan(TypedDict):
    index: int
    xy: list[float]
    occupied: bool
    top_mm: float  # highest point over the hole, above the table (a seated head: ~27)
    above_pose: Se3Pose
    grasp_pose: Se3Pose
    release_pose: Se3Pose


class Output(TypedDict):
    center_xy: list[float]
    axis_deg: float
    yaw_deg: float
    z_table: float
    n_screws: int
    holes: list[HolePlan]


def grasp_yaw(axis_deg: float) -> float:
    """Tool yaw [deg] in (-90, 90] whose jaws close across a row along ``axis_deg``: at yaw 0
    they close along base y (build.py VISION jaw_axis_deg 90), i.e. across a row along x."""
    y = math.fmod(axis_deg, 180.0)
    if y <= -90.0:
        y += 180.0
    elif y > 90.0:
        y -= 180.0
    return y


def run(
    ctx: NodeContext,
    require_all: bool = True,
    near_xy: list[float] | None = None,
    region: list[float] | None = None,
    table_z: float = 0.0046,
    max_table_gap_m: float = 0.008,
    grip_tip_above_block_m: float = 0.004,
    release_above_m: float = 0.002,
    approach_tip_above_table_m: float = 0.05,
    tip_below_tcp_m: float = 0.060,
    tool_correction_xy: list[float] | None = None,
    label: str = "holder",
    save_dir: str = "~/z3_rotor_images/screwing",
    popup: bool = True,
) -> Output:
    geom = HolderGeometry()
    obs = observe_still(ctx)
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    if depth is None:
        raise RuntimeError("the holder detection needs depth")
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser(), label)
    picture = frame.with_name(frame.stem + "_holder.png")

    def show(scene, lines=(), error=""):
        img = draw_holder(rgb, K, T_wc, scene, geom=geom, lines=lines, error=error)
        cv2.imwrite(str(picture), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if popup:
            pop_up(picture, f"{label}" + (" - NOT FOUND" if error else ""), frame.parent)

    try:
        scene = detect_holder(rgb, depth, K, T_wc, region=tuple(region) if region else None,
                              near_xy=np.asarray(near_xy, float) if near_xy else None, geom=geom,
                              table_z=table_z)
    except ScrewsNotFound as exc:
        show(None, error=str(exc))
        raise RuntimeError(f"holder detection failed: {exc} (picture: {picture})") from exc
    if abs(scene.z_table - table_z) > max_table_gap_m:
        msg = f"the depth puts the table at z {scene.z_table * 1e3:.1f} mm, expected {table_z * 1e3:.1f}"
        show(scene, error=msg)
        raise RuntimeError(f"{msg}: camera pose or depth off (picture: {picture})")
    yaw = grasp_yaw(scene.axis_deg)
    tcp = lambda tip: tip + tip_below_tcp_m  # noqa: E731
    grip_tip = table_z + geom.height_m + grip_tip_above_block_m
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    corr = np.array([[c, -s_], [s_, c]]) @ np.asarray(tool_correction_xy or [0.0, 0.0], float)
    holes: list[HolePlan] = []
    for h in scene.holes:
        xy = h.xy + corr
        holes.append({
            "index": h.index,
            "xy": [float(c) for c in h.xy],
            "occupied": h.occupied,
            "top_mm": h.top_m * 1e3,
            "above_pose": _pose(*xy, tcp(table_z + approach_tip_above_table_m), yaw),
            "grasp_pose": _pose(*xy, tcp(grip_tip), yaw),
            "release_pose": _pose(*xy, tcp(grip_tip + release_above_m), yaw),
        })
    n = sum(h["occupied"] for h in holes)
    lines = [f"{label}: block ({scene.center_xy[0]:.4f}, {scene.center_xy[1]:.4f}), row {scene.axis_deg:+.1f} deg, "
             f"yaw {yaw:+.1f}, heads moved the row {scene.row_shift_m[0] * 1e3:+.1f}/{scene.row_shift_m[1] * 1e3:+.1f} mm",
             f"table z {table_z * 1e3:.1f} mm (seen {scene.z_table * 1e3:.1f}), {n}/{len(holes)} holes hold a screw, "
             f"holes from {scene.refined.split(' (')[0]}, correction {corr[0] * 1e3:+.1f}/{corr[1] * 1e3:+.1f} mm"]
    empty = [h["index"] for h in holes if not h["occupied"]]
    if require_all and empty:
        msg = f"no screw in hole{'s' if len(empty) > 1 else ''} {', '.join(map(str, empty))}"
        show(scene, error=msg)
        raise RuntimeError(f"{msg} (picture: {picture})")
    show(scene, lines)
    print(f"[detect_holder] {lines[0]}; {lines[1]} -> {picture}")
    return {
        "center_xy": [float(c) for c in scene.center_xy],
        "axis_deg": float(scene.axis_deg),
        "yaw_deg": yaw,
        "z_table": table_z,
        "n_screws": n,
        "holes": holes,
    }
