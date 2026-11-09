"""Find the sun gear on the table (sun_vision.py) and return the grasp on it.

Grasped as it stands, from above, the jaws holding its top ``grip_m`` (the tool point is
``tip_below_tcp_m`` above the fingertip ends). Saves the frame and a quick-look picture
(_sun.png) into <save_dir>, pops it up, and raises (the graph stops) when nothing of the
sun's size stands within ``max_offset_m`` of the expected position.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose, pose_to_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_frame import observe_still, save_frame  # noqa: E402
from detect_rotor import pop_up  # noqa: E402
from sun_vision import SunNotFound, detect_sun, draw_sun_popup  # noqa: E402


class Output(TypedDict):
    center_xy: list[float]
    top_z: float
    height_m: float
    grasp_pose: Se3Pose
    above_pose: Se3Pose


def vertical(x: float, y: float, z: float) -> Se3Pose:
    return {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}


def run(
    ctx: NodeContext,
    nominal_xy: list[float],
    grip_m: float = 0.004,
    approach_m: float = 0.03,
    tip_below_tcp_m: float = 0.060,
    max_offset_m: float = 0.03,
    save_dir: str = "~/z3_rotor_images/sun",
    popup: bool = True,
) -> Output:
    obs = observe_still(ctx)
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser(), "look")
    picture = frame.with_name(frame.stem + "_sun.png")

    def show(det, error=""):
        img = draw_sun_popup(rgb, K, T_wc, det, nominal_xy, error=error)
        cv2.imwrite(str(picture), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if popup:
            pop_up(picture, "sun gear detection" + (" - NOT FOUND" if error else ""), frame.parent)

    if depth is None:
        raise RuntimeError("the sun gear detection needs depth")
    try:
        det = detect_sun(rgb, depth, K, T_wc, tuple(nominal_xy))
    except SunNotFound as exc:
        show(None, str(exc))
        raise RuntimeError(f"sun gear not found: {exc} (picture: {picture})") from exc
    show(det)
    cx, cy = (float(v) for v in det.center_xy)
    offset = math.hypot(cx - nominal_xy[0], cy - nominal_xy[1])
    print(f"[detect_sun] sun at ({cx:.4f}, {cy:.4f}), {offset * 1e3:.1f} mm from expected, top z "
          f"{det.z_top:.4f}, {det.height_m * 1e3:.1f} mm tall -> {picture}")
    if offset > max_offset_m:
        raise RuntimeError(f"sun gear {offset * 1e3:.0f} mm from where expected (> {max_offset_m * 1e3:.0f}); check {picture}")
    tip_z = det.z_top - grip_m  # fingertip ends
    if tip_z - det.z_table < 0.003:
        raise RuntimeError(f"grasp would put the fingertips {(tip_z - det.z_table) * 1e3:.1f} mm above the table")
    tcp_z = tip_z + tip_below_tcp_m
    return {
        "center_xy": [cx, cy],
        "top_z": float(det.z_top),
        "height_m": float(det.height_m),
        "grasp_pose": vertical(cx, cy, tcp_z),
        "above_pose": vertical(cx, cy, tcp_z + approach_m),
    }
