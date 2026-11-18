"""Find the gearbox axis with the wrist camera and return the insertion column on it.

Run from the look pose with the gripper empty (the rotor in the jaws would hide part of
the housing). The column poses keep their taught heights and the exact vertical
orientation; only x/y come from the camera (gearbox_vision.py), plus ``correction_xy``
(the held rotor's offset from the tool axis, if one is ever measured; 0 with the vision
grasp, which centres the tool on the rotor).

Saves the frame and a quick-look picture (_gearbox.png) into <save_dir> for every call,
found or not, and pops the picture up like the rotor detection. Raises (the graph stops,
nothing moves) when the gearbox is not found or is further than max_offset_m from the
taught position.
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
from gearbox_vision import GearboxNotFound, detect_gearbox, draw_gearbox_popup  # noqa: E402
from plan_sun_insert import surface_z  # noqa: E402


class Output(TypedDict):
    center_xy: list[float]
    offset_mm: float
    housing_top_z: float  # the housing's top face [m] (a seated rotor is flush with it)
    above_gearbox: Se3Pose
    pre_insert: Se3Pose
    touching_gearbox: Se3Pose
    in_gearbox: Se3Pose


def _vertical(x: float, y: float, z: float) -> Se3Pose:
    """Fingers straight down, rpy [180, 0, 0]."""
    return {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}


def run(
    ctx: NodeContext,
    nominal_xy: list[float],
    column_z: dict,
    z_top_guess: float = 0.034,
    max_offset_m: float = 0.006,
    max_top_dev_m: float = 0.005,
    correction_xy: list[float] | None = None,
    save_dir: str = "~/z3_rotor_images/gearbox",
    popup: bool = True,
) -> Output:
    obs = observe_still(ctx)
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser(), "look")
    picture = frame.with_name(frame.stem + "_gearbox.png")

    def show(det, error=""):
        img = draw_gearbox_popup(rgb, K, T_wc, det, nominal_xy, z_top=z_top_guess, error=error)
        cv2.imwrite(str(picture), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if popup:
            pop_up(picture, "gearbox detection" + (" - NOT FOUND" if error else ""), frame.parent)

    try:
        det = detect_gearbox(rgb, depth, K, T_wc, tuple(nominal_xy), z_top_guess)
    except GearboxNotFound as exc:
        show(None, str(exc))
        raise RuntimeError(f"gearbox not found: {exc} (picture: {picture})") from exc
    show(det)
    if abs(det.z_top - z_top_guess) > max_top_dev_m:
        raise RuntimeError(
            f"gearbox top measured at z {det.z_top:.4f}, usually {z_top_guess:.4f}: the camera pose "
            f"does not match the frame (arm still moving?) or the gearbox is not on its fixture. "
            f"Check {picture}"
        )
    corr = np.asarray(correction_xy if correction_xy is not None else [0.0, 0.0], float)
    cx, cy = (float(v) for v in det.center_xy + corr)
    offset = math.hypot(cx - nominal_xy[0], cy - nominal_xy[1])
    print(
        f"[detect_gearbox] axis ({cx:.4f}, {cy:.4f}) = {offset * 1e3:.1f} mm from the taught "
        f"column, top z {det.z_top:.4f}, hole square err {det.square_error_m * 1e3:.2f} mm, "
        f"ring vs holes {det.ring_offset_m * 1e3:.2f} mm -> {picture}"
    )
    if offset > max_offset_m:
        raise RuntimeError(
            f"gearbox found {offset * 1e3:.1f} mm from the taught column (> {max_offset_m * 1e3:.0f} mm); "
            f"not inserting. Check {picture}"
        )
    housing_top = det.z_top
    if depth is not None:  # median over an annulus outside the recess: steadier than z_top
        housing_top, _ = surface_z(np.asarray(depth, np.float32), K, T_wc, det.center_xy, 0.025, 0.027)
    names = ("above_gearbox", "pre_insert", "touching_gearbox", "in_gearbox")
    missing = [n for n in names if n not in column_z]
    if missing:
        raise RuntimeError(f"column_z lacks {missing}")
    return {
        "center_xy": [cx, cy],
        "offset_mm": offset * 1e3,
        "housing_top_z": float(housing_top),
        **{n: _vertical(cx, cy, float(column_z[n])) for n in names},
    }
