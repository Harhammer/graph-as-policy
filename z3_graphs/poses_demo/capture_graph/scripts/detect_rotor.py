"""Find the rotor in the wrist camera and return the grasp pose (rotor_vision.py).

The jaws close along a line through two opposite empty screw holes, centred on the rotor.
Saves the frame (.npz), an annotated picture (_detection.png) and, with save_steps, a
figure of every detection step (_steps.png) for every call, found or not, into
<save_dir>/<timestamp>* — the material for tuning. With popup, a quick-look picture
(_popup.png: holes, rotor circle, new centre, grasp line, taught position) opens in its own
window; the graph does not wait for it.

correction_xy is added to every detected centre: the hand-eye calibration is only good to
~5 mm, but from the fixed look pose that error is a constant offset, measured once against
the taught grasp (tune_rotor.py --reference). Pictures stay in the camera's frame (overlays
line up with the image); the taught position is drawn there as taught - correction_xy. Raises (the graph stops, nothing moves)
when the rotor or its hole pattern is not found, or when it is further than max_offset_m
from where it should be (more than the fixture's play means a wrong detection).
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose, pose_to_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_frame import save_frame  # noqa: E402
from rotor_vision import (  # noqa: E402
    RotorNotFound,
    detect_rotor,
    draw_detection,
    draw_popup,
    draw_steps,
)

logger = logging.getLogger(__name__)
VIEWER = Path(__file__).resolve().parent / "show_image.py"


def pop_up(png: Path, title: str, save_dir: Path) -> None:
    """Show ``png`` in its own window (a separate viewer process: never blocks the graph).
    Closes the previous detection's window first. Skipped without a display."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return
    pidfile = save_dir / ".viewer.pid"
    try:
        os.kill(int(pidfile.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
    try:
        proc = subprocess.Popen(
            [sys.executable, str(VIEWER), str(png), title],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        pidfile.write_text(str(proc.pid))
    except OSError:
        logger.warning("could not open the detection window", exc_info=True)


class Output(TypedDict):
    grasp_pose: Se3Pose
    above_pose: Se3Pose
    center_xy: list[float]
    yaw_deg: float
    offset_mm: float


def _pose(x: float, y: float, z: float, yaw_deg: float) -> Se3Pose:
    """Fingers down (roll 180), turned ``yaw_deg`` about base z — the build.py convention."""
    a = math.radians(yaw_deg) / 2
    # Rz(yaw) * Rx(180) as a wxyz quaternion: (0, cos a, sin a, 0)
    return {
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "rotation": {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0},
    }


def run(
    ctx: NodeContext,
    nominal_xy: list[float],
    z_top_guess: float,
    grasp_z: float,
    above_z: float,
    jaw_axis_deg: float,
    max_offset_m: float = 0.008,
    save_dir: str = "~/z3_rotor_images/detections",
    save_steps: bool = True,
    popup: bool = True,
    correction_xy: list[float] | None = None,
) -> Output:
    corr = np.asarray(correction_xy if correction_xy is not None else [0.0, 0.0], float)
    nominal_cam = [float(nominal_xy[0] - corr[0]), float(nominal_xy[1] - corr[1])]
    obs = ctx.tool("robot.get_observation")
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser())
    picture = frame.with_name(frame.stem + "_detection.png")
    steps = frame.with_name(frame.stem + "_steps.png")
    quick = frame.with_name(frame.stem + "_popup.png")
    trace: dict = {}

    def show(det, error=""):
        img = draw_popup(rgb, K, T_wc, det, nominal_cam, jaw_axis_deg,
                         z_top=trace.get("z_top"), error=error)
        cv2.imwrite(str(quick), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if popup:
            pop_up(quick, "rotor detection" + (" - NOT FOUND" if error else ""), frame.parent)

    def save_steps_figure(det, error=""):
        if save_steps:
            img = draw_steps(rgb, K, T_wc, det, trace, jaw_axis_deg, error=error)
            cv2.imwrite(str(steps), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    try:
        det = detect_rotor(rgb, depth, K, T_wc, tuple(nominal_cam), z_top_guess, trace=trace)
    except RotorNotFound as exc:
        cv2.imwrite(str(picture), cv2.cvtColor(
            draw_detection(rgb, K, T_wc, None, note=f"NOT FOUND: {exc}"), cv2.COLOR_RGB2BGR))
        show(None, str(exc))
        save_steps_figure(None, str(exc))
        raise RuntimeError(f"rotor not found: {exc} (pictures: {picture}, {steps})") from exc
    yaw = det.grasp_yaw_deg(jaw_axis_deg)
    cv2.imwrite(str(picture), cv2.cvtColor(
        draw_detection(rgb, K, T_wc, det, jaw_axis_deg), cv2.COLOR_RGB2BGR))
    show(det)
    save_steps_figure(det)
    cx, cy = (float(v) for v in det.center_xy + corr)  # camera estimate -> robot frame
    offset = math.hypot(cx - nominal_xy[0], cy - nominal_xy[1])
    print(
        f"[detect_rotor] centre ({cx:.4f}, {cy:.4f}) = {offset * 1e3:.1f} mm from nominal, "
        f"top z {det.z_top:.4f}, screws {'even' if det.screws_even else 'odd'} "
        f"(score {det.screw_score:+.1f}), grasp yaw {yaw:+.1f} deg -> {picture}"
    )
    for w in det.warnings:
        logger.warning("[detect_rotor] %s", w)
    if offset > max_offset_m:
        raise RuntimeError(
            f"rotor found {offset * 1e3:.1f} mm from nominal (> {max_offset_m * 1e3:.0f} mm); "
            f"not grasping. Check {picture}"
        )
    return {
        "grasp_pose": _pose(cx, cy, grasp_z, yaw),
        "above_pose": _pose(cx, cy, above_z, yaw),
        "center_xy": [cx, cy],
        "yaw_deg": yaw,
        "offset_mm": offset * 1e3,
    }
