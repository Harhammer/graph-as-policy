"""Find the sun gear roughly in a wide view and return a look pose over it.

From a high pose (e.g. start: the camera sees most of the table at ~0.8 mm/px) the sun is
the only small (~18 mm), tall (~20 mm), round, white part; depth finds it
(sun_vision.find_sun_candidates). The rotor's fixture and the gearbox are excluded (their
known places). The look pose puts the camera ~18 cm above it with it in the image centre,
for the precise detection (detect_sun.py). Pops up the wide view with the candidates.
"""

from __future__ import annotations

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
from detect_sun import vertical  # noqa: E402
from rotor_vision import project  # noqa: E402
from sun_vision import SunNotFound, find_sun_candidates  # noqa: E402


class Output(TypedDict):
    center_xy: list[float]
    look_pose: Se3Pose


def run(
    ctx: NodeContext,
    prefer_xy: list[float] | None = None,
    exclude: list[list[float]] | None = None,
    look_dx_m: float = 0.015,
    look_z_m: float = 0.105,
    save_dir: str = "~/z3_rotor_images/sun",
    popup: bool = True,
) -> Output:
    obs = observe_still(ctx)
    cam = obs["cameras"][0]
    rgb, depth, K = cam["rgb"], cam.get("depth"), np.asarray(cam["intrinsics"], float)
    if depth is None:
        raise RuntimeError("the sun search needs depth")
    T_wc = pose_to_matrix(cam["pose"])
    frame = save_frame(obs, Path(save_dir).expanduser(), "wide")
    picture = frame.with_name(frame.stem + "_find.png")
    cands = find_sun_candidates(depth, K, T_wc, prefer_xy=tuple(prefer_xy) if prefer_xy else None,
                                rgb=rgb, exclude=[tuple(e) for e in (exclude or [])])
    img = np.ascontiguousarray(rgb, np.uint8).copy()
    for i, c in enumerate(cands):
        uv, _ = project(T_wc, K, np.array([[*c.center_xy, 0.02]]))
        p = tuple(int(v) for v in uv[0])
        cv2.circle(img, p, 22, (0, 230, 255) if i == 0 else (255, 160, 0), 3 if i == 0 else 1, cv2.LINE_AA)
    for ex, ey, er in exclude or []:
        uv, _ = project(T_wc, K, np.array([[ex, ey, 0.02]]))
        cv2.drawMarker(img, tuple(int(v) for v in uv[0]), (120, 120, 120), cv2.MARKER_TILTED_CROSS, 18, 1)
    text = (f"sun at ({cands[0].center_xy[0]:.3f}, {cands[0].center_xy[1]:.3f}), {len(cands)} candidate(s)"
            if cands else "NOT FOUND: no small, tall, round, white part in view")
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if cands else (255, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(picture), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if popup:
        pop_up(picture, "sun gear search", frame.parent)
    if not cands:
        raise RuntimeError(f"sun gear not found in the wide view (picture: {picture})")
    c = cands[0]
    cx, cy = (float(v) for v in c.center_xy)
    print(f"[find_sun] sun roughly at ({cx:.4f}, {cy:.4f}), {c.height_m * 1e3:.0f} mm tall; "
          f"{len(cands) - 1} other candidate(s) -> {picture}")
    return {"center_xy": [cx, cy], "look_pose": vertical(cx + look_dx_m, cy, look_z_m)}
