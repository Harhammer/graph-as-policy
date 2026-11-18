"""Save the wrist camera's current frame for offline tuning of rotor_vision.

Writes <out_dir>/<timestamp>[_label].npz (rgb, depth [m], K, T_wc camera-to-base, joints)
and a .png next to it. Hold the arm still before this node (a robot.wait): the camera pose
travels with the image and is computed from the joints at send time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from gap import NodeContext
from gap_core.types import pose_to_matrix


class Output(TypedDict):
    path: str


def observe_still(ctx, still_s: float = 0.4, tol_rad: float = 3e-4, timeout_s: float = 6.0) -> dict:
    """An observation taken once the arm has not moved for ``still_s``.

    The camera pose sent with each frame is computed from the joints when it is sent; a
    frame captured while the arm still moves lands on the wrong pose (2026-11-09: the
    gearbox looked 7.7 mm low and 4.5 mm off, the rotor then hit the rim). A fixed wait
    after a move is not enough: the move tool returns within its joint tolerance.
    """
    t_end = time.monotonic() + timeout_s
    ref = None
    still_since = time.monotonic()
    while True:
        obs = ctx.tool("robot.get_observation")
        q = np.asarray(obs["arms"][0]["joint_state"]["positions"], dtype=float)[:7]
        now = time.monotonic()
        if ref is None or np.abs(q - ref).max() > tol_rad:
            ref, still_since = q, now
        elif now - still_since >= still_s:
            return ctx.tool("robot.get_observation")  # a frame from the still period
        if now > t_end:
            raise RuntimeError(f"the arm did not come to rest within {timeout_s:.0f} s")
        time.sleep(0.05)


def save_frame(obs: dict, out_dir: Path, label: str = "") -> Path:
    cam = obs["cameras"][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    if label:
        stem += f"_{label}"
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        rgb=np.asarray(cam["rgb"], dtype=np.uint8),
        depth=np.asarray(cam.get("depth", np.zeros(0)), dtype=np.float32),
        K=np.asarray(cam["intrinsics"], dtype=np.float64),
        T_wc=pose_to_matrix(cam["pose"]),
        joints=np.asarray(obs["arms"][0]["joint_state"]["positions"], dtype=np.float64),
    )
    cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(cam["rgb"], cv2.COLOR_RGB2BGR))
    return path


def run(ctx: NodeContext, out_dir: str = "~/z3_rotor_images", label: str = "") -> Output:
    obs = observe_still(ctx)
    path = save_frame(obs, Path(out_dir).expanduser(), label)
    print(f"[capture_frame] saved {path}")
    return {"path": str(path)}
