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
    obs = ctx.tool("robot.get_observation")
    path = save_frame(obs, Path(out_dir).expanduser(), label)
    print(f"[capture_frame] saved {path}")
    return {"path": str(path)}
