#!/usr/bin/env python
"""Turn joint poses into end-effector poses for build.py's EEF_POSES. No robot needed.

Uses GaP's own FK (FrankaRealEnv._compute_fk) with the same tool point as its IK
(GAP_FRANKA_TCP_FLANGE_M / GAP_FRANKA_TCP_ROT_Z_DEG), so the printed pose is exactly
what robot.go_to_pose aims for. Set those env vars the same way here as for `gap run`.

    cd ~/graph-as-policy
    .venv/bin/python z3_graphs/poses_demo/joints_to_eef.py              # every POSES entry
    .venv/bin/python z3_graphs/poses_demo/joints_to_eef.py "[-0.175, -0.498, 0.064, -2.572, -0.006, 2.098, -0.068]"

The second form takes a list as printed by ``uv run z3-gap-joints``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def joints_to_eef(q: list[float]) -> tuple[list[float], list[float]]:
    """(xyz [m], rpy_deg) of GaP's tool point in the robot base frame."""
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    from gap.envs.franka_real_env import FrankaRealEnv, franka_tcp

    env = FrankaRealEnv.__new__(FrankaRealEnv)  # FK only: no server, no threads
    env._urdf = load_robot_description("panda_description")
    env._TCP_OFFSET, env._TCP_ROTATION_Z = franka_tcp()
    pos, wxyz = env._compute_fk(np.asarray(q, dtype=np.float64))
    w, x, y, z = wxyz
    rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
    return [round(float(v), 4) for v in pos], [round(float(v), 2) for v in rpy]


def main() -> int:
    if len(sys.argv) > 1:
        poses = {f"pose_{i}": ast.literal_eval(a) for i, a in enumerate(sys.argv[1:])}
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build import POSES

        poses = POSES
    print("EEF_POSES = {")
    for name, q in poses.items():
        xyz, rpy = joints_to_eef(q)
        print(f'    "{name}": {{"xyz": {xyz}, "rpy_deg": {rpy}}},')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
