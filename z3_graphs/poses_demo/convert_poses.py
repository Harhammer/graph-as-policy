#!/usr/bin/env python
"""Carry taught EEF_POSES over to the bridge's corrected kinematics (--kinematics DIR).

A taught pose is really an arm configuration. In the corrected model that configuration
has a different end-effector pose (the joint zero offsets move it by a few mm), so the
numbers in EEF_POSES must change when the bridge switches to --kinematics — otherwise the
arm goes to a different physical place. This script recovers each pose's configuration
(IK on the nominal model, seeded with the recorded joints) and prints the pose that
configuration has in the corrected model, ready to paste.

    .venv/bin/python z3_graphs/poses_demo/convert_poses.py [--model ~/z3_handeye/combined_kinfit]

Caveat: a 7-joint arm reaches a pose with many configurations, and the offsets move each
of them slightly differently. The seed picks a configuration close to the recorded one,
but poses that need millimetre accuracy (the gearbox insertion) are better re-recorded
with `z3-gap-joints --kinematics <dir> --record` once the bridge runs corrected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "bridges" / "z3_franka" / "src"))
import build  # noqa: E402
from z3_gap_bridge.panda_fk import flange_pose  # noqa: E402

TCP_M = 0.157  # GaP's tool point on the flange z axis; tool orientation = flange orientation

# Which recorded joint pose (POSES) seeds each EEF pose's IK; others use the nearest listed.
SEEDS = {
    "start": "start",
    "above_gears": "on_top_off_gears",
    "at_gears": "at_the_gears",
}


def pose_matrix(spec: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    p = build.eef_pose(spec)
    t = np.eye(4)
    r = p["rotation"]
    t[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    t[:3, 3] = [p["position"][k] for k in "xyz"]
    return t


def tool_pose(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    f = flange_pose(q + dq)
    t = f.copy()
    t[:3, 3] = (f @ np.array([0.0, 0.0, TCP_M, 1.0]))[:3]
    return t


def ik_nominal(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def f(q):
        t = tool_pose(q, np.zeros(7))
        return np.r_[(t[:3, 3] - target[:3, 3]) * 10.0,
                     Rotation.from_matrix(target[:3, :3].T @ t[:3, :3]).as_rotvec(),
                     1e-3 * (q - seed)]

    r = least_squares(f, seed, xtol=1e-12, ftol=1e-12)
    if np.linalg.norm(r.fun[:6]) > 1e-5:
        raise RuntimeError(f"IK did not converge (residual {np.linalg.norm(r.fun[:6]):.2e})")
    return r.x


def main(argv: list[str]) -> int:
    from scipy.spatial.transform import Rotation

    model_dir = Path("~/z3_handeye/combined_kinfit").expanduser()
    for a in argv:
        if a.startswith("--model="):
            model_dir = Path(a.split("=", 1)[1]).expanduser()
    dq = np.radians(json.loads((model_dir / "report.json").read_text())["joint_offsets_deg"])
    print(f"# EEF_POSES in the corrected model ({model_dir.name}: offsets "
          f"{np.round(np.degrees(dq), 2).tolist()} deg). Paste when the bridge runs with "
          f"--kinematics {model_dir}")
    print("EEF_POSES = {")
    last_seed = np.asarray(build.POSES["start"], float)
    for name, spec in build.EEF_POSES.items():
        seed_name = SEEDS.get(name)
        seed = np.asarray(build.POSES[seed_name], float) if seed_name else last_seed
        q = ik_nominal(pose_matrix(spec), seed)
        last_seed = q
        t = tool_pose(q, dq)
        xyz = [round(float(v), 4) for v in t[:3, 3]]
        if "rpy_deg" in spec:
            rpy = Rotation.from_matrix(t[:3, :3]).as_euler("xyz", degrees=True)
            rot = f'"rpy_deg": [{", ".join(f"{v:.1f}" for v in rpy)}]'
        else:
            x, y, z, w = Rotation.from_matrix(t[:3, :3]).as_quat()
            rot = f'"quat_wxyz": [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]'
        shift = (t[:3, 3] - pose_matrix(spec)[:3, 3]) * 1e3
        print(f'    "{name}": {{"xyz": {xyz}, {rot}}},  # moved '
              f'({shift[0]:+.1f}, {shift[1]:+.1f}, {shift[2]:+.1f}) mm')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
