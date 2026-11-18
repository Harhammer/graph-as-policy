"""Droop test: how far the arm sits from a commanded pose once it is still.

Run right after the arm was sent to ``pose`` (and, for the soft phase, after the stiffness
switch). Waits until the tool point has moved less than ``tol_m`` over ``still_s``, then
compares the measured pose (GaP's FK of the measured joints, like robot.insert_twist) with
the commanded one, and the measured joints with the IK solution of the commanded pose (the
joints the arm was sent to). Per joint, stiffness x (commanded - measured) is the torque
the controller's spring holds: gravity the robot's model does not know about (load, cable,
the arm's own model error), plus friction. Appends one JSON line to ``session``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
from gap_core.types import Se3Pose
from scipy.spatial.transform import Rotation

from gap import NodeContext


class Output(TypedDict):
    droop_mm: float
    droop_deg: float
    tip_droop_mm: float
    settle_s: float


def _mat(pose: dict) -> np.ndarray:
    r, p = pose["rotation"], pose["position"]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    t[:3, 3] = [p["x"], p["y"], p["z"]]
    return t


def compute_droop(cmd: dict, meas: dict, q_cmd, q, k_theta, tip_m: float = 0.060) -> dict:
    """Measured vs commanded, base frame. dp/tip_dp [mm]: measured - commanded at the tool
    point and at the fingertip ends (``tip_m`` further along the tool axis); tilt [deg]:
    angle of measured rot * commanded rot^-1; dq [mrad] = measured - commanded joints;
    torque [Nm] = k * (commanded - measured) (what the spring pushes against)."""
    a, b = _mat(cmd), _mat(meas)
    dp = b[:3, 3] - a[:3, 3]
    lever = np.array([0.0, 0.0, tip_m])
    tip_dp = dp + (b[:3, :3] - a[:3, :3]) @ lever
    tilt = np.degrees(np.linalg.norm(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()))
    dq = np.asarray(q, float)[:7] - np.asarray(q_cmd, float)[:7]
    out = {
        "dp_mm": (dp * 1e3).round(3).tolist(),
        "droop_mm": float(np.linalg.norm(dp) * 1e3),
        "tip_dp_mm": (tip_dp * 1e3).round(3).tolist(),
        "tip_droop_mm": float(np.linalg.norm(tip_dp) * 1e3),
        "droop_deg": float(tilt),
        "dq_mrad": (dq * 1e3).round(3).tolist(),
    }
    if k_theta is not None:
        out["torque_nm"] = (-np.asarray(k_theta, float) * dq).round(3).tolist()
    return out


def _wait_still(ctx, still_s: float, tol_m: float, timeout_s: float) -> float:
    t0 = time.monotonic()
    hist: list[tuple[float, np.ndarray]] = []
    while True:
        p = ctx.tool("robot.get_ee_pose")["pose"]["position"]
        now = time.monotonic()
        hist.append((now, np.array([p["x"], p["y"], p["z"]])))
        hist = [h for h in hist if h[0] >= now - still_s]
        spread = max(np.linalg.norm(h[1] - hist[-1][1]) for h in hist)
        if now - t0 >= still_s and spread < tol_m:
            return now - t0
        if now - t0 > timeout_s:
            print(f"[measure_droop] still moving {spread * 1e3:.2f} mm after {timeout_s:.0f} s; measuring anyway")
            return now - t0
        time.sleep(0.02)


def run(
    ctx: NodeContext,
    pose: Se3Pose,
    label: str,
    phase: str,
    rep: int,
    session: str,
    k_theta: list[float] | None = None,
    still_s: float = 0.8,
    tol_m: float = 5e-5,
    timeout_s: float = 8.0,
    tip_m: float = 0.060,
) -> Output:
    settle_s = _wait_still(ctx, still_s, tol_m, timeout_s)
    meas = ctx.tool("robot.get_ee_pose")["pose"]
    obs = ctx.tool("robot.get_observation")
    q = [float(v) for v in obs["arms"][0]["joint_state"]["positions"][:7]]
    q_cmd = [float(v) for v in ctx.tool("robot.solve_ik", pose=pose)["joint_config"]["positions"][:7]]
    d = compute_droop(pose, meas, q_cmd, q, k_theta, tip_m)
    rec = {"t": time.time(), "label": label, "phase": phase, "rep": int(rep), "settle_s": round(settle_s, 2),
           "k_theta": k_theta, "q": q, "q_cmd": q_cmd, "cmd": pose, "meas": meas, **d}
    path = Path(session).expanduser()
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    tq = " ".join(f"{v:+.2f}" for v in d.get("torque_nm", []))
    print(f"[measure_droop] {label} {phase} #{rep}: {d['droop_mm']:.2f} mm (tip {d['tip_droop_mm']:.2f}), "
          f"{d['droop_deg']:.3f} deg, dp {d['dp_mm']} mm, settled {settle_s:.1f} s"
          + (f", torque [Nm] {tq}" if tq else ""))
    return {"droop_mm": d["droop_mm"], "droop_deg": d["droop_deg"],
            "tip_droop_mm": d["tip_droop_mm"], "settle_s": float(settle_s)}
