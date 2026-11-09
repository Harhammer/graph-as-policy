"""Pivot calibration of the gripper's tool point (TCP), and the table height with it.

Commands no motion. Close the gripper, then hand-guide so the closed fingertips rest on
ONE fixed point on the table (a pen dot, a board corner) from several wrist tilts, and
press Enter at each. The tool point ``p`` (flange frame) and the touched point ``c``
(base frame) solve ``R_i p + t_i = c`` for every flange pose ``(R_i, t_i)`` in the least
squares sense; the per-pose residual is the quality measure.

Outputs, printed and saved as JSON:

- ``GAP_FRANKA_TCP_FLANGE_M`` for GaP (FK and IK both read it), and
- ``--tool-length-m`` / ``--z-floor-m`` for the bridge's floor guard. The guard models the
  tool as a point on the flange axis; it is valid only when the touched point lies on the
  table surface. The old values (-0.045 / 0.20) belong to the previous cell — on
  2026-11-06 the hand-eye board put the table at z = +0.017, 6 cm above that floor.

    z3-gap-tcp                       # capture, then solve
    z3-gap-tcp --from-file F.json    # re-solve saved samples
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from z3_gap_bridge.handeye import tilt_coverage_deg

log = logging.getLogger(__name__)

MAX_RESIDUAL_MM = 1.5
MIN_TILT_DEG = 20.0


@dataclass
class TcpConfig:
    """Pivot-calibrate the gripper tool point by touching one table point from many tilts."""

    out: Path | None = None
    """Where samples and the result go. Default: ~/z3_tcp/<timestamp>.json."""
    from_file: Path | None = None
    """Skip capture and re-solve the samples saved in this JSON."""
    min_samples: int = 6
    floor_margin_m: float = 0.003
    """Floor guard = touched table height + this."""
    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    robot_ip: str = "192.168.98.121"


def solve_pivot(base_T_flange: list[np.ndarray]) -> dict[str, Any]:
    """Least-squares pivot: tool point in the flange frame, touched point in the base frame."""
    a = np.vstack([np.hstack([p[:3, :3], -np.eye(3)]) for p in base_T_flange])
    b = -np.concatenate([p[:3, 3] for p in base_T_flange])
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    tool, point = x[:3], x[3:]
    residuals = np.array(
        [np.linalg.norm(p[:3, :3] @ tool + p[:3, 3] - point) for p in base_T_flange]
    )
    return {
        "tool_in_flange_m": tool,
        "point_in_base_m": point,
        "residuals_mm": residuals * 1e3,
        "rms_mm": float(np.sqrt(np.mean(residuals**2)) * 1e3),
        "tilt_deg": tilt_coverage_deg(base_T_flange),
    }


def report(cfg: TcpConfig, result: dict[str, Any]) -> tuple[int, str]:
    """Print the result, the settings to use and the verdict."""
    tool, point = result["tool_in_flange_m"], result["point_in_base_m"]
    tilt = result["tilt_deg"]
    n = len(result["residuals_mm"])
    print(
        f"\n{n} touches; rotation about the three axes "
        + " / ".join(f"{a:.0f}" for a in tilt)
        + " deg"
    )
    print("per-touch residual (mm): " + " ".join(f"{r:.1f}" for r in result["residuals_mm"]))
    print(f"residual rms {result['rms_mm']:.2f} mm (pass <= {MAX_RESIDUAL_MM})")
    print(
        f"tool point in the flange frame: x {tool[0] * 1e3:+.1f}  y {tool[1] * 1e3:+.1f}  "
        f"z {tool[2] * 1e3:+.1f} mm   (stock Robotiq 2F-85 in GaP: 0 / 0 / +157)"
    )
    print(f"touched point in the base frame: {np.round(point * 1e3, 1).tolist()} mm")
    verdict = []
    if tilt[1] < MIN_TILT_DEG:
        verdict.append(f"too little tilt ({tilt[1]:.0f} deg) — tilt forward/back AND sideways")
    if result["rms_mm"] > MAX_RESIDUAL_MM:
        verdict.append(f"residual {result['rms_mm']:.1f} mm — the tip slid, or a touch was off")
    if not 0.05 < tool[2] < 0.35:
        verdict.append(f"tool z {tool[2] * 1e3:.0f} mm is implausible for a gripper")
    if verdict:
        print("NOT GOOD ENOUGH: " + "; ".join(verdict))
        return 1, verdict[0]
    floor = point[2] + cfg.floor_margin_m
    print(
        "\nGOOD. Use:\n"
        f"  GaP:     export GAP_FRANKA_TCP_FLANGE_M="
        f'"{tool[0]:.4f},{tool[1]:.4f},{tool[2]:.4f}"\n'
        f"  bridge:  --tool-length-m {tool[2]:.4f} --z-floor-m {floor:.4f}"
        f"   (table {point[2] * 1e3:+.1f} mm + {cfg.floor_margin_m * 1e3:.0f} mm; only if the touched"
        " point is ON the table surface)"
    )
    if np.hypot(tool[0], tool[1]) > 0.005:
        print(
            f"  note: the tip sits {np.hypot(tool[0], tool[1]) * 1e3:.0f} mm off the flange axis; the"
            " floor guard ignores that (it models the tool on the axis)"
        )
    return 0, "ok"


def _save(path: Path, flanges: list[np.ndarray], qs: list[np.ndarray], result: dict | None) -> None:
    data: dict[str, Any] = {
        "base_T_flange": [f.tolist() for f in flanges],
        "q": [q.tolist() for q in qs],
    }
    if result is not None:
        data["result"] = {k: np.asarray(v).tolist() for k, v in result.items()}
    path.write_text(json.dumps(data, indent=2))


def capture(cfg: TcpConfig) -> int:
    """Interactive capture over FCI (read-only), then solve."""
    from z3_gap_bridge.franky_client import FrankyClient

    path = cfg.out or Path.home() / "z3_tcp" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    client = FrankyClient(cfg.nuc_host, cfg.service_port, stale_threshold_s=0.1, stale_run_s=0.5)
    client.connect_robot(cfg.robot_ip)  # read-only from here on: read_q, flange_pose_of
    print(
        f"samples -> {path}\n"
        "Close the gripper (z3-gap-gripper --command close). Mark ONE point on the table.\n"
        "Hand-guide until the closed fingertips rest on it, let go, then:\n"
        "  Enter = capture   u = undo last   s = solve now   q = solve and quit\n"
        "Tilt the wrist 20-40 deg forward/back AND sideways between touches; keep the tip on the point.\n"
    )
    flanges: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    result = None
    try:
        while True:
            cmd = input(f"[{len(flanges)} touches] > ").strip().lower()
            if cmd == "u" and flanges:
                flanges.pop()
                qs.pop()
                continue
            if cmd in ("s", "q"):
                if len(flanges) >= cfg.min_samples:
                    result = solve_pivot(flanges)
                    report(cfg, result)
                    _save(path, flanges, qs, result)
                else:
                    print(f"need {cfg.min_samples} touches, have {len(flanges)}")
                if cmd == "q":
                    return 0
                continue
            q_a = np.asarray(client.read_q())
            time.sleep(0.3)
            q_b = np.asarray(client.read_q())
            if np.abs(q_b - q_a).max() > 2e-3:
                print("  arm moved — let go and hold still, then Enter")
                continue
            flanges.append(np.asarray(client.flange_pose_of(q_b), dtype=np.float64).reshape(4, 4))
            qs.append(q_b)
            _save(path, flanges, qs, None)
            if len(flanges) >= 3:
                partial = solve_pivot(flanges)
                print(
                    f"  touch {len(flanges)}: residual so far {partial['rms_mm']:.2f} mm, "
                    f"this touch {partial['residuals_mm'][-1]:.2f} mm, tilt "
                    + " / ".join(f"{a:.0f}" for a in partial["tilt_deg"])
                    + " deg"
                )
            else:
                print(f"  touch {len(flanges)} recorded")
    finally:
        client.close()


def main() -> None:
    """Console entry point: ``z3-gap-tcp``."""
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = tyro.cli(TcpConfig)
    if cfg.from_file is not None:
        data = json.loads(cfg.from_file.read_text())
        raise SystemExit(
            report(cfg, solve_pivot([np.asarray(f) for f in data["base_T_flange"]]))[0]
        )
    raise SystemExit(capture(cfg))
