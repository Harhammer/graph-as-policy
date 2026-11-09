"""Kinematic check of the arm itself, with no camera involved: ball-in-cone closure.

Hold a ball in the gripper (close on it once and do not open the gripper again during
the session) and hand-guide it into a fixed cone on the table from many different arm
configurations. The ball's centre is then at the same point every time, whatever the
robot's joint readings say. Fitting joint-angle offsets (plus the ball's position in the
flange frame and the cone's position) until the model agrees measures how far the
robot's own kinematics is off — the one thing a camera-based hand-eye calibration cannot
separate from the camera. The fitted ball offset is also the real tool point.

    z3-gap-kincal parts                    # the cone cup and the ball to print (STL)
    z3-gap-kincal capture                  # Enter records a pose; c = next cone; Ctrl-C ends
    z3-gap-kincal solve --files FILE [FILE ...]   # offsets, ball offset, residuals, uncertainty

Identifiability (what this test can and cannot see):

- joints 2-6: identifiable when the configurations vary widely (elbow left/right and
  high/low, wrist tilted both ways, the last joint turned by +-90 deg or more);
- joint 1: never — its offset only rotates the cone position about the base, which is a
  free parameter anyway (and it does not matter for camera-relative work either);
- joint 7: never — its offset only rotates the ball offset about the flange axis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from z3_gap_bridge.panda_fk import flange_pose

log = logging.getLogger(__name__)

DEFAULT_JOINTS = (1, 2, 3, 4, 5)  # 0-indexed: joints 2..6
STILL_RAD = 2e-3


# -- capture --------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Record ball-in-cone poses by hand-guiding (reads joints only, commands nothing)."""

    out: Path | None = None
    """Output JSON. Default ~/z3_kincal/<timestamp>.json."""
    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    robot_ip: str = "192.168.98.121"


def capture(cfg: CaptureConfig) -> int:
    from z3_gap_bridge.franky_client import FrankyClient

    out = cfg.out or Path.home() / "z3_kincal" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    client = FrankyClient(cfg.nuc_host, cfg.service_port, stale_threshold_s=0.1, stale_run_s=0.5)
    client.connect_robot(cfg.robot_ip)
    samples: list[dict[str, Any]] = []
    cone = "A"
    print(
        "Ball in the gripper (closed once, stays closed). Seat it in the cone, let go of\n"
        "the arm, then press Enter. 'c' + Enter = the cone was moved (next cone label),\n"
        "'r' + Enter = repeat of the previous configuration (seating-noise check).\n"
        "Vary the configuration a lot between samples: elbow left/right and high/low,\n"
        "wrist tilted forward/back and sideways, last joint turned +-90 deg or more.\n"
        "Ctrl-C ends. 15-25 samples per cone position; two cone positions are better."
    )
    try:
        while True:
            answer = input(f"[cone {cone}, {len(samples)} recorded] > ").strip().lower()
            if answer == "c":
                cone = chr(ord(cone) + 1)
                print(f"cone label now {cone}: move the cone, then record as before")
                continue
            q1 = np.asarray(client.read_q())
            time.sleep(0.3)
            q2 = np.asarray(client.read_q())
            if np.abs(q2 - q1).max() > STILL_RAD:
                print("  the arm is still moving — let go and press Enter again")
                continue
            samples.append({
                "q": q2.tolist(), "cone": cone, "repeat": answer == "r", "t": time.time(),
            })
            out.write_text(json.dumps({"samples": samples}, indent=1))
            tip = flange_pose(q2) @ np.array([0.0, 0.0, 0.2, 1.0])
            print(f"  recorded #{len(samples)}  (flange+0.2 m at {np.round(tip[:3], 3)})")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.close()
    print(f"\n{len(samples)} samples -> {out}\nsolve with:  z3-gap-kincal solve --files {out}")
    return 0


# -- solve ----------------------------------------------------------------------------


def load(paths: list[Path]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(q (N,7), cone label per sample, repeat flag per sample) from capture files.
    Cone labels are made unique per file (A in file 1 is not A in file 2)."""
    qs, cones, repeats = [], [], []
    for i, p in enumerate(paths):
        for s in json.loads(Path(p).read_text())["samples"]:
            qs.append(s["q"])
            cones.append(f"{i}:{s['cone']}")
            repeats.append(bool(s.get("repeat", False)))
    return np.asarray(qs, float), cones, np.asarray(repeats, bool)


def _ball_positions(q: np.ndarray, dq: np.ndarray, ball: np.ndarray) -> np.ndarray:
    b = np.array([*ball, 1.0])
    return np.array([(flange_pose(qi + dq) @ b)[:3] for qi in q])


def solve(
    q: np.ndarray,
    cones: list[str],
    joints: tuple[int, ...] = DEFAULT_JOINTS,
    ball0: tuple[float, float, float] = (0.0, 0.0, 0.2),
) -> dict[str, Any]:
    """Fit joint offsets (``joints``, 0-indexed), the ball offset in the flange frame and one
    centre per cone label so that every sample's ball lands on its cone's centre.

    Returns the offsets [rad], the ball offset [m], per-sample residuals [m] and, for
    comparison, the residuals of the nominal model (ball + cones fitted, offsets 0).
    """
    from scipy.optimize import least_squares

    labels = sorted(set(cones))
    idx = np.array([labels.index(c) for c in cones])
    joints = tuple(int(j) for j in joints)
    for j in joints:
        if j in (0, 6):
            raise ValueError("joint 1 and joint 7 offsets are not identifiable by this test")

    def unpack(p):
        dq = np.zeros(7)
        for k, j in enumerate(joints):
            dq[j] = p[k]
        n = len(joints)
        ball = p[n:n + 3]
        centres = p[n + 3:].reshape(len(labels), 3)
        return dq, ball, centres

    def residuals(p, fit_joints: bool):
        dq, ball, centres = unpack(p)
        if not fit_joints:
            dq = np.zeros(7)
        return (_ball_positions(q, dq, ball) - centres[idx]).ravel()

    def initial():
        pos0 = _ball_positions(q, np.zeros(7), np.asarray(ball0))
        centres0 = np.array([pos0[idx == k].mean(axis=0) for k in range(len(labels))])
        return np.concatenate([np.zeros(len(joints)), np.asarray(ball0, float), centres0.ravel()])

    nominal = least_squares(residuals, initial(), args=(False,), x_scale="jac")
    fitted = least_squares(residuals, nominal.x, args=(True,), x_scale="jac")
    dq, ball, centres = unpack(fitted.x)
    res = fitted.fun.reshape(-1, 3)
    res_nom = nominal.fun.reshape(-1, 3)

    # Jackknife standard error of each offset (leave one sample out).
    loo = []
    for k in range(len(q)):
        keep = np.arange(len(q)) != k
        if min(np.bincount(idx[keep], minlength=len(labels))) < 2:
            continue
        sub = least_squares(
            lambda p: (_ball_positions(q[keep], unpack(p)[0], unpack(p)[1])
                       - unpack(p)[2][idx[keep]]).ravel(),
            fitted.x, x_scale="jac",
        )
        loo.append(unpack(sub.x)[0][list(joints)])
    loo = np.asarray(loo)
    n = len(loo)
    se = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0)) if n > 1 else np.full(len(joints), np.nan)

    # Conditioning: a parameter whose column of the Jacobian is nearly dependent on the
    # others is not determined by these configurations.
    jac = fitted.jac
    sv = np.linalg.svd(jac, compute_uv=False)
    return {
        "joints": joints,
        "offsets_rad": dq[list(joints)],
        "offsets_se_rad": se,
        "ball_offset_m": ball,
        "cone_centres_m": {lab: centres[k] for k, lab in enumerate(labels)},
        "residual_mm": np.linalg.norm(res, axis=1) * 1e3,
        "nominal_residual_mm": np.linalg.norm(res_nom, axis=1) * 1e3,
        "condition": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "samples": len(q),
    }


@dataclass
class SolveConfig:
    """Fit joint offsets from ball-in-cone capture files."""

    files: tuple[Path, ...] = field(default_factory=tuple)
    joints: tuple[int, ...] = (2, 3, 4, 5, 6)
    """Joints to fit, 1-indexed (Franka A1..A7). 1 and 7 cannot be fitted by this test."""
    ball_z0_m: float = 0.2
    """Rough distance of the ball centre from the flange along the flange z axis, to start."""


def report(result: dict[str, Any], cones: list[str], repeats: np.ndarray) -> str:
    lines = [f"{result['samples']} samples, {len(result['cone_centres_m'])} cone position(s)"]
    nom, fit = result["nominal_residual_mm"], result["residual_mm"]
    lines.append(
        f"ball-centre scatter: nominal model {np.sqrt(np.mean(nom**2)):.2f} mm rms / "
        f"{nom.max():.2f} max  ->  with offsets {np.sqrt(np.mean(fit**2)):.2f} mm rms / "
        f"{fit.max():.2f} max"
    )
    lines.append("joint offsets (add to the reported angles), with leave-one-out uncertainty:")
    for j, v, s in zip(result["joints"], result["offsets_rad"], result["offsets_se_rad"], strict=True):
        flag = "   <- not determined by these configurations" if (np.isnan(s) or s > 0.3 * abs(v) + np.radians(0.1)) else ""
        lines.append(f"  joint {j + 1}: {np.degrees(v):+.3f} deg  (+- {np.degrees(s):.3f}){flag}")
    b = result["ball_offset_m"]
    lines.append(
        f"ball centre in the flange frame: x {b[0]*1e3:+.1f} y {b[1]*1e3:+.1f} z {b[2]*1e3:+.1f} mm "
        "(= the real tool point for this grasp)"
    )
    if repeats.any():
        rep_idx = np.flatnonzero(repeats)
        pairs = [(k - 1, k) for k in rep_idx if k > 0 and cones[k] == cones[k - 1]]
        if pairs:
            d = [abs(fit[a] - fit[b_]) for a, b_ in pairs]
            lines.append(f"seating repeatability ({len(pairs)} repeats): {np.mean(d):.2f} mm mean difference")
    worst = np.argsort(fit)[::-1][:3]
    lines.append("largest residuals: " + ", ".join(f"#{k + 1} {fit[k]:.2f} mm" for k in worst))
    if result["condition"] > 1e4:
        lines.append(f"WARNING: poorly conditioned fit (condition {result['condition']:.0f}) — vary the configurations more")
    return "\n".join(lines)


def solve_files(cfg: SolveConfig) -> int:
    if not cfg.files:
        raise SystemExit("solve needs at least one capture file")
    q, cones, repeats = load(list(cfg.files))
    if len(q) < 8:
        raise SystemExit(f"only {len(q)} samples; record at least 8 (15-25 per cone is good)")
    result = solve(q, cones, tuple(j - 1 for j in cfg.joints), (0.0, 0.0, cfg.ball_z0_m))
    print(report(result, cones, repeats))
    out = Path(cfg.files[0]).with_suffix(".fit.json")
    out.write_text(json.dumps({
        "files": [str(f) for f in cfg.files],
        "joints_1indexed": [j + 1 for j in result["joints"]],
        "joint_offsets_deg": np.degrees(result["offsets_rad"]).tolist(),
        "joint_offsets_se_deg": np.degrees(result["offsets_se_rad"]).tolist(),
        "ball_offset_m": result["ball_offset_m"].tolist(),
        "residual_rms_mm": float(np.sqrt(np.mean(result["residual_mm"] ** 2))),
        "nominal_residual_rms_mm": float(np.sqrt(np.mean(result["nominal_residual_mm"] ** 2))),
    }, indent=1))
    print(f"wrote {out}")
    return 0


def main() -> None:
    """Console entry point: ``z3-gap-kincal``."""
    from typing import Annotated

    import tyro

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(
        Annotated[CaptureConfig, tyro.conf.subcommand("capture")]
        | Annotated[SolveConfig, tyro.conf.subcommand("solve")]
        | Annotated[PartsConfig, tyro.conf.subcommand("parts")]
    )
    if isinstance(cmd, CaptureConfig):
        raise SystemExit(capture(cmd))
    if isinstance(cmd, PartsConfig):
        raise SystemExit(write_parts(cmd))
    raise SystemExit(solve_files(cmd))


# -- printable parts --------------------------------------------------------------------


def _revolve_stl(profile_rz: list[tuple[float, float]], segments: int = 96) -> str:
    """ASCII STL of the solid of revolution of a closed (r, z) polygon [mm] about z.

    Consecutive profile points are joined around the axis; a point on the axis (r = 0)
    makes the strip a fan. Watertight as long as the polygon is simple and closed.
    """
    import math

    pts = profile_rz + [profile_rz[0]]
    angles = [2 * math.pi * k / segments for k in range(segments + 1)]
    tris: list[tuple] = []

    def P(r, z, a):
        return (r * math.cos(a), r * math.sin(a), z)

    for (r0, z0), (r1, z1) in zip(pts[:-1], pts[1:], strict=True):
        for a0, a1 in zip(angles[:-1], angles[1:], strict=True):
            quad = (P(r0, z0, a0), P(r1, z1, a0), P(r1, z1, a1), P(r0, z0, a1))
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                a, b, c = (np.asarray(v) for v in tri)
                if np.linalg.norm(np.cross(b - a, c - a)) < 1e-9:
                    continue  # degenerate (on the axis)
                tris.append(tri)
    lines = ["solid part"]
    for a, b, c in tris:
        lines.append("  facet normal 0 0 0\n    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
        lines.append("    endloop\n  endfacet")
    lines.append("endsolid part")
    return "\n".join(lines) + "\n"


def cone_cup_profile(ball_d: float = 20.0) -> list[tuple[float, float]]:
    """A cup with a 90-degree conical seat for a ball of ``ball_d`` mm and a flat base flange
    to tape it down. Outer: flange radius 30, body radius 22, 18 mm tall; seat opening
    radius 16, apex on the axis 16 mm deep (a 20 mm ball's centre sits ~2 mm below the
    rim, the stem clear of it). [mm]"""
    r_open, depth = 16.0, 16.0
    return [
        (0.0, 0.0), (30.0, 0.0), (30.0, 3.0), (22.0, 3.0), (22.0, 18.0),
        (r_open, 18.0), (0.0, 18.0 - depth),
    ]


def ball_stem_profile(ball_d: float = 20.0, stem_d: float = 12.0, stem_len: float = 30.0):
    """A ball on a cylindrical stem for the jaws to hold. The ball's centre is what the test
    measures, and turning about the stem axis does not move it. Printed stem down. [mm]"""
    import math

    r_ball, r_stem = ball_d / 2, stem_d / 2
    z_c = stem_len + r_ball * 0.85  # ball centre: the stem enters the ball below its equator
    a_join = math.asin(min(r_stem / r_ball, 1.0))
    prof = [(0.0, 0.0), (r_stem, 0.0), (r_stem, z_c - r_ball * math.cos(a_join))]
    for k in range(1, 40):
        a = a_join + (math.pi - a_join) * k / 40
        prof.append((r_ball * math.sin(a), z_c - r_ball * math.cos(a)))
    prof.append((0.0, z_c + r_ball))
    return prof


def socket_stem_profile(ball_d: float = 20.0, stem_d: float = 12.0, stem_len: float = 30.0):
    """A stem with a spherical socket to glue a bought ball into (steel bearing ball or glass
    marble: far rounder than a print, and roundness is the one thing this test needs).
    The socket holds the ball to 40 % of its diameter; the collar around it is 1.8 mm thick.
    Printed stem down. [mm]"""
    import math

    r_ball, r_stem = ball_d / 2, stem_d / 2
    collar_h = 0.4 * ball_d + 2.0
    z_top = stem_len + collar_h
    z_c = z_top + 0.1 * ball_d  # ball centre 10 % of its diameter above the collar top
    r_open = math.sqrt(max(r_ball**2 - (z_c - z_top) ** 2, 0.0))
    r_collar = r_open + 1.8
    prof = [(0.0, 0.0), (r_stem, 0.0), (r_stem, stem_len), (r_collar, stem_len),
            (r_collar, z_top), (r_open, z_top)]
    a_top = math.asin(min(r_open / r_ball, 1.0))  # polar angle of the rim, from the bottom pole
    for k in range(1, 30):
        a = a_top * (1.0 - k / 30.0)  # down the inside of the socket to its bottom pole
        prof.append((r_ball * math.sin(a), z_c - r_ball * math.cos(a)))
    prof.append((0.0, z_c - r_ball))
    return prof


@dataclass
class PartsConfig:
    """Write the printable parts for the ball-in-cone test (ASCII STL, mm)."""

    out: Path = Path.home() / "z3_kincal" / "parts"
    ball_d: float = 20.0
    """Ball diameter: of the ball you glue in (socket stem) or of the printed ball."""


def write_parts(cfg: PartsConfig) -> int:
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "kincal_cone_cup.stl").write_text(_revolve_stl(cone_cup_profile(cfg.ball_d)))
    (cfg.out / "kincal_ball_stem.stl").write_text(_revolve_stl(ball_stem_profile(cfg.ball_d)))
    (cfg.out / "kincal_socket_stem.stl").write_text(_revolve_stl(socket_stem_profile(cfg.ball_d)))
    print(
        f"wrote to {cfg.out}:\n"
        "  kincal_cone_cup.stl     the seat; print as is, tape its flange to the table\n"
        f"  kincal_socket_stem.stl  PREFERRED: glue a {cfg.ball_d:.0f} mm steel ball or marble "
        "into it (stem down, no supports)\n"
        "  kincal_ball_stem.stl    if you have no ball: printed ball on a stem (stem down, no "
        "supports, 0.1-0.15 mm layers, sand the seam)"
    )
    return 0
