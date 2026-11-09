"""Plan the sun gear's insertion into the seated rotor (pure computation, no camera).

The rotor's top is taken as the housing's top face from the gearbox detection: a seated
rotor is flush with it (2026-11-09: plate 32.9 mm, housing 32.9 mm in the same frame), and
the rotor insertion only reports success once the rotor stays seated with the push
released. All heights below are fingertip-end heights unless named TCP; the tool point is
``tip_below_tcp_m`` above the fingertip ends.

1. insert (held): from ``pre`` (the sun's bottom ``start_gap_m`` above the rotor's top)
   down to ``end`` (the fingertips ``clearance_m`` above the rotor's top — the jaws can
   not follow the sun into the rotor), twisting while the teeth mesh with the planets;
2. release, lift to ``release_lift`` (fingertips ``lift_m`` above where they let go),
   close the gripper;
3. press: closed fingertips from ``press_start`` (``press_gap_m`` above the sun's top)
   down to the rotor's top (the sun flush), with a small wiggle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotor_vision import deproject, project, sample  # noqa: E402


class Output(TypedDict):
    rotor_top_z: float
    insert_stall_ok_m: float
    press_stall_ok_m: float
    above_rotor: Se3Pose
    pre: Se3Pose
    insert_depth_m: float
    release_lift: Se3Pose
    press_start: Se3Pose
    press_depth_m: float


def vertical(x: float, y: float, z: float) -> Se3Pose:
    return {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}


def surface_z(depth, K, T_wc, center_xy, r0: float, r1: float, z_guess: float = 0.035) -> tuple[float, float]:
    """(median, p90 - p10) of the depth-measured height [m] in the annulus r0..r1 around the axis."""
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    rr = np.linspace(r0, r1, 5)
    R, TH = np.meshgrid(rr, th)
    P = np.stack([center_xy[0] + R.ravel() * np.cos(TH.ravel()),
                  center_xy[1] + R.ravel() * np.sin(TH.ravel()), np.full(R.size, z_guess)], axis=1)
    uv, _ = project(T_wc, K, P)
    d = sample(depth, uv, nearest=True)
    ok = np.isfinite(d) & (d > 0.05)
    if ok.sum() < 100:
        raise RuntimeError(f"no depth in the annulus r {r0 * 1e3:.1f}-{r1 * 1e3:.1f} mm")
    z = deproject(T_wc, K, uv[ok], d[ok])[:, 2]
    return float(np.median(z)), float(np.percentile(z, 90) - np.percentile(z, 10))


def run(
    ctx: NodeContext,
    center_xy: list[float],
    housing_top_z: float,
    sun_height_m: float,
    rotor_above_housing_m: float = 0.0,
    grip_m: float = 0.004,
    clearance_m: float = 0.002,
    start_gap_m: float = 0.008,
    approach_m: float = 0.02,
    lift_m: float = 0.012,
    press_gap_m: float = 0.008,
    tip_below_tcp_m: float = 0.060,
    min_inside_m: float = 0.004,
) -> Output:
    top = float(housing_top_z) + rotor_above_housing_m
    if not 0.010 < sun_height_m < 0.030:
        raise RuntimeError(f"sun height {sun_height_m * 1e3:.1f} mm is implausible (17.5-21 expected)")
    x, y = center_xy
    tcp = lambda tip: tip + tip_below_tcp_m  # noqa: E731
    tip_end = top + clearance_m  # fingertips when the held insertion ends
    tip_pre = top + start_gap_m + sun_height_m - grip_m
    sun_top_after = tip_end + grip_m  # where the sun's top is when released
    tip_press_start = sun_top_after + press_gap_m
    print(f"[plan_sun_insert] rotor top z {top:.4f} (housing {housing_top_z:.4f}), sun "
          f"{sun_height_m * 1e3:.1f} mm: held move {(tip_pre - tip_end) * 1e3:.1f} mm, press "
          f"{(tip_press_start - top) * 1e3:.1f} mm")
    # Where the sun really stops is not known exactly (2026-11-09: in the bore, higher than
    # the CAD model says). A stall counts as inserted once the sun's bottom is at least
    # min_inside_m into the rotor; the press accepts any stall (it only pushes it home).
    tip_min_inside = top - min_inside_m + sun_height_m - grip_m
    return {
        "rotor_top_z": top,
        "insert_stall_ok_m": float(max(tip_min_inside - tip_end, 0.0)),
        "press_stall_ok_m": float(tip_press_start - top),
        "above_rotor": vertical(x, y, tcp(tip_pre + approach_m)),
        "pre": vertical(x, y, tcp(tip_pre)),
        "insert_depth_m": float(tip_pre - tip_end),
        "release_lift": vertical(x, y, tcp(tip_end + lift_m)),
        "press_start": vertical(x, y, tcp(tip_press_start)),
        "press_depth_m": float(tip_press_start - top),  # down to flush with the rotor's top
    }
