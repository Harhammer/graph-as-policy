"""Find the gearbox's axis in one wrist RGB-D frame.

The housing (the Printables NEMA17 planetary gearbox scaled to 133 %, M4 screws) seen from
above: a white 56 mm square in a black fixture, 4 corner holes on a 41.0 mm square
(measured on the print; model 41.23), and the ring gear (38 teeth, tip circle 38.3 mm)
starting 2.9 mm below the top face. The rotor goes in along the ring gear's axis.

As for the rotor (rotor_vision.py), everything is measured on the plane of the top face
in the robot base frame, so the camera's tilt never distorts the pattern:

1. top-face height from depth (the highest surface near the expected position);
2. the image resampled onto that plane;
3. the 4 corner holes: blobs darker than their surroundings, and of them the 4 that form
   a 41.0 mm square (the tooth ring has dark gaps too; the pattern rejects them) ->
   centre and rotation. This is the result;
4. cross-check: the 38-tooth pattern of the ring, searched around that centre on the
   plane of the ring's top, must be centred within ``ring_tol_m`` of it. Two
   independent features agreeing; otherwise the detection fails.

The dark bore is never used: seen at an angle, perspective shifts it by millimetres.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from rotor_vision import _gray, _highest_surface_z, RotorGeometry, project, sample


@dataclass(frozen=True)
class GearboxGeometry:
    housing_m: float = 0.0563  # outer square
    hole_square_m: float = 0.0410  # corner-hole spacing, measured on the print (model 41.23)
    hole_d_m: float = 0.004  # M4
    ring_teeth: int = 38
    ring_probe_r_m: float = 0.0205  # between the tooth tips (19.15) and roots (21.55)
    ring_below_top_m: float = 0.0029  # the recess above the ring gear


@dataclass(frozen=True)
class GearboxConfig:
    window_m: float = 0.045  # half-size of the search window around the nominal position
    res_m: float = 0.00015
    square_tol_m: float = 0.0015  # worst side/diagonal error of the 4-hole square
    ring_tol_m: float = 0.001  # ring centre vs hole centre
    ring_search_m: float = 0.002
    max_candidates: int = 40


@dataclass
class GearboxDetection:
    center_xy: np.ndarray  # gearbox axis, base frame [m]
    z_top: float  # top-face height [m]
    rotation_deg: float  # housing rotation about z, (-45, 45]
    holes_xy: np.ndarray  # (4, 2)
    square_error_m: float
    ring_offset_m: float  # ring-pattern centre vs hole centre
    ring_strength: float
    warnings: list[str] = field(default_factory=list)


class GearboxNotFound(RuntimeError):
    """No 4-hole square / no tooth ring where the gearbox was expected."""


def detect_gearbox(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    K: np.ndarray,
    T_wc: np.ndarray,
    nominal_xy: tuple[float, float],
    z_top_guess: float,
    geom: GearboxGeometry = GearboxGeometry(),
    cfg: GearboxConfig = GearboxConfig(),
    trace: dict | None = None,
) -> GearboxDetection:
    trace = {} if trace is None else trace
    K = np.asarray(K, dtype=np.float64)
    T_wc = np.asarray(T_wc, dtype=np.float64)
    nominal = np.asarray(nominal_xy, dtype=np.float64)
    gray = _gray(rgb)
    warnings: list[str] = []

    z = float(z_top_guess)
    if depth is not None:
        z_d = _highest_surface_z(np.asarray(depth, np.float32), K, T_wc, nominal, z,
                                 RotorGeometry(outer_d_m=geom.housing_m))
        if z_d is None:
            warnings.append("no usable depth on the gearbox; using the configured top height")
        else:
            if abs(z_d - z) > 0.01:
                warnings.append(f"top face from depth {z_d:.4f} vs configured {z:.4f} (>1 cm apart)")
            z = z_d
    trace.update(z_top=z, nominal_xy=nominal)

    # 2. top view
    ax = np.arange(-cfg.window_m, cfg.window_m + 1e-12, cfg.res_m)
    X, Y = np.meshgrid(nominal[0] + ax, nominal[1] + ax)  # row <-> y, col <-> x
    uv, _ = project(T_wc, K, np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1))
    top = sample(gray, uv).reshape(X.shape)
    if np.isnan(top).mean() > 0.3:
        raise GearboxNotFound("the search window is mostly outside the image")
    g = cv2.GaussianBlur(np.nan_to_num(top, nan=float(np.nanmedian(top))).astype(np.float32), (3, 3), 0)
    trace.update(top=g, top_axis=ax)

    def to_xy(col: float, row: float) -> np.ndarray:
        return np.array([nominal[0] + ax[0] + col * cfg.res_m, nominal[1] + ax[0] + row * cfg.res_m])

    # 3. corner holes: locally dark, hole-sized; then the 4 that make the square
    lo, hi = np.percentile(g, [40, 97])
    k = int(round(0.007 / cfg.res_m)) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contrast = (bg - g) / max(hi - lo, 1.0)
    hole_area = math.pi * (geom.hole_d_m / 2 / cfg.res_m) ** 2
    expected_r = geom.hole_square_m / math.sqrt(2)
    side, diag = geom.hole_square_m, geom.hole_square_m * math.sqrt(2)
    best = None
    for thr in (0.25, 0.15, 0.10):
        nl, _labels, stats, cents = cv2.connectedComponentsWithStats((contrast > thr).astype(np.uint8))
        cands = [
            (to_xy(*cents[i]), stats[i, cv2.CC_STAT_AREA]) for i in range(1, nl)
            if 0.15 * hole_area < stats[i, cv2.CC_STAT_AREA] < 3.0 * hole_area
        ]
        # corner holes are ~29 mm from the axis, which is within the window of the nominal
        cands = [c for c, _ in cands if abs(np.hypot(*(c - nominal)) - expected_r) < cfg.window_m * 0.6]
        cands = sorted(cands, key=lambda c: abs(np.hypot(*(c - nominal)) - expected_r))[: cfg.max_candidates]
        for combo in itertools.combinations(range(len(cands)), 4):
            P = np.array([cands[i] for i in combo])
            d = np.sort([np.hypot(*(P[a] - P[b])) for a in range(4) for b in range(a + 1, 4)])
            err = max(np.abs(d[:4] - side).max(), np.abs(d[4:] - diag).max())
            if best is None or err < best[0]:
                best = (err, P)
        if best is not None and best[0] < cfg.square_tol_m:
            break
    trace["contrast"] = contrast
    if best is None or best[0] >= cfg.square_tol_m:
        err = "none" if best is None else f"{best[0] * 1e3:.1f} mm off"
        raise GearboxNotFound(f"no 4 corner holes on a {side * 1e3:.1f} mm square (best: {err})")
    sq_err, holes = best
    center = holes.mean(axis=0)
    rel = holes - center
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    rot = (ang - 45.0 + 45.0) % 90.0 - 45.0  # each corner's deviation from the 45-deg diagonal
    rotation = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(rot * 4))))) / 4)
    trace.update(holes=holes, hole_center=center)

    # 4. cross-check with the tooth ring, on the plane of the ring's top
    z_ring = z - geom.ring_below_top_m
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    step = 0.00025
    offs = np.arange(-cfg.ring_search_m, cfg.ring_search_m + 1e-12, step)
    best_c, best_s, strengths = center, -1.0, []
    for dx in offs:
        for dy in offs:
            c = center + [dx, dy]
            ring = np.stack([c[0] + geom.ring_probe_r_m * np.cos(th),
                             c[1] + geom.ring_probe_r_m * np.sin(th), np.full_like(th, z_ring)], axis=1)
            uvr, _ = project(T_wc, K, ring)
            v = sample(gray, uvr)
            if np.isnan(v).any():
                continue
            spec = np.abs(np.fft.rfft(v - v.mean())) / len(v)
            s = float(spec[geom.ring_teeth])
            strengths.append(s)
            if s > best_s:
                best_s, best_c = s, c
    ring_off = float(np.hypot(*(best_c - center)))
    trace.update(ring_center=best_c)
    if best_s <= 0:
        raise GearboxNotFound("the tooth ring is outside the image")
    if ring_off > cfg.ring_tol_m:
        raise GearboxNotFound(
            f"tooth ring centred {ring_off * 1e3:.1f} mm from the corner holes' centre "
            f"(> {cfg.ring_tol_m * 1e3:.1f} mm): not this gearbox, or a hole was misread"
        )
    return GearboxDetection(
        center_xy=center, z_top=z, rotation_deg=rotation, holes_xy=holes,
        square_error_m=float(sq_err), ring_offset_m=ring_off, ring_strength=best_s,
        warnings=warnings,
    )


def draw_gearbox_popup(
    rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    det: GearboxDetection | None,
    nominal_xy,
    geom: GearboxGeometry = GearboxGeometry(),
    z_top: float | None = None,
    error: str = "",
    scale: int = 2,
) -> np.ndarray:
    """Quick-look picture: holes (red), the ring's tip/root circles (orange/green) around
    the detected axis (cyan +), the taught position (magenta x), the offset."""
    nominal = np.asarray(nominal_xy, float)
    z = det.z_top if det is not None else (z_top if z_top is not None else 0.03)
    center = det.center_xy if det is not None else nominal
    uv_c, dist = project(T_wc, K, np.array([[*center, z]]))
    half = int(np.clip(0.75 * geom.housing_m * K[0, 0] / max(dist[0], 1e-3), 60, 220))
    h, w = rgb.shape[:2]
    u0 = int(np.clip(round(uv_c[0, 0]) - half, 0, max(w - 2 * half, 0)))
    v0 = int(np.clip(round(uv_c[0, 1]) - half, 0, max(h - 2 * half, 0)))
    img = cv2.resize(np.ascontiguousarray(rgb[v0:v0 + 2 * half, u0:u0 + 2 * half], dtype=np.uint8),
                     None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def px(points):
        uv, _ = project(T_wc, K, np.atleast_2d(points))
        return [(int(round((a - u0) * scale)), int(round((b - v0) * scale))) for a, b in uv]

    def marker(p, color, kind):
        cv2.drawMarker(img, p, (0, 0, 0), kind, 26, 5, cv2.LINE_AA)
        cv2.drawMarker(img, p, color, kind, 26, 2, cv2.LINE_AA)

    lines = []
    if det is not None:
        th = np.linspace(0, 2 * np.pi, 120)
        zr = z - geom.ring_below_top_m
        for r, col in ((0.01915, (255, 160, 0)), (0.02155, (40, 220, 40))):
            ring = px(np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th), np.full_like(th, zr)], 1))
            cv2.polylines(img, [np.array(ring, np.int32)], True, col, 2, cv2.LINE_AA)
        hole_r = max(4, int(round(geom.hole_d_m / 2 * K[0, 0] / dist[0] * scale)) + 4)
        for p in px(np.c_[det.holes_xy, np.full(4, z)]):
            cv2.circle(img, p, hole_r, (255, 40, 40), 2, cv2.LINE_AA)
        marker(px([*center, z])[0], (0, 230, 255), cv2.MARKER_CROSS)
        off = (center - nominal) * 1e3
        lines += [
            f"axis {np.hypot(*off):.1f} mm from taught (dx {off[0]:+.1f}, dy {off[1]:+.1f})",
            f"hole square err {det.square_error_m * 1e3:.2f} mm, rot {det.rotation_deg:+.1f} deg",
            f"tooth ring vs holes {det.ring_offset_m * 1e3:.2f} mm, top z {det.z_top * 1e3:.1f} mm",
        ]
    pn = px([*nominal, z])[0]
    marker(pn, (255, 0, 255), cv2.MARKER_TILTED_CROSS)
    cv2.putText(img, "taught", (pn[0] + 14, pn[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "taught", (pn[0] + 14, pn[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    if error:
        lines = [f"NOT FOUND: {error}"]
    bar = np.full((26 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 90, 90) if error else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])
