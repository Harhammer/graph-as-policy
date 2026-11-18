"""Find the sun gear standing on the table (no fixture) in one wrist RGB-D frame.

The sun gear: a small white cylinder (Ø17.6 mm hub / Ø17.0 mm teeth, ~16-21 mm tall) with
a bore, standing upright on the light wooden table. The wood is nearly as bright as the
part, so depth finds it first and colour only refines it:

1. height map around the expected position (depth deprojected per cell); the table is
   the low level, the part everything > ``min_height_m`` above it. The blob of the part's
   size nearest the expected position gives a rough centre, the top height (median of
   its top) and the part's height above the table;
2. the image resampled onto the plane of the top, around the rough centre; the bright
   top face's outline (halfway between the face's and the surrounding table's
   brightness), fitted as a circle -> the centre, checked against the expected radius;
3. cross-check: the dark bore's centroid must be near the circle's centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from rotor_vision import _fit_circle, _gray, deproject, project, sample


@dataclass(frozen=True)
class SunGeometry:
    outer_d_m: float = 0.0176  # hub (the teeth are Ø17.0)
    bore_d_m: float = 0.006


@dataclass(frozen=True)
class SunConfig:
    window_m: float = 0.035  # half-size of the search window
    height_res_m: float = 0.0005
    min_height_m: float = 0.006  # above the table = part
    top_res_m: float = 0.0001
    radius_tol: float = 0.15
    bore_tol_m: float = 0.003  # the dark of the bore leans toward the camera too (~2 mm seen)
    # Known-radius fit ignoring the side wall seen from the look pose: geometrically cleaner, but
    # with it the sun arrived ~1 mm off the far rim of the bore (2026-11-09 17:24), with the free
    # fit it went in (16:32, 16:44) — something else in the chain offsets the wall bias. Off.
    fixed_radius_fit: bool = False
    keep_fraction: float = 0.6  # outline points used by the known-radius fit (the rest: side wall)


@dataclass
class SunDetection:
    center_xy: np.ndarray
    z_top: float  # top face [m]
    z_table: float  # table around it [m]
    radius_m: float
    bore_offset_m: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def height_m(self) -> float:
        return self.z_top - self.z_table


class SunNotFound(RuntimeError):
    """No standing part of the sun gear's size near the expected position."""


def _height_map(depth, K, T_wc, center, half, res, z_plane):
    ax = np.arange(-half, half + 1e-12, res)
    X, Y = np.meshgrid(center[0] + ax, center[1] + ax)
    uv, _ = project(T_wc, K, np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_plane)], axis=1))
    d = sample(depth, uv, nearest=True)
    ok = np.isfinite(d) & (d > 0.03)
    h = np.full(len(d), np.nan)
    W = deproject(T_wc, K, uv[ok], d[ok])
    h[ok] = W[:, 2]
    return ax, h.reshape(X.shape)


def detect_sun(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    nominal_xy: tuple[float, float],
    geom: SunGeometry = SunGeometry(),
    cfg: SunConfig = SunConfig(),
    trace: dict | None = None,
) -> SunDetection:
    trace = {} if trace is None else trace
    K = np.asarray(K, float)
    T_wc = np.asarray(T_wc, float)
    depth = np.asarray(depth, np.float32)
    nominal = np.asarray(nominal_xy, float)
    warnings: list[str] = []

    # 1. depth: table level, the standing part
    ax, H = _height_map(depth, K, T_wc, nominal, cfg.window_m, cfg.height_res_m, 0.01)
    if np.isnan(H).mean() > 0.5:
        raise SunNotFound("little depth in the search window")
    z_table = float(np.nanpercentile(H, 30))
    above = np.nan_to_num(H - z_table, nan=0.0) > cfg.min_height_m
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nl, lab, stats, cents = cv2.connectedComponentsWithStats(above)
    area_px = math.pi * (geom.outer_d_m / 2 / cfg.height_res_m) ** 2
    best, best_cost = None, math.inf
    for k in range(1, nl):
        a = stats[k, cv2.CC_STAT_AREA]
        if not 0.35 * area_px < a < 2.2 * area_px:
            continue
        c = np.array([nominal[0] + ax[0] + cents[k][0] * cfg.height_res_m,
                      nominal[1] + ax[0] + cents[k][1] * cfg.height_res_m])
        cost = np.hypot(*(c - nominal)) + 0.01 * abs(a / area_px - 1)
        if cost < best_cost:
            best, best_cost = (k, c), cost
    trace.update(height_axis=ax, height_map=H, z_table=z_table, nominal_xy=nominal)
    if best is None:
        raise SunNotFound(
            f"no standing part of D{geom.outer_d_m * 1e3:.1f} mm in the depth around "
            f"({nominal[0]:.3f}, {nominal[1]:.3f})"
        )
    k, c0 = best
    tops = H[lab == k]
    z_top = float(np.nanpercentile(tops, 75))  # the top face, not its blurred edge
    trace.update(rough_center=c0, z_top=z_top)

    # 2. the bright top face's outline, on the plane of the top
    half = 1.6 * geom.outer_d_m / 2
    tax = np.arange(-half, half + 1e-12, cfg.top_res_m)
    X, Y = np.meshgrid(c0[0] + tax, c0[1] + tax)
    uv, _ = project(T_wc, K, np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_top)], axis=1))
    top = sample(_gray(rgb), uv).reshape(X.shape)
    if np.isnan(top).any():
        raise SunNotFound("the part is at the edge of the image")
    g = cv2.GaussianBlur(top.astype(np.float32), (5, 5), 0)
    R = np.hypot(X - c0[0], Y - c0[1])
    r_out = geom.outer_d_m / 2
    face = np.percentile(g[(R > 0.6 * r_out) & (R < 0.85 * r_out)], 60)
    table = np.median(g[R > 1.35 * r_out])
    if face - table < 8:
        raise SunNotFound(f"the top face is not brighter than the table ({face:.0f} vs {table:.0f})")
    mask = ((g > (face + table) / 2) & (R < 1.4 * r_out)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise SunNotFound("no bright top face")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    pts = np.stack([c0[0] + tax[0] + cnt[:, 0] * cfg.top_res_m,
                    c0[1] + tax[0] + cnt[:, 1] * cfg.top_res_m], axis=1)
    center, radius = _fit_circle(pts)
    resid = np.abs(np.hypot(*(pts - center).T) - radius)
    keep = resid < max(3 * np.median(resid), 2 * cfg.top_res_m)
    if keep.sum() > 20:
        center, radius = _fit_circle(pts[keep])
    trace.update(top=g, top_axis=tax, top_center=c0, outline=pts, free_fit_center=center.copy())
    if abs(radius / r_out - 1) > cfg.radius_tol:
        raise SunNotFound(
            f"top face radius {radius * 1e3:.1f} mm, expected {r_out * 1e3:.1f} (wrong part, "
            "or lying on its side)"
        )

    # 2b. the camera looks at the part from one side, so the outline also takes in the side
    # wall facing it (teeth below the top face): a free circle fit grows and its centre moves
    # toward the camera — 0.8 mm on 2026-11-09, more than the bore's 0.5 mm clearance. A circle
    # of the known radius through the outline points closest to it ignores the wall.
    from scipy.optimize import least_squares

    c_fix = center.copy()
    for _ in range(6):
        res = np.abs(np.hypot(*(pts - c_fix).T) - r_out)
        keep = np.argsort(res)[: int(cfg.keep_fraction * len(pts))]
        c_fix = least_squares(lambda cc: np.hypot(*(pts[keep] - cc).T) - r_out, c_fix).x
    trace["fixed_r_center"] = c_fix
    if cfg.fixed_radius_fit:
        if np.hypot(*(c_fix - center)) > 0.002:
            warnings.append(f"fixed-radius centre {np.hypot(*(c_fix - center)) * 1e3:.1f} mm from the free fit")
        center = c_fix

    # 3. the bore
    bore_off = None
    Rc = np.hypot(X - center[0], Y - center[1])
    dark = (g < table - 0.3 * (face - table)) & (Rc < 0.6 * r_out)
    if dark.sum() > 10:
        bc = np.array([X[dark].mean(), Y[dark].mean()])
        bore_off = float(np.hypot(*(bc - center)))
        if bore_off > cfg.bore_tol_m:
            warnings.append(f"bore {bore_off * 1e3:.1f} mm off the outline's centre")
    else:
        warnings.append("bore not seen")
    return SunDetection(center_xy=center, z_top=z_top, z_table=z_table, radius_m=radius,
                        bore_offset_m=bore_off, warnings=warnings)


def draw_sun_popup(rgb, K, T_wc, det: SunDetection | None, nominal_xy, z_guess=0.018,
                   error: str = "", geom: SunGeometry = SunGeometry(), scale: int = 4) -> np.ndarray:
    nominal = np.asarray(nominal_xy, float)
    z = det.z_top if det is not None else z_guess
    c = det.center_xy if det is not None else nominal
    uv_c, dist = project(T_wc, K, np.array([[*c, z]]))
    half = int(np.clip(2.2 * geom.outer_d_m * K[0, 0] / max(dist[0], 1e-3), 40, 120))
    h, w = rgb.shape[:2]
    u0 = int(np.clip(round(uv_c[0, 0]) - half, 0, max(w - 2 * half, 0)))
    v0 = int(np.clip(round(uv_c[0, 1]) - half, 0, max(h - 2 * half, 0)))
    img = cv2.resize(np.ascontiguousarray(rgb[v0:v0 + 2 * half, u0:u0 + 2 * half], np.uint8), None,
                     fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def px(p):
        uv, _ = project(T_wc, K, np.atleast_2d(p))
        return [(int(round((a - u0) * scale)), int(round((b - v0) * scale))) for a, b in uv]

    lines = []
    if det is not None:
        th = np.linspace(0, 2 * np.pi, 90)
        ring = px(np.stack([c[0] + det.radius_m * np.cos(th), c[1] + det.radius_m * np.sin(th),
                            np.full_like(th, z)], 1))
        cv2.polylines(img, [np.array(ring, np.int32)], True, (255, 230, 0), 2, cv2.LINE_AA)
        p = px([*c, z])[0]
        cv2.drawMarker(img, p, (0, 0, 0), cv2.MARKER_CROSS, 26, 5)
        cv2.drawMarker(img, p, (0, 230, 255), cv2.MARKER_CROSS, 26, 2)
        lines += [f"sun at ({c[0]:.4f}, {c[1]:.4f}), {np.hypot(*(c - nominal)) * 1e3:.1f} mm from expected",
                  f"top z {det.z_top * 1e3:.1f} mm, {det.height_m * 1e3:.1f} mm tall, r {det.radius_m * 1e3:.2f} mm",
                  *det.warnings]
    if error:
        lines = [f"NOT FOUND: {error}"]
    bar = np.full((24 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 90, 90) if error else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


# -- coarse search: where on the table is it? ------------------------------------------------


@dataclass
class SunCandidate:
    center_xy: np.ndarray
    height_m: float
    area_ratio: float  # footprint area / the expected disc's
    roundness: float  # footprint area / its enclosing circle's
    score: float  # lower is better


def find_sun_candidates(
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    prefer_xy: tuple[float, float] | None = None,
    half_m: float = 0.22,
    res_m: float = 0.0012,
    rgb: np.ndarray | None = None,
    exclude: list[tuple[float, float, float]] = (),
    height_range_m: tuple[float, float] = (0.012, 0.032),
    geom: SunGeometry = SunGeometry(),
) -> list[SunCandidate]:
    """Standing parts of the sun gear's footprint and height, in a wide view (e.g. from the
    start pose: 0.8 mm/px). Depth only: a height map around where the optical axis meets the
    table, everything ``height_range_m`` above the table, blobs of the expected footprint
    (0.4-2.2x its area) that are round. Wider parts (end cap, rotor, gearbox) and flat ones
    (plates) drop out. With ``rgb``, the top must also be brighter than the table around
    it (the white part, not a dark fixture's rim). Sorted best first; ``prefer_xy`` breaks
    ties toward a position. ``exclude``: (x, y, radius) areas to ignore (the rotor's
    empty fixture looks sun-like from afar: round, ~18 mm high)."""
    K = np.asarray(K, float)
    T_wc = np.asarray(T_wc, float)
    o, z_dir = T_wc[:3, 3], T_wc[:3, 2]
    t = (0.0 - o[2]) / z_dir[2]
    ground = o + t * z_dir  # where the optical axis meets z = 0
    ax, H = _height_map(np.asarray(depth, np.float32), K, T_wc, ground[:2], half_m, res_m, 0.005)
    valid = np.isfinite(H)
    if valid.mean() < 0.3:
        raise SunNotFound("little depth in the wide view")
    table = float(np.nanpercentile(H, 40))
    rel = np.where(valid, H - table, 0.0)
    mask = ((rel > height_range_m[0]) & (rel < height_range_m[1] + 0.01)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nl, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    want = math.pi * (geom.outer_d_m / 2 / res_m) ** 2
    out = []
    for k in range(1, nl):
        a = stats[k, cv2.CC_STAT_AREA]
        if not 0.4 * want < a < 2.2 * want:
            continue
        ys, xs = np.nonzero(lab == k)
        (_, _), r_enc = cv2.minEnclosingCircle(np.stack([xs, ys], 1).astype(np.float32))
        roundness = a / max(math.pi * r_enc**2, 1.0)
        h = float(np.percentile(rel[lab == k], 75))
        if roundness < 0.6 or not height_range_m[0] < h < height_range_m[1]:
            continue
        c = np.array([ground[0] + ax[0] + cents[k][0] * res_m, ground[1] + ax[0] + cents[k][1] * res_m])
        if any(np.hypot(c[0] - ex, c[1] - ey) < er for ex, ey, er in exclude):
            continue
        if rgb is not None:
            gray = _gray(rgb)
            th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            def ring(r0, r1, z):
                R, TH = np.meshgrid(np.linspace(r0, r1, 3), th)
                P = np.stack([c[0] + R.ravel() * np.cos(TH.ravel()), c[1] + R.ravel() * np.sin(TH.ravel()),
                              np.full(R.size, z)], 1)
                uv, _ = project(T_wc, K, P)
                return float(np.nanmedian(sample(gray, uv)))
            top = ring(0.004, 0.007, table + h)  # the face between bore and edge
            around = ring(0.013, 0.018, table)
            if not top > around + 8:
                continue
        score = abs(a / want - 1) + (1 - roundness)
        if prefer_xy is not None:
            score += 2.0 * float(np.hypot(*(c - np.asarray(prefer_xy)))) / half_m
        out.append(SunCandidate(c, h, a / want, roundness, score))
    return sorted(out, key=lambda s: s.score)
