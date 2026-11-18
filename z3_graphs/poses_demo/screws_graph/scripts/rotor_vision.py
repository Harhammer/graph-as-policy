"""Find the gearbox rotor in one wrist RGB-D frame and choose where to grasp it.

The rotor is a white disk (Ø36.7 mm) with 8 holes (Ø3.5 mm) on a Ø28.2 mm bolt circle,
alternating: 4 hold black screws, 4 are empty (you see the fixture through them). The jaws
must close along a line through two opposite EMPTY holes; any of those lines works (the
rotor's angle doesn't matter for the insertion), so a valid grasp exists every 90 deg.

Everything is measured in the robot base frame on the plane of the rotor's top surface:
image points are found by projecting world points into the camera (K + camera pose from
the hand-eye calibration), so the camera's tilt never distorts the circle. Steps:

1. top view: resample the image onto a metric grid on that plane around the nominal
   position; the white disk on the non-white fixture is the largest bright blob there.
   A least-squares circle fit of its outline gives the centre (and the radius check).
2. the plane height is re-measured from depth on the disk, and step 1 repeated on it.
3. hole pattern: brightness along the bolt circle has 8 dips; the phase of its 8th
   Fourier component gives their angles.
4. screws vs empty: alternating, so it is one of two hypotheses (even or odd holes are
   screws). Screws are higher (head at the surface; empty holes read the fixture, deeper)
   and black. Both cues are scored as even-minus-odd differences, no absolute thresholds.
5. grasp yaw: the jaw axis must lie on an empty-hole diameter (every 90 deg); take the one
   needing the smallest wrist turn from rpy [180, 0, 0].

Pure numpy + OpenCV, no GaP imports: the tests and offline tuning use it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class RotorGeometry:
    outer_d_m: float = 0.0367
    bolt_circle_d_m: float = 0.0282
    hole_d_m: float = 0.0035
    n_holes: int = 8


@dataclass(frozen=True)
class DetectConfig:
    window_m: float = 0.035  # half-size of the top-view search window around the nominal
    res_m: float = 0.00025  # top-view resolution (the camera sees ~0.34 mm/px at 15 cm)
    radius_tol: float = 0.12  # fitted radius must be within 12 % of the true one
    min_hole_contrast: float = 0.15  # holes vs between-holes, as a fraction of disk brightness
    min_screw_score: float = 2.0  # |even-vs-odd| separation needed to call screws vs empty


@dataclass
class RotorDetection:
    center_xy: np.ndarray  # base frame, metres
    z_top: float  # top-surface height, metres (from depth when available)
    radius_m: float  # fitted outer radius
    hole_phase_rad: float  # angle of hole 0 (holes at phase + k * 45 deg), base frame
    screws_even: bool  # True: holes 0,2,4,6 have screws; odd ones are empty
    screw_score: float  # signed even-vs-odd separation (+ = even are screws)
    cue_scores: dict[str, float]
    hole_gray: np.ndarray  # (8,) mean brightness in each hole
    hole_dz: np.ndarray  # (8,) hole depth relative to the top surface, metres (nan = none)
    hole_contrast: float
    empty_axis_rad: float  # angle of one empty-hole diameter; the others are +90 deg
    z_from_depth: bool
    warnings: list[str] = field(default_factory=list)

    def grasp_yaw_deg(self, jaw_axis_deg: float) -> float:
        """Tool yaw (rpy [180, 0, yaw]) that puts the jaw axis on an empty-hole diameter,
        within (-45, 45]. ``jaw_axis_deg``: the jaws' closing direction in the base frame
        at yaw 0 (0 = along base x, 90 = along base y)."""
        return _wrap(math.degrees(self.empty_axis_rad) - jaw_axis_deg, 90.0)


def _wrap(angle_deg: float, period: float) -> float:
    """Wrap to (-period/2, period/2]."""
    a = (angle_deg + period / 2.0) % period - period / 2.0
    return period / 2.0 if math.isclose(a, -period / 2.0) else a


# -- camera geometry ----------------------------------------------------------------------


def project(T_wc: np.ndarray, K: np.ndarray, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) -> pixels (N, 2) and camera-z depth (N,)."""
    T_cw = np.linalg.inv(T_wc)
    p = pts_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = p[:, 2]
    uv = np.stack([K[0, 0] * p[:, 0] / z + K[0, 2], K[1, 1] * p[:, 1] / z + K[1, 2]], axis=1)
    return uv, z


def deproject(T_wc: np.ndarray, K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) with camera-z depth (N,) -> world points (N, 3)."""
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * depth
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * depth
    p = np.stack([x, y, depth], axis=1)
    return p @ T_wc[:3, :3].T + T_wc[:3, 3]


def sample(img: np.ndarray, uv: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Values of ``img`` at pixels (N, 2); outside the image -> nan."""
    img = np.asarray(img, dtype=np.float32)
    n = len(uv)
    side = max(1, math.ceil(math.sqrt(n)))  # remap's map must stay under 32767 per side
    pad = side * side - n
    mx = np.pad(uv[:, 0].astype(np.float32), (0, pad), constant_values=-1).reshape(side, side)
    my = np.pad(uv[:, 1].astype(np.float32), (0, pad), constant_values=-1).reshape(side, side)
    out = cv2.remap(
        img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
    )
    return out.reshape(-1)[:n]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )


# -- step 1: the disk ---------------------------------------------------------------------


def _top_view(gray, K, T_wc, center_xy, z, cfg: DetectConfig):
    n = int(round(2 * cfg.window_m / cfg.res_m)) + 1
    axis = np.linspace(-cfg.window_m, cfg.window_m, n)
    X, Y = np.meshgrid(center_xy[0] + axis, center_xy[1] + axis)  # row <-> y, col <-> x
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    uv, _ = project(T_wc, K, pts)
    return sample(gray, uv).reshape(n, n), axis


def _fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares (Kasa) circle through points (N, 2)."""
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def _find_disk(top, axis, center_xy, geom: RotorGeometry, cfg: DetectConfig, trace=None):
    trace = {} if trace is None else trace
    trace.update(top=top, top_axis=axis, window_center=np.asarray(center_xy, float))
    valid = np.isfinite(top)
    if valid.mean() < 0.5:
        raise RotorNotFound("the search window is mostly outside the image")
    img = np.where(valid, top, np.nanmedian(top)).astype(np.float32)
    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img8 = cv2.GaussianBlur(img8, (5, 5), 0)
    # The rotor is the brightest thing in the window and covers ~20 % of it; the rest is
    # background of any number of shades (black fixture, wood, shadows). Split halfway
    # between the background's median and the rotor's level. (Otsu's two-class split put
    # the wood with the rotor on the 2026-11-09 multi-view frames.)
    lo, hi = np.percentile(img8[valid], [50, 97])
    mask = ((img8 > (lo + hi) / 2) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    trace["mask"] = mask > 0
    r_px = geom.outer_d_m / 2 / cfg.res_m
    want_area = math.pi * r_px**2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mid = (len(axis) - 1) / 2.0
    best, best_cost = None, math.inf
    for c in contours:
        area = cv2.contourArea(c)
        if not 0.5 * want_area < area < 1.6 * want_area:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        cost = abs(area / want_area - 1.0) + math.hypot(cx - mid, cy - mid) / r_px
        if cost < best_cost:
            best, best_cost = c, cost
    if best is None:
        raise RotorNotFound("no bright disk of the rotor's size in the search window")
    pts = best.reshape(-1, 2).astype(np.float64)
    xy = np.stack([center_xy[0] + axis[0] + pts[:, 0] * cfg.res_m,
                   center_xy[1] + axis[0] + pts[:, 1] * cfg.res_m], axis=1)
    # outline points touching the window edge or the image border are not the disk's edge
    inside = (pts.min(axis=1) > 1) & (pts.max(axis=1) < len(axis) - 2)
    if inside.sum() < 30:
        raise RotorNotFound("the rotor's outline is cut off by the window or image edge")
    center, radius = _fit_circle(xy[inside])
    # drop outliers (a finger or shadow touching the edge) and refit once
    resid = np.abs(np.hypot(*(xy[inside] - center).T) - radius)
    keep = resid < max(3 * np.median(resid), 2 * cfg.res_m)
    if keep.sum() >= 30:
        center, radius = _fit_circle(xy[inside][keep])
    trace.update(outline_xy=xy[inside], outline_keep=keep, outline_center=center,
                 outline_radius=radius)
    r_true = geom.outer_d_m / 2
    if abs(radius / r_true - 1.0) > cfg.radius_tol:
        raise RotorNotFound(
            f"fitted radius {radius * 1e3:.1f} mm, expected {r_true * 1e3:.1f} mm "
            "(wrong object, or the top-surface height is far off)"
        )
    return center, radius


# -- step 2: the plane height -------------------------------------------------------------


def _height_map(depth, K, T_wc, window_center, z, axis) -> np.ndarray:
    """World height [m] seen by depth at each top-view cell (nan = no depth)."""
    X, Y = np.meshgrid(window_center[0] + axis, window_center[1] + axis)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    uv, _ = project(T_wc, K, pts)
    d = sample(depth, uv, nearest=True)
    ok = np.isfinite(d) & (d > 0.03)
    h = np.full(len(d), np.nan)
    h[ok] = deproject(T_wc, K, uv[ok], d[ok])[:, 2]
    return h.reshape(X.shape)


def _surface_z(depth, K, T_wc, center, z, geom: RotorGeometry) -> float | None:
    """Median world z of the disk's top surface from depth (annulus clear of the holes)."""
    rs = np.linspace(0.25, 0.85, 8) * geom.bolt_circle_d_m / 2  # inside the bolt circle
    rs = np.concatenate([rs, [0.5 * (geom.bolt_circle_d_m + geom.outer_d_m) / 2]])
    th = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    R, T = np.meshgrid(rs, th)
    pts = np.stack([center[0] + R.ravel() * np.cos(T.ravel()),
                    center[1] + R.ravel() * np.sin(T.ravel()), np.full(R.size, z)], axis=1)
    uv, _ = project(T_wc, K, pts)
    d = sample(depth, uv, nearest=True)
    ok = np.isfinite(d) & (d > 0.03)
    if ok.sum() < 50:
        return None
    zs = deproject(T_wc, K, uv[ok], d[ok])[:, 2]
    return float(np.median(zs))


def _highest_surface_z(depth, K, T_wc, nominal_xy, z_guess, geom: RotorGeometry):
    """Rough top height before the disk is found: the rotor's top is the highest surface
    near the nominal position (fixture and table are lower). 90th percentile of the world
    heights within 1.5 rotor radii, re-centred twice (where nominal x/y lands in the image
    depends on the assumed height)."""
    z = z_guess
    r = 1.5 * geom.outer_d_m / 2
    gx, gy = np.meshgrid(np.linspace(-r, r, 41), np.linspace(-r, r, 41))
    inside = np.hypot(gx, gy) <= r
    for _ in range(3):
        pts = np.stack([nominal_xy[0] + gx[inside], nominal_xy[1] + gy[inside],
                        np.full(inside.sum(), z)], axis=1)
        uv, _ = project(T_wc, K, pts)
        d = sample(depth, uv, nearest=True)
        ok = np.isfinite(d) & (d > 0.03)
        if ok.sum() < 50:
            return None
        z = float(np.percentile(deproject(T_wc, K, uv[ok], d[ok])[:, 2], 90))
    return z


# -- steps 3-4: the holes -----------------------------------------------------------------


def _ring(center, r, th, z):
    return np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th),
                     np.full_like(th, z)], axis=1)


def _hole_phase(gray, K, T_wc, center, z, geom: RotorGeometry, cfg: DetectConfig, trace=None):
    n = geom.n_holes
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rb, rh = geom.bolt_circle_d_m / 2, geom.hole_d_m / 2
    prof = np.zeros_like(th)
    for dr in np.linspace(-0.5 * rh, 0.5 * rh, 5):
        uv, _ = project(T_wc, K, _ring(center, rb + dr, th, z))
        prof += sample(gray, uv)
    prof /= 5
    if not np.isfinite(prof).all():
        raise RotorNotFound("the bolt circle is partly outside the image")
    dark = prof.max() - prof
    c = (dark * np.exp(-1j * n * th)).sum()  # dark ~ cos(n (th - phase)) -> c ~ e^{-i n phase}
    phase = (-math.atan2(c.imag, c.real) / n) % (2 * np.pi / n)
    # contrast: brightness between holes minus in holes, relative to the disk brightness
    holes = phase + 2 * np.pi * np.arange(n) / n
    def at(angles):
        idx = np.round(np.mod(angles, 2 * np.pi) / (2 * np.pi) * len(th)).astype(int) % len(th)
        return prof[idx]
    contrast = float(np.median(at(holes + np.pi / n) - at(holes)) / max(prof.max(), 1.0))
    if trace is not None:
        trace.update(profile_th=th, profile=prof, profile_phase=phase, profile_contrast=contrast,
                     profile_center=np.asarray(center, float))
    if contrast < cfg.min_hole_contrast:
        raise RotorNotFound(
            f"no 8-hole pattern on the bolt circle (contrast {contrast:.2f} < "
            f"{cfg.min_hole_contrast})"
        )
    return phase, contrast


def _disk_offsets(r_in: float, r_out: float, n_r: int = 4, n_t: int = 16) -> np.ndarray:
    rr, tt = np.meshgrid(np.linspace(r_in, r_out, n_r), np.linspace(0, 2 * np.pi, n_t, endpoint=False))
    return np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()], axis=1)


def _refine_with_holes(gray, K, T_wc, center, z, phase, geom: RotorGeometry, trace=None):
    """Centre and phase from the 8 holes' own centroids.

    The holes sit on a circle concentric with the rotor axis, a better reference than the
    outline (planet teeth stick out past the plate on some sides and pull the outline fit
    off-centre). Every hole, screw or empty, is darker than the plate around it: each one's
    darkness-weighted centroid is found in a small patch, and a circle is fitted to them.
    Returns (center, phase, bolt radius) or None when fewer than 6 holes are clear.
    """
    n = geom.n_holes
    rb, rh = geom.bolt_circle_d_m / 2, geom.hole_d_m / 2
    g = np.linspace(-1.8 * rh, 1.8 * rh, 25)
    gx, gy = np.meshgrid(g, g)
    near = np.hypot(gx, gy) <= 1.8 * rh
    patch = np.stack([gx[near], gy[near]], axis=1)
    cents, ks = [], []
    for k in range(n):
        a = phase + 2 * np.pi * k / n
        hc = center + rb * np.array([math.cos(a), math.sin(a)])
        uv, _ = project(T_wc, K, np.column_stack([hc + patch, np.full(len(patch), z)]))
        val = sample(gray, uv)
        if not np.isfinite(val).all():
            continue
        plate = np.percentile(val, 85)
        w = np.clip(0.85 * plate - val, 0.0, None)
        if w.sum() <= 0 or (w > 0).mean() < 0.1:
            continue
        cents.append(hc + (w[:, None] * patch).sum(axis=0) / w.sum())
        ks.append(k)
    if trace is not None:
        trace["hole_centroids"] = np.asarray(cents).reshape(-1, 2)
    if len(cents) < 6:
        return None
    cents = np.asarray(cents)
    new_center, r_bolt = _fit_circle(cents)
    if trace is not None:
        trace["bolt_fit"] = (new_center, r_bolt)
    if abs(r_bolt - rb) > 0.001:
        return None  # not the bolt circle: keep the outline's centre
    rel = np.arctan2(cents[:, 1] - new_center[1], cents[:, 0] - new_center[0])
    resid = rel - 2 * np.pi * np.asarray(ks) / n
    m = np.exp(1j * n * resid).mean()
    new_phase = (math.atan2(m.imag, m.real) / n) % (2 * np.pi / n)
    return new_center, new_phase, r_bolt


def _hole_features(gray, depth, K, T_wc, center, z, phase, geom: RotorGeometry):
    """Per hole: brightness relative to the plate right around it (cancels shading from a
    hand or the light), and depth relative to the top surface."""
    n = geom.n_holes
    rb, rh = geom.bolt_circle_d_m / 2, geom.hole_d_m / 2
    inner = _disk_offsets(0.0, 0.6 * rh)
    ring = _disk_offsets(1.4 * rh, 1.9 * rh)
    gray_k, dz_k = np.zeros(n), np.full(n, np.nan)
    for k in range(n):
        a = phase + 2 * np.pi * k / n
        hc = center + rb * np.array([math.cos(a), math.sin(a)])
        uv, _ = project(T_wc, K, np.column_stack([hc + inner, np.full(len(inner), z)]))
        uv_ring, _ = project(T_wc, K, np.column_stack([hc + ring, np.full(len(ring), z)]))
        vals = np.sort(sample(gray, uv))
        hole = np.nanmean(vals[: max(1, int(0.6 * len(vals)))])  # darkest 60 %: off-centre safe
        plate = np.nanmedian(sample(gray, uv_ring))
        gray_k[k] = hole / max(plate, 1.0)
        if depth is not None:
            d = sample(depth, uv, nearest=True)
            ok = np.isfinite(d) & (d > 0.03)
            if ok.sum() >= 3:
                dz_k[k] = float(np.median(deproject(T_wc, K, uv[ok], d[ok])[:, 2])) - z
    return gray_k, dz_k


def _separation(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    """(mean a - mean b) / pooled spread; 0 when either side has no data."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    spread = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + floor
    return float((a.mean() - b.mean()) / spread)


# -- the whole thing ----------------------------------------------------------------------


class RotorNotFound(RuntimeError):
    """No rotor (or no readable hole pattern) where it was expected."""


def detect_rotor(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    K: np.ndarray,
    T_wc: np.ndarray,
    nominal_xy: tuple[float, float],
    z_top_guess: float,
    geom: RotorGeometry = RotorGeometry(),
    cfg: DetectConfig = DetectConfig(),
    trace: dict | None = None,
) -> RotorDetection:
    """Find the rotor. ``trace``: a dict to fill with the intermediate results (top view,
    outline fit, heights, brightness profile, hole centroids) for :func:`draw_steps` —
    filled as far as the detection got when it raises."""
    trace = {} if trace is None else trace
    gray = _gray(rgb)
    K = np.asarray(K, dtype=np.float64)
    T_wc = np.asarray(T_wc, dtype=np.float64)
    depth = None if depth is None else np.asarray(depth, dtype=np.float32)
    warnings: list[str] = []

    z = float(z_top_guess)
    if depth is not None:
        z_rough = _highest_surface_z(depth, K, T_wc, np.asarray(nominal_xy, float), z, geom)
        if z_rough is not None:
            if abs(z_rough - z) > 0.01:
                warnings.append(
                    f"top height from depth {z_rough:.4f} vs configured {z:.4f} (>1 cm apart)"
                )
            z = z_rough
    trace.update(z_guess=float(z_top_guess), z_rough=z if depth is not None else None,
                 nominal_xy=np.asarray(nominal_xy, float), warnings=warnings)
    top, axis = _top_view(gray, K, T_wc, np.asarray(nominal_xy, float), z, cfg)
    if depth is not None:  # for the step figure, also when the disk is not found
        trace["z_top"] = z
        trace["height_map"] = _height_map(depth, K, T_wc, np.asarray(nominal_xy, float), z, axis)
    center, radius = _find_disk(top, axis, np.asarray(nominal_xy, float), geom, cfg, trace)
    z_from_depth = False
    if depth is not None:
        z_meas = _surface_z(depth, K, T_wc, center, z, geom)
        if z_meas is None:
            warnings.append("no usable depth on the rotor; using the configured top height")
        else:
            z, z_from_depth = z_meas, True
            top, axis = _top_view(gray, K, T_wc, center, z, cfg)
            center, radius = _find_disk(top, axis, center, geom, cfg, trace)
    trace["z_top"] = z
    if depth is not None:
        trace["height_map"] = _height_map(depth, K, T_wc, trace["window_center"], z, axis)

    phase, contrast = _hole_phase(gray, K, T_wc, center, z, geom, cfg, trace)
    for _ in range(2):
        refined = _refine_with_holes(gray, K, T_wc, center, z, phase, geom, trace)
        if refined is None:
            warnings.append("hole centroids unclear; centre from the outline only")
            break
        center, phase, _ = refined
    gray_k, dz_k = _hole_features(gray, depth, K, T_wc, center, z, phase, geom)
    even = np.arange(geom.n_holes) % 2 == 0
    cues = {
        # screws are black; an empty hole shows whatever is below, lighter (relative to the
        # plate around each hole)
        "gray": _separation(gray_k[~even], gray_k[even], floor=0.03),
        # screw heads sit higher than what an empty hole shows. The D405 rarely resolves a
        # 3.5 mm hole at 20 cm: this only votes when it is clearly separated.
        "depth": _separation(dz_k[even], dz_k[~even], floor=0.0005),
    }
    score = cues["gray"] + (cues["depth"] if abs(cues["depth"]) >= 3.0 else 0.0)
    if cues["depth"] * cues["gray"] < 0 and min(abs(cues["depth"]), abs(cues["gray"])) >= 3.0:
        warnings.append(f"depth and colour disagree about which holes have screws: {cues}")
    if abs(score) < cfg.min_screw_score:
        raise RotorNotFound(
            f"cannot tell screws from empty holes (score {score:+.2f}, cues {cues}); "
            "tune min_screw_score on captured images"
        )
    screws_even = score > 0
    empty0 = phase + (2 * np.pi / geom.n_holes) * (1 if screws_even else 0)
    return RotorDetection(
        center_xy=center, z_top=z, radius_m=radius, hole_phase_rad=phase,
        screws_even=screws_even, screw_score=score, cue_scores=cues,
        hole_gray=gray_k, hole_dz=dz_k, hole_contrast=contrast,
        empty_axis_rad=float(np.mod(empty0, np.pi / 2)),
        z_from_depth=z_from_depth, warnings=warnings,
    )


# -- debug picture ------------------------------------------------------------------------


def draw_detection(
    rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    det: RotorDetection | None,
    jaw_axis_deg: float = 0.0,
    geom: RotorGeometry = RotorGeometry(),
    note: str = "",
) -> np.ndarray:
    """RGB image with the fitted circle, holes (red = screw, green = empty), the chosen
    jaw axis (cyan) and the numbers."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8).copy()
    lines = [note] if note else []
    if det is not None:
        c, z = det.center_xy, det.z_top
        th = np.linspace(0, 2 * np.pi, 180)
        for r, col in ((det.radius_m, (255, 255, 0)), (geom.bolt_circle_d_m / 2, (255, 160, 0))):
            uv, _ = project(T_wc, K, _ring(c, r, th, z))
            cv2.polylines(img, [uv.astype(np.int32)], True, col, 1, cv2.LINE_AA)
        for k in range(geom.n_holes):
            a = det.hole_phase_rad + 2 * np.pi * k / geom.n_holes
            uv, _ = project(T_wc, K, _ring(c, geom.bolt_circle_d_m / 2, np.array([a]), z))
            screw = (k % 2 == 0) == det.screws_even
            cv2.circle(img, tuple(int(v) for v in uv[0]), 9, (255, 0, 0) if screw else (0, 255, 0),
                       1, cv2.LINE_AA)  # a ring around the hole, so the hole stays visible
        yaw = det.grasp_yaw_deg(jaw_axis_deg)
        a = math.radians(jaw_axis_deg + yaw)
        half = 0.6 * geom.outer_d_m
        ends = np.array([[c[0] - half * math.cos(a), c[1] - half * math.sin(a), z],
                         [c[0] + half * math.cos(a), c[1] + half * math.sin(a), z]])
        uv, _ = project(T_wc, K, ends)
        cv2.line(img, tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1]),
                 (0, 255, 255), 2, cv2.LINE_AA)
        lines += [
            f"center ({c[0]:.4f}, {c[1]:.4f}) z_top {z:.4f}{'' if det.z_from_depth else ' (cfg)'}",
            f"radius {det.radius_m * 1e3:.2f} mm  hole contrast {det.hole_contrast:.2f}",
            f"screws {'even' if det.screws_even else 'odd'}  score {det.screw_score:+.1f} "
            f"(depth {det.cue_scores['depth']:+.1f}, gray {det.cue_scores['gray']:+.1f})",
            f"grasp yaw {yaw:+.1f} deg",
            *det.warnings,
        ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_steps(
    rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    det: RotorDetection | None,
    trace: dict,
    jaw_axis_deg: float = 0.0,
    geom: RotorGeometry = RotorGeometry(),
    error: str = "",
) -> np.ndarray:
    """One RGB picture of every detection step (filled as far as the trace got):

    1 camera crop with the result, 2 metric top view with the bright-disk outline fit,
    3 heights from depth, 4 brightness along the bolt circle (the 8 dips -> hole angles),
    5 hole centroids -> centre, 6 screw-or-empty decision per hole -> grasp.
    Top views are in mm relative to the nominal position, base x right, base y up.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), dpi=90)
    axs = axs.ravel()
    nominal = trace.get("nominal_xy", np.zeros(2))
    mm = lambda xy: (np.asarray(xy, float) - nominal) * 1e3  # noqa: E731
    th_c = np.linspace(0, 2 * np.pi, 200)

    def circle(ax, c, r, **kw):
        c = mm(c)
        ax.plot(c[0] + r * 1e3 * np.cos(th_c), c[1] + r * 1e3 * np.sin(th_c), **kw)

    def top_extent():
        wc, axis = mm(trace["window_center"]), trace["top_axis"] * 1e3
        return (wc[0] + axis[0], wc[0] + axis[-1], wc[1] + axis[0], wc[1] + axis[-1])

    # 1 camera
    ax = axs[0]
    over = draw_detection(rgb, K, T_wc, det, jaw_axis_deg, geom)
    c3 = np.array([*(det.center_xy if det is not None else nominal),
                   det.z_top if det is not None else trace.get("z_top", trace.get("z_guess", 0.0))])
    uv, z = project(T_wc, K, c3[None])
    half = int(np.clip(1.4 * geom.outer_d_m * K[0, 0] / max(z[0], 1e-3), 60, 240))
    u, v = (int(round(x)) for x in uv[0])
    h, w = over.shape[:2]
    y0, y1, x0, x1 = max(v - half, 0), min(v + half, h), max(u - half, 0), min(u + half, w)
    ax.imshow(over[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else over)
    ax.set_title("1  camera (red = screw, green = empty, cyan = jaws)")
    ax.axis("off")

    # 2 top view + outline fit
    ax = axs[1]
    if "top" in trace:
        ax.imshow(trace["top"], cmap="gray", origin="lower", extent=top_extent())
        if "mask" in trace:
            ax.contour(trace["mask"].astype(float), levels=[0.5], colors="deepskyblue",
                       linewidths=0.6, origin="lower", extent=top_extent())
        if "outline_xy" in trace:
            pts, keep = mm(trace["outline_xy"]), trace["outline_keep"]
            ax.plot(*pts[keep].T, ".", ms=1.5, color="lime", label="outline (used)")
            ax.plot(*pts[~keep].T, ".", ms=2.5, color="red", label="outline (rejected)")
            circle(ax, trace["outline_center"], trace["outline_radius"], color="yellow", lw=1,
                   label=f"fit r={trace['outline_radius'] * 1e3:.1f} mm "
                         f"(true {geom.outer_d_m * 500:.1f})")
        ax.plot(0, 0, "w+", ms=12, mew=1.5, label="nominal")
        ax.legend(loc="lower left", fontsize=7, framealpha=0.6)
    ax.set_title(f"2  top view at z={trace.get('z_top', trace.get('z_rough') or 0):.4f}: "
                 "bright blob -> outline circle")
    ax.set_xlabel("base x - nominal [mm]")
    ax.set_ylabel("base y - nominal [mm]")

    # 3 heights
    ax = axs[2]
    if "height_map" in trace:
        zt = trace["z_top"]
        im = ax.imshow(trace["height_map"] * 1e3, origin="lower", extent=top_extent(),
                       cmap="viridis", vmin=(zt - 0.015) * 1e3, vmax=(zt + 0.005) * 1e3)
        fig.colorbar(im, ax=ax, fraction=0.046, label="height [mm]")
        if "outline_center" in trace:
            circle(ax, trace["outline_center"], trace["outline_radius"], color="w", lw=0.8)
        zr = trace.get("z_rough")
        plate = f", plate {zt * 1e3:.1f}" if "outline_center" in trace else ""
        ax.set_title(f"3  depth heights: guess {trace['z_guess'] * 1e3:.1f}, highest "
                     f"{(zr or float('nan')) * 1e3:.1f}{plate} mm", fontsize=10)
    else:
        ax.text(0.5, 0.5, "no depth", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("3  depth heights")
    ax.set_xlabel("base x - nominal [mm]")

    # 4 profile
    ax = axs[3]
    if "profile" in trace:
        th = np.degrees(trace["profile_th"])
        ax.plot(th, trace["profile"], color="0.3", lw=1)
        ph = math.degrees(trace["profile_phase"])
        for k in range(geom.n_holes):
            a = (ph + 360.0 * k / geom.n_holes) % 360
            col = "0.5" if det is None else ("red" if (k % 2 == 0) == det.screws_even else "green")
            ax.axvline(a, color=col, lw=1.2, alpha=0.8)
        ax.set_xlim(0, 360)
        ax.set_title(f"4  brightness along the bolt circle: 8 dips at {ph:.1f} + k*45 deg "
                     f"(contrast {trace['profile_contrast']:.2f})", fontsize=10)
        ax.set_xlabel("angle about the rotor centre, from base x [deg]")
        ax.set_ylabel("gray")

    # 5 holes -> centre
    ax = axs[4]
    if "top" in trace and "hole_centroids" in trace:
        ax.imshow(trace["top"], cmap="gray", origin="lower", extent=top_extent())
        cents = mm(trace["hole_centroids"])
        ax.plot(*cents.T, ".", ms=7, color="cyan", label="hole centroids")
        oc = mm(trace["outline_center"])
        ax.plot(*oc, "x", color="yellow", ms=10, mew=2, label="outline centre")
        if "bolt_fit" in trace:
            bc, br = trace["bolt_fit"]
            circle(ax, bc, br, color="orange", lw=1, label=f"bolt circle r={br * 1e3:.2f} mm")
            b = mm(bc)
            ax.plot(*b, "+", color="red", ms=12, mew=2, label="hole centre (used)")
            ax.set_title(f"5  holes -> centre: moved {np.hypot(*(b - oc)):.2f} mm from the outline's")
        else:
            ax.set_title("5  holes -> centre: too few clear holes")
        c = cents.mean(axis=0)
        ax.set_xlim(c[0] - 22, c[0] + 22)
        ax.set_ylim(c[1] - 22, c[1] + 22)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.6)
    ax.set_xlabel("base x - nominal [mm]")

    # 6 decision
    ax = axs[5]
    if det is not None:
        n = geom.n_holes
        cols = ["red" if (k % 2 == 0) == det.screws_even else "green" for k in range(n)]
        ax.bar(range(n), det.hole_gray, color=cols)
        for k in range(n):
            dz = det.hole_dz[k]
            ax.text(k, det.hole_gray[k] + 0.02, "" if np.isnan(dz) else f"{dz * 1e3:+.1f}mm",
                    ha="center", fontsize=7)
        ax.set_xticks(range(n), [f"{k}\n{'even' if k % 2 == 0 else 'odd'}" for k in range(n)])
        ax.set_ylim(0, max(1.1, float(np.nanmax(det.hole_gray)) + 0.15))
        ax.set_ylabel("hole brightness / plate around it")
        cues = det.cue_scores
        ax.set_title(
            f"6  screws = {'even' if det.screws_even else 'odd'} holes (score "
            f"{det.screw_score:+.1f}: gray {cues['gray']:+.1f}, depth {cues['depth']:+.1f})\n"
            f"grasp yaw {det.grasp_yaw_deg(jaw_axis_deg):+.1f} deg; numbers = hole depth "
            "vs top", fontsize=10)
    else:
        ax.axis("off")
    for ax, key in ((axs[3], "profile"), (axs[4], "hole_centroids")):
        if key not in trace:
            ax.clear()
            ax.axis("off")
            ax.text(0.5, 0.5, "not reached", ha="center", va="center", color="0.5",
                    transform=ax.transAxes)
    if error:
        fig.suptitle(f"NOT FOUND: {error}", color="red", fontsize=12)
    elif det is not None:
        c = det.center_xy
        fig.suptitle(
            f"rotor at ({c[0]:.4f}, {c[1]:.4f}) = {np.hypot(*mm(c)):.1f} mm from nominal, "
            f"top z {det.z_top:.4f}" + (f"   WARN: {'; '.join(det.warnings)}" if det.warnings else ""),
            fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def draw_popup(
    rgb: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    det: RotorDetection | None,
    nominal_xy,
    jaw_axis_deg: float = 0.0,
    geom: RotorGeometry = RotorGeometry(),
    z_top: float | None = None,
    error: str = "",
    scale: int = 3,
) -> np.ndarray:
    """The quick-look picture: a zoomed crop with the rotor circle (yellow), the screw (red)
    and empty (green) holes, the new centre (cyan +), the grasp line where the jaws close
    (cyan) and the taught position (magenta x), plus offset and yaw."""
    nominal = np.asarray(nominal_xy, float)
    z = det.z_top if det is not None else (z_top if z_top is not None else 0.0)
    center = det.center_xy if det is not None else nominal
    uv_c, depth_c = project(T_wc, K, np.array([[*center, z]]))
    half = int(np.clip(1.5 * geom.outer_d_m * K[0, 0] / max(depth_c[0], 1e-3), 60, 200))
    h, w = rgb.shape[:2]
    u0 = int(np.clip(round(uv_c[0, 0]) - half, 0, max(w - 2 * half, 0)))
    v0 = int(np.clip(round(uv_c[0, 1]) - half, 0, max(h - 2 * half, 0)))
    crop = np.ascontiguousarray(rgb[v0:v0 + 2 * half, u0:u0 + 2 * half], dtype=np.uint8)
    img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def px(points_w):
        uv, _ = project(T_wc, K, np.atleast_2d(points_w))
        return [(int(round((u - u0) * scale)), int(round((v - v0) * scale))) for u, v in uv]

    def cross(p, color, size, thick, kind=cv2.MARKER_CROSS):
        cv2.drawMarker(img, p, (0, 0, 0), kind, size, thick + 3, cv2.LINE_AA)
        cv2.drawMarker(img, p, color, kind, size, thick, cv2.LINE_AA)

    lines = []
    if det is not None:
        th = np.linspace(0, 2 * np.pi, 120)
        outline = px(_ring(center, geom.outer_d_m / 2, th, z))
        cv2.polylines(img, [np.array(outline, np.int32)], True, (255, 230, 0), 2, cv2.LINE_AA)
        hole_r = max(4, int(round(0.9 * geom.hole_d_m / 2 * K[0, 0] / depth_c[0] * scale)))
        for k in range(geom.n_holes):
            a = det.hole_phase_rad + 2 * np.pi * k / geom.n_holes
            p = px(_ring(center, geom.bolt_circle_d_m / 2, np.array([a]), z))[0]
            screw = (k % 2 == 0) == det.screws_even
            cv2.circle(img, p, hole_r + 3, (255, 40, 40) if screw else (40, 255, 40), 2, cv2.LINE_AA)
        yaw = det.grasp_yaw_deg(jaw_axis_deg)
        a = math.radians(jaw_axis_deg + yaw)
        d = np.array([math.cos(a), math.sin(a)])
        r_out = 0.75 * geom.outer_d_m
        ends = px(np.array([[*(center - r_out * d), z], [*(center + r_out * d), z]]))
        cv2.line(img, ends[0], ends[1], (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, ends[0], ends[1], (0, 230, 255), 2, cv2.LINE_AA)
        for e in ends:  # where the jaws come from
            cross(e, (0, 230, 255), 18, 2, cv2.MARKER_TRIANGLE_UP)
        offset = np.hypot(*(center - nominal)) * 1e3
        lines += [
            f"offset {offset:.1f} mm from taught  (dx {(center[0] - nominal[0]) * 1e3:+.1f}, "
            f"dy {(center[1] - nominal[1]) * 1e3:+.1f})",
            f"grasp yaw {yaw:+.1f} deg   screws/empty score {abs(det.screw_score):.1f}",
        ]
    pn = px([*nominal, z])[0]
    pc = px([*center, z])[0]
    cross(pn, (255, 0, 255), 26, 2, cv2.MARKER_TILTED_CROSS)
    # label on the side away from the new centre, so the two never overlap
    away = np.subtract(pn, pc).astype(float)
    away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1 else np.array([-1.0, 1.0]) / 2**0.5
    (tw, th_), _ = cv2.getTextSize("taught", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    lx = int(pn[0] + away[0] * 28 - tw / 2)
    ly = int(pn[1] + away[1] * 28 + th_ / 2)
    cv2.putText(img, "taught", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "taught", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2,
                cv2.LINE_AA)
    if det is not None:
        cv2.arrowedLine(img, pn, pc, (255, 0, 255), 1, cv2.LINE_AA, tipLength=0.15)
        cross(pc, (0, 230, 255), 26, 2)
    if error:
        lines = [f"NOT FOUND: {error}"]
    bar = np.full((26 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 90, 90) if error else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])
