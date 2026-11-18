"""Find M4 socket head cap screws lying on the table, tell them apart by length, and pick
one the gripper can reach (z3 screw kitting, 2026-11-10).

The screws are black (oxide) on the light wooden table, lying flat in loose piles where
most of them touch a neighbour. One wrist RGB-D frame:

1. the table: a plane fitted to the depth (base frame), and its brightness;
2. a top-down view of the table (``res_m`` per cell, on the plane ``probe_height_m`` above
   it, so the tilted camera's parallax is small at the screws' height): brightness and
   height above the table per cell. Screw cells: much darker than the wood and low
   (< ``max_height_m``: the fingers and the gearbox are taller);
3. templates of a lying screw (head 7 x 4 mm, shank 4 mm, each nominal length) at every
   ``angle_step_deg``, correlated with the screw cells: ``fill`` = how much of the
   template's core is screw, ``spill`` = how much of the band around it (beside the shank,
   past the tip, behind the head) is screw too. A template that fits one screw fills ~1
   and spills ~0; a short template on a long screw spills past its tip; a long one on a
   short screw leaves its shank half empty. Touching screws keep their own fit as long
   as the neighbour only crosses a corner of the band;
4. the best fits, greedily, without overlapping each other -> one ``Screw`` each, with its
   length measured along its axis in the screw cells;
5. per screw: lying flat on the table (not on another one: the shank's height), and room
   for the fingers — the two strips the fingertips come down into and close through must
   hold nothing but the screw itself.

All positions in the base frame of ``T_wc``. The gripper approaches vertically.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from rotor_vision import _gray, deproject, project, sample


@dataclass(frozen=True)
class ScrewGeometry:
    """ISO 4762 M4: head D7 x 4 mm. ``lengths_m``: the nominal (under-head) lengths in play."""

    head_d_m: float = 0.007
    head_len_m: float = 0.004
    shank_d_m: float = 0.004
    lengths_m: tuple[tuple[str, float], ...] = (("M4x18", 0.018), ("M4x25", 0.025))

    def total_m(self, label: str) -> float:
        return self.head_len_m + dict(self.lengths_m)[label]

    def balance_m(self, total_m: float) -> float:
        """Centre of mass from the head's end: head (less its hex socket) and a threaded
        shank (~85 % of a full rod). M4x18: 8.5 mm, M4x25: 11.7 mm."""
        rh, rs, hl = self.head_d_m / 2, self.shank_d_m / 2, self.head_len_m
        v_head = math.pi * rh**2 * hl * 0.87
        v_shank = 0.85 * math.pi * rs**2 * (total_m - hl)
        return (v_head * hl / 2 + v_shank * (hl + total_m) / 2) / (v_head + v_shank)


@dataclass(frozen=True)
class GripperGeometry:
    """The fingertips as seen from above, for the clearance check. ``finger_width_m`` along
    the screw, ``finger_thick_m`` across it; they come down ``approach_open_m`` apart.
    Where along the screw they close, in this order: its balance point (centre of mass),
    moved toward the tip until the fingertips are clear of the head (M4x18: 11.5 mm from the
    head's end = its middle; M4x25: 11.7) — balanced, and the 4 mm shank leaves more room in
    the gap than the 7 mm head, but its equator is only 2 mm above the table; then the head's
    middle (``head_grip_m``: 3.5 mm to its equator); then the rest of the shank every
    ``shank_step_m``, nearest the balance point first, until ``tip_margin_m`` before the tip.
    ``prefer_head`` puts the head first."""

    finger_width_m: float = 0.007  # measured 2026-11-10: contact width at the tip
    finger_thick_m: float = 0.0015  # measured: 8 mm outside to outside at a 5 mm gap
    approach_open_m: float = 0.013  # close fraction 0.75 (measured)
    margin_m: float = 0.0015  # around the strips: where the fingers may land off the plan
    head_grip_m: float = 0.002
    shank_step_m: float = 0.002
    tip_margin_m: float = 0.004
    allow_shank_grip: bool = True
    prefer_head: bool = False

    def grip_points(self, geom: ScrewGeometry, total_m: float) -> list[float]:
        if not self.allow_shank_grip:
            return [self.head_grip_m]
        first = geom.head_len_m + self.finger_width_m / 2 + 0.0005  # fingertips clear of the head
        last = total_m - self.tip_margin_m
        if last < first:
            return [self.head_grip_m]
        centre = min(max(geom.balance_m(total_m), first), last)
        rest = sorted((float(u) for u in np.arange(first, last + 1e-9, self.shank_step_m)
                       if abs(u - centre) > 1e-4), key=lambda u: abs(u - centre))
        if self.prefer_head:
            return [self.head_grip_m, centre, *rest]
        return [centre, self.head_grip_m, *rest]


@dataclass(frozen=True)
class ScrewConfig:
    res_m: float = 0.0005
    probe_height_m: float = 0.002  # the top view's plane above the table (screw mid-height)
    max_height_m: float = 0.012  # taller than this is not a lying screw
    dark_fraction: float = 0.5  # darker than this x the table's brightness = screw
    angle_step_deg: float = 5.0
    # the templates can be matched on coarser cells (the mask averaged into them). Tried 1 mm
    # (2026-11-10): 6x faster, but in touching screws fits went wrong that 0.5 mm gets right
    match_res_m: float = 0.0005
    threads: int = 0  # template matching in parallel (0 = one per CPU)
    min_fill: float = 0.8  # the worse of head and shank core coverage
    max_spill: float = 0.12
    spill_weight: float = 2.0
    side_weight: float = 0.5
    max_side: float = 0.6
    # 80th percentile of the shank's height above the table: 4 mm lying flat (the D405 sees
    # 3-3.5 from 30 cm); on another screw it is 7 mm or more
    max_lying_height_m: float = 0.0055
    max_zone_obstacle_m2: float = 4e-6  # largest blob in the finger strips: a neighbour's tip is ~16 mm2
    # taller than this and not a screw = in the fingers' way. The depth on bare wood from the
    # 30 cm look pose: 99 % below 3 mm, up to 4.6 (10:12, 10:16); the parts near the piles
    # stand >= 5 mm; screws are found by colour, not by height
    obstacle_height_m: float = 0.006
    depth_halo_m: float = 0.003  # the depth bleeds this far around a screw
    length_tol_m: float = 0.0035  # measured vs nominal length


@dataclass
class Screw:
    label: str  # "M4x18" ...
    head_xy: np.ndarray  # the head's end face
    axis_deg: float  # direction head -> tip, base frame, (-180, 180]
    length_m: float  # measured, head end to tip (nominal + 4 mm head)
    fill: float
    spill: float
    score: float
    lying_height_m: float = math.nan
    grip_along_m: float = math.nan  # where the fingers close, from the head's end
    grip_rank: int = 99  # its place in GripperGeometry.grip_points (0 = the first choice)
    zone_obstacle_m2: float = math.nan  # at the best grip point
    reasons: list[str] = field(default_factory=list)  # why it can't be grasped; [] = can

    @property
    def axis(self) -> np.ndarray:
        a = math.radians(self.axis_deg)
        return np.array([math.cos(a), math.sin(a)])

    def point(self, along_m: float) -> np.ndarray:
        return self.head_xy + along_m * self.axis

    @property
    def graspable(self) -> bool:
        return not self.reasons


@dataclass
class ScrewScene:
    screws: list[Screw]
    z_table: float  # the table's height at the screws [m]
    plane: np.ndarray  # z = a x + b y + c of the table
    table_gray: float
    grid_origin: np.ndarray  # world xy of cell (0, 0); rows <-> +y, cols <-> +x
    res_m: float
    screw_mask: np.ndarray
    obstacle: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def table_z_at(self, xy) -> float:
        return float(self.plane[0] * xy[0] + self.plane[1] * xy[1] + self.plane[2])

    def pick(self, label: str) -> Screw | None:
        """The graspable screw of ``label`` with the best grip (first choice: the balance
        point), the best template fit among those; None when there is none."""
        ok = [s for s in self.screws if s.label == label and s.graspable]
        return min(ok, key=lambda s: (s.grip_rank, -s.score)) if ok else None


class ScrewsNotFound(RuntimeError):
    """No table or no screws in view."""


# -- 1-2: table and top view -------------------------------------------------------------


def fit_table(depth, K, T_wc, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Plane z = a x + b y + c of the table (largest horizontal level in view) and the
    world points it was fitted to. Raises when there is no such level."""
    depth = np.asarray(depth, np.float32)
    v, u = np.mgrid[0 : depth.shape[0] : step, 0 : depth.shape[1] : step]
    d = depth[v, u].ravel()
    ok = np.isfinite(d) & (d > 0.05)
    if ok.sum() < 500:
        raise ScrewsNotFound("little depth in the frame")
    W = deproject(T_wc, K, np.stack([u.ravel()[ok], v.ravel()[ok]], 1).astype(float), d[ok])
    hist, edges = np.histogram(W[:, 2], bins=np.arange(W[:, 2].min(), W[:, 2].max() + 0.002, 0.002))
    z0 = edges[np.argmax(hist)] + 0.001
    inl = np.abs(W[:, 2] - z0) < 0.004
    for tol in (0.003, 0.0015):
        A = np.c_[W[inl, :2], np.ones(inl.sum())]
        coef, *_ = np.linalg.lstsq(A, W[inl, 2], rcond=None)
        inl = np.abs(np.c_[W[:, :2], np.ones(len(W))] @ coef - W[:, 2]) < tol
    if inl.mean() < 0.25:
        raise ScrewsNotFound(f"no table level in view ({inl.mean():.0%} of the depth on it)")
    tilt = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
    if tilt > 3.0:
        raise ScrewsNotFound(f"the table plane is tilted {tilt:.1f} deg in the base frame: wrong T_wc?")
    return coef, W[inl]


def _footprint(T_wc, K, shape, z) -> tuple[np.ndarray, np.ndarray]:
    """xy bounds of the image's footprint on the plane z."""
    h, w = shape
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    o = T_wc[:3, 3]
    rays = np.c_[(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)]
    rays = rays @ T_wc[:3, :3].T
    t = (z - o[2]) / rays[:, 2]
    P = o[:2] + t[:, None] * rays[:, :2]
    return P.min(0), P.max(0)


def top_view(rgb, depth, K, T_wc, plane, cfg: ScrewConfig, region=None):
    """Gray, height above the table and the world xy grid on the table, ``res_m`` per cell."""
    lo, hi = _footprint(T_wc, K, depth.shape, plane[2])
    if region is not None:
        lo = np.maximum(lo, region[:2])
        hi = np.minimum(hi, region[2:])
    if np.any(hi - lo < 0.01):
        raise ScrewsNotFound("the search region is not in view")
    xs = np.arange(lo[0], hi[0], cfg.res_m)
    ys = np.arange(lo[1], hi[1], cfg.res_m)
    X, Y = np.meshgrid(xs, ys)  # rows <-> y, cols <-> x
    Zt = plane[0] * X + plane[1] * Y + plane[2]
    P = np.stack([X.ravel(), Y.ravel(), (Zt + cfg.probe_height_m).ravel()], 1)
    uv, _ = project(T_wc, K, P)
    gray = sample(_gray(rgb), uv).reshape(X.shape)
    d = sample(np.asarray(depth, np.float32), uv, nearest=True)
    ok = np.isfinite(d) & (d > 0.05)
    Hm = np.full(len(d), np.nan)
    Wp = deproject(T_wc, K, uv[ok], d[ok])
    Hm[ok] = Wp[:, 2] - (plane[0] * Wp[:, 0] + plane[1] * Wp[:, 1] + plane[2])
    return gray, Hm.reshape(X.shape), np.array([xs[0], ys[0]])


# -- 3: templates --------------------------------------------------------------------------


def _kernels(total_m: float, angle_deg: float, geom: ScrewGeometry, res: float, ss: int = 1):
    """(head core, shank core, end zones, side band) kernels of a screw ``total_m`` long
    pointing ``angle_deg`` (head -> tip), centred on the screw's midpoint, ``res`` per cell.
    Row offset = +y, column offset = +x. ``ss``: sub-samples per cell side — the kernels then
    hold each zone's area fraction per cell (for matching on a coarser grid than the mask's)."""
    half = int(math.ceil((total_m / 2 + 0.007) / res))
    n = 2 * half + 1
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    off = (np.arange(-half, half + 1)[:, None] + sub[None, :]).ravel() * res
    DX, DY = np.meshgrid(off, off)
    a = math.radians(angle_deg)
    u = DX * math.cos(a) + DY * math.sin(a) + total_m / 2  # 0 = head end, total = tip
    v = -DX * math.sin(a) + DY * math.cos(a)
    hl, rh, rs = geom.head_len_m, geom.head_d_m / 2, geom.shank_d_m / 2
    m = 0.0007  # the mask's blur: core inside, band outside the true outline by this
    head = (u > m) & (u < hl - m) & (np.abs(v) < rh - m)
    shank = (u > hl + m) & (u < total_m - m) & (np.abs(v) < rs - m)
    outline = ((u > -m) & (u < hl + m) & (np.abs(v) < rh + m)) | (
        (u > -m) & (u < total_m + m) & (np.abs(v) < rs + m)
    )
    # past the tip / behind the head: must be clear (this is what tells the lengths apart)
    ends = (((u > total_m + m) & (u < total_m + 0.004) & (np.abs(v) < rs + 0.001))
            | ((u < -m) & (u > -0.004) & (np.abs(v) < rh)))
    # beside it: a neighbour lying against it (a mild penalty only: room for the fingers is
    # checked on its own, against the real finger strips)
    side = (u > -m) & (u < total_m + m) & (np.abs(v) < rh + 0.002) & ~outline & ~ends
    side &= ~((u > hl + m) & (np.abs(v) < rh + m) & (u < hl + 0.0025))
    # (the corner just past the head's shoulder: the head's dark edge blurs into it)
    def k(b):
        b = b.reshape(n, ss, n, ss).mean(axis=(1, 3))
        return (b / max(b.sum(), 1e-9)).astype(np.float32)

    return k(head), k(shank), k(ends), k(side)


def _measure_length(mask, origin, res, head_xy, axis, max_m) -> float:
    """Head end -> last screw cell along the axis (gaps up to 1 mm bridged)."""
    t = np.arange(-0.002, max_m, res / 2)
    P = head_xy + np.outer(t, axis)
    c = np.round((P - origin) / res).astype(int)
    inside = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < mask.shape[1]) & (c[:, 1] < mask.shape[0])
    on = np.zeros(len(t), bool)
    on[inside] = mask[c[inside, 1], c[inside, 0]] > 0
    if not on.any():
        return 0.0
    first = np.argmax(on)
    gap, last = 0, first
    for i in range(first, len(t)):
        if on[i]:
            last, gap = i, 0
        else:
            gap += 1
            if gap * res / 2 > 0.001:
                break
    return float(t[last] - t[first])


def _rasterize(shape, origin, res, poly_xy) -> np.ndarray:
    pts = np.round((np.asarray(poly_xy) - origin) / res).astype(np.int32)
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [pts], 1)
    return out.astype(bool)


def _window(shape, origin, res, center, half_m):
    """(slices, origin) of the grid cells within ``half_m`` of ``center`` (clipped to the
    grid): work on a few cm2 instead of the whole view."""
    c = np.asarray(center, float)
    lo = np.maximum(np.floor((c - half_m - origin) / res).astype(int), 0)
    hi = np.minimum(np.ceil((c + half_m - origin) / res).astype(int) + 1, [shape[1], shape[0]])
    hi = np.maximum(hi, lo)
    return np.s_[lo[1]:hi[1], lo[0]:hi[0]], origin + lo * res


def _rect(center, axis, along_half, across_lo, across_hi):
    """Corners of the rectangle |u - c| <= along_half, across_lo <= v <= across_hi."""
    n = np.array([-axis[1], axis[0]])
    return [center + s * along_half * axis + w * n for s, w in ((-1, across_lo), (1, across_lo),
                                                                  (1, across_hi), (-1, across_hi))]


def screw_body(screw: Screw, geom: ScrewGeometry, grow_m: float = 0.0):
    """The screw's outline from above (head + shank polygons), grown by ``grow_m``."""
    a = screw.axis
    head = _rect(screw.point(geom.head_len_m / 2), a, geom.head_len_m / 2 + grow_m,
                 -geom.head_d_m / 2 - grow_m, geom.head_d_m / 2 + grow_m)
    L = screw.length_m
    shank = _rect(screw.point((geom.head_len_m + L) / 2), a, (L - geom.head_len_m) / 2 + grow_m,
                  -geom.shank_d_m / 2 - grow_m, geom.shank_d_m / 2 + grow_m)
    return head, shank


def finger_zones(screw: Screw, geom: ScrewGeometry, grip: GripperGeometry, along_m: float | None = None):
    """The two strips each fingertip comes down into and closes through, as polygons, for
    the grip ``along_m`` from the head's end (default: the screw's chosen grip)."""
    along_m = screw.grip_along_m if along_m is None else along_m
    c = screw.point(along_m)
    r_obj = (geom.head_d_m if along_m < geom.head_len_m else geom.shank_d_m) / 2
    lo, hi = r_obj + 0.0005, grip.approach_open_m / 2 + grip.finger_thick_m + grip.margin_m
    half = grip.finger_width_m / 2 + grip.margin_m
    return [_rect(c, screw.axis, half, lo, hi), _rect(c, screw.axis, half, -hi, -lo)]


def _refine_axis(s: Screw, mask, origin, res, geom: ScrewGeometry, rounds: int = 2) -> None:
    win, origin = _window(mask.shape, origin, res, s.point(s.length_m / 2), s.length_m / 2 + 0.005)
    mask = mask[win]
    for _ in range(rounds):
        inside = np.zeros(mask.shape, bool)
        for poly in screw_body(s, geom, grow_m=0.0007):
            inside |= _rasterize(mask.shape, origin, res, poly)
        rr, cc = np.nonzero(inside & (mask > 0))
        if len(rr) < 20:
            return
        P = origin + np.stack([cc, rr], 1) * res
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        a = vt[0] if vt[0] @ s.axis > 0 else -vt[0]
        if abs(math.degrees(math.acos(min(1.0, float(a @ s.axis))))) > 1.5 * 5.0:
            return  # pulled away by a neighbour: keep the template's
        n = np.array([-a[1], a[0]])
        u, v = (P - s.head_xy) @ a, (P - s.head_xy) @ n
        shank = u > geom.head_len_m + 0.001
        if shank.sum() < 10:
            return
        s.head_xy = s.head_xy + float(np.median(v[shank])) * n + float(np.percentile(u, 1)) * a
        s.axis_deg = math.degrees(math.atan2(a[1], a[0]))


_KERNEL_CACHE: dict = {}
_POOLS: dict = {}


def _pool(threads: int):
    """A thread pool kept for the process (cv2.filter2D releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    n = threads or os.cpu_count() or 4
    if n not in _POOLS:
        _POOLS[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="screw_match")
    return _POOLS[n]


def _kernel_set(geom: ScrewGeometry, angles, mres: float, ss: int) -> dict:
    """All templates (length class, angle) -> kernels; built once per process (~0.1 s)."""
    key = (geom, tuple(angles), mres, ss)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = {(n, a): _kernels(geom.total_m(n), a, geom, mres, ss)
                              for n, _ in geom.lengths_m for a in angles}
    return _KERNEL_CACHE[key]


# -- the detection ---------------------------------------------------------------------------


def screw_cells(rgb, depth, K, T_wc, region=None, cfg: ScrewConfig = ScrewConfig(), trace: dict | None = None):
    """Steps 1-2: (screw cells, tall non-screw cells, height map, grid origin, table plane,
    table brightness) on the top view (``region`` = x0, y0, x1, y1 limits it)."""
    trace = {} if trace is None else trace
    K, T_wc = np.asarray(K, float), np.asarray(T_wc, float)
    plane, _ = fit_table(depth, K, T_wc)
    gray, Hm, origin = top_view(rgb, depth, K, T_wc, plane, cfg, region)
    res = cfg.res_m
    wood = np.isfinite(Hm) & (np.abs(Hm) < 0.001) & np.isfinite(gray)
    if wood.sum() < 1000:
        raise ScrewsNotFound("no free table in view")
    table_gray = float(np.median(gray[wood]))
    low = np.isfinite(Hm) & (Hm < cfg.max_height_m)
    mask = (np.nan_to_num(gray, nan=255) < cfg.dark_fraction * table_gray) & low
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    tall = (np.nan_to_num(Hm, nan=1.0) > cfg.obstacle_height_m).astype(np.uint8)
    # the D405's stereo streaks: 1-2 cells wide, no object is that thin (screws are dark anyway)
    tall = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    # the depth bleeds ~2-3 mm around every screw, more or less from frame to frame; the screws
    # themselves are in the (exact) dark mask, so tall only counts away from them
    halo = int(round(cfg.depth_halo_m / res))
    tall &= cv2.dilate(mask, np.ones((2 * halo + 1, 2 * halo + 1), np.uint8)) == 0
    trace.update(gray=gray, height=Hm, mask=mask, origin=origin, plane=plane)

    return mask, tall, Hm, origin, plane, table_gray


def detect_screws(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    region: tuple[float, float, float, float] | None = None,
    exclude: list[tuple[float, float, float, float]] = (),
    geom: ScrewGeometry = ScrewGeometry(),
    grip: GripperGeometry = GripperGeometry(),
    cfg: ScrewConfig = ScrewConfig(),
    trace: dict | None = None,
) -> ScrewScene:
    """Every screw in view (``region`` = x0, y0, x1, y1 limits the search), with its
    length class and whether it can be grasped. Screws in an ``exclude`` box (x0, y0, x1,
    y1: the kit) are left out. Raises ScrewsNotFound without a table in view."""
    trace = {} if trace is None else trace
    mask, tall, Hm, origin, plane, table_gray = screw_cells(rgb, depth, K, T_wc, region, cfg, trace)
    res = cfg.res_m
    obstacle = tall | (mask > 0)

    # 3. templates, per cluster of screw cells (a few cm each: cheap)
    fmask = mask.astype(np.float32)
    nl, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    angles = np.arange(-180.0, 180.0, cfg.angle_step_deg)
    fits = []  # (score, fill, spill, label, angle, centre_xy)
    f = max(1, int(round(cfg.match_res_m / res)))  # matched on cells f x f as large
    mres = f * res
    pad = int(math.ceil((max(geom.total_m(n) for n, _ in geom.lengths_m) / 2 + 0.008) / res))
    min_area = 0.5 * geom.shank_d_m * geom.lengths_m[0][1] / res**2
    kernels = _kernel_set(geom, angles, mres, f)
    nms = 5 if f == 1 else 3  # local-maximum window: 2.5-3 mm
    pool = _pool(cfg.threads)
    for k in range(1, nl):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        cx, cy = origin + cents[k] * res
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in exclude):
            continue  # the kit, the gearbox: never taken from, not worth matching
        r0, c0 = max(y - pad, 0), max(x - pad, 0)
        r1, c1 = min(y + h + pad, mask.shape[0]), min(x + w + pad, mask.shape[1])
        r1, c1 = r0 + (r1 - r0) // f * f, c0 + (c1 - c0) // f * f
        size = ((c1 - c0) // f, (r1 - r0) // f)
        crop = cv2.resize(fmask[r0:r1, c0:c1], size, interpolation=cv2.INTER_AREA)
        here = cv2.resize((lab[r0:r1, c0:c1] == k).astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        def match(item, crop=crop, here=here, r0=r0, c0=c0):
            (name, ang), (kh, ks, ke, kside) = item
            conv = lambda kern: cv2.filter2D(crop, -1, kern, borderType=cv2.BORDER_CONSTANT)  # noqa: E731
            fill = np.minimum(conv(kh), conv(ks))
            ends, side = conv(ke), conv(kside)
            score = fill - cfg.spill_weight * ends - cfg.side_weight * side
            good = (fill >= cfg.min_fill) & (ends <= cfg.max_spill) & (side <= cfg.max_side) & here
            if not good.any():
                return []
            # local maxima of the score within this template
            peak = (score >= cv2.dilate(score, np.ones((nms, nms), np.uint8))) & good
            return [(float(score[rr, cc]), float(fill[rr, cc]), float(ends[rr, cc]), name, float(ang),
                     origin + np.array([c0 + f * cc + (f - 1) / 2, r0 + f * rr + (f - 1) / 2]) * res)
                    for rr, cc in zip(*np.nonzero(peak), strict=True)]

        for found in pool.map(match, kernels.items()):
            fits.extend(found)

    # 4. greedy: best fit first, no overlap with one already taken
    fits.sort(key=lambda f: -f[0])
    taken: list[tuple[Screw, np.ndarray]] = []
    claimed = np.zeros(mask.shape, bool)  # the bodies taken so far
    for score, fill, spill, name, ang, ctr in fits:
        total = geom.total_m(name)
        a = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        head_xy = ctr - a * total / 2
        s = Screw(name, head_xy, ang, total, fill, spill, score)
        win, o_w = _window(mask.shape, origin, res, ctr, total / 2 + 0.005)
        body = np.zeros(claimed[win].shape, bool)
        for poly in screw_body(s, geom):
            body |= _rasterize(body.shape, o_w, res, poly)
        if (body & claimed[win]).sum() > 0.15 * body.sum():
            continue
        claimed[win] |= body
        taken.append((s, body))
    trace["fits"] = len(fits)

    # 4b. the template's angle is quantised (angle_step_deg) and its position to a cell: refine
    # both from the screw cells inside its outline (principal axis; head end = the end of the
    # cells on the head's side, the shank's cells centre it sideways)
    for s, _ in taken:
        _refine_axis(s, mask, origin, res, geom)

    # 5. measured length, lying flat, room for the fingers, not in the kit
    screws = []
    for s, _ in taken:
        s.length_m = _measure_length(mask, origin, res, s.head_xy, s.axis, s.length_m + 0.008)
        if abs(s.length_m - geom.total_m(s.label)) > cfg.length_tol_m:
            s.reasons.append(f"measured {s.length_m * 1e3:.1f} mm long, {s.label} is "
                             f"{geom.total_m(s.label) * 1e3:.0f}")
        mid = s.point(0.5 * (geom.head_len_m + s.length_m))
        if any(x0 <= mid[0] <= x1 and y0 <= mid[1] <= y1 for x0, y0, x1, y1 in exclude):
            continue
        win, o_w = _window(mask.shape, origin, res, s.point(s.length_m / 2), s.length_m / 2 + 0.02)
        m_w, t_w, H_w = mask[win], tall[win], Hm[win]
        shank = _rasterize(m_w.shape, o_w, res, screw_body(s, geom)[1])
        hs = H_w[shank & np.isfinite(H_w)]
        s.lying_height_m = float(np.percentile(hs, 80)) if hs.size else math.nan
        if not s.lying_height_m <= cfg.max_lying_height_m:
            s.reasons.append(f"not flat on the table (shank {s.lying_height_m * 1e3:.1f} mm high)")
        def grown(m, s=s, shape=m_w.shape, o_w=o_w):
            out = np.zeros(shape, bool)
            for poly in screw_body(s, geom, grow_m=m):
                out |= _rasterize(shape, o_w, res, poly)
            return out
        own = grown(0.001)
        # dark = another screw, up to 1 mm from this one's outline; tall = anything else, beyond
        # the depth's halo around this screw (~2 mm at 30 cm)
        blocked = ((m_w > 0) & ~own) | (t_w & ~grown(cfg.depth_halo_m))
        best = None
        for rank, along in enumerate(grip.grip_points(geom, s.length_m)):
            zone = np.zeros(m_w.shape, bool)
            for poly in finger_zones(s, geom, grip, along):
                zone |= _rasterize(m_w.shape, o_w, res, poly)
            if zone.sum() < 0.8 * 2 * grip.finger_width_m * (grip.approach_open_m / 2 + grip.finger_thick_m
                                                             - geom.head_d_m / 2 + grip.margin_m) / res**2:
                continue  # the strips leave the view
            hit = cv2.morphologyEx((blocked & zone).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(hit)
            area = float(st[1:, cv2.CC_STAT_AREA].max() * res**2) if n > 1 else 0.0
            if best is None or area < best[1] - 1e-9:
                best = (along, area, rank)
            if area <= cfg.max_zone_obstacle_m2:
                break  # the first free one in order of preference
        if best is None:
            s.reasons.append("finger strips leave the view")
        else:
            s.grip_along_m, s.zone_obstacle_m2, s.grip_rank = best
            if best[1] > cfg.max_zone_obstacle_m2:
                s.reasons.append(f"no room for the fingers (a {best[1] * 1e6:.0f} mm2 obstacle in their way)")
        screws.append(s)
    z_table = float(plane[0] * np.mean([s.head_xy[0] for s in screws] or [origin[0]])
                    + plane[1] * np.mean([s.head_xy[1] for s in screws] or [origin[1]]) + plane[2])
    return ScrewScene(screws=screws, z_table=z_table, plane=plane, table_gray=table_gray, grid_origin=origin,
                      res_m=res, screw_mask=mask, obstacle=obstacle)


# -- picture ---------------------------------------------------------------------------------


COLORS = {"M4x18": (0, 200, 255), "M4x25": (255, 120, 0)}


def draw_screws(rgb, K, T_wc, scene: ScrewScene | None, chosen: Screw | None = None,
                geom: ScrewGeometry = ScrewGeometry(), grip: GripperGeometry = GripperGeometry(),
                lines: list[str] = (), error: str = "", slots: list[np.ndarray] = ()) -> np.ndarray:
    """The frame with every screw found (line head -> tip in its class colour, a dot on the
    head; red cross = not graspable), the chosen one's finger strips, the kit slots."""
    img = np.ascontiguousarray(rgb, np.uint8).copy()
    z = (scene.z_table if scene is not None else 0.0) + 0.002

    def px(p):
        uv, _ = project(T_wc, K, np.c_[np.atleast_2d(p), np.full(len(np.atleast_2d(p)), z)])
        return [tuple(int(round(c)) for c in q) for q in uv]

    for s in scene.screws if scene is not None else []:
        a, b = px([s.head_xy, s.point(s.length_m)])
        col = COLORS.get(s.label, (255, 255, 255))
        cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        cv2.circle(img, a, 3, col, -1, cv2.LINE_AA)
        if not s.graspable:
            m = px(s.point(s.length_m / 2))[0]
            cv2.drawMarker(img, m, (255, 40, 40), cv2.MARKER_TILTED_CROSS, 9, 2)
    if chosen is not None and not math.isfinite(chosen.grip_along_m):  # a place, not a grasp
        a, b = px([chosen.head_xy, chosen.point(chosen.length_m)])
        cv2.line(img, a, b, (60, 255, 60), 1, cv2.LINE_AA)
        cv2.circle(img, a, 5, (60, 255, 60), 1, cv2.LINE_AA)
    elif chosen is not None:
        for poly in finger_zones(chosen, geom, grip):
            cv2.polylines(img, [np.array(px(poly), np.int32)], True, (60, 255, 60), 1, cv2.LINE_AA)
        cv2.drawMarker(img, px(chosen.point(chosen.grip_along_m))[0], (60, 255, 60),
                       cv2.MARKER_CROSS, 14, 2)
    for i, p in enumerate(slots):
        q = px(p)[0]
        cv2.circle(img, q, 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (q[0] + 5, q[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    text = [f"NOT FOUND: {error}"] if error else list(lines)
    if scene is not None and not error:
        for name, _ in geom.lengths_m:
            n = sum(s.label == name for s in scene.screws)
            g = sum(s.label == name and s.graspable for s in scene.screws)
            text.append(f"{name}: {n} found, {g} graspable")
    bar = np.full((22 * len(text) + 8, img.shape[1], 3), 30, np.uint8)
    for i, t in enumerate(text):
        cv2.putText(bar, t, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 90, 90) if error else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])
