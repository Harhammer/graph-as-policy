"""Find the screw holder (a 3D-printed block with four vertical M4 holes) and the M4x18 screws
standing in it (z3 screwing, 2026-11-16).

CAD: "Screw Holder 18mm M4 - Part 1.step" (repo root). A 60 x 20 mm block, 17 mm high, with
four blind holes in a row along its long side: 15 mm pitch, centred across it, 4.2 mm
across, 12 mm deep, chamfered to 5.8 mm at the top. An M4x18 in a hole stands on its floor
with 6 mm of shank out and its head (7 mm across, 4 mm high) from 23 to 27 mm above the
table. The block and the screws are both black, so the colour does not tell them apart;
the depth does.

One wrist RGB-D frame:

1. the table: a plane fitted to the depth (screw_vision.fit_table, base frame);
2. a height map: every depth pixel deprojected and binned by its own x/y (``res_m`` cells,
   the highest point per cell) — no parallax at the heads' height, which a top view
   sampled on the table plane would have (25 mm up, 10-20 mm off the optical axis -> mm);
3. the block: cells less than ``body_below_top_m`` below its top (half its height was
   too low once the holder was taped down: the tape over its long edges slopes down to the
   table, 15:45 the blob grew to 69 x 28 mm), the largest blob (closest to ``near_xy`` if
   given); its rectangle (cv2.minAreaRect) gives the centre and the row's direction, and
   the holes sit where the CAD has them;
4. the row is then shifted onto the heads seen in the depth (cells above the heads' lower
   half): across the block by the heads' mean offset, along it by the middle of their
   extent when all four holes hold a screw;
5. per hole: a screw in it or not (the highest point within the head's radius above the
   block's top + ``head_min_m``), and that point's height (the head's top);
6. when the block's top is light (the cream holder, 2026-11-16 15:40) the heads stand out
   dark in the colour image: each head's centre from it (a top view at the heads' height,
   dark cells, the blob's centroid), and the CAD row (15 mm pitch) fitted onto those centres
   (rotation + shift, least squares) replaces the depth's. Not done, and said why in
   ``refined``, for a dark block (the first, black holder), fewer than two heads, or a fit
   that leaves a head more than ``colour_max_residual_m`` off.

All positions in the base frame of ``T_wc``. The D405 smooths the depth over a few mm at
22 cm, so heads 8 mm apart merge into one ridge: the heads' positions come from the CAD
pattern fitted to the block and to that ridge, not from four separate blobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from rotor_vision import _gray, deproject, project, sample
from screw_vision import ScrewsNotFound, fit_table


@dataclass(frozen=True)
class HolderGeometry:
    length_m: float = 0.060
    width_m: float = 0.020
    height_m: float = 0.017
    n_holes: int = 4
    pitch_m: float = 0.015
    hole_depth_m: float = 0.012
    hole_d_m: float = 0.0042
    chamfer_d_m: float = 0.0058
    shank_m: float = 0.018  # M4x18, under the head
    head_h_m: float = 0.004
    head_d_m: float = 0.007

    def head_bottom_m(self) -> float:
        """Above the table, a screw standing on the hole's floor."""
        return self.height_m - self.hole_depth_m + self.shank_m

    def head_top_m(self) -> float:
        return self.head_bottom_m() + self.head_h_m

    def along_m(self, k: int) -> float:
        """Hole k's offset from the block's centre along the row."""
        return (k - (self.n_holes - 1) / 2) * self.pitch_m


@dataclass(frozen=True)
class HolderConfig:
    res_m: float = 0.0005
    max_height_m: float = 0.045  # points above: the fingers, not the holder (27 mm)
    body_below_top_m: float = 0.004  # block cells: higher than its top minus this
    size_tol_m: float = 0.008  # the blob's rectangle vs the CAD (depth smear, a side seen)
    head_min_m: float = 0.005  # above the block's top: a head (they stand 10 mm above it)
    max_shift_m: float = 0.004  # the row moved off the CAD centre by the heads, at most
    near_tol_m: float = 0.06  # near_xy given: the block's centre within this of it
    # 6: the heads in colour
    colour_res_m: float = 0.0002
    dark_fraction: float = 0.5  # head cells: darker than this times the block's top
    colour_max_residual_m: float = 0.001  # a head off the fitted row by more: colour not used
    colour_max_turn_deg: float = 4.0  # the colour row vs the block's axis from the depth


@dataclass
class Hole:
    index: int
    xy: np.ndarray
    occupied: bool
    top_m: float  # highest point within the head's radius, above the table


@dataclass
class HolderScene:
    center_xy: np.ndarray
    axis_deg: float  # along the row, hole 0 -> hole 3
    size_m: tuple[float, float]  # the blob's rectangle (long, short)
    holes: list[Hole]
    z_table: float
    plane: np.ndarray
    row_shift_m: tuple[float, float]  # (along, across) the heads moved the row off the CAD
    warnings: list[str] = field(default_factory=list)
    refined: str = "depth"  # where the holes' x/y come from: "colour ..." or "depth (why)"
    heads_seen: list = field(default_factory=list)  # colour centres per hole (None = not found)

    @property
    def axis(self) -> np.ndarray:
        a = math.radians(self.axis_deg)
        return np.array([math.cos(a), math.sin(a)])

    def table_z_at(self, xy) -> float:
        return float(self.plane[0] * xy[0] + self.plane[1] * xy[1] + self.plane[2])


def height_map(depth, K, T_wc, plane, res_m: float, region=None, max_height_m: float = 0.045):
    """Highest point above the table per ``res_m`` cell (NaN = no point), and the grid's
    origin (x, y of cell [0, 0]; rows <-> y, cols <-> x). Points higher than
    ``max_height_m`` are left out: the fingers hang in view right above the block."""
    depth = np.asarray(depth, np.float32)
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    d = depth.ravel()
    ok = np.isfinite(d) & (d > 0.05)
    W = deproject(T_wc, K, np.stack([u.ravel()[ok], v.ravel()[ok]], 1).astype(float), d[ok])
    h = W[:, 2] - (plane[0] * W[:, 0] + plane[1] * W[:, 1] + plane[2])
    W, h = W[h < max_height_m], h[h < max_height_m]
    if region is not None:
        keep = (W[:, 0] >= region[0]) & (W[:, 1] >= region[1]) & (W[:, 0] <= region[2]) & (W[:, 1] <= region[3])
        W, h = W[keep], h[keep]
    if len(W) < 100:
        raise ScrewsNotFound("no depth in the search region")
    origin = W[:, :2].min(0)
    ij = np.floor((W[:, :2] - origin) / res_m).astype(int)
    shape = (ij[:, 1].max() + 1, ij[:, 0].max() + 1)
    H = np.full(shape, -np.inf)
    np.maximum.at(H, (ij[:, 1], ij[:, 0]), h)
    H[~np.isfinite(H)] = np.nan
    return H, origin


def _cells_xy(mask, origin, res):
    r, c = np.nonzero(mask)
    return origin + (np.c_[c, r] + 0.5) * res


def detect_holder(rgb, depth, K, T_wc, region=None, near_xy=None,
                  geom: HolderGeometry = HolderGeometry(), cfg: HolderConfig = HolderConfig(),
                  table_z: float | None = None) -> HolderScene:
    """``table_z``: the table's height in the arm's model (touched), for the colour step: the
    heads' rays are cut at their height above it, not above the depth's table (which reads
    ~2 mm low from 22 cm; 0.5 mm on the heads, 2026-11-16)."""
    plane, _ = fit_table(depth, K, T_wc)
    H, origin = height_map(depth, K, T_wc, plane, cfg.res_m, region, cfg.max_height_m)
    res = cfg.res_m
    filled = np.nan_to_num(H, nan=0.0)
    # the deprojected points leave single empty cells between them close to the camera
    filled = np.maximum(filled, cv2.morphologyEx(filled.astype(np.float32), cv2.MORPH_CLOSE, np.ones((3, 3))))

    # 3: the block
    body = (filled > geom.height_m - cfg.body_below_top_m).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(body, 8)
    want = np.array([geom.length_m, geom.width_m])
    cands = []
    for i in range(1, n):
        pts = _cells_xy(lab == i, origin, res).astype(np.float32)
        if len(pts) < 50:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        size = np.array(sorted((w, h), reverse=True))
        if np.any(np.abs(size - want) > cfg.size_tol_m):
            continue
        c = np.array([cx, cy])
        if near_xy is not None and np.hypot(*(c - near_xy)) > cfg.near_tol_m:
            continue
        long_deg = ang if w >= h else ang + 90.0
        cost = 0.0 if near_xy is None else float(np.hypot(*(c - near_xy)))
        cands.append((cost, -len(pts), c, long_deg, tuple(size), lab == i))
    if not cands:
        raise ScrewsNotFound(f"no block of {geom.length_m * 1e3:.0f} x {geom.width_m * 1e3:.0f} mm, "
                             f"{geom.height_m * 1e3:.0f} mm high in view")
    _, _, center, long_deg, size, blob = min(cands, key=lambda t: t[:2])
    a = math.radians(long_deg)
    u = np.array([math.cos(a), math.sin(a)])
    # hole 0 at the robot's left (+y), or the far end (+x) when the row runs along y... the
    # row's direction is hole 0 -> 3, i.e. -y (or -x for a row along x)
    if abs(u[1]) >= abs(u[0]):
        u = u if u[1] < 0 else -u
    else:
        u = u if u[0] < 0 else -u
    v = np.array([-u[1], u[0]])
    axis_deg = math.degrees(math.atan2(u[1], u[0]))

    # the block's top as seen (median over the blob without the heads)
    flat = blob & (filled < geom.height_m + cfg.head_min_m)
    top = float(np.median(filled[flat])) if flat.sum() > 20 else float("nan")
    warnings = []
    if not abs(top - geom.height_m) <= 0.004:
        warnings.append(f"block top seen {top * 1e3:.1f} mm above the table, CAD {geom.height_m * 1e3:.0f}")

    # 4: the heads' ridge
    head_thr = geom.height_m + cfg.head_min_m
    heads = (filled > head_thr) & cv2.dilate(blob.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    shift = np.zeros(2)
    if heads.sum() > 20:
        P = _cells_xy(heads, origin, res) - center
        s_along, s_across = P @ u, P @ v
        shift[1] = float(np.median(s_across))
    holes = []

    def holes_at(sh):
        return [center + (geom.along_m(k) + sh[0]) * u + sh[1] * v for k in range(geom.n_holes)]

    def occupancy(xys):
        out = []
        r = geom.head_d_m / 2
        for k, xy in enumerate(xys):
            c0 = np.floor((xy - r - origin) / res).astype(int)
            c1 = np.ceil((xy + r - origin) / res).astype(int)
            win = filled[max(c0[1], 0) : c1[1] + 1, max(c0[0], 0) : c1[0] + 1]
            if win.size == 0:
                out.append(Hole(k, xy, False, float("nan")))
                continue
            rr, cc = np.mgrid[max(c0[1], 0) : max(c0[1], 0) + win.shape[0], max(c0[0], 0) : max(c0[0], 0) + win.shape[1]]
            d = np.hypot(origin[0] + (cc + 0.5) * res - xy[0], origin[1] + (rr + 0.5) * res - xy[1])
            hi = float(np.max(win[d <= r])) if np.any(d <= r) else float("nan")
            out.append(Hole(k, xy, bool(hi > head_thr), hi))
        return out

    holes = occupancy(holes_at(shift))
    if heads.sum() > 20 and all(h.occupied for h in holes):
        # the ridge runs from hole 0's head to hole 3's: its middle is the row's
        lo, hi = np.percentile(s_along, [2, 98])
        shift[0] = float((lo + hi) / 2)
        holes = occupancy(holes_at(shift))
    if np.any(np.abs(shift) > cfg.max_shift_m):
        warnings.append(f"the heads sit {shift[0] * 1e3:+.1f} / {shift[1] * 1e3:+.1f} mm (along/across) off the "
                        "block's centre line: not trusted, CAD centre used")
        shift = np.zeros(2)
        holes = occupancy(holes_at(shift))
    scene = HolderScene(center_xy=center, axis_deg=axis_deg, size_m=size, holes=holes,
                        z_table=float(plane[2] + plane[0] * center[0] + plane[1] * center[1]), plane=plane,
                        row_shift_m=(float(shift[0]), float(shift[1])), warnings=warnings)

    # 6: the heads in colour
    if rgb is not None:
        ray_plane = plane.copy()
        if table_z is not None:
            ray_plane[2] += table_z - scene.z_table
        scene.heads_seen, why = heads_in_colour(rgb, K, T_wc, ray_plane, holes, u, v, center, geom, cfg)
        found = [(k, m) for k, m in enumerate(scene.heads_seen) if m is not None]
        if why:
            scene.refined = f"depth ({why})"
        elif len(found) < 2:
            scene.refined = f"depth (only {len(found)} head found in colour)"
        else:
            q = np.array([[geom.along_m(k), 0.0] for k, _ in found])  # CAD, row frame
            m = np.array([np.r_[(mm - center) @ u, (mm - center) @ v] for _, mm in found])
            qc, mc = q.mean(0), m.mean(0)
            A, B = q - qc, m - mc
            th = math.atan2(np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]), np.sum(A * B))
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            t = mc - R @ qc
            resid = np.hypot(*(m - (q @ R.T + t)).T)
            if resid.max() > cfg.colour_max_residual_m:
                scene.refined = f"depth (a head {resid.max() * 1e3:.1f} mm off the row fitted in colour)"
            elif abs(math.degrees(th)) > cfg.colour_max_turn_deg:
                scene.refined = f"depth (colour row turned {math.degrees(th):+.1f} deg from the block)"
            else:
                u2 = R[0, 0] * u + R[1, 0] * v
                c2 = center + t[0] * u + t[1] * v
                scene.holes = occupancy([c2 + geom.along_m(k) * u2 for k in range(geom.n_holes)])
                scene.axis_deg = math.degrees(math.atan2(u2[1], u2[0]))
                moved = max(np.hypot(*(a.xy - b.xy)) for a, b in zip(scene.holes, holes, strict=True))
                scene.refined = (f"colour ({len(found)} heads, worst {resid.max() * 1e3:.2f} mm off the row, "
                                 f"holes moved {moved * 1e3:.1f} mm from the depth's)")
    return scene


def heads_in_colour(rgb, K, T_wc, plane, holes: list[Hole], u, v, center, geom: HolderGeometry,
                    cfg: HolderConfig) -> tuple[list, str]:
    """Each occupied hole's head centre [x, y] from the colour image (None where not found),
    and why none were looked for ("" = they were)."""
    gray = _gray(rgb)
    zt = lambda xy: plane[0] * xy[..., 0] + plane[1] * xy[..., 1] + plane[2]  # noqa: E731
    # the block's top, away from the heads
    a = np.linspace(-geom.length_m / 2 + 0.003, geom.length_m / 2 - 0.003, 40)
    b = np.linspace(-geom.width_m / 2 + 0.003, geom.width_m / 2 - 0.003, 9)
    A, B = np.meshgrid(a, b)
    P = center + A.ravel()[:, None] * u + B.ravel()[:, None] * v
    far = np.min([np.hypot(*(P - h.xy).T) for h in holes], axis=0) > geom.head_d_m / 2 + 0.002
    P = P[far]
    uv, _ = project(T_wc, K, np.c_[P, zt(P) + geom.height_m])
    top = float(np.nanmedian(sample(gray, uv)))
    core = []
    for h in holes:
        if h.occupied:
            uvh, _ = project(T_wc, K, np.r_[h.xy, zt(h.xy) + geom.head_top_m()][None])
            core.append(float(sample(gray, uvh)[0]))
    if not core:
        return [None] * len(holes), "no head"
    if not np.nanmedian(core) < cfg.dark_fraction * top:
        return [None] * len(holes), f"block top {top:.0f} gray, heads {np.nanmedian(core):.0f}: no contrast"
    r = geom.head_d_m / 2 + 0.0025
    n = int(round(2 * r / cfg.colour_res_m)) + 1
    ax = np.linspace(-r, r, n)
    out = []
    for h in holes:
        if not h.occupied:
            out.append(None)
            continue
        X, Y = np.meshgrid(h.xy[0] + ax, h.xy[1] + ax)
        W = np.c_[X.ravel(), Y.ravel()]
        uvw, _ = project(T_wc, K, np.c_[W, zt(W) + geom.head_top_m()])
        dark = (sample(gray, uvw).reshape(X.shape) < cfg.dark_fraction * top).astype(np.uint8)
        k, lab, st, ce = cv2.connectedComponentsWithStats(dark, 8)
        area = np.pi * (geom.head_d_m / 2) ** 2 / cfg.colour_res_m**2
        best = None
        for i in range(1, k):
            if not 0.5 * area <= st[i, cv2.CC_STAT_AREA] <= 2.0 * area:
                continue
            d = math.hypot(ce[i][0] - (n - 1) / 2, ce[i][1] - (n - 1) / 2)
            if best is None or d < best[0]:
                best = (d, ce[i])
        out.append(None if best is None else
                   np.array([h.xy[0] - r + best[1][0] * cfg.colour_res_m, h.xy[1] - r + best[1][1] * cfg.colour_res_m]))
    return out, ""


def draw_holder(rgb, K, T_wc, scene: HolderScene | None, chosen: int | None = None,
                geom: HolderGeometry = HolderGeometry(), lines=(), error: str = "") -> np.ndarray:
    """The frame with the block's outline (on its top), each hole's head circle (green =
    screw in it, red = empty), its number, the chosen one thicker."""
    img = np.ascontiguousarray(rgb, np.uint8).copy()

    def px(xy, z):
        xy = np.atleast_2d(xy)
        uv, _ = project(T_wc, K, np.c_[xy, np.full(len(xy), z)])
        return np.round(uv).astype(np.int32)

    if scene is not None:
        u = scene.axis
        v = np.array([-u[1], u[0]])
        L, W = geom.length_m / 2, geom.width_m / 2
        corners = [scene.center_xy + a * L * u + b * W * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        cv2.polylines(img, [px(corners, scene.z_table + geom.height_m)], True, (255, 200, 0), 1, cv2.LINE_AA)
        t = np.linspace(0, 2 * np.pi, 24)
        for h in scene.holes:
            col = (60, 255, 60) if h.occupied else (255, 60, 60)
            ring = h.xy + geom.head_d_m / 2 * np.c_[np.cos(t), np.sin(t)]
            z = scene.z_table + (geom.head_top_m() if h.occupied else geom.height_m)
            cv2.polylines(img, [px(ring, z)], True, col, 2 if h.index == chosen else 1, cv2.LINE_AA)
            c = px(h.xy, z)[0]
            cv2.putText(img, str(h.index), (int(c[0]) - 4, int(c[1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                        cv2.LINE_AA)
    text = [f"NOT FOUND: {error}"] if error else list(lines)
    if scene is not None and not error:
        text.append("holes: " + "  ".join(
            f"{h.index}:{'screw' if h.occupied else 'empty'} {h.top_m * 1e3:.0f}mm" for h in scene.holes))
        text.append(f"holes from {scene.refined}")
        text += scene.warnings
    bar = np.full((22 * max(len(text), 1) + 8, img.shape[1], 3), 30, np.uint8)
    for i, t in enumerate(text):
        cv2.putText(bar, t, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 90, 90) if error else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])
