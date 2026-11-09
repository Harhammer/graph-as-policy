"""Eye-in-hand calibration of the wrist D405: the flange -> colour-optical transform.

Commands no motion. You hand-guide the arm over a ChArUco board taped to the table; at
each pose the tool reads the joints and the robot model's own flange pose over FCI, grabs
a colour + depth frame, and fits the board. ``cv2.calibrateHandEye`` then solves
``flange_T_camera`` with five methods.

Every result carries its own checks, because a wrong transform does not fail — it shifts
every grasp:

- **board layout**: the detected MARKERS are fitted to where this board definition puts
  them. A wrong ``legacy_pattern`` or square count still yields ChArUco corners with a
  tiny reprojection error, just in the wrong place; the marker fit exposes it (measured
  on the real board: 1.2 px with the right layout, 97 px with the wrong one).
- **scale**: the board fit's distance against the depth camera's, which catches a wrong
  ``square_m`` (a printer's rescale).
- **stillness**: the joints are read twice; a pose is rejected if the arm moved.
- **consistency (the result's positive control)**: the board is fixed, so
  ``base_T_flange_i @ X @ camera_T_board_i`` must land on the same place for every sample.
  Its spread in mm is the calibration's real-world error.

Run it with the bridge stopped (the camera allows one reader):

    z3-gap-handeye                      # capture, then solve
    z3-gap-handeye --from-dir DIR       # re-solve saved samples, e.g. with other board params
    z3-gap-handeye --from-dir DIR --fit-joints 2 5 6 --fit-scale
                                        # ... and fit joint zero offsets + board scale too

**Fitting joint offsets** (``--fit-joints``): the closed-form solve trusts the robot's
kinematics. When the per-pose board spread stays at several mm however many samples are
added, the arm's reported joint angles may be off; a reprojection bundle adjustment then
fits zero offsets of the chosen joints together with the camera transform and the board
pose(s). The leave-one-out error says whether that is real structure or over-fitting.
Joints 1 and 7 cannot be fitted this way (their offsets look like a board / camera
rotation). The result is only valid together with its offsets: the bridge must apply them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from z3_gap_bridge.realsense_client import WRIST_D405_SERIAL, RealSenseFrame, undistort_pixels

log = logging.getLogger(__name__)

METHODS = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")
#: ANDREFF is reported but never chosen or used for agreement: on the first real capture
#: (10 poses) it sat 17 mm off the other four, which agreed within 0.5 mm.
ROBUST = ("TSAI", "PARK", "HORAUD", "DANIILIDIS")
#: Pass thresholds on the jackknife standard error of the result itself.
MAX_SE_MM = 1.5
MAX_SE_DEG = 0.3
MIN_TILT_DEG = 25.0


@dataclass
class HandEyeConfig:
    """Calibrate the wrist D405 against the flange by hand-guiding over a ChArUco board."""

    out_dir: Path | None = None
    """Where samples and the result go. Default: ~/z3_handeye/<timestamp>."""
    from_dir: tuple[Path, ...] = ()
    """Skip capture and re-solve the samples saved in these directories (combinable when
    the board did not move between them — the report shows where each put it)."""
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """Offline only: added to the recorded joint angles before forward kinematics. 2026-11-06:
    +0.98 deg on joint 4 (0-indexed, Franka's A5) cut the per-pose scatter 5.4 -> 3.2 mm,
    fitted independently on two sessions (+0.89 / +0.98) and cross-validated."""
    fit_joints: tuple[int, ...] = ()
    """Offline only: fit zero offsets of these joints (1-indexed, e.g. 2 5 6) by bundle
    adjustment on top of ``joint_offsets_deg``. 2026-11-09 on the two 11-06 sessions:
    joints 2/5/6 at +1.8/+1.0/+0.5 deg took the held-out error from 5.8 to 1.1 mm."""
    fit_scale: bool = False
    """Offline only: also fit one scale factor of the board (a print not exactly square_m)."""

    squares_x: int = 11
    squares_y: int = 8
    square_m: float = 0.0212
    """Measured on the print, not the PDF value (21.2 mm with calipers, 2026-11-09; the
    2026-11-06 fits with 21.0 needed a 1.0099 scale factor to agree with it)."""
    marker_m: float = 0.0155
    dictionary: str = "DICT_4X4_50"
    legacy_pattern: bool = True
    """OpenCV >= 4.6 lays out even-row boards differently; True matches the z3 board."""

    min_corners: int = 12
    max_reproj_px: float = 1.5
    max_marker_fit_px: float = 5.0
    max_depth_scale_error: float = 0.04
    min_samples: int = 10

    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    robot_ip: str = "192.168.98.121"
    camera_serial: str = WRIST_D405_SERIAL
    camera_width: int = 848
    camera_height: int = 480
    camera_fps: int = 30


# -- board ----------------------------------------------------------------------------


def make_board(cfg: HandEyeConfig):
    """The ChArUco board as OpenCV models it."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y), cfg.square_m, cfg.marker_m, dictionary
    )
    board.setLegacyPattern(cfg.legacy_pattern)
    return board


def _pnp(obj: np.ndarray, img: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, float]:
    """Pose of ``obj`` points seen at undistorted pixels ``img``; returns (4x4, RMS px)."""
    import cv2

    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.ravel()
    return pose, rms


def detect_board(frame: RealSenseFrame, board, cfg: HandEyeConfig, detector=None) -> dict[str, Any]:
    """Fit the board in one frame. Returns the pose and every check's value.

    ``ok`` is False with a ``reason`` when the frame must not be used. The raw detections
    (``charuco_px``/``charuco_ids``/``marker_px``/``marker_ids``) are included for drawing.
    Pass a reused ``cv2.aruco.CharucoDetector`` when calling at frame rate.
    """
    import cv2

    k = np.asarray(frame.intrinsics, dtype=np.float64)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    detector = detector or cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)
    n = 0 if ch_ids is None else len(ch_ids)
    out: dict[str, Any] = {
        "ok": False,
        "corners": n,
        "charuco_px": ch_corners,
        "charuco_ids": ch_ids,
        "marker_px": mk_corners,
        "marker_ids": mk_ids,
    }
    if n < cfg.min_corners:
        out["reason"] = f"{n} board corners < {cfg.min_corners} — more of the board in view"
        return out

    undistort = lambda px: undistort_pixels(  # noqa: E731
        px, k, frame.distortion, frame.distortion_model
    )
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    obj = obj.reshape(-1, 3).astype(np.float64)
    img_raw = img.reshape(-1, 2).astype(np.float64)
    camera_T_board, rms = _pnp(obj, undistort(img_raw), k)
    out.update(camera_T_board=camera_T_board, reproj_px=rms)

    # Layout check: markers fitted where THIS board definition puts their ids.
    board_ids = list(np.asarray(board.getIds()).ravel())
    board_objs = board.getObjPoints()
    m_obj = np.concatenate(
        [np.asarray(board_objs[board_ids.index(int(i))]).reshape(4, 3) for i in mk_ids.ravel()]
    ).astype(np.float64)
    m_img = undistort(np.concatenate([c.reshape(4, 2) for c in mk_corners]).astype(np.float64))
    _, marker_rms = _pnp(m_obj, m_img, k)
    out["marker_fit_px"] = marker_rms

    # Scale check: depth camera vs board fit, at the corner pixels (depth is aligned to
    # the raw colour image, so it is sampled at the raw pixel positions).
    z_fit = (camera_T_board @ np.c_[obj, np.ones(len(obj))].T)[2]
    u = np.clip(np.round(img_raw[:, 0]).astype(int), 0, frame.depth_m.shape[1] - 1)
    v = np.clip(np.round(img_raw[:, 1]).astype(int), 0, frame.depth_m.shape[0] - 1)
    z_depth = frame.depth_m[v, u]
    valid = z_depth > 0
    out["depth_scale"] = (
        float(np.median(z_depth[valid] / z_fit[valid])) if valid.sum() >= 4 else None
    )

    if rms > cfg.max_reproj_px:
        out["reason"] = f"reprojection {rms:.2f} px > {cfg.max_reproj_px} — blur or glare?"
    elif marker_rms > cfg.max_marker_fit_px:
        out["reason"] = (
            f"marker fit {marker_rms:.1f} px: the board does not match this definition "
            "(squares_x/y, legacy_pattern?)"
        )
    elif out["depth_scale"] is not None and abs(out["depth_scale"] - 1) > cfg.max_depth_scale_error:
        out["reason"] = (
            f"depth says the board is {out['depth_scale']:.3f}x as far as the fit — "
            "is square_m/marker_m measured on the print?"
        )
    else:
        out["ok"] = True
    return out


# -- solve ----------------------------------------------------------------------------


def _rot_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def _mean_rotation(rs: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(rs, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def board_spread(
    base_T_flange: list[np.ndarray], camera_T_board: list[np.ndarray], flange_T_camera: np.ndarray
) -> dict[str, float]:
    """Where each sample puts the (fixed) board in the base frame, and how much that varies."""
    boards = [b @ flange_T_camera @ c for b, c in zip(base_T_flange, camera_T_board, strict=True)]
    t = np.array([p[:3, 3] for p in boards])
    dev = np.linalg.norm(t - t.mean(axis=0), axis=1)
    r_mean = _mean_rotation([p[:3, :3] for p in boards])
    rot_dev = [_rot_angle_deg(r_mean.T @ p[:3, :3]) for p in boards]
    return {
        "trans_rms_mm": float(np.sqrt(np.mean(dev**2)) * 1e3),
        "trans_max_mm": float(dev.max() * 1e3),
        "rot_rms_deg": float(np.sqrt(np.mean(np.square(rot_dev)))),
        "rot_max_deg": float(np.max(rot_dev)),
        "board_xyz_m": t.mean(axis=0).round(4).tolist(),
    }


def solve(base_T_flange: list[np.ndarray], camera_T_board: list[np.ndarray]) -> dict[str, Any]:
    """Solve flange_T_camera with every method; pick the most self-consistent one."""
    import cv2

    rg = [p[:3, :3] for p in base_T_flange]
    tg = [p[:3, 3] for p in base_T_flange]
    rt = [p[:3, :3] for p in camera_T_board]
    tt = [p[:3, 3] for p in camera_T_board]
    results: dict[str, Any] = {}
    for name in METHODS:
        r, t = cv2.calibrateHandEye(rg, tg, rt, tt, method=getattr(cv2, f"CALIB_HAND_EYE_{name}"))
        x = np.eye(4)
        x[:3, :3] = r
        x[:3, 3] = t.ravel()
        results[name] = {"flange_T_camera": x, **board_spread(base_T_flange, camera_T_board, x)}
    best = min(ROBUST, key=lambda m: results[m]["trans_rms_mm"])
    xb = results[best]["flange_T_camera"]
    agree_mm = max(
        np.linalg.norm(results[m]["flange_T_camera"][:3, 3] - xb[:3, 3]) * 1e3 for m in ROBUST
    )
    agree_deg = max(
        _rot_angle_deg(xb[:3, :3].T @ results[m]["flange_T_camera"][:3, :3]) for m in ROBUST
    )
    return {"best": best, "methods": results, "agree_mm": agree_mm, "agree_deg": agree_deg}


def jackknife(fit, n: int, full: np.ndarray) -> tuple[float, float]:
    """Standard error of a transform by leave-one-out: (mm, deg). ``fit(keep)`` -> 4x4."""
    loo = [fit([j for j in range(n) if j != k]) for k in range(n)]
    t = np.array([x[:3, 3] for x in loo])
    se_t = np.sqrt((n - 1) / n * np.sum(np.linalg.norm(t - t.mean(axis=0), axis=1) ** 2))
    se_r = np.sqrt(
        (n - 1) / n * np.sum([_rot_angle_deg(full[:3, :3].T @ x[:3, :3]) ** 2 for x in loo])
    )
    return float(se_t * 1e3), float(se_r)


def tilt_coverage_deg(base_T_flange: list[np.ndarray]) -> np.ndarray:
    """RMS relative rotation between poses about each principal axis, largest first.

    Hand-eye needs rotation about at least two well-separated axes; turning only the
    last joint (rotation about the flange's own z) leaves the camera's offset along
    that axis undetermined. Measured 2026-11-06: 58 / 22 / 16 deg gave +-3 mm.
    """
    rots = [p[:3, :3] for p in base_T_flange]
    m = np.zeros((3, 3))
    pairs = 0
    for i, a in enumerate(rots):
        for b in rots[i + 1 :]:
            rel = a.T @ b
            angle = np.radians(_rot_angle_deg(rel))
            if angle < 1e-9:
                continue
            axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
            axis /= np.linalg.norm(axis) or 1.0
            m += np.outer(axis, axis) * angle**2
            pairs += 1
    if not pairs:
        return np.zeros(3)
    eig = np.clip(np.linalg.eigvalsh(m / pairs), 0.0, None)  # rounding can dip below 0
    return np.degrees(np.sqrt(np.sort(eig)[::-1]))


# -- refinement: joint offsets + board scale by bundle adjustment --------------------------


def _pose6(p: np.ndarray) -> np.ndarray:
    import cv2

    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(p[:3], dtype=np.float64))[0]
    t[:3, 3] = p[3:6]
    return t


def _vec6(t: np.ndarray) -> list[float]:
    import cv2

    return [*cv2.Rodrigues(np.ascontiguousarray(t[:3, :3]))[0].ravel(), *t[:3, 3]]


def refine(
    samples: list[dict[str, Any]],
    x0: np.ndarray,
    joint_idx: tuple[int, ...],
    fit_scale: bool,
    base_offsets: np.ndarray,
    leave_one_out: bool = True,
) -> dict[str, Any]:
    """Bundle-adjust the hand-eye chain.

    ``samples``: per usable pose ``{"q", "obj", "img", "K", "session"}`` — joints, board
    corner object points [m] and their undistorted pixels, intrinsics, which capture
    session (each session has its own board pose). Unknowns: ``flange_T_camera``, one board
    pose per session, a zero offset per joint in ``joint_idx`` (0-indexed; 0 and 6 are
    refused), optionally one board scale. Minimises corner reprojection error.

    Returns the refined transform, the offsets [rad] (on top of ``base_offsets``), the
    scale, reprojection rms [px], the per-pose 3D board error [mm] and its leave-one-out
    version (each pose located by a model fitted without it).
    """
    from scipy.optimize import least_squares

    from z3_gap_bridge.panda_fk import flange_pose

    for j in joint_idx:
        if j in (0, 6):
            raise ValueError("joint 1 and joint 7 offsets cannot be fitted from camera data")
    sessions = sorted({s["session"] for s in samples})
    n_j = len(joint_idx)

    def unpack(p):
        x = _pose6(p[:6])
        boards = {k: _pose6(p[6 + 6 * i: 12 + 6 * i]) for i, k in enumerate(sessions)}
        i = 6 + 6 * len(sessions)
        dq = base_offsets.copy()
        for j, v in zip(joint_idx, p[i:i + n_j], strict=True):
            dq[j] += v
        scale = p[i + n_j] if fit_scale else 1.0
        return x, boards, dq, scale

    def residuals(p, subset):
        x, boards, dq, scale = unpack(p)
        out = []
        for s in subset:
            cam = flange_pose(s["q"] + dq) @ x
            pts = (np.linalg.inv(cam) @ boards[s["session"]]
                   @ np.c_[s["obj"] * scale, np.ones(len(s["obj"]))].T)[:3]
            k = s["K"]
            uv = np.stack([k[0, 0] * pts[0] / pts[2] + k[0, 2], k[1, 1] * pts[1] / pts[2] + k[1, 2]], 1)
            out.append((uv - s["img"]).ravel())
        return np.concatenate(out)

    def board_error(p, subset):
        """3D rms [mm] per pose: the board located from that pose's own image through the
        fitted chain, vs the fitted board pose."""
        import cv2

        x, boards, dq, scale = unpack(p)
        errs = []
        for s in subset:
            ok, rv, tv = cv2.solvePnP(s["obj"] * scale, s["img"], s["K"], None, flags=cv2.SOLVEPNP_SQPNP)
            rv, tv = cv2.solvePnPRefineLM(s["obj"] * scale, s["img"], s["K"], None, rv, tv)
            seen = flange_pose(s["q"] + dq) @ x @ _pose6([*rv.ravel(), *tv.ravel()])
            pts = np.c_[s["obj"] * scale, np.ones(len(s["obj"]))].T
            d = (seen @ pts)[:3] - (boards[s["session"]] @ pts)[:3]
            errs.append(float(np.sqrt(np.mean(np.sum(d**2, axis=0))) * 1e3))
        return np.array(errs)

    # start: the closed-form camera, the mean board pose per session, zero offsets
    p0 = list(_vec6(x0))
    for k in sessions:
        stack = [flange_pose(s["q"] + base_offsets) @ x0 @ s["camera_T_board"]
                 for s in samples if s["session"] == k]
        mean = np.eye(4)
        mean[:3, :3] = _mean_rotation([b[:3, :3] for b in stack])
        mean[:3, 3] = np.mean([b[:3, 3] for b in stack], axis=0)
        p0 += _vec6(mean)
    p0 += [0.0] * n_j + ([1.0] if fit_scale else [])
    p0 = np.asarray(p0, dtype=np.float64)

    def fit(subset):
        return least_squares(residuals, p0, args=(subset,), x_scale="jac", method="lm").x

    p = fit(samples)
    x, _, dq, scale = unpack(p)
    out = {
        "flange_T_camera": x,
        "joint_offsets_rad": dq,
        "fitted_offsets_rad": {j: float(dq[j] - base_offsets[j]) for j in joint_idx},
        "board_scale": float(scale),
        "reproj_rms_px": float(np.sqrt(np.mean(residuals(p, samples) ** 2))),
        "board_error_mm": board_error(p, samples),
    }
    if leave_one_out:
        loo = []
        for k in range(len(samples)):
            rest = [s for i, s in enumerate(samples) if i != k]
            if len({s["session"] for s in rest}) < len(sessions):
                continue
            loo.append(board_error(fit(rest), [samples[k]])[0])
        out["loo_error_mm"] = np.asarray(loo)
    return out


# -- samples on disk ------------------------------------------------------------------


def save_sample(directory: Path, index: int, frame: RealSenseFrame, q, base_T_flange) -> Path:
    """One npz per pose: everything needed to re-detect and re-solve offline."""
    path = directory / f"sample_{index:03d}.npz"
    np.savez_compressed(
        path,
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        intrinsics=frame.intrinsics,
        distortion=frame.distortion,
        distortion_model=np.array(frame.distortion_model),
        q=np.asarray(q, dtype=np.float64),
        base_T_flange=np.asarray(base_T_flange, dtype=np.float64),
    )
    return path


def load_samples(directory: Path) -> list[tuple[RealSenseFrame, np.ndarray, np.ndarray]]:
    """(frame, base_T_flange, q) for every saved sample, in capture order."""
    out = []
    for path in sorted(Path(directory).glob("sample_*.npz")):
        with np.load(path) as d:
            frame = RealSenseFrame(
                rgb=d["rgb"],
                depth_m=d["depth_m"],
                intrinsics=d["intrinsics"],
                frame_number=0,
                hw_timestamp_ms=0.0,
                timestamp_domain="",
                host_time_s=0.0,
                distortion_model=str(d["distortion_model"]),
                distortion=d["distortion"],
            )
            out.append((frame, d["base_T_flange"], d["q"]))
    return out


def solve_directory(cfg: HandEyeConfig, directories: tuple[Path, ...]) -> int:
    """Detect, check, solve and report for saved samples; 0 when usable."""
    from z3_gap_bridge.panda_fk import flange_pose

    board = make_board(cfg)
    offsets = np.radians(np.asarray(cfg.joint_offsets_deg, dtype=np.float64))
    if offsets.shape != (7,):
        raise ValueError("joint_offsets_deg needs 7 values")
    if np.any(offsets):
        print(f"joint offsets applied (deg): {np.round(np.degrees(offsets), 3).tolist()}")
    flanges, boards, refine_samples = [], [], []
    for si, directory in enumerate(directories):
        before = len(flanges)
        for i, (frame, base_T_flange, q) in enumerate(load_samples(directory)):
            det = detect_board(frame, board, cfg)
            if not det["ok"]:
                print(f"  {directory.name}/sample_{i:03d} skipped: {det['reason']}")
                continue
            # The robot model's recorded flange pose, unless the joint angles are corrected.
            flanges.append(flange_pose(q + offsets) if np.any(offsets) else base_T_flange)
            boards.append(det["camera_T_board"])
            obj, img = board.matchImagePoints(det["charuco_px"], det["charuco_ids"])
            k = np.asarray(frame.intrinsics, dtype=np.float64)
            refine_samples.append({
                "q": np.asarray(q, dtype=np.float64), "session": si, "K": k,
                "obj": obj.reshape(-1, 3).astype(np.float64),
                "img": undistort_pixels(img.reshape(-1, 2).astype(np.float64), k,
                                        frame.distortion, frame.distortion_model),
                "camera_T_board": det["camera_T_board"],
            })
        if len(directories) > 1 and len(flanges) > before:
            alone = solve(flanges[before:], boards[before:])
            xyz = alone["methods"][alone["best"]]["board_xyz_m"]
            print(
                f"  {directory.name}: {len(flanges) - before} samples, board at "
                f"{np.round(np.asarray(xyz) * 1e3, 1).tolist()} mm"
            )
    if len(flanges) < cfg.min_samples:
        print(f"only {len(flanges)} usable samples, need {cfg.min_samples}")
        return 1
    out = cfg.out_dir or (
        directories[0]
        if len(directories) == 1
        else Path.home() / "z3_handeye" / f"combined_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    refined = None
    if cfg.fit_joints or cfg.fit_scale:
        joint_idx = tuple(int(j) - 1 for j in cfg.fit_joints)
        x0 = solve(flanges, boards)
        x0 = x0["methods"][x0["best"]]["flange_T_camera"]
        print("\nfitting " + (f"zero offsets of joints {list(cfg.fit_joints)}" if cfg.fit_joints else "")
              + (" + board scale" if cfg.fit_scale else "") + " by bundle adjustment ...")
        refined = refine(refine_samples, x0, joint_idx, cfg.fit_scale, offsets)
    return _report(cfg, out, flanges, boards, refined)[0]


def _report(cfg, directory: Path, flanges, boards, refined=None) -> tuple[int, str]:
    """Solve, judge, print, save. Returns (exit code, one-line verdict)."""
    result = solve(flanges, boards)
    best_name = result["best"]
    best = result["methods"][best_name]
    x = best["flange_T_camera"]
    n = len(flanges)

    def fit(keep: list[int]) -> np.ndarray:
        sub = solve([flanges[i] for i in keep], [boards[i] for i in keep])
        return sub["methods"][best_name]["flange_T_camera"]

    se_mm, se_deg = jackknife(fit, n, x)
    tilt = tilt_coverage_deg(flanges)

    print(
        f"\n{n} samples; rotation between poses about the three principal axes: "
        + " / ".join(f"{a:.0f}" for a in tilt)
        + " deg rms"
    )
    print(f"{'method':11s} {'board spread rms/max':>22s} {'rot rms/max':>16s}")
    for name, r in result["methods"].items():
        mark = "  <- result" if name == best_name else ("  (ignored)" if name not in ROBUST else "")
        print(
            f"{name:11s} {r['trans_rms_mm']:9.2f} / {r['trans_max_mm']:5.2f} mm "
            f"{r['rot_rms_deg']:7.2f} / {r['rot_max_deg']:4.2f} deg{mark}"
        )
    print(
        f"robust methods agree within {result['agree_mm']:.1f} mm / {result['agree_deg']:.2f} deg; "
        f"camera {np.linalg.norm(x[:3, 3]) * 1e3:.0f} mm from the flange"
    )
    print(
        f"UNCERTAINTY of the result (jackknife): {se_mm:.2f} mm, {se_deg:.2f} deg "
        f"(pass: <= {MAX_SE_MM} mm, <= {MAX_SE_DEG} deg)"
    )
    print(
        f"(board spread = one pose's localisation error, {best['trans_rms_mm']:.1f} mm rms; "
        "it does not shrink with more samples — the uncertainty does)"
    )
    print("flange_T_camera =\n" + np.array2string(x, precision=4, suppress_small=True))

    fitted_offsets_deg = None
    if refined is not None:
        dq_total = np.degrees(refined["joint_offsets_rad"])
        err, loo = refined["board_error_mm"], refined.get("loo_error_mm")
        print(
            f"\nREFINED (bundle adjustment): reprojection {refined['reproj_rms_px']:.2f} px, "
            f"board error {np.sqrt(np.mean(err**2)):.2f} mm rms / {err.max():.2f} max"
            + (f", held-out {np.sqrt(np.mean(loo**2)):.2f} mm rms" if loo is not None and len(loo) else "")
            + f"   (closed form above: {best['trans_rms_mm']:.2f} mm rms)"
        )
        print("  joint zero offsets, total (add to the reported angles): "
              + ", ".join(f"j{j + 1} {dq_total[j]:+.2f} deg" for j in range(7) if abs(dq_total[j]) > 1e-9))
        if cfg.fit_scale:
            print(f"  board scale {refined['board_scale']:.4f} (square {cfg.square_m * refined['board_scale'] * 1e3:.2f} mm)")
        print("  refined flange_T_camera =\n" + np.array2string(refined["flange_T_camera"], precision=4, suppress_small=True))
        print("  NOTE: this transform is only right together with the offsets — the bridge must apply them.")
        x = refined["flange_T_camera"]
        fitted_offsets_deg = dq_total.tolist()

    verdict = []
    if tilt[-1] < MIN_TILT_DEG:
        verdict.append(
            f"too little rotation about a second axis ({tilt[1]:.0f}/{tilt[2]:.0f} deg) — tilt "
            "the wrist forward/back AND left/right; turning only the last joint does not help"
        )
    if refined is not None:
        loo = refined.get("loo_error_mm")
        if loo is not None and len(loo) and np.sqrt(np.mean(loo**2)) > MAX_SE_MM:
            verdict.append(f"held-out error of the refined model {np.sqrt(np.mean(loo**2)):.1f} mm — add poses")
    elif se_mm > MAX_SE_MM or se_deg > MAX_SE_DEG:
        verdict.append(f"uncertainty {se_mm:.1f} mm / {se_deg:.2f} deg — add tilted poses")
    if result["agree_mm"] > 3.0:
        verdict.append(f"robust methods disagree by {result['agree_mm']:.1f} mm")
    if np.linalg.norm(x[:3, 3]) > 0.25:
        verdict.append("camera > 25 cm from the flange — physically implausible")

    out = directory / "flange_T_camera.npy"
    np.save(out, x)
    report = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "samples": n,
        "tilt_coverage_deg": tilt.tolist(),
        "joint_offsets_deg": fitted_offsets_deg or list(cfg.joint_offsets_deg),
        "refined": None if refined is None else {
            "fit_joints": list(cfg.fit_joints),
            "board_scale": refined["board_scale"],
            "reproj_rms_px": refined["reproj_rms_px"],
            "board_error_rms_mm": float(np.sqrt(np.mean(refined["board_error_mm"] ** 2))),
            "loo_error_rms_mm": (float(np.sqrt(np.mean(refined["loo_error_mm"] ** 2)))
                                 if len(refined.get("loo_error_mm", [])) else None),
            "flange_T_camera": refined["flange_T_camera"].tolist(),
        },
        "sources": [str(d) for d in cfg.from_dir],
        "best": best_name,
        "jackknife_se_mm": se_mm,
        "jackknife_se_deg": se_deg,
        "agree_mm": result["agree_mm"],
        "agree_deg": result["agree_deg"],
        "methods": {
            m: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for m, r in result["methods"].items()
        },
        "verdict": verdict or ["ok"],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}\n      {directory / 'report.json'}")
    headline = f"+-{se_mm:.1f} mm / {se_deg:.2f} deg"
    if verdict:
        print("NOT GOOD ENOUGH: " + "; ".join(verdict))
        return 1, f"NOT GOOD ({headline}): {verdict[0]}"
    if refined is not None:
        print(f"GOOD. Use it with:  z3-gap-bridge --kinematics {directory} ...   (offsets + this transform)")
    else:
        print(f"GOOD. Use it with:  z3-gap-bridge --flange-t-camera {out} ...")
    return 0, f"GOOD ({headline}) -> {out.name}"


# -- capture --------------------------------------------------------------------------


WINDOW = "z3 hand-eye"
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (60, 60, 230), (0, 220, 255), (240, 240, 240)


def draw_preview(
    frame: RealSenseFrame, det: dict[str, Any], lines: list[tuple[str, tuple]], board=None
):
    """Greyscale frame with the board detection and a status bar drawn over it (BGR)."""
    import cv2

    img = cv2.cvtColor(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
    if det.get("marker_px") is not None and len(det["marker_px"]):
        cv2.polylines(
            img, [np.int32(c.reshape(-1, 2)) for c in det["marker_px"]], True, (255, 160, 0), 1
        )
    if det.get("charuco_ids") is not None and len(det["charuco_ids"]):
        colour = GREEN if det["ok"] else RED
        for u, v in det["charuco_px"].reshape(-1, 2):
            cv2.circle(img, (int(round(u)), int(round(v))), 3, colour, -1, cv2.LINE_AA)
    if "camera_T_board" in det:
        pose = det["camera_T_board"]
        # Axes at the centre of the VISIBLE corners: the board origin is often off-image.
        centre = np.zeros(3)
        if board is not None and det.get("charuco_ids") is not None:
            centre = np.asarray(board.getChessboardCorners())[det["charuco_ids"].ravel()].mean(
                axis=0
            )
        rvec = cv2.Rodrigues(pose[:3, :3])[0]
        tvec = pose[:3, :3] @ centre + pose[:3, 3]
        k = np.asarray(frame.intrinsics, dtype=np.float64)
        # The colour stream's coefficients are OpenCV's own (see undistort_pixels).
        cv2.drawFrameAxes(img, k, np.asarray(frame.distortion), rvec, tvec, 0.04, 2)
    bar = 24 * len(lines) + 8
    cv2.rectangle(img, (0, 0), (img.shape[1], bar), (30, 30, 30), -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(
            img, text, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA
        )
    return img


def capture(cfg: HandEyeConfig) -> int:
    """Live window over FCI + the D405: capture by key, then solve."""
    import cv2

    from z3_gap_bridge.bridge import FK_TOLERANCE_DEG, FK_TOLERANCE_M, fk_error
    from z3_gap_bridge.franky_client import FrankyClient
    from z3_gap_bridge.panda_fk import flange_pose
    from z3_gap_bridge.realsense_client import RealSenseClient

    directory = cfg.out_dir or Path.home() / "z3_handeye" / time.strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    board = make_board(cfg)
    detector = cv2.aruco.CharucoDetector(board)
    client = FrankyClient(cfg.nuc_host, cfg.service_port, stale_run_s=0.15)
    camera = RealSenseClient(cfg.camera_serial, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
    client.connect_robot(cfg.robot_ip)  # read-only from here on: read_q, flange_pose_of
    q0 = client.read_q()
    trans_err, rot_err = fk_error(
        flange_pose(q0), np.asarray(client.flange_pose_of(q0)).reshape(4, 4)
    )
    if trans_err > FK_TOLERANCE_M or rot_err > FK_TOLERANCE_DEG:
        raise RuntimeError(f"FK mismatch {trans_err * 1e3:.1f} mm / {rot_err:.2f} deg")
    camera.start()
    print(
        f"samples -> {directory}\n"
        "Hand-guide the arm so the board is in view and let go. In the WINDOW:\n"
        "  SPACE/Enter = capture   u = undo last   s = solve now   q/Esc = solve and quit\n"
        "Vary the poses: different heights, sides, and wrist tilts of 20-40 deg.\n"
    )
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    flanges: list[np.ndarray] = []
    boards: list[np.ndarray] = []
    paths: list[Path] = []
    history: list[tuple[float, np.ndarray]] = []
    message, message_colour = "hand-guide over the board, let go, press SPACE", WHITE
    last_frame_number, det = None, {"ok": False}
    tilt_for = -1
    try:
        while True:
            now = time.monotonic()
            q = np.asarray(client.read_q())
            history = [(t, h) for t, h in history if now - t < 0.3] + [(now, q)]
            moving = np.abs(q - history[0][1]).max() > 2e-3
            frame = camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame.frame_number != last_frame_number:
                last_frame_number = frame.frame_number
                det = detect_board(frame, board, cfg, detector)
            if det["ok"]:
                flange = np.asarray(flange_pose(q))
                gap = min(
                    (_rot_angle_deg(f[:3, :3].T @ flange[:3, :3]) for f in flanges),
                    default=float("inf"),
                )
                scale = "n/a" if det["depth_scale"] is None else f"{det['depth_scale']:.3f}"
                detail = (
                    f"{det['corners']} corners  {det['reproj_px']:.2f}px  markers "
                    f"{det['marker_fit_px']:.1f}px  depth/fit {scale}  "
                    + ("new pose" if gap == float("inf") else f"{gap:.0f} deg to nearest")
                )
                detail_colour = GREEN if not moving else YELLOW
            else:
                detail, detail_colour = det.get("reason", "no board"), RED
            state = (
                "MOVING - let go"
                if moving
                else ("READY - SPACE to capture" if det["ok"] else "still")
            )
            if tilt_for != len(flanges):
                tilt_for = len(flanges)
                tilt = tilt_coverage_deg(flanges)
                tilt_ok = len(flanges) >= 3 and tilt[-1] >= MIN_TILT_DEG
                tilt_text = (
                    "rotation about 3 axes: "
                    + " / ".join(f"{a:.0f}" for a in tilt)
                    + f" deg  (want all >= {MIN_TILT_DEG:.0f}: tilt fwd/back AND left/right)"
                )
            lines = [
                (
                    f"{len(flanges)} samples (need {cfg.min_samples})   {state}",
                    YELLOW if moving else (GREEN if det["ok"] else WHITE),
                ),
                (detail, detail_colour),
                (tilt_text, GREEN if tilt_ok else YELLOW),
                (message, message_colour),
            ]
            cv2.imshow(WINDOW, draw_preview(frame, det, lines, board))
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                key = ord("q")
            if key == ord("u") and paths:
                paths.pop().unlink()
                flanges.pop()
                boards.pop()
                message, message_colour = f"removed sample {len(flanges)}", WHITE
            elif key in (ord("s"), ord("q"), 27):
                if len(flanges) >= cfg.min_samples:
                    code, message = _report(cfg, directory, flanges, boards)
                    message_colour = GREEN if code == 0 else RED
                else:
                    message, message_colour = f"need {cfg.min_samples} samples to solve", RED
                if key != ord("s"):
                    return 0
            elif key in (ord(" "), 13, 10):
                message, message_colour = _capture_one(
                    cfg, client, camera, board, detector, directory, flanges, boards, paths
                )
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        client.close()


def _capture_one(cfg, client, camera, board, detector, directory, flanges, boards, paths):
    """Stillness check, a frame taken after it, full checks, save. Returns (message, colour)."""
    q_a = np.asarray(client.read_q())
    time.sleep(0.3)
    t_still = time.time()
    q_b = np.asarray(client.read_q())
    if np.abs(q_b - q_a).max() > 2e-3:
        return "rejected: the arm moved - let go and hold still", RED
    deadline = time.time() + 1.0
    frame = camera.latest()
    while frame is None or frame.host_time_s < t_still:
        if time.time() > deadline:
            return "rejected: no fresh camera frame", RED
        time.sleep(0.01)
        frame = camera.latest()
    det = detect_board(frame, board, cfg, detector)
    if not det["ok"]:
        return f"rejected: {det['reason']}", RED
    base_T_flange = np.asarray(client.flange_pose_of(q_b), dtype=np.float64).reshape(4, 4)
    paths.append(save_sample(directory, len(flanges), frame, q_b, base_T_flange))
    flanges.append(base_T_flange)
    boards.append(det["camera_T_board"])
    text = f"captured sample {len(flanges)} ({det['corners']} corners, {det['reproj_px']:.2f}px)"
    print("  " + text)
    return text, GREEN


def main() -> None:
    """Console entry point: ``z3-gap-handeye``."""
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = tyro.cli(HandEyeConfig)
    raise SystemExit(solve_directory(cfg, cfg.from_dir) if cfg.from_dir else capture(cfg))
