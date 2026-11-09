"""D405 RGB-D capture for the GaP bridge, and a standalone camera check.

One ``pyrealsense2`` pipeline on a background thread, colour + depth, depth aligned to
colour so a depth pixel and an RGB pixel are the same ray. GaP wants depth as float32
**metres** with 0 meaning "no depth" and a 3x3 pinhole ``K`` of the colour stream
(OpenCV convention, as ``gap.envs.utils.depth_color_to_pointcloud`` deprojects it), so
the conversion happens here, once, with the device's own depth scale — the D405's
default unit is 0.1 mm, not the 1 mm of the D4xx family, so it is read, never assumed.

Run ``z3-gap-realsense --serial <sn>`` to check a camera without the robot or GaP.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: The wrist D405 that looks at the Robotiq fingers (identified 2026-11-04). The Wuji-hand
#: palm D405 is 352122274180 (experiments/scripts/cam-2026-09-21.sh in oktonex).
WRIST_D405_SERIAL = "409122274238"


@dataclass(frozen=True)
class RealSenseFrame:
    """One aligned capture.

    Attributes:
        rgb: HxWx3 uint8, RGB order.
        depth_m: HxW float32 metres, aligned to ``rgb``; 0 = no depth.
        intrinsics: 3x3 float64 pinhole matrix of the colour stream.
        frame_number: RealSense colour frame counter.
        hw_timestamp_ms: RealSense colour timestamp (domain in ``timestamp_domain``).
        timestamp_domain: e.g. ``"Global Time"`` or ``"Hardware Clock"``.
        host_time_s: ``time.time()`` when the frame reached this process.
        distortion_model: RealSense model name of the colour stream, e.g.
            ``"inverse_brown_conrady"`` (the D405's), ``"brown_conrady"``, ``"none"``.
        distortion: The 5 RealSense coefficients for ``distortion_model``; usable as
            OpenCV (k1, k2, p1, p2, k3) — see ``undistort_pixels``.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    frame_number: int
    hw_timestamp_ms: float
    timestamp_domain: str
    host_time_s: float
    distortion_model: str = "none"
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))


def intrinsics_matrix(fx: float, fy: float, ppx: float, ppy: float) -> np.ndarray:
    """3x3 pinhole matrix from RealSense intrinsics fields."""
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_pixels(
    pixels: np.ndarray, intrinsics: np.ndarray, coeffs: np.ndarray, model: str
) -> np.ndarray:
    """Distorted pixel coordinates -> the pixels an ideal pinhole with the same K would see.

    librealsense (2.58) deprojects its ``inverse_brown_conrady`` model — the D405 colour
    stream's — by iteratively inverting the standard Brown–Conrady polynomial, so the
    coefficients mean exactly what OpenCV's (k1, k2, p1, p2, k3) mean. Checked against
    ``rs2_deproject_pixel_to_point`` / ``rs2_project_point_to_pixel``: within 0.01 px
    (tests/test_handeye.py). The name suggests the opposite direction; it is not.
    """
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(coeffs, dtype=np.float64)
    model = model.lower().replace("distortion.", "")
    if model == "none" or not np.any(c):
        return pts.copy()
    if model not in ("inverse_brown_conrady", "brown_conrady"):
        raise NotImplementedError(f"distortion model {model!r} is not supported")
    import cv2

    k = np.asarray(intrinsics, dtype=np.float64)
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), k, c, P=k).reshape(-1, 2)


def depth_to_metres(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """uint16 device units -> float32 metres; 0 stays 0 (no depth)."""
    if depth_scale <= 0.0:
        raise ValueError(f"depth scale must be positive, got {depth_scale}")
    return depth_raw.astype(np.float32) * np.float32(depth_scale)


def list_devices() -> list[dict[str, str]]:
    """Serial, name and USB type of every connected RealSense."""
    import pyrealsense2 as rs

    out = []
    for dev in rs.context().query_devices():
        info = {}
        for key, attr in (
            ("serial", rs.camera_info.serial_number),
            ("name", rs.camera_info.name),
            ("usb", rs.camera_info.usb_type_descriptor),
            ("firmware", rs.camera_info.firmware_version),
        ):
            info[key] = dev.get_info(attr) if dev.supports(attr) else ""
        out.append(info)
    return out


class RealSenseClient:
    """Latest-frame RGB-D reader for one RealSense (D405 by default).

    Args:
        serial: Device serial; None takes the only connected device (and refuses
            to guess if several are connected).
        width: Stream width in pixels (colour and depth).
        height: Stream height in pixels.
        fps: Stream rate.
    """

    def __init__(
        self,
        serial: str | None = WRIST_D405_SERIAL,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale: float | None = None
        self._pipeline: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: RealSenseFrame | None = None
        self._error: BaseException | None = None
        self.frames_received = 0

    def start(self, first_frame_timeout_s: float = 5.0) -> None:
        """Open the pipeline and block until the first aligned frame arrives.

        A D405 can open its streams and then deliver nothing (seen 2026-11-06: twice in a
        row, cleared by restarting its streams). On that one failure the camera gets a
        ``hardware_reset`` and a single second attempt; a second failure raises.
        """
        try:
            self._start_once(first_frame_timeout_s)
        except TimeoutError:
            log.warning(
                "RealSense %s opened but sent no frames — hardware reset and one retry",
                self.serial,
            )
            self._hardware_reset()
            self._start_once(first_frame_timeout_s)

    def _hardware_reset(self, reappear_timeout_s: float = 15.0) -> None:
        import pyrealsense2 as rs

        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) == self.serial:
                dev.hardware_reset()
                break
        gone_by = time.monotonic() + 5.0  # wait for it to drop off the bus...
        while self.serial in {d["serial"] for d in list_devices()}:
            if time.monotonic() > gone_by:
                log.warning("RealSense %s never left the bus after reset", self.serial)
                break
            time.sleep(0.2)
        deadline = time.monotonic() + reappear_timeout_s  # ...and come back
        while self.serial not in {d["serial"] for d in list_devices()}:
            if time.monotonic() > deadline:
                raise RuntimeError(f"RealSense {self.serial} did not come back after reset")
            time.sleep(0.5)

    def _start_once(self, first_frame_timeout_s: float) -> None:
        import pyrealsense2 as rs

        if self.serial is None:
            devices = list_devices()
            if len(devices) != 1:
                raise RuntimeError(
                    f"{len(devices)} RealSense devices connected — pass a serial: "
                    + ", ".join(f"{d['serial']} ({d['name']})" for d in devices)
                )
            self.serial = devices[0]["serial"]

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self._pipeline = pipeline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="realsense", daemon=True)
        self._thread.start()
        log.info(
            "RealSense %s streaming %dx%d@%d, depth scale %.6f m/unit",
            self.serial,
            self.width,
            self.height,
            self.fps,
            self.depth_scale,
        )
        deadline = time.monotonic() + first_frame_timeout_s
        while self.latest() is None:
            if self._error is not None:
                self.stop()
                raise RuntimeError("RealSense read thread died") from self._error
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(
                    f"no frame from RealSense {self.serial} within {first_frame_timeout_s}s"
                )
            time.sleep(0.01)

    def _run(self) -> None:
        import pyrealsense2 as rs

        align = rs.align(rs.stream.color)
        assert self.depth_scale is not None
        try:
            while not self._stop.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(1000)
                except RuntimeError:
                    log.warning("RealSense %s: no frames for 1 s", self.serial)
                    continue
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    continue
                k = color.profile.as_video_stream_profile().intrinsics
                frame = RealSenseFrame(
                    rgb=np.asanyarray(color.get_data()).copy(),
                    depth_m=depth_to_metres(np.asanyarray(depth.get_data()), self.depth_scale),
                    intrinsics=intrinsics_matrix(k.fx, k.fy, k.ppx, k.ppy),
                    frame_number=int(color.get_frame_number()),
                    hw_timestamp_ms=float(color.get_timestamp()),
                    timestamp_domain=str(color.get_frame_timestamp_domain()),
                    host_time_s=time.time(),
                    distortion_model=str(k.model).replace("distortion.", ""),
                    distortion=np.asarray(k.coeffs, dtype=np.float64),
                )
                with self._lock:
                    self._latest = frame
                    self.frames_received += 1
        except BaseException as exc:  # surfaced by start()/latest(), never swallowed
            self._error = exc
            log.exception("RealSense %s read thread died", self.serial)

    def latest(self, max_age_s: float | None = None) -> RealSenseFrame | None:
        """Newest frame, or None if there is none yet or it is older than ``max_age_s``."""
        if self._error is not None:
            raise RuntimeError(f"RealSense {self.serial} read thread died") from self._error
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        if max_age_s is not None and time.time() - frame.host_time_s > max_age_s:
            return None
        return frame

    def stop(self) -> None:
        """Stop the thread and the pipeline (idempotent)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None

    def __enter__(self) -> RealSenseClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


@dataclass
class CheckConfig:
    """Stream a RealSense for a few seconds and report whether it is usable for GaP."""

    serial: str | None = WRIST_D405_SERIAL
    """Device serial; None = the only connected device. Default: the wrist D405."""
    width: int = 848
    height: int = 480
    fps: int = 30
    seconds: float = 3.0
    """How long to stream for the rate measurement."""
    save: Path | None = None
    """Write the last frame here as .npz (rgb, depth_m, intrinsics) plus a .png preview."""
    list: bool = False
    """Only list connected devices."""


def check(cfg: CheckConfig) -> int:
    """Stream, measure, and report; 0 when every check passes."""
    if cfg.list:
        for dev in list_devices():
            print(f"{dev['serial']}  {dev['name']}  usb {dev['usb']}  fw {dev['firmware']}")
        return 0

    with RealSenseClient(cfg.serial, cfg.width, cfg.height, cfg.fps) as cam:
        n0, t0 = cam.frames_received, time.monotonic()
        time.sleep(cfg.seconds)
        rate = (cam.frames_received - n0) / (time.monotonic() - t0)
        frame = cam.latest()
        assert frame is not None
        valid = frame.depth_m > 0
        h, w = frame.depth_m.shape
        centre = frame.depth_m[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5]
        centre_valid = centre[centre > 0]
        checks = {
            f"rate {rate:.1f} Hz >= 80% of {cfg.fps}": rate >= 0.8 * cfg.fps,
            f"depth aligned to colour {frame.depth_m.shape} == {frame.rgb.shape[:2]}": (
                frame.depth_m.shape == frame.rgb.shape[:2]
            ),
            f"valid depth {valid.mean():.0%} of pixels >= 30%": valid.mean() >= 0.3,
            f"depth scale {cam.depth_scale} m/unit set": bool(cam.depth_scale),
        }
        print(f"serial {cam.serial}  timestamps: {frame.timestamp_domain}")
        print("K =\n" + np.array2string(frame.intrinsics, precision=2, suppress_small=True))
        if centre_valid.size:
            print(f"centre depth {float(np.median(centre_valid)):.4f} m")
        else:
            print("centre depth: none (too close/far for the D405's ~7-50 cm range?)")
        for name, ok in checks.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if cfg.save is not None:
            out = cfg.save.with_suffix(".npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out, rgb=frame.rgb, depth_m=frame.depth_m, intrinsics=frame.intrinsics
            )
            try:
                from PIL import Image

                d = frame.depth_m
                hi = float(np.percentile(d[d > 0], 99)) if valid.any() else 1.0
                grey = (np.clip(d / hi, 0, 1) * 255).astype(np.uint8)
                preview = np.concatenate([frame.rgb, np.repeat(grey[..., None], 3, -1)], axis=1)
                Image.fromarray(preview).save(out.with_suffix(".png"))
            except ImportError:
                log.info("pillow not installed — skipped the .png preview")
            print(f"saved {out}")
        return 0 if all(checks.values()) else 1


def main() -> None:
    """Console entry point: ``z3-gap-realsense``."""
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(check(tyro.cli(CheckConfig)))
