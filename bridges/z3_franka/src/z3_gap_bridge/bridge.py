"""Z3 cell <-> GaP: drive the Panda + Robotiq + wrist D405 from ``gap run --real franka``.

Takes the place of the ``robots_realtime`` rr-session client. GaP's ``FrankaRealEnv``
listens on 127.0.0.1:9000; this process connects to it and, every tick, sends one
observation and receives GaP's latest action:

    observation  {left: {joint_pos: f32[8]},                       7 joints + gripper OPEN fraction
                  camera_top: {images: {rgb}, depth_data, intrinsics: {left: {intrinsics_matrix}},
                               pose_mat}}                         every 1/camera_hz, base-frame pose
    action       {timestamp, left: {joint_pos: [7], gripper}}       absolute target, gripper OPEN fraction

The arm runs through the NUC franky service (``FrankyClient``, Ruckig-interpolated
``move_joints``), the gripper through the service's Robotiq driver, the camera through
``RealSenseClient``. GaP itself is not modified.

Conventions this file exists to get right:

- **Gripper polarity.** GaP: 1 = open, 0 = closed. Franky service: 0 = open, 1 = closed.
  Inverted in exactly one place each way (``_gripper_open_fraction``, ``_command_gripper``).
- **GaP's first action is not the arm's pose.** ``FrankaRealEnv`` pre-seeds its action
  with a hard-coded home configuration so the robots_realtime client does not fall back
  to its gizmo. Forwarding that seed would drive the arm to GaP's home the moment the
  bridge connects. Arm targets are therefore held until GaP sends one that differs from
  its seed (``ChangeGate``), i.e. until the graph actually commands a motion. The
  gripper is held until GaP's ``gripper_seq`` (count of real gripper commands, 0 in the
  seed) is non-zero — a value gate cannot see the graph's first "open", which equals
  the seed's 1.0 — or, from a GaP without that field, until its value changes.
- **Camera pose is world <- colour-optical (OpenCV: +z forward, +y down)**, recomputed
  from FK for every frame because the D405 rides on the wrist.

Dry run unless ``--execute``: everything streams, GaP plans and logs, and every command
the bridge would send is logged instead. ``--execute`` needs a person at the e-stop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from z3_gap_bridge.panda_fk import flange_pose, tool_tip_position
from z3_gap_bridge.realsense_client import WRIST_D405_SERIAL
from z3_gap_bridge.wire import field as wire_field

if TYPE_CHECKING:
    from z3_gap_bridge.franky_client import FrankyClient
    from z3_gap_bridge.realsense_client import RealSenseClient
    from z3_gap_bridge.robotiq import LocalGripper, ServiceGripper
    from z3_gap_bridge.wire import WireClient

log = logging.getLogger(__name__)

# Values copied from oktonex okto_robot (config_franka_panda_franky.py / franka_panda_franky.py
# @ c83e4ed9) — the Robotiq-era cell configuration. Rationale for each lives there.
DEFAULT_K_THETA = (300.0, 200.0, 300.0, 300.0, 50.0, 100.0, 50.0)
DEFAULT_COLLISION_TORQUE_THRESHOLDS = (30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0)
DEFAULT_COLLISION_FORCE_THRESHOLDS = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0)
DEFAULT_MAX_JOINT_VELOCITY = (1.8, 2.0, 1.8, 1.5, 2.2, 2.2, 2.2)
# A joint-stiffness change waits until the arm has been still this long (libfranka refuses
# set_joint_impedance while a motion runs): measured joints moved < STILL_RAD per tick.
STILL_TICKS = 10
STILL_RAD = 2e-4
JOINT_POS_MIN = np.array([-2.877, -1.743, -2.877, -3.052, -2.877, 0.002, -2.877])
JOINT_POS_MAX = np.array([2.877, 1.743, 2.877, -0.090, 2.877, 3.733, 2.877])
FK_TOLERANCE_M = 0.002
FK_TOLERANCE_DEG = 0.2


def fk_error(ours: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(max translation error [m], rotation angle error [deg]) between two 4x4 poses."""
    trans = float(np.abs(reference[:3, 3] - ours[:3, 3]).max())
    rel = ours[:3, :3].T @ reference[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
    return trans, float(angle)


@dataclass
class BridgeConfig:
    """Bridge the z3 cell to a running ``gap run ... --real franka --no-rr-autostart``."""

    flange_t_camera: Path | None = None
    """4x4 flange -> D405 COLOUR OPTICAL frame (.npy, or .json with key
    ``flange_T_camera_optical``). Required unless --kinematics is given: a wrong one
    silently offsets every grasp."""
    kinematics: Path | None = None
    """A ``z3-gap-handeye`` result directory whose report.json carries joint zero offsets
    (``--fit-joints`` or ``--joint-offsets-deg``): the offsets are added to every joint
    reading sent to GaP and subtracted from every GaP target, and its flange_T_camera.npy
    is the camera transform (do not pass --flange-t-camera as well: the two are only
    valid together). GaP, its IK and the camera then work in the corrected kinematics;
    the robot still receives its own angles. Off = the robot's model as reported."""
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """The offsets --kinematics loads [deg, joints 1-7]. Explicit use needs a
    flange_t_camera solved WITH the same offsets."""
    execute: bool = False
    """Actually move the arm and gripper. Without it this is a dry run."""

    gap_host: str = "127.0.0.1"
    gap_port: int = 9000
    gap_wait_s: float = 120.0
    """How long to wait for GaP's server to come up."""

    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    robot_ip: str = "192.168.98.121"
    gripper: Literal["workstation", "nuc", "none"] = "workstation"
    """Where the Robotiq's RS-485 adapter is plugged in; "none" reports it as open."""
    gripper_port: str | None = None
    """Serial port; default: the adapter's /dev/serial/by-id path on the workstation,
    /dev/ttyUSB0 on the NUC."""
    gripper_speed: float = 0.5
    gripper_force: float = 0.1
    gripper_deadband: float = 0.02

    camera_serial: str = WRIST_D405_SERIAL
    camera_width: int = 848
    camera_height: int = 480
    camera_fps: int = 30
    camera_hz: float = 15.0
    """Rate camera frames are sent to GaP (robots_realtime sends its ZED at 15)."""
    camera_max_age_s: float = 0.5
    camera_key: str = "camera_top"
    """Wire key of the camera dict; GaP auto-discovers it and renames it robot0_robotview."""

    loop_hz: float = 50.0
    stale_threshold_s: float = 0.1
    """Round-trip budget per NUC read. Teleop's 20 ms hair trigger is wrong here: GaP
    sends absolute targets, so a late read only delays one observation. 2026-11-06: a
    single 72.5 ms read_q (while GaP loaded its models) killed the 20 ms bridge."""
    max_stale_s: float = 1.0
    """Abort only when reads stay over budget this long; single stalls skip a tick."""
    motion: Literal["nuc", "stream"] = "nuc"
    """How a GaP target becomes motion. ``nuc``: one move_joints per target; the service's
    POSITION controller plans it with Ruckig in its 1 kHz loop (needs the service started
    WITHOUT --controller tracking — checked at connect, refused otherwise). ``stream``: the
    bridge plans it (trajectory.py) and streams small steps to a TRACKING controller, which
    does not interpolate at all — the cause of the 2026-11-06 jumps when targets went raw."""
    dynamics: tuple[float, float, float] = (0.4, 0.3, 0.3)
    """Ruckig (velocity, acceleration, jerk) factors: on the NUC's limits for ``nuc``, on
    max_joint_velocity / _acceleration / _jerk below for ``stream`` (0.7 rad/s,
    0.9 rad/s^2, 4.5 rad/s^3)."""
    status_poll_s: float = 0.5
    """--execute: read the robot mode this often and latch on Reflex / user stop. An
    asynchronous move's reflex otherwise surfaces only with the NEXT command."""
    max_joint_velocity: tuple[float, ...] = DEFAULT_MAX_JOINT_VELOCITY
    max_joint_acceleration: tuple[float, ...] = (3.0,) * 7
    max_joint_jerk: tuple[float, ...] = (15.0,) * 7
    k_theta: tuple[float, ...] = DEFAULT_K_THETA
    collision_torque_thresholds: tuple[float, ...] = DEFAULT_COLLISION_TORQUE_THRESHOLDS
    collision_force_thresholds: tuple[float, ...] = DEFAULT_COLLISION_FORCE_THRESHOLDS
    z_floor_m: float | None = 0.020
    """Tool-tip floor [m, base frame]. The z3 cell (2026-11-06): table +0.016 (pivot touch)
    / +0.017 (hand-eye board, 25 cm away) -> +0.020 with 3 mm margin. The -0.045 it
    replaced was the previous cell's, 6 cm too low here. Valid only paired with
    tool_length_m; None disables."""
    tool_length_m: float = 0.220
    """Flange -> fingertip end along the flange axis [m]. z3 custom Robotiq fingertips:
    pivot calibration 0.217 +- 0.003 (z3-gap-tcp, 2026-11-06); 0.220 errs long, so the
    guard engages early."""


def load_kinematics(directory: str | Path) -> tuple[tuple[float, ...], np.ndarray]:
    """(joint zero offsets [deg], flange_T_camera) from a ``z3-gap-handeye`` result dir."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    offsets = tuple(float(v) for v in report["joint_offsets_deg"])
    if len(offsets) != 7:
        raise ValueError(f"{directory}: joint_offsets_deg needs 7 values, got {len(offsets)}")
    if max(abs(v) for v in offsets) > 5.0:
        raise ValueError(f"{directory}: joint offsets {offsets} deg — more than 5 deg is not a zero offset")
    return offsets, load_flange_t_camera(directory / "flange_T_camera.npy")


def load_flange_t_camera(path: str | Path) -> np.ndarray:
    """Load and validate a 4x4 flange -> colour-optical transform."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        matrix = np.asarray(data["flange_T_camera_optical"], dtype=np.float64)
    else:
        matrix = np.asarray(np.load(path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite 4x4 matrix, got shape {matrix.shape}")
    rot = matrix[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or abs(np.linalg.det(rot) - 1) > 1e-4:
        raise ValueError(f"{path}: rotation block is not a proper rotation")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{path}: last row must be [0, 0, 0, 1]")
    if np.linalg.norm(matrix[:3, 3]) > 0.5:
        raise ValueError(f"{path}: camera is {np.linalg.norm(matrix[:3, 3]):.2f} m off the flange")
    return matrix


class ChangeGate:
    """Opens on the first value that differs from the first value it saw.

    GaP's first action is a seed, not a command; the first value after it that differs
    is the graph's first real command. Stays open once opened.
    """

    def __init__(self, atol: float = 1e-6) -> None:
        self._atol = atol
        self._seed: np.ndarray | None = None
        self.open = False

    def update(self, value: np.ndarray | float) -> bool:
        """Feed one value; True once the gate is open."""
        value = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if self.open:
            return True
        if self._seed is None:
            self._seed = value.copy()
            return False
        if not np.allclose(value, self._seed, atol=self._atol, rtol=0.0):
            self.open = True
        return self.open


class Bridge:
    """One GaP session on the cell: connect, tick at ``loop_hz``, close.

    Args:
        cfg: Bridge configuration.
        client: Franky service client (not yet connected to the robot).
        camera: RealSense client (not yet started).
        wire: GaP wire client (not yet connected).
        flange_t_camera: 4x4 flange -> colour-optical transform.
        gripper: Robotiq driver (not yet connected), or None for gripper-less.
    """

    def __init__(
        self,
        cfg: BridgeConfig,
        client: FrankyClient,
        camera: RealSenseClient,
        wire: WireClient,
        flange_t_camera: np.ndarray,
        gripper: LocalGripper | ServiceGripper | None = None,
    ) -> None:
        self.cfg = cfg
        self._gripper = gripper
        self._client = client
        self._camera = camera
        self._wire = wire
        self._flange_t_camera = flange_t_camera
        # Joint zero offsets: everything inside the bridge and on the GaP side is in the
        # CORRECTED model (reported angle + offset, the arm's true geometry as far as the
        # calibration knows it); the robot's own angles appear only at the two boundaries,
        # read_q() (+offset) and move_joints() (-offset). The joint limits are the robot's.
        if len(cfg.joint_offsets_deg) != 7:
            raise ValueError("joint_offsets_deg needs 7 values")
        self._dq = np.radians(np.asarray(cfg.joint_offsets_deg, dtype=np.float64))
        self._arm_gate = ChangeGate()
        self._gripper_gate = ChangeGate()
        self._gripper_connected = False
        self._last_requested: np.ndarray | None = None
        self._last_dispatched: np.ndarray | None = None
        self._last_gripper_cmd: float | None = None
        self._last_gripper_seq: int | None = None
        self._gripper_live = False
        # GaP's joint-stiffness requests: (seq, k_theta) waiting for the arm to be still;
        # the last applied seq is echoed in every observation (left.impedance_seq).
        self._impedance_pending: tuple[int, tuple[float, ...]] | None = None
        self._impedance_applied_seq = 0
        self._still_ticks = 0
        self._last_camera_sent_t = 0.0
        self._last_camera_frame: int | None = None
        self.fault: str | None = None
        self.floor_clamp_count = 0
        self.dispatch_count = 0
        self._ticks = 0
        self._camera_sends = 0
        self._last_q = np.zeros(7)
        self._warn_t: dict[str, float] = {}
        self._stale_since: float | None = None
        self.stale_ticks = 0
        self.stream_steps = 0
        self._status_t = 0.0
        self._reference = None  # JointReference, built in connect() for --execute

    # -- lifecycle ----------------------------------------------------------------------

    def connect(self) -> None:
        """Connect robot, gripper, camera and GaP — in that order, failing closed."""
        cfg = self.cfg
        mode = "EXECUTE" if cfg.execute else "DRY RUN (no motion; --execute to move)"
        log.info("z3 GaP bridge — %s", mode)
        if cfg.execute:
            # Supervised: a reflex comes back as an error (and latches the fault here)
            # instead of the service recovering and resuming on its own — which on
            # 2026-11-06 turned one torque step into jump, reflex, recover, jump.
            try:
                self._client.claim_control()
            except Exception as exc:
                raise RuntimeError(
                    "could not claim exclusive control of the NUC service: "
                    f"{str(exc).splitlines()[0]}\n  Close every other program connected to it "
                    "(z3-gap-gripper, z3-gap-handeye, z3-gap-tcp, teleop) and retry."
                ) from None
            if cfg.motion == "stream":
                from z3_gap_bridge.trajectory import JointReference

                dv, da, dj = cfg.dynamics
                self._reference = JointReference(
                    1.0 / cfg.loop_hz,
                    np.asarray(cfg.max_joint_velocity) * dv,
                    np.asarray(cfg.max_joint_acceleration) * da,
                    np.asarray(cfg.max_joint_jerk) * dj,
                )
        try:
            self._client.connect_robot(cfg.robot_ip)
        except Exception as exc:
            first = str(exc).splitlines()[0]
            if "Reflex" in first or "User stopped" in first or "UserStopped" in first:
                raise RuntimeError(
                    f"the robot is not ready: {first}\n  Clear it at the robot (see the README, "
                    "'Robot in Reflex'), then rerun — the bridge never recovers it for you."
                ) from None
            raise
        q = np.asarray(self._client.read_q(), dtype=np.float64)
        ref = np.asarray(self._client.flange_pose_of(q), dtype=np.float64).reshape(4, 4)
        err, rot_err = fk_error(flange_pose(q), ref)
        if err > FK_TOLERANCE_M or rot_err > FK_TOLERANCE_DEG:
            raise RuntimeError(
                f"FK mismatch vs robot model: {err * 1e3:.1f} mm, {rot_err:.2f} deg — the "
                "floor guard and the camera pose both depend on it; refusing to start"
            )
        log.info("FK self-check %.2f mm, %.3f deg", err * 1e3, rot_err)
        if np.any(self._dq):
            log.info(
                "kinematic correction ON: joint zero offsets %s deg (reported angles + these "
                "go to GaP; GaP targets - these go to the robot)",
                np.round(np.degrees(self._dq), 3).tolist(),
            )

        if cfg.execute:
            controller = (self._client.control_config() or {}).get("controller")
            if cfg.motion == "nuc" and controller != "position":
                raise RuntimeError(
                    f"--motion nuc needs the NUC service's POSITION controller, it runs "
                    f"{controller!r}: a raw target into the tracking controller is the 2026-11-06 "
                    "jump. Restart the service without --controller tracking (README), or use "
                    "--motion stream."
                )
            log.info("motion: %s (NUC controller %s)", cfg.motion, controller)
            self._client.set_joint_impedance(cfg.k_theta)
            self._client.set_collision_behavior(
                cfg.collision_torque_thresholds, cfg.collision_force_thresholds
            )
            self._client.set_joint_velocity_limit(cfg.max_joint_velocity)
        if self._gripper is not None:
            if cfg.execute:
                self._gripper.connect()
                self._gripper_connected = True
            elif self._gripper.probe():
                # Never activate the Robotiq in a dry run (activation moves the fingers).
                self._gripper_connected = True
            else:
                log.info("gripper not activated — reporting it as open (dry runs never activate)")

        self._camera.start()
        self._wire.connect(wait_s=cfg.gap_wait_s)
        self._client.begin_warmup()

    def run(self) -> None:
        """Tick until GaP disconnects or Ctrl-C; always stops the arm on the way out."""
        period = 1.0 / self.cfg.loop_hz
        next_t = time.perf_counter()
        beat_t = time.perf_counter()
        beat_ticks = beat_cams = 0
        try:
            while True:
                self.tick()
                now = time.perf_counter()
                if now - beat_t >= 1.0:
                    log.info(
                        "loop %.0f Hz  cam %.1f Hz  arm %s  gripper %s  dispatched %d  "
                        "floor clamps %d  stalled reads %d  %s%s",
                        (self._ticks - beat_ticks) / (now - beat_t),
                        (self._camera_sends - beat_cams) / (now - beat_t),
                        "live" if self._arm_gate.open else "holding (GaP seed)",
                        "live" if self._gripper_live else "holding",
                        self.dispatch_count,
                        self.floor_clamp_count,
                        self.stale_ticks,
                        "MOVING" if self._reference is not None and self._reference.active else "",
                        f"  FAULT: {self.fault}" if self.fault else "",
                    )
                    beat_t, beat_ticks, beat_cams = now, self._ticks, self._camera_sends
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            log.info("interrupted")
        except ConnectionError as exc:
            log.info("GaP connection ended: %s", exc)

    def close(self) -> None:
        """Stop motion (execute mode), camera and sockets."""
        if self.cfg.execute:
            try:
                self._client.stop()
            except Exception:
                log.warning("stop at close failed", exc_info=True)
        self._camera.stop()
        if self._gripper is not None:
            self._gripper.close()
        self._wire.close()
        self._client.close()

    # -- one tick -----------------------------------------------------------------------

    def tick(self) -> None:
        """Measure, send one observation, apply GaP's reply.

        A read over the staleness budget skips this tick (GaP keeps its last observation);
        reads that stay stale for ``max_stale_s`` raise.
        """
        from z3_gap_bridge.franky_client import FrankyStaleError

        try:
            q = np.asarray(self._client.read_q(), dtype=np.float64)
        except FrankyStaleError as exc:
            now = time.monotonic()
            if self._stale_since is None:
                self._stale_since = now
            self.stale_ticks += 1
            if now - self._stale_since > self.cfg.max_stale_s:
                raise
            self._warn("stale", "NUC read stalled, tick skipped: %s", str(exc).splitlines()[0])
            return
        self._stale_since = None
        q = q + self._dq  # -> the corrected model; the robot's own angles stay in read_q()
        moved = float(np.abs(q - self._last_q).max())
        self._still_ticks = self._still_ticks + 1 if moved < STILL_RAD else 0
        self._last_q = q
        if self.cfg.execute and self.fault is None:
            self._poll_status()  # BEFORE anything this tick could command
        obs: dict[str, Any] = {
            # GaP's heartbeat ages observations by this wall-clock stamp; without it every
            # observation reads as infinitely old and GaP logs STALE all run long.
            "timestamp": time.time(),
            "left": {
                "joint_pos": np.append(q, self._gripper_open_fraction()).astype(np.float32),
                "impedance_seq": self._impedance_applied_seq,
            },
        }
        camera = self._camera_payload(q)
        if camera is not None:
            obs[self.cfg.camera_key] = camera
        action = self._wire.exchange(obs)
        self._ticks += 1
        self._apply(action)
        self._apply_impedance()
        if self._reference is not None and self._reference.active and self.fault is None:
            self._stream_step()

    def _poll_status(self) -> None:
        """Latch on Reflex / user stop as soon as the robot reports it."""
        now = time.monotonic()
        if now - self._status_t < self.cfg.status_poll_s:
            return
        self._status_t = now
        try:
            mode, errors = self._client.read_status()
        except Exception as exc:
            self._warn("status", "read_status failed: %s", str(exc).splitlines()[0])
            return
        mode = str(mode).split(".")[-1]
        try:
            errs = json.loads(str(errors))
        except ValueError:
            errs = [str(errors)] if str(errors) not in ("", "[]") else []
        if mode in ("Reflex", "UserStopped") or errs:
            if self._reference is not None:
                self._reference.active = False
            self._latch_fault(f"robot reports {mode}, errors {errs}")

    def _gripper_open_fraction(self) -> float:
        if not self._gripper_connected or self._gripper is None:
            return 1.0
        closed, _ = self._gripper.state()
        return float(np.clip(1.0 - float(closed), 0.0, 1.0))

    def _camera_payload(self, q: np.ndarray) -> dict[str, Any] | None:
        now = time.perf_counter()
        if now - self._last_camera_sent_t < 1.0 / self.cfg.camera_hz:
            return None
        frame = self._camera.latest(max_age_s=self.cfg.camera_max_age_s)
        if frame is None:
            self._warn("camera", "no camera frame younger than %.1f s", self.cfg.camera_max_age_s)
            return None
        if frame.frame_number == self._last_camera_frame:
            return None
        self._last_camera_sent_t = now
        self._last_camera_frame = frame.frame_number
        self._camera_sends += 1
        pose = flange_pose(q) @ self._flange_t_camera
        return {
            "images": {"rgb": frame.rgb},
            "depth_data": frame.depth_m,
            "intrinsics": {"left": {"intrinsics_matrix": frame.intrinsics.astype(np.float32)}},
            "pose_mat": pose.astype(np.float32),
        }

    def _apply(self, action: dict) -> None:
        left = wire_field(action, "left")
        if left is None:
            return  # GaP has not produced an action yet
        target = np.asarray(wire_field(left, "joint_pos"), dtype=np.float64).reshape(-1)
        gripper = float(wire_field(left, "gripper"))
        if target.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(gripper):
            raise ValueError(f"malformed GaP action: joint_pos {target}, gripper {gripper}")
        arm_live = self._arm_gate.update(target)
        # GaP counts its real gripper commands in gripper_seq (0 = the seed). With it, the
        # graph's first "open" goes through even though 1.0 equals the seed value — the
        # value-change gate alone drops it (2026-11-09: the gripper never opened at the
        # start). Older GaP without the field: value-change gate only.
        seq = wire_field(left, "gripper_seq")
        if seq is None:
            gripper_live = self._gripper_gate.update(gripper)
            new_command = False
        else:
            seq = int(seq)
            gripper_live = seq > 0
            new_command = seq != self._last_gripper_seq
            self._last_gripper_seq = seq
        self._gripper_live = gripper_live
        imp_seq = wire_field(left, "impedance_seq")
        if imp_seq is not None and int(imp_seq) > self._impedance_applied_seq:
            if self._impedance_pending is None or self._impedance_pending[0] != int(imp_seq):
                raw = wire_field(left, "joint_impedance")
                k = tuple(float(v) for v in (raw if raw is not None else ()))
                self._impedance_pending = (int(imp_seq), k or tuple(self.cfg.k_theta))
                log.info(
                    "joint stiffness request %d: %s", int(imp_seq),
                    list(np.round(self._impedance_pending[1])) if k else "startup values",
                )
        if self.fault is not None:
            return
        if arm_live:
            self._command_arm(target)
        if gripper_live:
            self._command_gripper(float(np.clip(gripper, 0.0, 1.0)), force=new_command)

    # -- commands -----------------------------------------------------------------------

    def _apply_impedance(self) -> None:
        """Apply GaP's pending joint-stiffness request once the arm is still."""
        if self._impedance_pending is None:
            return
        seq, k = self._impedance_pending
        if len(k) != 7 or not all(np.isfinite(v) and 0.0 < v <= 15000.0 for v in k):
            self._warn("impedance", "joint stiffness request %d rejected: %s", seq, k)
            return
        if not self.cfg.execute:
            log.info("DRY RUN joint stiffness %s", list(np.round(k)))
        elif self.fault is not None:
            return
        elif self.cfg.motion != "nuc":
            self._warn(
                "impedance", "joint stiffness request %d ignored: only --motion nuc uses the "
                "firmware joint impedance (the tracking controller has its own)", seq,
            )
            return
        elif self._still_ticks < STILL_TICKS:
            return
        else:
            try:
                self._client.set_joint_impedance(k)
            except Exception as exc:
                self._still_ticks = 0  # a motion is still running; try again once still
                self._warn("impedance", "set_joint_impedance: %s", str(exc).splitlines()[0])
                return
            log.info("joint stiffness now %s (request %d)", list(np.round(k)), seq)
        self._impedance_applied_seq = seq
        self._impedance_pending = None

    def _command_arm(self, target: np.ndarray) -> None:
        if self._last_requested is not None and np.array_equal(target, self._last_requested):
            return  # GaP re-stamps the same target at 50 Hz; Ruckig already has it
        self._last_requested = target.copy()
        reference = self._last_dispatched if self._last_dispatched is not None else self._last_q
        within = np.clip(target - self._dq, JOINT_POS_MIN, JOINT_POS_MAX) + self._dq  # robot's limits
        safe = self._floor_clamp(within, reference)
        step = float(np.abs(safe - self._last_q).max())
        if not self.cfg.execute:
            log.info("DRY RUN arm target %s  (max |dq| %.3f rad)", np.round(safe, 3), step)
            self._last_dispatched = safe
            return
        if self.cfg.motion == "nuc":
            # One preempting Ruckig JointMotion, planned and run in the NUC's 1 kHz loop.
            try:
                self._client.move_joints(
                    tuple(float(v) for v in safe - self._dq), rel_dyn=self.cfg.dynamics
                )
            except Exception as exc:
                self._latch_fault(f"move_joints rejected: {str(exc).splitlines()[0]}")
                return
            self._last_dispatched = safe
            self.dispatch_count += 1
            self._still_ticks = 0
            log.info("arm target %s  (max |dq| %.3f rad)", np.round(safe, 3), step)
            return
        assert self._reference is not None
        if not self._reference.initialised:
            self._reference.reset(self._last_q)  # start from the measured pose, at rest
        self._reference.set_target(safe)  # streamed in small steps by _stream_step
        self._last_dispatched = safe
        self.dispatch_count += 1
        log.info("arm target %s  (max |dq| %.3f rad)", np.round(safe, 3), step)

    def _stream_step(self) -> None:
        """Send the next jerk-limited reference step to the tracker."""
        assert self._reference is not None
        q_ref, dq_ref = self._reference.step()
        try:
            self._client.move_joints(
                tuple(float(v) for v in q_ref - self._dq),
                rel_dyn=self.cfg.dynamics,
                velocity=tuple(float(v) for v in dq_ref),
            )
        except Exception as exc:
            self._reference.active = False
            self._latch_fault(f"move_joints rejected: {str(exc).splitlines()[0]}")
            return
        self.stream_steps += 1

    def _floor_clamp(self, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Clamp a target whose tool tip is below the floor onto it (driver semantics).

        Bisects the joint-space step from ``reference`` (the last dispatched target, or
        the measured pose) — the same rule as oktonex ``FrankaPandaFranky.send_action``.
        """
        if self._floor_ok(target):
            return target
        lo, hi = 0.0, 1.0
        if self._floor_ok(reference):
            step = target - reference
            for _ in range(12):
                mid = (lo + hi) / 2.0
                if self._floor_ok(reference + mid * step):
                    lo = mid
                else:
                    hi = mid
        self.floor_clamp_count += 1
        clamped = reference + lo * (target - reference)
        self._warn(
            "floor",
            "floor guard: target tool tip z=%.3f m < %.3f m — clamped to %.0f%% of the step",
            tool_tip_position(target, self.cfg.tool_length_m)[2],
            self.cfg.z_floor_m,
            lo * 100,
        )
        return clamped

    def _floor_ok(self, q: np.ndarray) -> bool:
        if self.cfg.z_floor_m is None:
            return True
        return bool(tool_tip_position(q, self.cfg.tool_length_m)[2] >= self.cfg.z_floor_m)

    def _command_gripper(self, open_fraction: float, force: bool = False) -> None:
        """Send a gripper target. ``force``: a new GaP command (gripper_seq changed) — send
        it even within the deadband, e.g. "open" again after the fingers were pushed."""
        closed = 1.0 - open_fraction  # GaP 1 = open; service 1 = closed
        if (
            not force
            and self._last_gripper_cmd is not None
            and abs(closed - self._last_gripper_cmd) < self.cfg.gripper_deadband
        ):
            return
        self._last_gripper_cmd = closed
        if not self.cfg.execute:
            log.info("DRY RUN gripper -> %.2f open", open_fraction)
            return
        if not self._gripper_connected or self._gripper is None:
            self._warn("gripper", "gripper command but no gripper connected — ignored")
            return
        self._gripper.move(
            closed,
            speed=round(self.cfg.gripper_speed * 255),
            force=round(self.cfg.gripper_force * 255),
        )

    def _latch_fault(self, reason: str) -> None:
        """Stop forwarding commands for the rest of the session; keep streaming state.

        No automatic recover-and-retry toward a GaP target: after a reflex the graph's
        assumptions about where the arm is are wrong. Recover by hand and restart.
        """
        self.fault = reason
        try:
            mode, errors = self._client.read_status()
            reason = f"{reason} (robot mode {mode}, errors {errors})"
        except Exception:
            pass
        log.error(
            "FAULT LATCHED — no further commands this session: %s. Ctrl-C, clear the "
            "robot, restart the bridge and the graph.",
            reason,
        )

    def _warn(self, key: str, msg: str, *args: object) -> None:
        now = time.monotonic()
        if now - self._warn_t.get(key, 0.0) >= 1.0:
            self._warn_t[key] = now
            log.warning(msg, *args)


def main() -> None:
    """Console entry point: ``z3-gap-bridge``."""
    import tyro

    from z3_gap_bridge.franky_client import FrankyClient
    from z3_gap_bridge.realsense_client import RealSenseClient
    from z3_gap_bridge.robotiq import LocalGripper, ServiceGripper
    from z3_gap_bridge.wire import WireClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = tyro.cli(BridgeConfig)
    if cfg.kinematics is not None:
        if cfg.flange_t_camera is not None or any(cfg.joint_offsets_deg):
            raise SystemExit("--kinematics carries its own camera transform and offsets; drop "
                             "--flange-t-camera / --joint-offsets-deg")
        offsets, _ = load_kinematics(cfg.kinematics)
        cfg.joint_offsets_deg = offsets
        cfg.flange_t_camera = Path(cfg.kinematics) / "flange_T_camera.npy"
    if cfg.flange_t_camera is None:
        raise SystemExit(
            "--flange-t-camera (or --kinematics DIR) is required: the 4x4 flange -> D405 "
            "colour-optical transform for the mount on the arm now (see bridges/z3_franka/README.md)"
        )
    client = FrankyClient(
        cfg.nuc_host,
        cfg.service_port,
        stale_threshold_s=cfg.stale_threshold_s,
        stale_consecutive_limit=5,
        stale_run_s=0.15,
    )
    gripper: LocalGripper | ServiceGripper | None = None
    if cfg.gripper == "workstation":
        gripper = LocalGripper(cfg.gripper_port or LocalGripper.DEFAULT_PORT)
    elif cfg.gripper == "nuc":
        gripper = ServiceGripper(client, cfg.gripper_port or "/dev/ttyUSB0")
    bridge = Bridge(
        cfg,
        client,
        RealSenseClient(cfg.camera_serial, cfg.camera_width, cfg.camera_height, cfg.camera_fps),
        WireClient(cfg.gap_host, cfg.gap_port),
        load_flange_t_camera(cfg.flange_t_camera),
        gripper,
    )
    try:
        bridge.connect()
        bridge.run()
    finally:
        bridge.close()
