"""Connector base — the absorbed sim_bridge servicer logic, as plain Python.

The original research codebase ran four gRPC servicers (SimBridge,
Observation, Gripper, RobotControl) over one shared ``SimState``. This module
absorbs the *method bodies* of those servicers into a plain-Python
:class:`Connector`: the proto request/response packing becomes ordinary
arguments and :mod:`gap.types` TypedDicts, while every control loop —
convergence stepping, settle counts, velocity-mode action synthesis, TCP
offset/rotation application — is kept verbatim.

A connector owns one environment instance (built by ``gap.envs``) and exposes:

- ``tool_registry`` — a fresh :class:`gap.tools.ToolRegistry` with the
  connector's ``robot.*`` (and subclass ``sim.*``) tools registered;
- ``get_observation()`` — env obs dict → :class:`gap.types.Observation`;
- ``capabilities`` — what this backend can do (reset/success/video/world);
- ``close()`` + context-manager support.

Quaternion convention: gap is wxyz scalar-first throughout. Env classes
already convert sim-native xyzw quaternions to wxyz inside their obs dict
(``robot_cartesian_pos_*[3:7]``, camera ``pose[3:]``, object poses) — the
connector preserves that and never re-encodes image buffers.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.types import (
    ArmState,
    CameraFrame,
    Observation,
    Se3Pose,
    Trajectory,
    make_pose,
)

from gap import env_config

logger = logging.getLogger(__name__)

# Franka Panda home joint configuration (radians)
_FRANKA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# DROID Sim Evals home (matches third_party/droid-sim-evals nvidia_droid.py
# init_state). Differs from canonical Franka home in joint7=0 (vs π/4).
_DROID_HOME_JOINTS = [
    0.0,
    -math.pi / 5,      # -0.6283
    0.0,
    -4 * math.pi / 5,  # -2.5133
    0.0,
    3 * math.pi / 5,   # 1.8850
    0.0,
]


@dataclass(frozen=True)
class Capabilities:
    """What a connector backend can do.

    ``gap.execute`` and the benchmark harness branch on these instead of
    isinstance checks.
    """

    reset: bool = False
    success_check: bool = False
    video: bool = False
    world_state: bool = False


def _as_vec3(value: Any) -> np.ndarray | None:
    """Coerce a Vec3-ish value (dict / sequence / ndarray) to float64 (3,)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return np.array(
            [float(value["x"]), float(value["y"]), float(value["z"])], dtype=np.float64
        )
    return np.asarray(value, dtype=np.float64).reshape(3)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _twist_pattern(amplitude_deg: float, step_deg: float, cycles: int) -> list[float]:
    """Yaw angles for a back-and-forth search: 0 -> +A -> -A -> 0, ``cycles`` times."""
    n = max(1, round(abs(amplitude_deg) / max(abs(step_deg), 1e-6)))
    s = abs(amplitude_deg) / n
    one = (
        [s * i for i in range(1, n + 1)]
        + [s * i for i in range(n - 1, -n - 1, -1)]
        + [s * i for i in range(-n + 1, 1)]
    )
    return one * max(0, int(cycles))


def _search_rings(radius_m: float, step_m: float) -> list[np.ndarray]:
    """Sideways offsets for a peg-in-hole search: rings at step_m, 2*step_m, ... out to radius_m,
    points ~step_m apart around each, alternating direction; ends back at the centre."""
    if radius_m <= 0 or step_m <= 0:
        return []
    out = []
    n_rings = max(1, int(round(radius_m / step_m)))
    for i in range(1, n_rings + 1):
        r = radius_m * i / n_rings
        n = max(6, int(math.ceil(2 * math.pi * r / step_m)))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) * (1 if i % 2 else -1)
        out += [np.array([r * math.cos(a), r * math.sin(a)]) for a in angles]
    out.append(np.zeros(2))
    return out


def _as_positions(value: Any) -> list[float]:
    """Coerce a JointState-ish value (dict with 'positions' / sequence) to list."""
    if isinstance(value, dict):
        value = value.get("positions", [])
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


class Connector:
    """Base connector: shared robot-control + observation logic.

    Args:
        env: An environment instance with the pinned env surface
            (``reset``/``step``/``get_observation`` → sim-native obs dict,
            ``task_completed``/``compute_reward``, video methods).
        config: An ``gap.envs.registry.EnvConfig`` (duck-typed — anything
            exposing ``arm_dof``, ``num_arms``, ``action_mode``,
            ``control_freq``, ``home_joints``, ``tcp_offset``,
            ``tcp_rotation_z``, ``arm_bases``, ``robot_urdf_path``,
            ``default_cameras``, ``is_real``).
        camera_names: Camera override; defaults to
            ``config.default_cameras``.
        ik: Optional pre-built IK backend. Defaults to a
            :class:`gap.connector.ik.CuRoboBackend` built from *config* (GPU,
            v0.8 MotionPlanner — linear motion with IK fallback via
            ``plan_to_pose``). Pass an explicit
            :class:`gap.connector.ik.PyRokiBackend` to opt into the in-process
            CPU-JAX path (e.g. for YAM/6-DOF or no-GPU machines).
    """

    def __init__(
        self,
        env: Any,
        config: Any,
        *,
        camera_names: list[str] | None = None,
        ik: Any | None = None,
    ) -> None:
        self.env = env
        self.config = config

        # --- SimState fields, populated from EnvConfig ---------------------
        self.action_mode: str = getattr(config, "action_mode", "absolute_joints")
        self.control_freq: float = float(getattr(config, "control_freq", 20.0))
        self.camera_names: list[str] = list(
            camera_names
            or getattr(config, "default_cameras", None)
            or ["agentview", "robot0_eye_in_hand"]
        )
        self._gripper_fraction: float = 1.0  # open
        self._arm_dof: int = int(getattr(config, "arm_dof", 7))
        self.num_arms: int = int(getattr(config, "num_arms", 1) or 1)
        home = getattr(config, "home_joints", None)
        self._home_joints: list[float] | None = list(home) if home is not None else None
        self._robot_urdf_path: str | None = getattr(config, "robot_urdf_path", None)
        # TCP offset from EE link to tool tip (meters, in EE frame)
        self._tcp_offset: np.ndarray | None = _as_vec3(getattr(config, "tcp_offset", None))
        # TCP rotation from EE link to tool tip (scipy Rotation or None)
        tcp_rot_z = getattr(config, "tcp_rotation_z", None)
        if tcp_rot_z is not None:
            from scipy.spatial.transform import Rotation as _Rot

            self._tcp_rotation = _Rot.from_euler("z", float(tcp_rot_z))
        else:
            self._tcp_rotation = None
        # Per-arm base positions in world frame (bimanual)
        bases = getattr(config, "arm_bases", None)
        self._arm_bases: list[tuple[float, float, float]] | None = (
            [tuple(float(v) for v in b) for b in bases] if bases is not None else None
        )
        self.is_real: bool = bool(getattr(config, "is_real", False))
        # Opt-in policy-like OSC Cartesian servo for straight-line moves
        # (``go_to_pose_cartesian``): streams clamped OSC deltas through the
        # env's ``apply_policy_action`` with NO IK/planning. Off by default —
        # it drops collision-checking on the servo'd segment, so enable only
        # where the straight-line path is known clear. Falls back to the
        # cuRobo linear plan on stall/non-convergence.
        self._servo_enabled = env_config.libero_servo()
        self._cam_suspend_depth = 0  # reentrancy guard for _quiet_motion

        # Pluggable IK backend (cuRobo by default; lazy). Pass ``ik=`` to
        # opt into PyRoKi or any other backend implementing the same surface.
        self._ik = ik

        # Step-callback seam: the DataCollector (and anything else that
        # wants synchronized per-control-step samples) registers a callable
        # ``fn(action, obs, reward, done)`` here. Fired by ``step_once`` and
        # the connector-owned move loops.
        self._step_callbacks: list[Callable[[np.ndarray | None, dict, float, bool], None]] = []

        self._tool_registry: ToolRegistry | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # IK backend (lazy)
    # ------------------------------------------------------------------

    @property
    def ik(self) -> Any:
        if self._ik is None:
            from gap.connector.ik import CuRoboBackend

            urdf_path = self._robot_urdf_path
            # "panda_description" is a robot_descriptions name, not a path.
            kwargs: dict[str, Any] = {}
            if urdf_path and not urdf_path.endswith((".urdf", ".xacro")):
                kwargs["robot_urdf"] = urdf_path
            elif urdf_path:
                kwargs["robot_urdf_path"] = urdf_path
            try:
                self._ik = CuRoboBackend(
                    arm_dof=self._arm_dof,
                    tcp_offset=self._tcp_offset,
                    tcp_rotation=self._tcp_rotation,
                    home_joints=self._home_joints,
                    arm_bases=self._arm_bases,
                    **kwargs,
                )
                logger.info(
                    "Connector.ik: using cuRobo backend (default; arm_dof=%d)",
                    self._arm_dof,
                )
            except NotImplementedError as e:
                # cuRobo doesn't bundle this arm (e.g. YAM 6-DOF). Surface a
                # clear pointer rather than silently auto-falling-back: per
                # the design decision, PyRoKi is opt-in only.
                raise RuntimeError(
                    f"Default IK backend (cuRobo) cannot serve this config: {e} "
                    f"Construct the connector with ik=PyRokiBackend(...) to "
                    f"opt into the in-process PyRoKi path."
                ) from e
        return self._ik

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        env = self.env
        return Capabilities(
            reset=hasattr(env, "reset"),
            success_check=hasattr(env, "task_completed"),
            video=hasattr(env, "enable_video_capture"),
            world_state=False,
        )

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tool_registry(self) -> ToolRegistry:
        """A fresh ToolRegistry with this connector's tools registered.

        Built once per connector. Holds exactly the connector-owned
        ``robot.*`` / ``sim.*`` tools; skill-bundle ``@tool`` registrations
        are drained into it by ``gap.execute(..., skills=...)``.
        """
        if self._tool_registry is None:
            reg = ToolRegistry()
            self._register_robot_tools(reg)
            self._register_extra_tools(reg)
            self._tool_registry = reg
        return self._tool_registry

    def _register_robot_tools(self, reg: ToolRegistry) -> None:
        rc = reg.register_callable
        # Getters — no guard tag.
        rc("robot.get_observation", self.get_observation,
           summary="Capture the current observation: all cameras + arm states.")
        rc("robot.get_camera_pose", self.get_camera_pose,
           summary="Get one camera's world pose by name.")
        rc("robot.get_ee_pose", self._tool_get_ee_pose,
           summary="Get the end-effector pose in world frame.")
        rc("robot.get_gripper", self._tool_get_gripper,
           summary="Get the gripper open fraction (0 closed, 1 open).")
        rc("robot.get_gripper_pose", self._tool_get_gripper_pose,
           summary="Get the gripper (end-effector) pose in world frame.")
        # Control — sim_step guard tag.
        rc("robot.go_to_pose", self.go_to_pose,
           summary="Move the end-effector to a world-frame pose via IK.",
           tags=("sim_step",))
        rc("robot.go_to_pose_cartesian", self.go_to_pose_cartesian,
           summary="Move the end-effector along a straight Cartesian line.",
           tags=("sim_step",))
        rc("robot.move_to_joints", self._tool_move_to_joints,
           summary="Move the arm to a joint configuration, blocking until converged.",
           tags=("sim_step",))
        rc("robot.execute_trajectory", self._tool_execute_trajectory,
           summary="Execute a joint trajectory waypoint-by-waypoint.",
           tags=("sim_step",))
        rc("robot.go_home", self.go_home,
           summary="Move all arms to the home joint configuration.",
           tags=("sim_step",))
        rc("robot.open_gripper", self.open_gripper,
           summary="Open the gripper and let physics settle.",
           tags=("sim_step",))
        rc("robot.wait", self.wait,
           summary="Pause for `seconds`, holding the last arm and gripper command.",
           tags=("sim_step",))
        rc("robot.close_gripper", self.close_gripper,
           summary="Close the gripper (optionally only `fraction` of the way, 1 = fully) "
                   "and let it settle.",
           tags=("sim_step",))
        rc("robot.set_joint_stiffness", self.set_joint_stiffness,
           summary="Change the real arm's joint stiffness (7 Nm/rad values; omit = back to "
                   "the startup values). Applied at rest; no-op in sim.",
           tags=("sim_step",))
        rc("robot.insert_twist", self.insert_twist,
           summary="Push the held part `depth_m` straight down, twisting about the tool "
                   "axis while it sticks, until it seats (gear/spline insertion).",
           tags=("sim_step",))
        # Planning.
        rc("robot.solve_ik", self._tool_solve_ik,
           summary="Solve IK for a world-frame pose; returns a joint configuration.",
           tags=("planning",))

    def _register_extra_tools(self, reg: ToolRegistry) -> None:
        """Subclass hook for ``sim.*`` / backend-specific tools."""

    # ------------------------------------------------------------------
    # Step-callback seam (DataCollector et al.)
    # ------------------------------------------------------------------

    def add_step_callback(
        self, fn: Callable[[np.ndarray | None, dict, float, bool], None]
    ) -> None:
        """Register ``fn(action, obs, reward, done)``, fired once per sim step.

        ``action`` is the low-level action applied for the step (``None``
        when the env stepped internally, e.g. via its own blocking-move
        loop); ``obs`` is the env's structured obs dict.
        """
        self._step_callbacks.append(fn)

    def remove_step_callback(self, fn: Callable) -> None:
        if fn in self._step_callbacks:
            self._step_callbacks.remove(fn)

    def _emit_step(
        self, action: np.ndarray | None, obs: dict, reward: float, done: bool
    ) -> None:
        for fn in self._step_callbacks:
            try:
                fn(action, obs, reward, done)
            except Exception:
                logger.warning("step callback %r failed", fn, exc_info=True)

    # ------------------------------------------------------------------
    # Frame helpers (verbatim from SimState)
    # ------------------------------------------------------------------

    def world_pose_to_base_frame(self, pose: Se3Pose, arm_id: int) -> Se3Pose:
        """Transform a world-frame Se3Pose into the specified arm's base frame.

        For YAM the arms are mounted vertically (identity base rotation), so
        only a translational offset is needed.
        """
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pose
        bx, by, bz = self._arm_bases[arm_id]
        p = pose["position"]
        return {
            "position": {"x": p["x"] - bx, "y": p["y"] - by, "z": p["z"] - bz},
            "rotation": dict(pose["rotation"]),
        }

    def _base_frame_to_world(self, pos: np.ndarray, arm_id: int) -> np.ndarray:
        """Translate a base-frame position to world frame."""
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pos
        base = np.array(self._arm_bases[arm_id])
        return pos + base

    # ------------------------------------------------------------------
    # Observation assembly (absorbed ObservationServicer)
    # ------------------------------------------------------------------

    def _build_camera_frame(self, cam_data: dict) -> CameraFrame:
        frame: dict[str, Any] = {
            "name": cam_data.get("name", ""),
            "rgb": cam_data["images"]["rgb"],
        }
        # Depth and intrinsics may not be available (e.g. sim without depth)
        depth = cam_data.get("images", {}).get("depth")
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            # Sim backends (robosuite/MuJoCo) emit depth as [H, W, 1];
            # gap.types.CameraFrame mandates float32 [H, W] — squeeze the
            # trailing channel once, here at the boundary.
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            frame["depth"] = d
        intrinsics = cam_data.get("intrinsics")
        if intrinsics is not None:
            frame["intrinsics"] = np.asarray(intrinsics)
        pose = cam_data.get("pose")
        if pose is not None:
            # Env convention: [x, y, z, qw, qx, qy, qz] (wxyz, converted from
            # sim-native xyzw inside the env class).
            frame["pose"] = make_pose(pose[:3], pose[3:])
        return frame  # type: ignore[return-value]

    def _build_observation(self, obs: dict) -> Observation:
        """Assemble a gap Observation from the env's structured obs dict.

        Verbatim port of ``SimState.build_observation_response`` minus the
        proto packing: rgb/depth stay numpy (no byte re-encoding), and the
        quaternions in ``robot_cartesian_pos_*`` / camera ``pose`` arrive
        wxyz from the env class.
        """
        cameras: list[CameraFrame] = []
        for cam_key in self.camera_names:
            if cam_key in obs and "images" in obs[cam_key]:
                cam = obs[cam_key]
                cam["name"] = cam_key
                cameras.append(self._build_camera_frame(cam))

        arms: list[ArmState] = []
        num_arms = getattr(self, "num_arms", 1)
        dof = self._arm_dof
        for arm_id in range(num_arms):
            joint_pos = obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1))
            joint_pos = np.asarray(joint_pos)
            joints = joint_pos[:dof] if len(joint_pos) >= dof else joint_pos
            gripper_frac = float(joint_pos[dof]) if len(joint_pos) > dof else 1.0

            cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
            ee_pos = cart[:3].copy()
            ee_quat = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
            # FK returns base-frame; convert to world frame
            ee_pos = self._base_frame_to_world(ee_pos, arm_id)

            # Surface the simulator's raw per-finger gripper qpos when
            # available. LIBERO π-series checkpoints consume this verbatim
            # as the last 2 entries of their state vector.
            gripper_qpos: list[float] = []
            if hasattr(self.env, "handle"):
                try:
                    raw = self.env._current_obs.get(  # type: ignore[attr-defined]
                        f"robot{arm_id}_gripper_qpos"
                    )
                    if raw is not None:
                        gripper_qpos = [float(x) for x in np.asarray(raw).reshape(-1)]
                except Exception:
                    gripper_qpos = []

            # Raw VLA proprioception state (policy-training-exact layout).
            proprio = obs.get(f"robot_proprio_pi05_libero_{arm_id}")

            arm: dict[str, Any] = {
                "joint_state": {"positions": np.asarray(joints, dtype=np.float64)},
                "gripper_fraction": gripper_frac,
                "ee_pose": make_pose(ee_pos, ee_quat),
            }
            if gripper_qpos:
                arm["gripper_qpos"] = np.asarray(gripper_qpos, dtype=np.float64)
            if proprio is not None:
                arm["proprio_state"] = np.asarray(proprio, dtype=np.float64).reshape(-1)
            arms.append(arm)  # type: ignore[arg-type]

        return {"cameras": cameras, "arms": arms}

    def get_observation(self) -> Observation:
        """Capture the current observation (all cameras + all arm states)."""
        obs = self.env.get_observation()
        return self._build_observation(obs)

    def get_camera_pose(self, camera_name: str) -> dict:
        """Get one camera's world pose. Raises ToolError when unknown."""
        obs = self.env.get_observation()
        if camera_name in obs and "pose" in obs[camera_name]:
            pose = obs[camera_name]["pose"]
            return {"pose": make_pose(pose[:3], pose[3:])}
        raise ToolError("robot.get_camera_pose", f"Camera '{camera_name}' not found")

    # ------------------------------------------------------------------
    # EE pose / gripper getters (verbatim from SimState)
    # ------------------------------------------------------------------

    def get_ee_pose(self, arm_id: int = 0) -> Se3Pose:
        obs = self.env.get_observation()
        cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
        ee_pos = cart[:3].copy()
        ee_quat = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
        # FK returns base-frame; convert to world frame for public API
        ee_pos = self._base_frame_to_world(ee_pos, arm_id)
        return make_pose(ee_pos, ee_quat)

    def _tool_get_ee_pose(self, arm_id: int = 0) -> dict:
        return {"pose": self.get_ee_pose(arm_id=arm_id)}

    def _tool_get_gripper_pose(self, arm_id: int = 0) -> dict:
        return {"pose": self.get_ee_pose(arm_id=arm_id)}

    def get_gripper_fraction(self, arm_id: int = 0) -> float:
        obs = self.env.get_observation()
        dof = self._arm_dof
        joint_pos = np.asarray(obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1)))
        return float(joint_pos[dof]) if len(joint_pos) > dof else self._gripper_fraction

    def _tool_get_gripper(self, arm_id: int = 0) -> dict:
        return {"position": self.get_gripper_fraction(arm_id=arm_id)}

    # ------------------------------------------------------------------
    # Low-level sim operations (verbatim from SimState)
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        target_joints: list[float],
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
        settle_steps: int | None = None,
    ) -> None:
        """Move to target joints, blocking until converged, then settle.

        ``max_steps == 0`` is a streaming fire-and-forget path: publish the
        joint target once and return without polling or settling. Only
        meaningful on envs whose ``move_to_joints_blocking`` honors the same
        convention (e.g. the real Franka env).
        """
        dof = self._arm_dof
        target = np.array(target_joints[:dof], dtype=np.float64)

        if hasattr(self.env, "move_to_joints_blocking"):
            self.env.move_to_joints_blocking(
                target, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id
            )
            if max_steps == 0:
                return
            # Let physics settle after convergence (YAM needs this; Panda doesn't)
            if settle_steps is None:
                settle_steps = 30 if dof == 6 else 0
            self._settle(settle_steps)
            return

        # Manual convergence loop (single-arm fallback for sim envs)
        for step in range(max_steps):
            obs = self.env.get_observation()
            current = np.asarray(
                obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1))
            )[:dof]
            if np.linalg.norm(current - target) < tolerance and step > 0:
                break

            if self.action_mode == "velocity_joints":
                delta = (target - current) * self.control_freq
                action = np.concatenate([delta, [1.0 - self._gripper_fraction * 2.0]])
            else:
                gv = 1.0 - self._gripper_fraction * 2.0
                action = np.concatenate([target, [gv, gv]])[: dof + 1]

            self.env.handle.step(action)
            self._emit_step(
                action,
                obs,
                float(self.env.compute_reward() or 0.0),
                bool(getattr(self.env, "_current_done", False)),
            )

    def stream_trajectory(
        self,
        waypoints: list[list[float]],
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        """Feed-forward stream a joint trajectory (1 sim tick per waypoint).

        Returns True if the env supports streaming and the call ran; False
        if the caller should fall back to per-waypoint ``move_to_joints``.
        """
        if not hasattr(self.env, "stream_joint_trajectory"):
            return False
        dof = self._arm_dof
        wps = [list(np.asarray(w, dtype=np.float64)[:dof]) for w in waypoints]
        self.env.stream_joint_trajectory(
            wps,
            settle_tolerance=settle_tolerance,
            settle_max_steps=settle_max_steps,
            arm_id=arm_id,
        )
        return True

    def step_once(self) -> tuple[dict, float, bool, bool, dict]:
        """Execute one sim step holding current position + current gripper.

        Calls ``env._step_once()`` which advances physics with zero joint
        velocity and the current gripper fraction (set via
        :meth:`set_gripper`). Returns ``(obs, reward, terminated, truncated,
        info)``.
        """
        self.env._step_once()
        obs = self.env.get_observation()

        reward = self.env.compute_reward()
        done = getattr(self.env, "_current_done", False)
        truncated = self.env._sim_step_count >= self.env.max_steps
        self._emit_step(None, obs, float(reward or 0.0), bool(done))
        info: dict = {}
        latency_fn = getattr(self.env, "get_latency_info", None)
        if latency_fn is not None:
            try:
                info.update(latency_fn())
            except Exception:
                pass
        return obs, reward, bool(done), truncated, info

    def _settle(self, n: int) -> None:
        """Advance ``n`` hold steps (physics + video frame only). Skips the
        per-step ``get_observation`` + ``compute_reward`` + callback emit that
        ``step_once`` does — a hold-in-place settle needs none of it — UNLESS a
        step callback (e.g. a DataCollector) is attached, in which case each
        step is recorded via the full ``step_once`` path."""
        if self._step_callbacks or not hasattr(self.env, "_step_once"):
            for _ in range(int(n)):
                self.step_once()
        else:
            for _ in range(int(n)):
                self.env._step_once()

    def set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set gripper open fraction (0 closed, 1 open). Local call."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        if hasattr(self.env, "_set_gripper"):
            self.env._set_gripper(self._gripper_fraction, arm_id=arm_id)

    # ------------------------------------------------------------------
    # Gripper (absorbed GripperServicer)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _quiet_motion(self):
        """Suspend the env's per-step camera rendering for a motion/settle
        segment (``GAP_LIBERO_MOTION_RENDER=0``), restoring + refreshing on exit
        so the next perception read sees fresh frames. Reentrant; a no-op when
        the env doesn't support it or rendering-in-motion is on."""
        env = self.env
        setf = getattr(env, "set_cameras_active", None)
        if (
            setf is None
            or getattr(env, "_motion_render", True)
            or self._cam_suspend_depth > 0
            or self._step_callbacks  # a DataCollector needs per-step frames
        ):
            self._cam_suspend_depth += 1
            try:
                yield
            finally:
                self._cam_suspend_depth -= 1
            return
        self._cam_suspend_depth += 1
        setf(False)
        try:
            yield
        finally:
            setf(True)
            self._cam_suspend_depth -= 1
            try:
                env.refresh_camera_obs()
            except Exception:
                logger.debug("refresh_camera_obs failed", exc_info=True)

    def _wait_gripper(
        self, target_open: float, arm_id: int = 0, timeout_s: float = 5.0
    ) -> float:
        """Real hardware: block until the fingers reach ``target_open`` or stop moving
        (closed on an object). Returns the measured open fraction.

        "Stopped" only counts after the fingers have moved: right after the command the
        reading is still the old position (the command takes a few ticks to reach the
        gripper). If they never move — already there, blocked, or a dry run — it returns
        after ``never_moved_s``.
        """
        reached_tol = 0.02  # open fraction; Robotiq counts are 1/255
        still_tol = 0.004
        still_s = 0.3  # longer than the gripper state's ~20 Hz refresh
        never_moved_s = 1.0
        t0 = time.monotonic()
        first = self.get_gripper_fraction(arm_id=arm_id)
        moved = False
        history: list[tuple[float, float]] = []  # (t, position) over the last still_s
        while True:
            pos = self.get_gripper_fraction(arm_id=arm_id)
            now = time.monotonic()
            if abs(pos - target_open) <= reached_tol:
                return pos
            moved = moved or abs(pos - first) > still_tol
            history.append((now, pos))
            while history and history[0][0] < now - still_s:
                history.pop(0)
            span = max(p for _, p in history) - min(p for _, p in history)
            # still = moved less than still_tol over a whole still_s window (a slow
            # gripper creeps less than that between two polls, but not over 0.3 s)
            full_window = now - t0 >= still_s and now - history[0][0] >= still_s - 0.03
            if moved and full_window and span <= still_tol:
                return pos  # stopped short of the target: closed on the part
            if not moved and now - t0 >= never_moved_s:
                logger.info(
                    "gripper did not move in %.1f s (target %.2f, at %.2f)",
                    never_moved_s, target_open, pos,
                )
                return pos
            if now - t0 >= timeout_s:
                logger.warning(
                    "gripper still moving after %.1f s (target %.2f, at %.2f); continuing",
                    timeout_s, target_open, pos,
                )
                return pos
            time.sleep(0.02)

    def open_gripper(
        self, settle_steps: int = 40, arm_id: int = 0, timeout_s: float = 5.0
    ) -> dict:
        """Open the gripper. Simulation: run the settle loop (default 40 steps). Real
        hardware: block until the fingers are open (at most ``timeout_s``);
        ``settle_steps`` is ignored."""
        settle = settle_steps if settle_steps > 0 else 40
        self.set_gripper(1.0, arm_id=arm_id)
        if self.is_real:
            position = self._wait_gripper(1.0, arm_id=arm_id, timeout_s=timeout_s)
            logger.info("robot.open_gripper arm=%d -> position=%.4f", arm_id, position)
            return {"position": position}
        with self._quiet_motion():
            self._settle(settle)
        position = self.get_gripper_fraction(arm_id=arm_id)
        logger.info(
            "robot.open_gripper arm=%d settle=%d -> position=%.4f",
            arm_id, settle, position,
        )
        return {"position": position}

    def wait(self, seconds: float = 1.0) -> None:
        """Pause for ``seconds`` without commanding anything new.

        Real hardware: wall-clock sleep. The env's republisher keeps sending the
        last arm/gripper target, so a move that is still arriving finishes during
        the wait (a settle loop would re-target the arm to where it is right now).
        Simulation: advance ``seconds * control_freq`` hold steps, since sim time
        only moves with physics steps.
        """
        seconds = max(0.0, float(seconds))
        logger.info("robot.wait %.2f s", seconds)
        if self.is_real:
            time.sleep(seconds)
            return
        with self._quiet_motion():
            self._settle(round(seconds * self.control_freq))

    def close_gripper(
        self,
        settle_steps: int = 60,
        arm_id: int = 0,
        fraction: float = 1.0,
        timeout_s: float = 5.0,
    ) -> dict:
        """Close the gripper. Simulation: run the settle loop (default 60 steps). Real
        hardware: block until the fingers reach the target or stop on the part (at most
        ``timeout_s``); ``settle_steps`` is ignored.

        ``fraction`` is how far to close: 1.0 (default) closes fully, 0.5 closes
        halfway, 0.0 leaves it open. The returned ``position`` is the measured
        *open* fraction, as everywhere else in GaP.
        """
        settle = settle_steps if settle_steps > 0 else 60
        fraction = float(np.clip(fraction, 0.0, 1.0))
        self.set_gripper(1.0 - fraction, arm_id=arm_id)
        if self.is_real:
            position = self._wait_gripper(1.0 - fraction, arm_id=arm_id, timeout_s=timeout_s)
            logger.info(
                "robot.close_gripper arm=%d fraction=%.2f -> position=%.4f",
                arm_id, fraction, position,
            )
            return {"position": position}
        with self._quiet_motion():
            self._settle(settle)
        position = self.get_gripper_fraction(arm_id=arm_id)
        logger.info(
            "robot.close_gripper arm=%d fraction=%.2f settle=%d -> position=%.4f",
            arm_id, fraction, settle, position,
        )
        return {"position": position}

    def set_joint_stiffness(
        self, k_theta: list[float] | None = None, timeout_s: float = 5.0
    ) -> dict:
        """Change the real arm's joint stiffness (firmware joint impedance, Nm/rad,
        joints 1-7) — e.g. soft for an insertion, stiff again for precise moves.

        ``None`` restores the robot client's startup values (the z3 bridge's
        ``--k-theta``). The client applies it once the arm is at rest and confirms;
        this blocks until then and raises after ``timeout_s``. Simulation: no-op.
        """
        if k_theta is not None:
            k = [float(v) for v in k_theta]
            if len(k) != 7 or not all(math.isfinite(v) and 0.0 < v <= 15000.0 for v in k):
                raise ToolError(
                    "robot.set_joint_stiffness",
                    f"k_theta needs 7 values in (0, 15000] Nm/rad, got {k_theta!r}",
                )
            k_theta = k
        fn = getattr(self.env, "set_joint_impedance", None)
        if not self.is_real or fn is None:
            logger.info("robot.set_joint_stiffness: no joint impedance here (sim); ignored")
            return {"applied": False}
        logger.info(
            "robot.set_joint_stiffness %s",
            "startup values" if k_theta is None else [round(v) for v in k_theta],
        )
        if not fn(k_theta, timeout_s=timeout_s):
            raise ToolError(
                "robot.set_joint_stiffness",
                f"robot client did not confirm within {timeout_s:.1f} s. The arm must be at "
                "rest; the z3 bridge must be new enough and run --motion nuc (the tracking "
                "controller has its own stiffness).",
            )
        return {"applied": True}

    def insert_twist(
        self,
        depth_m: float,
        push_m: float = 0.003,
        twist_deg: float = 5.0,
        twist_step_deg: float = 1.0,
        max_cycles: int = 3,
        descend_step_m: float = 0.001,
        contact_lag_m: float = 0.0015,
        seated_tol_m: float = 0.0015,
        dwell_s: float = 0.15,
        relax_close_fraction: float = -1.0,
        arm_id: int = 0,
        start_pose: Se3Pose | None = None,
        max_droop_m: float = 0.01,
        lift_to_twist_m: float = 0.0,
        stall_ok_m: float = 0.0,
        stall_accept_at_contact: bool = False,
        search_radius_m: float = 0.0,
        search_step_m: float = 0.0003,
        settle_tol_m: float = 0.0001,
        settle_window_s: float = 0.4,
        settle_timeout_s: float = 4.0,
    ) -> dict:
        """Push the held part ``depth_m`` straight down, twisting about the tool axis
        whenever it sticks, until it seats.

        ``start_pose``: the pose the arm was last sent to (the column's top, e.g. the
        graph's pre_insert). A soft joint stiffness lets the arm droop under its load
        (z3 cell 2026-11-09: joint 2 sagged 0.6 deg -> 2.3 mm sideways, 4.4 mm down), so
        the measured pose is not the intended one. With ``start_pose`` the droop is
        measured once (measured vs start_pose, in the base frame: a position offset and
        an orientation offset about the tool point — gravity droop does not turn with
        the wrist) and every command is offset by it, so the tool follows the intended
        line from start_pose down; more than ``max_droop_m`` means the arm is not at
        start_pose and the tool refuses. The droop is measured only once the arm has
        settled (moved < ``settle_tol_m`` over ``settle_window_s``): right after a switch to a
        soft stiffness it is still sagging, and a droop measured then is mostly missed
        (2026-11-09: the rotor arrived off-axis and caught the ring's rim).
        Without it the line starts at the measured pose (no compensation).

        ``lift_to_twist_m`` > 0: turn only while lifted that far off the contact, then press
        again. Needed when what the part rests on can turn with it: the sun gear pressed on
        free-spinning planets drags them round by friction, so twisting under load never
        changes the teeth's relative phase (2026-11-09: 72 steps on top of the planets).

        ``stall_ok_m`` > 0: when the part's real end is not known exactly, a stall (twist
        search exhausted) within that far above the seat height still counts as inserted,
        if it holds with the push released — reported with ``"stalled": True``. Above it,
        the tool fails as before. With ``stall_accept_at_contact`` such a stall is accepted
        as soon as it happens, without twisting first (the sun gear sinks in to its end stop
        on its own; twisting there only wastes time).

        ``search_radius_m`` > 0: a part stuck on the hole's rim (stopped above the accepted
        band) is moved sideways in rings of ``search_step_m`` out to that radius, still
        pushing lightly, until it drops — the classic peg-in-hole spiral. Only then does the
        twist search start. The sideways offset where it dropped is kept.

        Start directly above the hole, part aligned. The tool steps down
        ``descend_step_m`` at a time. Contact is when the measured tool point lags the
        commanded one by more than ``contact_lag_m``; it needs a compliant arm (the z3
        bridge's soft joint impedance), and the lag bounds the push force. While stuck,
        it commands ``push_m`` below the measured height and steps the yaw through
        0 -> +twist_deg -> -twist_deg -> 0 (``max_cycles`` times) until the part drops.
        Seated = measured z within ``seated_tol_m`` of start z - ``depth_m``.

        ``relax_close_fraction`` >= 0 re-sets the gripper to that close fraction at first
        contact (e.g. 0.45 after grasping at 0.55) so the part can self-align in the
        fingers; -1 keeps the grasp. On success the push is released at the seated
        height. On failure the arm holds where it is (no pull-out) and the tool raises.
        """
        if depth_m <= 0:
            raise ToolError("robot.insert_twist", "depth_m must be > 0")
        from scipy.spatial.transform import Rotation

        def mat(pose: Se3Pose) -> np.ndarray:
            r_ = pose["rotation"]
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat([r_["x"], r_["y"], r_["z"], r_["w"]]).as_matrix()
            t[:3, 3] = _as_vec3(pose["position"])
            return t

        settle_s = 0.0
        if start_pose and self.is_real:
            t0 = time.monotonic()
            hist: list[tuple[float, np.ndarray]] = []
            while True:
                p_now = _as_vec3(self.get_ee_pose(arm_id=arm_id)["position"])
                now = time.monotonic()
                hist.append((now, p_now))
                hist = [h for h in hist if h[0] >= now - settle_window_s]
                spread = max(np.linalg.norm(h[1] - p_now) for h in hist)
                if now - t0 >= settle_window_s and spread < settle_tol_m:
                    break
                if now - t0 > settle_timeout_s:
                    logger.warning(
                        "robot.insert_twist: arm still moving %.2f mm after %.1f s; measuring the "
                        "droop anyway", spread * 1e3, settle_timeout_s,
                    )
                    break
                time.sleep(0.02)
            settle_s = time.monotonic() - t0
        start = self.get_ee_pose(arm_id=arm_id)
        # droop, base frame: measured position = commanded + dp, measured rotation = dR @ commanded
        dp, dR = np.zeros(3), np.eye(3)
        d_mm = d_deg = 0.0
        line = start_pose if start_pose else start
        if start_pose:
            cmd0, meas0 = mat(start_pose), mat(start)
            dp = meas0[:3, 3] - cmd0[:3, 3]
            dR = meas0[:3, :3] @ cmd0[:3, :3].T
            d_mm = float(np.linalg.norm(dp) * 1e3)
            d_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())))
            logger.info(
                "robot.insert_twist droop vs start_pose: %.1f mm (%+.1f, %+.1f, %+.1f), %.2f deg, "
                "measured after %.2f s settling (compensated)",
                d_mm, *(dp * 1e3), d_deg, settle_s,
            )
            if d_mm > max_droop_m * 1e3:
                raise ToolError(
                    "robot.insert_twist",
                    f"the arm is {d_mm:.1f} mm from start_pose (> {max_droop_m * 1e3:.0f} mm): "
                    "not at the column's top; refusing to insert",
                )
        p0 = _as_vec3(line["position"])
        r = line["rotation"]
        q0 = np.array([r["w"], r["x"], r["y"], r["z"]], dtype=np.float64)
        z_seat = float(p0[2]) - float(depth_m)
        off = self._default_tcp_offset()
        tcp = {"x": float(off[0]), "y": float(off[1]), "z": float(off[2])}

        offset = np.zeros(2)  # sideways search offset (kept once the part drops)

        def command(z: float, yaw_deg: float) -> float:
            half = math.radians(yaw_deg) / 2.0
            q = _quat_mul(q0, np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))
            want = mat(make_pose([float(p0[0] + offset[0]), float(p0[1] + offset[1]), z], q))
            # send it so that, drooped, the arm ends up at want
            qs = Rotation.from_matrix(dR.T @ want[:3, :3]).as_quat()  # xyzw
            pose = make_pose(want[:3, 3] - dp, [qs[3], qs[0], qs[1], qs[2]])
            joints = self._solve_ik(pose, tcp, arm_id=arm_id)
            if joints is None:
                raise ToolError(
                    "robot.insert_twist", f"IK failed at z={z:.4f} yaw={yaw_deg:+.1f} deg"
                )
            if self.is_real:
                self.move_to_joints(joints, max_steps=0, arm_id=arm_id)
                time.sleep(max(0.0, dwell_s))
            else:
                ticks = max(1, round(dwell_s * self.control_freq))
                self.move_to_joints(joints, tolerance=0.002, max_steps=ticks, arm_id=arm_id)
            return float(self.get_ee_pose(arm_id=arm_id)["position"]["z"])

        pattern = _twist_pattern(twist_deg, twist_step_deg, max_cycles)
        search = _search_rings(search_radius_m, search_step_m)
        search_i = 0
        yaw = 0.0
        contact_z: float | None = None
        twists = 0
        with self._quiet_motion():
            z_cmd = float(p0[2])
            z = command(z_cmd, yaw)
            sag = z - z_cmd  # free-space offset (gravity / model error), not contact
            z_floor = z_seat - sag - abs(push_m)  # never command deeper than this
            logger.info(
                "robot.insert_twist start z=%.4f seat=%.4f sag=%+.4f pattern=%d steps",
                p0[2], z_seat, sag, len(pattern),
            )
            false_seats = 0
            while True:
                if z <= z_seat + seated_tol_m:
                    # Confirm with the push released: command the seat height itself and
                    # check the arm gets there. Pressing can squeeze a rotor that is still
                    # on top of the teeth into the tolerance (2026-11-09: 1.3 mm short,
                    # "seated" while pushing, 1.6 mm short again once released).
                    z_cmd = z_seat - sag
                    z = command(z_cmd, yaw)
                    if self.is_real:
                        time.sleep(max(0.0, dwell_s))
                        z = float(self.get_ee_pose(arm_id=arm_id)["position"]["z"])
                    if z <= z_seat + seated_tol_m:
                        logger.info(
                            "robot.insert_twist SEATED z=%.4f yaw=%+.1f deg after %d twist steps",
                            z, yaw, twists,
                        )
                        return {
                            "inserted": True, "final_z": z, "contact_z": contact_z,
                            "yaw_deg": yaw, "twist_steps": twists, "false_seats": false_seats,
                            "droop_mm": d_mm, "droop_deg": d_deg,
                        }
                    false_seats += 1
                    logger.info(
                        "robot.insert_twist not seated once released: z=%.4f, %.1f mm above "
                        "seat; twisting on", z, (z - z_seat) * 1e3,
                    )
                    # fall through: stuck at z with the arm commanded below it -> twist
                lag = z - z_cmd - sag
                if lag <= contact_lag_m and z_cmd > z_floor + 1e-6:
                    z_cmd = max(z_cmd - descend_step_m, z_floor)  # free: keep going down
                    z = command(z_cmd, yaw)
                    continue
                if contact_z is None:
                    contact_z = z
                    logger.info(
                        "robot.insert_twist contact at z=%.4f (%.1f mm above seat)",
                        z, (z - z_seat) * 1e3,
                    )
                    if relax_close_fraction >= 0:
                        relaxed = 1.0 - float(np.clip(relax_close_fraction, 0.0, 1.0))
                        self.set_gripper(relaxed, arm_id=arm_id)
                        if self.is_real:
                            self._wait_gripper(relaxed, arm_id=arm_id, timeout_s=2.0)
                        else:
                            self._settle(round(0.5 * self.control_freq))
                if stall_accept_at_contact and stall_ok_m > 0 and z <= z_seat + stall_ok_m:
                    z = command(z - sag, yaw)  # release the push: does it stay there?
                    if self.is_real:
                        time.sleep(max(0.0, dwell_s))
                        z = float(self.get_ee_pose(arm_id=arm_id)["position"]["z"])
                    if z <= z_seat + stall_ok_m:
                        logger.info(
                            "robot.insert_twist stopped %.1f mm above the planned end, inside the "
                            "accepted %.1f mm: inserted as far as it goes", (z - z_seat) * 1e3,
                            stall_ok_m * 1e3,
                        )
                        return {
                            "inserted": True, "stalled": True, "final_z": z,
                            "contact_z": contact_z, "yaw_deg": yaw, "twist_steps": twists,
                            "false_seats": false_seats, "droop_mm": d_mm, "droop_deg": d_deg,
                        }
                on_rim = stall_ok_m <= 0 or z > z_seat + stall_ok_m
                if search_radius_m > 0 and on_rim and search_i < len(search):
                    offset[:] = search[search_i]
                    search_i += 1
                    z_cmd = max(z - sag - abs(push_m), z_floor)
                    z = command(z_cmd, yaw)
                    logger.info(
                        "robot.insert_twist search %d/%d offset (%+.1f, %+.1f) mm z=%.4f (%.1f mm to seat)",
                        search_i, len(search), offset[0] * 1e3, offset[1] * 1e3, z, (z - z_seat) * 1e3,
                    )
                    continue
                if twists >= len(pattern):
                    break
                if lift_to_twist_m > 0:
                    free = z - sag + lift_to_twist_m
                    command(free, yaw)  # off the contact at the old angle
                    yaw = pattern[twists]
                    command(free, yaw)  # turn while nothing is touching
                else:
                    yaw = pattern[twists]
                twists += 1
                z_cmd = max(z - sag - abs(push_m), z_floor)  # push_m past where it is
                z = command(z_cmd, yaw)
                logger.info(
                    "robot.insert_twist twist %d/%d yaw=%+.1f deg z=%.4f (%.1f mm to seat)",
                    twists, len(pattern), yaw, z, (z - z_seat) * 1e3,
                )
            z_hold = command(z - sag, yaw)  # stop pushing, stay put
            if stall_ok_m > 0:
                if self.is_real:
                    time.sleep(max(0.0, dwell_s))
                    z_hold = float(self.get_ee_pose(arm_id=arm_id)["position"]["z"])
                if z_hold <= z_seat + stall_ok_m:
                    logger.info(
                        "robot.insert_twist stopped %.1f mm above the planned end, within the "
                        "accepted %.1f mm: inserted as far as it goes", (z_hold - z_seat) * 1e3,
                        stall_ok_m * 1e3,
                    )
                    return {
                        "inserted": True, "stalled": True, "final_z": z_hold,
                        "contact_z": contact_z, "yaw_deg": yaw, "twist_steps": twists,
                        "false_seats": false_seats, "droop_mm": d_mm, "droop_deg": d_deg,
                    }
        raise ToolError(
            "robot.insert_twist",
            f"not seated after {twists} twist steps: z={z_hold:.4f}, seat={z_seat:.4f} "
            f"({(z_hold - z_seat) * 1e3:.1f} mm short); holding here",
        )

    # ------------------------------------------------------------------
    # IK orchestration (absorbed RobotControlServicer)
    # ------------------------------------------------------------------

    def _default_tcp_offset(self) -> np.ndarray:
        """Fallback TCP offset used when the caller doesn't provide one.

        Prefer the value the env config supplied (``self._tcp_offset`` —
        franka_real uses -0.157 m for the Robotiq tool, libero uses -0.1 m
        for the panda hand). Fall back to the historical Panda fingertip
        default of -0.1 m along Z only when no config offset is configured.
        YAM's fixed TCP is applied inside the backend itself.
        """
        if self._tcp_offset is not None:
            return np.asarray(self._tcp_offset, dtype=np.float64).reshape(3)
        return np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def _current_joints(self, arm_id: int) -> list[float] | None:
        """Read the current joint positions (simulator order) for seeding IK."""
        if self.env is None:
            return None
        obs = self.env.get_observation()
        cur = obs.get(f"robot_joint_pos_{arm_id}")
        if cur is None:
            return None
        return list(np.asarray(cur[: self._arm_dof], dtype=np.float64))

    def _solve_ik(
        self,
        pose: Se3Pose,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
    ) -> list[float] | None:
        """Delegate to the configured IK backend.

        Returns joint positions in simulator-native order, or None.
        """
        if self.ik is None:
            raise RuntimeError("IK backend not configured")
        off = _as_vec3(tcp_offset)
        if off is None:
            off = self._default_tcp_offset()
        seed = self._current_joints(arm_id)
        joints = self.ik.solve_ik(
            pose,
            arm_id=arm_id,
            seed_joints=seed,
            tcp_offset=off,
        )
        if joints is None:
            logger.warning("_solve_ik FAILED for arm=%d", arm_id)
            return None
        p = pose["position"]
        logger.info(
            "_solve_ik arm=%d tcp=(%.4f,%.4f,%.4f) joints=[%s]",
            arm_id, p["x"], p["y"], p["z"],
            ", ".join(f"{j:.4f}" for j in joints),
        )
        return joints

    def _tool_solve_ik(
        self,
        pose: Se3Pose,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
    ) -> dict:
        """Tool wrapper: raises ToolError on failure (source: INVALID_ARGUMENT)."""
        joints = self._solve_ik(pose, tcp_offset, arm_id=arm_id)
        if joints is None:
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        return {"joint_config": {"positions": list(joints)}, "success": True}

    def _execute_trajectory(
        self,
        trajectory: Trajectory,
        subsample: int,
        tolerance: float,
        max_steps_per_wp: int,
        arm_id: int = 0,
        reverse_joints: bool = False,
    ) -> None:
        """Execute a joint trajectory waypoint-by-waypoint.

        Args:
            reverse_joints: If True, reverse joint order per waypoint (needed
                for PyRoKI which returns joints in reversed URDF parse order
                for 6-DOF arms).

        Forces the env into ``closed_loop`` joint motion mode for the
        duration of the trajectory: qpos-teleport per waypoint bypasses
        contact integration, which is fine for contact-free single moves but
        breaks the descent into the target object. Single moves keep their
        configured mode.
        """
        saved_mode = getattr(self.env, "_joint_motion_mode", None)
        if saved_mode is not None and saved_mode != "closed_loop":
            self.env._joint_motion_mode = "closed_loop"
        try:
            subsample = max(subsample, 1)
            waypoints = trajectory["waypoints"]
            total = len(waypoints)
            # Streaming path: ignore `subsample` — the dense interpolation IS
            # the control sequence; one final settle on the last waypoint
            # replaces the per-waypoint convergence loop.
            if getattr(self.env, "_stream_enabled", True) and hasattr(
                self.env, "stream_joint_trajectory"
            ):
                wps_all: list[list[float]] = []
                for waypoint in waypoints:
                    joints = _as_positions(waypoint)
                    if reverse_joints:
                        joints.reverse()
                    wps_all.append(joints)
                self.stream_trajectory(
                    wps_all,
                    settle_tolerance=tolerance,
                    settle_max_steps=max(max_steps_per_wp * 4, 60),
                    arm_id=arm_id,
                )
                logger.info(
                    "robot.execute_trajectory streamed %d waypoints (subsample ignored)",
                    total,
                )
                return
            # Per-waypoint blocking fallback.
            executed = 0
            for i, waypoint in enumerate(waypoints):
                if i % subsample != 0 and i != total - 1:
                    continue
                joints = _as_positions(waypoint)
                if reverse_joints:
                    joints.reverse()
                executed += 1
                self.move_to_joints(
                    joints, tolerance=tolerance, max_steps=max_steps_per_wp, arm_id=arm_id,
                )
            logger.info(
                "robot.execute_trajectory: executed %d/%d waypoints", executed, total,
            )
        finally:
            if saved_mode is not None:
                self.env._joint_motion_mode = saved_mode

    # ------------------------------------------------------------------
    # High-level motion tools (absorbed RobotControlServicer handlers)
    # ------------------------------------------------------------------

    def go_to_pose(
        self,
        pose: Se3Pose,
        tolerance: float = 0.0,
        max_steps: int = 0,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
        z_approach: float = 0.0,
        nonblocking: bool = False,
    ) -> None:
        """Move the end-effector to a world-frame pose via IK.

        ``tolerance``/``max_steps`` <= 0 take the source defaults (0.01 rad /
        120 steps for Panda; 0.03 / 300 for 6-DOF arms). ``z_approach > 0``
        first moves to a point that far above the target.
        """
        if not pose:
            raise ToolError("robot.go_to_pose", "pose required")
        # NB: ``go_to_pose`` is NOT servo'd. It is the contact-rich grasp
        # descend (and precision hand-off) path, which needs cuRobo's IK +
        # controlled approach; an OSC straight-line servo onto the object
        # stalls on contact and misses the grasp (measured reward 1.0 -> 0.0).
        # Only the free-space straight legs (``go_to_pose_cartesian``) servo.

        if tcp_offset is None:
            # Use the env's configured TCP (franka_real: -0.157, libero: -0.1)
            # rather than a hardcoded Panda default — the latter silently
            # shifts the IK target by ~5.7 mm on the real Robotiq stack.
            off = self._default_tcp_offset()
            tcp_offset = {"x": float(off[0]), "y": float(off[1]), "z": float(off[2])}
        is_yam = self._arm_dof == 6
        tolerance = tolerance if tolerance > 0 else (0.03 if is_yam else 0.01)
        # nonblocking=True forces max_steps=0, which makes move_to_joints_blocking
        # publish the command once and return. Each new go_to_pose call
        # supersedes the last.
        if nonblocking:
            max_steps = 0
        else:
            max_steps = max_steps if max_steps > 0 else (300 if is_yam else 120)

        with self._quiet_motion():
            # z_approach: first move above target
            if z_approach > 0 and pose.get("position"):
                p = pose["position"]
                approach_pose: Se3Pose = {
                    "position": {"x": p["x"], "y": p["y"], "z": p["z"] + z_approach},
                    "rotation": dict(pose["rotation"]),
                }
                joints = self._solve_ik(approach_pose, tcp_offset, arm_id=arm_id)
                if joints is None:
                    raise ToolError("robot.go_to_pose", "IK failed for approach pose")
                self.move_to_joints(joints, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id)

            joints = self._solve_ik(pose, tcp_offset, arm_id=arm_id)
            if joints is None:
                raise ToolError("robot.go_to_pose", "IK failed for target pose")
            self.move_to_joints(joints, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id)

        # Post-move verification: log actual EE position vs target. Skip in
        # nonblocking mode — the arm hasn't reached the target yet.
        if not nonblocking:
            actual_ee = self.get_ee_pose(arm_id=arm_id)
            tp = pose["position"]
            ap = actual_ee["position"]
            err = (
                (tp["x"] - ap["x"]) ** 2
                + (tp["y"] - ap["y"]) ** 2
                + (tp["z"] - ap["z"]) ** 2
            ) ** 0.5
            logger.info(
                "go_to_pose arm=%d target=(%.4f,%.4f,%.4f) actual_ee=(%.4f,%.4f,%.4f) err=%.4f",
                arm_id, tp["x"], tp["y"], tp["z"], ap["x"], ap["y"], ap["z"], err,
            )

    def _servo_to_pose(
        self,
        pose: Se3Pose,
        *,
        arm_id: int = 0,
        pos_tol: float = 0.005,
        rot_tol: float = 0.05,
        max_ticks: int = 200,
        stall_ticks: int = 25,
    ) -> None:
        """Drive the EE to ``pose`` (world frame) by streaming clamped OSC
        deltas through the env's ``apply_policy_action`` — the policy's own
        actuator, NO IK/planning. Policy-like and fast, but collision-UNAWARE,
        so it is used only for known-safe straight-line segments; on stall or
        non-convergence it raises :class:`ToolError` so the caller can fall
        back to the collision-aware cuRobo linear plan.

        Frames: ``get_ee_pose`` and the target are both world-frame, and the
        LIBERO Franka base is axis-aligned with world, so a world-frame
        position delta is exactly the OSC ``control_delta`` the controller
        expects (output_max 0.05 m / 0.5 rad per step). Orientation uses
        robosuite's own ``orientation_error`` to match the OSC convention.
        """
        apply = getattr(self.env, "apply_policy_action", None)
        if apply is None:
            raise ToolError(
                "robot.go_to_pose_cartesian",
                "servo path requires env.apply_policy_action (OSC passthrough)",
            )
        from robosuite.utils.control_utils import orientation_error
        from robosuite.utils.transform_utils import quat2mat

        def _mat(q: Any) -> np.ndarray:
            if not q:
                return np.eye(3)
            return quat2mat(
                np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
            )

        tp = pose["position"]
        target_pos = np.array(
            [float(tp["x"]), float(tp["y"]), float(tp["z"])], dtype=np.float64
        )
        rot = pose.get("rotation")
        target_mat = _mat(rot) if rot else None  # None → hold current orientation
        gcmd = 1.0 - 2.0 * float(getattr(self.env, "_gripper_fraction", 1.0))
        out_p, out_r = 0.05, 0.5  # OSC output_max (m, rad)

        best = float("inf")
        stuck = 0
        pos_err = float("inf")
        for _ in range(int(max_ticks)):
            cur = self.get_ee_pose(arm_id=arm_id)
            cp = cur["position"]
            cur_pos = np.array(
                [float(cp["x"]), float(cp["y"]), float(cp["z"])], dtype=np.float64
            )
            dpos = target_pos - cur_pos
            dori = (
                orientation_error(target_mat, _mat(cur.get("rotation")))
                if target_mat is not None
                else np.zeros(3)
            )
            pos_err = float(np.linalg.norm(dpos))
            rot_err = float(np.linalg.norm(dori))
            if pos_err < pos_tol and rot_err < rot_tol:
                return
            action = np.empty(7, dtype=np.float64)
            action[:3] = np.clip(dpos / out_p, -1.0, 1.0)
            action[3:6] = np.clip(dori / out_r, -1.0, 1.0)
            action[6] = gcmd
            apply(action)
            if pos_err < best - 1e-4:
                best, stuck = pos_err, 0
            else:
                stuck += 1
                if stuck >= int(stall_ticks):
                    raise ToolError(
                        "robot.go_to_pose_cartesian",
                        f"servo stalled at pos_err={pos_err:.4f} "
                        "(joint limit / singularity?)",
                    )
        raise ToolError(
            "robot.go_to_pose_cartesian",
            f"servo did not converge in {max_ticks} ticks (pos_err={pos_err:.4f})",
        )

    def go_to_pose_cartesian(self, pose: Se3Pose, arm_id: int = 0) -> None:
        """Move along a straight Cartesian line to ``pose``.

        The caller is responsible for providing target poses already in the
        IK link frame (panda_hand for Franka, link_6 for YAM) — matching the
        source's GoToPoseCartesian contract.
        """
        if not pose:
            raise ToolError("robot.go_to_pose_cartesian", "pose required")
        # Policy-like OSC servo fast-path (opt-in via GAP_LIBERO_SERVO): stream
        # clamped OSC deltas with NO IK/planning for this straight-line move.
        # Falls back to the collision-aware cuRobo linear plan on stall.
        if self._servo_enabled and hasattr(self.env, "apply_policy_action"):
            start_joints = self._current_joints(arm_id)  # pre-servo config
            try:
                with self._quiet_motion():
                    self._servo_to_pose(pose, arm_id=arm_id)
                return
            except ToolError as exc:
                logger.warning(
                    "[servo] cartesian servo failed (%s); restoring pre-servo "
                    "config and falling back to cuRobo linear plan", exc,
                )
                # A stalled servo may have left the arm partway — and possibly
                # in collision, which makes the cuRobo fallback fail with a
                # "start state in collision" (e.g. a cluttered grasp leg in
                # grocery_packing). Return to the pre-servo config so the
                # planner solves from the same state it would have without the
                # servo attempt.
                if start_joints is not None:
                    try:
                        with self._quiet_motion():
                            self.move_to_joints(
                                start_joints, tolerance=0.02, max_steps=200,
                                arm_id=arm_id,
                            )
                    except Exception:
                        logger.debug("[servo] restore-to-start failed", exc_info=True)
        if self.ik is None:
            raise ToolError("robot.go_to_pose_cartesian", "IK backend not configured")

        off = self._default_tcp_offset()
        is_yam = self._arm_dof == 6
        ee_pose = self.get_ee_pose(arm_id=arm_id)
        # Pass the current joint config as the linear seed — cuRobo's
        # MotionPlanner needs the actual start_config for FK; PyRoKi
        # accepts and ignores it.
        seed = self._current_joints(arm_id)
        trajectory = self.ik.plan_linear(
            ee_pose, pose, arm_id=arm_id, tcp_offset=off, seed_joints=seed,
        )
        if trajectory is None:
            raise ToolError("robot.go_to_pose_cartesian", "linear plan failed")
        tolerance = 0.03 if is_yam else 0.01
        max_steps = 200 if is_yam else 120
        with self._quiet_motion():
            self._execute_trajectory(
                trajectory, 1, tolerance, max_steps,
                arm_id=arm_id,
                reverse_joints=self.ik.trajectory_needs_joint_reverse,
            )

    def _tool_execute_trajectory(
        self,
        trajectory: Trajectory,
        subsample: int = 0,
        tolerance: float = 0.0,
        max_steps_per_waypoint: int = 0,
    ) -> None:
        """Execute a joint trajectory (source ExecuteJointTrajectory defaults)."""
        if not trajectory or not trajectory.get("waypoints"):
            raise ToolError("robot.execute_trajectory", "trajectory required")
        subsample = subsample if subsample > 0 else 1
        tolerance = tolerance if tolerance > 0 else 0.01
        max_steps = max_steps_per_waypoint if max_steps_per_waypoint > 0 else 120
        self._execute_trajectory(trajectory, subsample, tolerance, max_steps)

    def _tool_move_to_joints(
        self,
        joint_config: Any,
        tolerance: float = 0.0,
        max_steps: int = 0,
    ) -> None:
        """Move the arm to a joint configuration (source MoveToJoints defaults)."""
        positions = _as_positions(joint_config) if joint_config is not None else []
        if not positions:
            raise ToolError("robot.move_to_joints", "joint_config required")
        tolerance = tolerance if tolerance > 0 else 0.01
        max_steps = max_steps if max_steps > 0 else 120
        self.move_to_joints(positions, tolerance=tolerance, max_steps=max_steps)

    def go_home(self) -> None:
        """Move every arm to the home joint configuration."""
        # Safety: skip GoHome on real robot suites to avoid collisions
        if self.is_real:
            logger.warning("go_home skipped on real robot suite (safety)")
            return
        home = self._home_joints or _FRANKA_HOME_JOINTS
        with self._quiet_motion():
            for arm_id in range(self.num_arms):
                self.move_to_joints(home, tolerance=0.01, max_steps=200, arm_id=arm_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying environment (idempotent)."""
        if self._closed:
            return
        self._closed = True
        env = self.env
        if env is None:
            return
        try:
            if hasattr(env, "close"):
                env.close()
            elif hasattr(env, "handle"):
                env.handle.env.close()
        except Exception:
            logger.debug("env close failed", exc_info=True)

    def __enter__(self) -> Connector:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Capabilities", "Connector"]
