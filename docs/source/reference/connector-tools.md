# Connector Tools

Every connector ships a small, fixed set of embodiment tools: thirteen
`robot.*` tools registered by the connector base class, plus six `sim.*`
tools added by simulation connectors. These are the engine's entire
built-in tool surface — everything else (perception, geometry, motion
planning) comes from skill bundles, documented in the
[Tool catalog](../skills/tool-catalog.md).

Graphs reach these tools two ways:

- a `type: "tool"` node in `workflow.json`, e.g. `"tool": "robot.go_to_pose"`;
- `ctx.tool("robot.go_to_pose", pose=..., ...)` from inside a script node.

Both go through the same flat dispatch surface, so the names below are
exactly what you write in either place. The registry is built once per
connector ([`gap/connector/core.py`](gh-engine:gap/connector/core.py));
bundle `@tool` registrations are drained into it by
`gap.execute(..., skills=...)`.

## Conventions

- **Poses** are `Se3Pose` dicts: `{"position": {"x", "y", "z"}, "rotation":
  {"w", "x", "y", "z"}}` — quaternions are **wxyz scalar-first**
  everywhere in GaP.
- **Gripper fraction** is `0.0` = fully closed, `1.0` = fully open.
- **Joint configs** are `{"positions": [...]}` dicts (a bare list is also
  accepted); a **trajectory** is `{"waypoints": [joint_config, ...]}`.
- Failures raise `ToolError(tool, detail)` — a `PipelineError` subclass
  that workflow `on_error` exits can catch.

### Guard tags

Each tool carries zero or one guard tag. Tags feed the call-count guards
in `gap.tools.guards`: when a cap is configured — via the
`GAP_MAX_SIM_STEPS` / `GAP_MAX_PLANNING_CALLS` /
`GAP_MAX_PERCEPTION_CALLS` environment variables or per-workflow
`guards.set_limits()` — every dispatch of a tagged tool increments a
process-wide counter against it (uncapped categories skip the counter
entirely). Exceeding a cap raises
`GuardLimitExceeded`, which deliberately extends `BaseException` so a
runaway loop cannot swallow it with `except Exception`. See
[Environment variables](environment-variables.md#safety-guards) and
[Executor semantics](executor.md).

| Tag | Cap variable | Tools |
|---|---|---|
| *(none)* | — | the five `robot.get_*` getters, `sim.reset`, `sim.check_success`, `sim.enable_video`, `sim.save_video` |
| `sim_step` | `GAP_MAX_SIM_STEPS` | all seven `robot.*` motion tools, `sim.step`, `sim.apply_policy_action` |
| `planning` | `GAP_MAX_PLANNING_CALLS` | `robot.solve_ik` |

No connector tool carries the `perception` tag — that one is used by
perception bundles (see the [Tool catalog](../skills/tool-catalog.md)).

## robot.* — observation getters

Untagged, read-only, available on every connector (sim and real).

| Tool | Parameters | Returns |
|---|---|---|
| `robot.get_observation` | — | `Observation`: `{"cameras": [CameraFrame, ...], "arms": [ArmState, ...]}` |
| `robot.get_camera_pose` | `camera_name: str` | `{"pose": Se3Pose}` (camera-to-world); `ToolError` if the camera is unknown |
| `robot.get_ee_pose` | `arm_id: int = 0` | `{"pose": Se3Pose}` — end-effector in world frame |
| `robot.get_gripper` | `arm_id: int = 0` | `{"position": float}` — open fraction, 0 closed → 1 open |
| `robot.get_gripper_pose` | `arm_id: int = 0` | `{"pose": Se3Pose}` — same pose as `robot.get_ee_pose` |

## robot.* — motion (tag: `sim_step`)

| Tool | Parameters | Returns |
|---|---|---|
| `robot.go_to_pose` | `pose: Se3Pose`, `tolerance: float = 0.0`, `max_steps: int = 0`, `tcp_offset = None`, `arm_id: int = 0`, `z_approach: float = 0.0`, `nonblocking: bool = False` | `None` |
| `robot.go_to_pose_cartesian` | `pose: Se3Pose`, `arm_id: int = 0` | `None` |
| `robot.move_to_joints` | `joint_config`, `tolerance: float = 0.0`, `max_steps: int = 0` | `None` |
| `robot.execute_trajectory` | `trajectory: Trajectory`, `subsample: int = 0`, `tolerance: float = 0.0`, `max_steps_per_waypoint: int = 0` | `None` |
| `robot.go_home` | — | `None` |
| `robot.open_gripper` | `settle_steps: int = 40`, `arm_id: int = 0`, `timeout_s: float = 5.0` | `{"position": float}` — measured fraction after settling |
| `robot.close_gripper` | `settle_steps: int = 60`, `arm_id: int = 0`, `fraction: float = 1.0`, `timeout_s: float = 5.0` | `{"position": float}` |
| `robot.wait` | `seconds: float = 1.0` | `None` |
| `robot.set_joint_stiffness` | `k_theta: list[float] \| None = None`, `timeout_s: float = 5.0` | `{"applied": bool}`; `ToolError` when the robot client does not confirm |
| `robot.insert_twist` | `depth_m: float`, `push_m: float = 0.003`, `twist_deg: float = 5.0`, `twist_step_deg: float = 1.0`, `max_cycles: int = 3`, `descend_step_m: float = 0.001`, `contact_lag_m: float = 0.0015`, `seated_tol_m: float = 0.0015`, `dwell_s: float = 0.15`, `relax_close_fraction: float = -1.0`, `arm_id: int = 0` | `{"inserted": True, "final_z", "contact_z", "yaw_deg", "twist_steps"}`; `ToolError` when not seated |

Behavioral details, verified against
[`gap/connector/core.py`](gh-engine:gap/connector/core.py):

- **`robot.go_to_pose`** solves IK for the world-frame pose, then runs a
  blocking joint move. Zero-valued parameters take embodiment defaults:
  `tolerance <= 0` → 0.01 rad (7-DOF Panda) or 0.03 rad (6-DOF arms);
  `max_steps <= 0` → 120 (Panda) or 300 (6-DOF). `tcp_offset=None`
  defaults to `{"x": 0, "y": 0, "z": -0.1}` (meters, EE frame).
  `z_approach > 0` first moves to a point that far above the target.
  `nonblocking=True` forces `max_steps=0`, publishing the command once
  and returning immediately — each new call supersedes the last.
  IK failure raises `ToolError`.
- **`robot.go_to_pose_cartesian`** plans a straight Cartesian line from
  the current EE pose and executes it. The target must already be in the
  **IK link frame** (`panda_hand` for Franka, `link_6` for 6-DOF arms).
  Raises `ToolError` when the linear plan fails.
- **`robot.move_to_joints`** blocks until convergence; `tolerance <= 0` →
  0.01 rad, `max_steps <= 0` → 120. Empty `joint_config` raises
  `ToolError`.
- **`robot.execute_trajectory`** runs waypoints with defaults
  `subsample` → 1, `tolerance` → 0.01, `max_steps_per_waypoint` → 120.
  It forces the env into `closed_loop` joint-motion mode for the
  duration (qpos-teleporting per waypoint would bypass contact
  integration and break descents into objects). On envs with a streaming
  joint interface, `subsample` is ignored — the dense interpolation *is*
  the control sequence.
- **`robot.go_home`** moves every arm to the `EnvConfig.home_joints`
  configuration (Franka default when unset). **On real hardware it is a
  no-op**: when `config.is_real` is true the call logs a warning and
  returns without moving, to avoid blind multi-joint sweeps into the
  workspace.
- **`robot.open_gripper` / `robot.close_gripper`** set the gripper
  fraction and run a physics settle loop (`settle_steps <= 0` falls back
  to the 40/60-step defaults), then return the measured fraction.
  `close_gripper`'s `fraction` is how far to close (1.0 = fully, the
  default; 0.5 = halfway), clipped to [0, 1]; the returned `position` is
  still the measured *open* fraction. On real hardware `settle_steps` is
  ignored: the call polls the measured finger position and returns when
  the fingers reach the target or stop on a part (once they have moved),
  after 1 s if they never move (already there, or a dry run), and at
  `timeout_s` at the latest.
- **`robot.wait`** pauses without commanding anything new. On real
  hardware it sleeps; the env keeps republishing the last arm/gripper
  target, so a move still arriving finishes during the wait. In
  simulation it advances `seconds * control_freq` hold steps.
- **`robot.set_joint_stiffness`** changes the real arm's firmware joint
  impedance (7 values, Nm/rad), e.g. soft for an insertion and stiff
  again for precise moves. `None` restores the robot client's startup
  values (the z3 bridge's `--k-theta`). libfranka accepts the change only
  while no motion runs, so the client applies it once the arm is at rest
  and echoes the request number; the tool blocks until then and raises
  after `timeout_s`. Only the bridge's `--motion nuc` uses this
  impedance. In simulation it is a no-op returning `{"applied": False}`.
- **`robot.insert_twist`** inserts a held part (a gear, spline or
  keyed shaft) from directly above the hole. It steps the tool point
  straight down `descend_step_m` at a time. When the measured height lags
  the commanded one by more than `contact_lag_m`, the part is stuck. It
  then commands `push_m` below the measured height and steps the yaw
  about the tool axis 0 → +`twist_deg` → −`twist_deg` → 0, `max_cycles`
  times, until the part drops. It is seated when the measured height is
  within `seated_tol_m` of the start height minus `depth_m`; the push is
  then released. It needs a compliant arm: the lag bounds the push
  force only if the controller lets the arm fall behind its target. On
  failure it holds where it is (no pull-out) and raises `ToolError`.
  `relax_close_fraction >= 0` re-sets the gripper at first contact so
  the part can self-align in the fingers.

## robot.* — planning (tag: `planning`)

| Tool | Parameters | Returns |
|---|---|---|
| `robot.solve_ik` | `pose: Se3Pose`, `tcp_offset = None`, `arm_id: int = 0` | `{"joint_config": {"positions": [...]}, "success": True}`; `ToolError("Target pose is unreachable")` on failure |

IK runs in-process via the pyroki backend, seeded from the current joint
positions. The first solve per robot JIT-compiles (a few seconds);
subsequent solves take milliseconds.

## sim.* — simulation control

Registered only by `SimConnector`
([`gap/connector/sim.py`](gh-engine:gap/connector/sim.py)). **Real
connectors never get `sim.*` tools** — the base-class registration hook
is a no-op, so a graph that calls `sim.check_success` simply has no such
tool on hardware (validation flags it as a warning before the run).

| Tool | Parameters | Returns | Tag |
|---|---|---|---|
| `sim.reset` | `seed: int = 0` | `{"observation": Observation}` | — |
| `sim.step` | `gripper_fraction: float = -1.0` | `{"observation", "reward", "terminated", "truncated"}` | `sim_step` |
| `sim.check_success` | — | `{"task_completed": bool, "reward": float}` | — |
| `sim.apply_policy_action` | `action: list[float]` | `None` | `sim_step` |
| `sim.enable_video` | `enabled: bool = True`, `clear: bool = True` | `None` | — |
| `sim.save_video` | `output_path: str`, `fps: int = 20`, `clear: bool = False` | `{"success": bool, "file_path": str, "num_frames": int}` | — |

- **`sim.reset`** — `seed=0` means *unseeded*; any other value seeds the
  episode's initial state.
- **`sim.step`** advances one control step holding the current pose;
  `gripper_fraction >= 0` sets the gripper first (negative leaves it).
- **`sim.apply_policy_action`** forwards one low-level action verbatim to
  the env's passthrough controller. Only envs exposing
  `apply_policy_action` support it — currently `FrankaLiberoEnv`, whose
  action space is the 7-dim OSC_POSE layout
  `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` used by pi05-libero (positive
  gripper = close). Other envs raise `ToolError`. This is the hand-off
  point for learned policies — see [Policies](../benchmarks/policies.md).
- **`sim.save_video`** encodes the buffered frames with libx264 and also
  writes per-camera extras as `<stem>_<cam>.mp4` next to the main file.
  An empty buffer returns `success: True` with `num_frames: 0`; encoding
  errors return `success: False` rather than raising.

## Availability by backend

| Backend | `robot.*` getters (5) | `robot.*` motion (7) | `robot.solve_ik` | `sim.*` (6) |
|---|---|---|---|---|
| `gap.connector.sim("libero", ...)` | yes | yes | yes | yes |
| `gap.connector.real("franka")` | yes | yes (`robot.go_home` is a no-op) | yes | **no** |
| `gap.connector.real("ur_zed")` | yes | **no** | **no** | **no** |

The UR+ZED connector is perception-only: it registers exactly the five
observation getters, so accidental motion is structurally impossible —
there is no motion tool to call
([`gap/connector/real.py`](gh-engine:gap/connector/real.py)). See
[Connectors](../real-robots/connectors.md) for backend setup and
[Safety](../real-robots/safety.md) for the full real-robot gating story.

:::{warning}
Real connectors also report `Capabilities(reset=False, success_check=False,
video=False, world_state=False)` even though the env classes expose shims
for interface parity. Harness code must branch on `connector.capabilities`,
not on whether a method exists.
:::

## Related pages

- [Tool catalog](../skills/tool-catalog.md) — bundle-provided tools
  (`sam3.*`, `geometry.*`, `curobo.*`, ...).
- [Workflow schema](workflow-schema.md) — how `type: "tool"` nodes bind
  inputs and outputs.
- [Executor semantics](executor.md) — dispatch, guards, and tracing of
  `ctx.tool` calls.
- [Environments](../running/environments.md) — the env registry behind
  each connector.
