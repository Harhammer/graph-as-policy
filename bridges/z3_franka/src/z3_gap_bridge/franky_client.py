# Copied verbatim from oktonex packages/robot/src/okto_robot/franky_client.py @ c83e4ed9.
# Fix it there first; keep this copy byte-identical below this header.
"""Workstation client for the NUC franky control service.

Phase 1 of the franky teleop migration (docs/franky-teleop-migration.md).

Design constraints, all measured (docs/nuc-robot-setup.md):
- TCP_NODELAY on the client socket (the service sets it server-side too).
- All values cross the wire as tuples of native floats — never lists (rpyc passes
  mutable objects by reference → per-element callbacks), never np.float64 elements
  (one netref each, ~1.5 ms apiece).
- Every call runs through a staleness watchdog: round trips slower than
  ``stale_threshold_s`` raise ``FrankyStaleError`` so a teleop loop can never
  silently act on stale state. A SINGLE slow round trip does not raise —
  ``stale_consecutive_limit`` of them in a row does, and any in-budget call
  resets the run. One late read is already bounded by the driver's per-tick
  ``max_delta`` clamp and Ruckig; sustained staleness is the actual hazard.
  A round trip over ``2 x stale_threshold_s`` still raises immediately, so a
  genuine stall keeps its hair trigger.
- Legacy dropped connections trigger bounded reconnect-with-backoff, then one retry.
  A supervised claim disables reconnect/replay; command outcome can be unknown.

Usage:
    client = FrankyClient("192.168.99.100")
    client.connect_robot()
    ts, q, dq, tau, tau_ext = client.read_teleop()
    client.move_joints(target_q, rel_dyn=0.2)
"""

from __future__ import annotations

import contextlib
import logging
import math
import socket
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

log = logging.getLogger(__name__)

DEFAULT_PORT = 18861
DEFAULT_ROBOT_IP = "192.168.98.121"

_RECONNECTABLE = (EOFError, ConnectionError, socket.timeout, OSError)


class FrankyClientError(RuntimeError):
    """Base error for the franky client."""


class FrankyStaleError(FrankyClientError):
    """A round trip exceeded the staleness threshold — state may be outdated."""


class FrankyConnectionError(FrankyClientError):
    """Connection to the NUC service could not be (re-)established."""


def _default_connection_factory(host: str, port: int):
    """Connect to the service with TCP_NODELAY set. Returns an rpyc connection."""
    import rpyc

    conn = rpyc.connect(host, port, config={"sync_request_timeout": 10})
    conn._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def pack_target(target: Iterable[float]) -> tuple[float, ...]:
    """Validate and convert a 7-joint target to a tuple of native floats."""
    out = tuple(float(v) for v in target)
    if len(out) != 7:
        raise ValueError(f"joint target must have 7 elements, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError("joint target must be finite")
    return out


class FrankyClient:
    """Thin, watchdogged client for the FrankyControlService on the NUC.

    Args:
        host: NUC management IP.
        port: Service port.
        stale_threshold_s: Round-trip budget for loop-rate calls.
        stale_consecutive_limit: Consecutive over-budget round trips tolerated
            before raising FrankyStaleError. 1 restores the old hair trigger.
        stale_run_s: A run of over-budget calls must also span this many seconds
            before it raises (0 = the count alone decides; see the config comment).
        stale_warmup_s: Grace period after connect during which over-budget round
            trips are logged but never raise. 0 disables the grace.
        reconnect_attempts: Reconnects tried (with backoff) on a dropped connection.
        connection_factory: Injection point for tests; must return an rpyc-like
            connection with a ``root`` attribute and a ``close()`` method.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        stale_threshold_s: float = 0.020,
        stale_consecutive_limit: int = 3,
        stale_warmup_s: float = 2.0,
        stale_run_s: float = 0.0,
        rtt_log_interval_s: float = 0.0,
        reconnect_attempts: int = 3,
        connection_factory: Callable | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._stale_threshold_s = stale_threshold_s
        self._stale_consecutive_limit = max(1, int(stale_consecutive_limit))
        self._stale_run = 0
        self._stale_run_since = 0.0
        # A run of over-budget calls must ALSO span this long before it raises (0 = the
        # count alone decides). At 120 Hz dispatch, 3 consecutive calls are 25 ms of wall
        # time — one workstation hiccup, not a degrading service (2026-09-16 13:43).
        self._stale_run_s = max(0.0, float(stale_run_s))
        self._stale_warmup_s = stale_warmup_s
        # Set on the first watchdogged call after connect_robot(); see _check_staleness.
        self._warmup_until: float | None = None
        # Round-trip histogram for the periodic summary (0 = disabled).
        self._rtt_log_interval_s = rtt_log_interval_s
        self._rtt_samples: list[float] = []
        self._rtt_window_t0 = time.perf_counter()
        self._reconnect_attempts = reconnect_attempts
        self._factory = connection_factory or _default_connection_factory
        self._conn = None
        self._supervised = False
        self._connection_lost = False
        self._robot_ip: str | None = None
        self.last_rtt_s: float = 0.0
        # None = not probed yet; probed once via capture_info (absent on older services).
        self._supports_capture: bool | None = None

    # -- connection management ----------------------------------------------

    def _ensure_conn(self):
        if self._connection_lost:
            raise FrankyConnectionError("supervised connection lost; start a new explicit session")
        if self._conn is None:
            self._conn = self._factory(self._host, self._port)
            log.info("connected to franky service at %s:%d", self._host, self._port)
            if self._robot_ip is not None and not self._supervised:
                self._conn.root.connect(self._robot_ip)
        return self._conn

    def _reconnect(self) -> None:
        self.close()
        delay = 0.2
        for attempt in range(1, self._reconnect_attempts + 1):
            try:
                self._ensure_conn()
                return
            except Exception as exc:  # factory errors vary by transport
                log.warning("reconnect %d/%d failed: %s", attempt, self._reconnect_attempts, exc)
                time.sleep(delay)
                delay *= 2
        raise FrankyConnectionError(
            f"could not reconnect to {self._host}:{self._port} "
            f"after {self._reconnect_attempts} attempts"
        )

    def _call(self, method: str, *args, watchdog: bool = True, **kwargs):
        """Invoke a service method with watchdog timing and one reconnect+retry.

        ``watchdog=False`` exempts lifecycle calls (connect, server_info) whose
        first invocation legitimately pays one-time costs; the watchdog exists
        for loop-rate calls where a slow round trip means stale state. Supervised
        sessions never reconnect/replay.
        """
        for attempt in (1, 2):
            conn = self._ensure_conn()
            t0 = time.perf_counter()
            try:
                result = getattr(conn.root, method)(*args, **kwargs)
            except _RECONNECTABLE as exc:
                if self._supervised:
                    self._connection_lost = True
                    raise FrankyConnectionError(
                        f"supervised {method} lost its connection; outcome unknown, no replay"
                    ) from exc
                if attempt == 2:
                    raise FrankyConnectionError(f"{method} failed after reconnect: {exc}") from exc
                log.warning("connection lost during %s (%s) — reconnecting", method, exc)
                self._reconnect()
                continue
            self.last_rtt_s = time.perf_counter() - t0
            if watchdog:
                if self._rtt_log_interval_s > 0.0:
                    self._record_rtt()
                self._check_staleness(method)
            return result
        raise AssertionError("unreachable")

    def begin_warmup(self) -> None:
        """(Re)start the post-connect grace window from NOW.

        Call this once the WHOLE station is connected, not when the arm is.
        Lazy arming on the first watchdogged call is wrong for composed robots:
        ``FrankaPandaFranky.connect()`` itself issues watchdogged ``read_q`` /
        ``read_obs`` calls (the FK self-check, the clock anchor), so the window
        would open at arm-connect time and expire while the Wuji hand and glove
        take another ~6 s to come up — measured 2026-09-04, arm connected then
        hand bridge up at 15:54:47 and the loop's first call blew the budget at
        15:54:49, with the grace already spent.
        """
        self._warmup_until = time.perf_counter() + self._stale_warmup_s
        self._stale_run = 0

    def _record_rtt(self) -> None:
        """Accumulate round trips and log percentiles once per window.

        The watchdog only ever reports the ONE call that tripped it, which cannot
        distinguish a slow server from a stalled client thread from an isolated
        spike in an otherwise healthy loop. This gives the distribution instead,
        so an A/B (capture tee on vs off, say) is a comparison of percentiles
        rather than of anecdotes. Off unless ``rtt_log_interval_s`` is set.
        """
        self._rtt_samples.append(self.last_rtt_s)
        now = time.perf_counter()
        if now - self._rtt_window_t0 < self._rtt_log_interval_s:
            return
        s = sorted(self._rtt_samples)
        n = len(s)
        pick = lambda q: s[min(n - 1, int(q * n))] * 1e3  # noqa: E731
        log.info(
            "rtt over %.1fs: n=%d rate=%.1f/s | p50 %.2f p90 %.2f p99 %.2f max %.2f ms "
            "| over-budget %d",
            now - self._rtt_window_t0,
            n,
            n / max(now - self._rtt_window_t0, 1e-9),
            pick(0.5),
            pick(0.9),
            pick(0.99),
            s[-1] * 1e3,
            sum(1 for x in s if x > self._stale_threshold_s),
        )
        self._rtt_samples.clear()
        self._rtt_window_t0 = now

    def _check_staleness(self, method: str) -> None:
        """Raise once round trips have been over budget ``stale_consecutive_limit`` times.

        Measured on the bench 2026-09-04 (service under SCHED_FIFO, capture tee
        publishing at ~496 Hz): ``read_q`` and ``read_obs`` both sit at ~3.98 ms
        mean, ~6 ms p99, ~8 ms max over 400 calls, with nothing above 20 ms. The
        breaches seen in teleop were 20.8 ms and 35.7 ms — isolated spikes while
        the WORKSTATION was contended (the hand SDK logged ``joint_states lagged
        80`` in the same second), not the NUC going stale. Since the timing wraps
        the whole call, a client-side descheduling shows up here as a slow "round
        trip", so a hair trigger kills teleop for a hiccup the delta clamp
        already covers.

        MEASURED 2026-09-04, tee-on teleop, 5 s windows. First full window after
        connect: ``p50 1.04 p90 5.41 p99 23.06 max 47.76 ms``. All 25 later
        windows: p99 ~1.9 ms, max 2-7 ms. So the spikes are a startup transient
        (device SDK bridges coming up, ~183 threads spawning, 1 kHz subscriptions
        catching up) and live nowhere else — which is why all four crashes that
        day fired on the FIRST get_action()/get_observation() within seconds of
        connect. ``stale_warmup_s`` exempts exactly that window: the same
        reasoning as ``watchdog=False`` on lifecycle calls "whose first invocation
        legitimately pays one-time costs". It is safe because the clutch is still
        disengaged during it, so a stale read cannot move the robot.
        """
        now = time.perf_counter()
        if self._warmup_until is None and self._stale_warmup_s > 0.0:
            # First loop-rate call after connect — the loop is starting now.
            self._warmup_until = now + self._stale_warmup_s
        if self.last_rtt_s <= self._stale_threshold_s:
            self._stale_run = 0
            return
        if self._stale_run == 0:
            self._stale_run_since = now
        if self._warmup_until is not None and now < self._warmup_until:
            log.warning(
                "%s round trip %.1f ms over the %.1f ms budget during the %.1f s "
                "post-connect warm-up — not raising",
                method,
                self.last_rtt_s * 1e3,
                self._stale_threshold_s * 1e3,
                self._stale_warmup_s,
            )
            return
        self._stale_run += 1
        over = self.last_rtt_s > 2.0 * self._stale_threshold_s
        run_s = now - self._stale_run_since
        sustained = self._stale_run >= self._stale_consecutive_limit and run_s >= self._stale_run_s
        if over or sustained:
            run, self._stale_run = self._stale_run, 0
            raise FrankyStaleError(
                f"{method} round trip took {self.last_rtt_s * 1e3:.1f} ms "
                f"(threshold {self._stale_threshold_s * 1e3:.1f} ms"
                + (
                    ", over 2x — stalled)"
                    if over
                    else f", {run} in a row over {run_s * 1e3:.0f} ms)"
                )
            )
        log.warning(
            "%s round trip %.1f ms over the %.1f ms budget (%d/%d in a row)",
            method,
            self.last_rtt_s * 1e3,
            self._stale_threshold_s * 1e3,
            self._stale_run,
            self._stale_consecutive_limit,
        )

    def close(self) -> None:
        """Close the connection (safe to call repeatedly)."""
        if self._conn is not None:
            with contextlib.suppress(Exception):  # best-effort close
                self._conn.close()
            self._conn = None
        self._supervised = False
        self._connection_lost = False

    def claim_control(self) -> None:
        """Require protocol-v1 exclusive control; no fallback on an old service.

        Reconnection/replay is disabled before the claim, including an ambiguous
        claim acknowledgement. Close this client after any failed claim.
        """
        self._supervised = True
        try:
            version = self._call("claim_control", watchdog=False)
            if type(version) is not int or version != 1:
                raise FrankyClientError("service does not support supervised control protocol 1")
        except Exception:
            self._connection_lost = True
            raise

    # -- service API ----------------------------------------------------------

    def require_hold_support(self) -> None:
        """Refuse old NUC services before robot connection/preparation."""
        config = self.control_config()
        if (
            not config
            or config.get("pause_hold_protocol") != 1
            or config.get("controller") != "tracking"
        ):
            raise FrankyClientError(
                "pause hold requires the updated NUC tracking service (hold protocol 1)"
            )
        if config.get("tracking_shaper", "none") != "none":
            raise FrankyClientError("pause hold does not support a reference shaper")

    def require_reflex_recovery_support(self) -> None:
        """Refuse automatic reflex recovery before device preparation on an older NUC."""
        self.require_hold_support()
        if (self.control_config() or {}).get("reflex_recovery_protocol") != 2:
            raise FrankyClientError("automatic reflex recovery requires updated NUC protocol 2")

    def recover_reflex_hold(self) -> dict:
        """Recover the owner's dead tracker and return a new hold at the measured pose."""
        if not self._supervised:
            raise FrankyClientError("reflex recovery requires supervised control; no replay")
        token, stamp, q = self._call("recover_reflex_hold", watchdog=False)
        return {"token": int(token), "arm_time_ms": int(stamp), "q": [float(v) for v in q]}

    def hold_current(self) -> dict:
        """Freeze measured joints once in the running supervised impedance loop."""
        token, stamp, q = self._call("hold_current", watchdog=False)
        return {"token": int(token), "arm_time_ms": int(stamp), "q": [float(v) for v in q]}

    def hold_status(self) -> dict:
        """Read the latched reference and health without motion or re-anchoring."""
        token, stamp, q, healthy = self._call("hold_status", watchdog=False)
        return {
            "token": int(token),
            "arm_time_ms": stamp,
            "q": None if q is None else [float(v) for v in q],
            "healthy": bool(healthy),
        }

    def release_hold(self, token: int) -> None:
        """Explicitly release one verified hold without issuing a new target."""
        if self._call("release_hold", token, watchdog=False) is not True:
            raise FrankyClientError("hold release was not acknowledged")

    def server_info(self) -> tuple[str, str, bool, float]:
        """Return (python_version, franky_version, robot_connected, uptime_s)."""
        return self._call("server_info", watchdog=False)

    def control_config(self) -> dict | None:
        """``Robot.move`` kwargs the service applies (franky >= 2.0.0: limit_rate,
        cutoff_frequency) and whether the installed franky accepted them; None on a service
        predating the call (before 2026-09-15)."""
        try:
            pairs = self._call("control_config", watchdog=False)
            return {str(k): v for k, v in pairs}
        except Exception:
            log.debug("control_config unavailable", exc_info=True)
            return None

    def mass_diag(self, q: Sequence[float] | None = None) -> tuple[float, ...] | None:
        """Diagonal of the mass matrix [kg m²] at ``q`` (7 rad) or, with None, at the current
        pose; None on an old service."""
        try:
            args = () if q is None else (pack_target(q),)
            return tuple(float(v) for v in self._call("mass_diag", *args, watchdog=False))
        except Exception:
            log.debug("mass_diag unavailable", exc_info=True)
            return None

    def connect_robot(self, robot_ip: str = DEFAULT_ROBOT_IP) -> None:
        """Connect the service to the robot (idempotent) and clear errors."""
        self._robot_ip = robot_ip
        self._call("connect", robot_ip, watchdog=False)
        self._warmup_until = None  # re-armed on the first loop-rate call

    def read_q(self) -> tuple[float, ...]:
        """Joint positions [rad], 7-tuple."""
        return self._call("read_q")

    def read_teleop(self) -> tuple:
        """Return (ts_unix, q, dq, tau_J, tau_ext) — one round trip."""
        return self._call("read_teleop")

    def read_obs(self) -> tuple:
        """Return (ts_host_unix, ts_device_s, q, dq, tau_J, tau_ext) — one round trip."""
        return self._call("read_obs")

    def capture_info(self) -> tuple[bool, str]:
        """Return (tee_enabled, json_status) from the NUC capture tee.

        ``(False, "{}")`` when the service runs without ``--capture-endpoint`` or
        predates the tee (the exposed method is missing → treated as disabled).
        """
        try:
            enabled, status = self._call("capture_info", watchdog=False)
        except AttributeError:
            self._supports_capture = False
            return (False, "{}")
        self._supports_capture = True
        return (bool(enabled), str(status))

    def capture_flush(self) -> bool:
        """Ask the NUC tee to emit its stats frame now; False when there is no tee."""
        if not self.supports_capture:
            return False
        try:
            return bool(self._call("capture_flush", watchdog=False))
        except AttributeError:
            return False

    @property
    def supports_capture(self) -> bool:
        """True when the service accepts ``seq``/``t_send_ns`` on ``move_joints``."""
        if self._supports_capture is None:
            self.capture_info()
        return bool(self._supports_capture)

    def start_sidecar(self, t0_host: float, t0_device: float) -> str:
        """Start the NUC-side 1 kHz sidecar logger; returns its HDF5 path on the NUC."""
        return self._call("start_sidecar", float(t0_host), float(t0_device), watchdog=False)

    def stop_sidecar(self) -> str:
        """Stop the NUC-side sidecar logger; returns the finished HDF5 path."""
        return self._call("stop_sidecar", watchdog=False)

    def move_joints(
        self,
        target: Iterable[float],
        rel_dyn: float | tuple[float, float, float] = 0.2,
        *,
        asynchronous: bool = True,
        velocity: Sequence[float] | None = None,
        seq: int | None = None,
        t_send_ns: int | None = None,
    ) -> None:
        """Dispatch a preempting JointMotion to `target` (7 joint values, rad).

        rel_dyn is a scalar factor or a (velocity, acceleration, jerk) 3-tuple
        for split Ruckig scaling. velocity is an optional target ARRIVAL velocity
        [rad/s] (feedforward — Ruckig plans through the target instead of
        stopping at it). Synchronous moves block until the motion finishes, so
        the staleness watchdog only applies to asynchronous (loop-rate) dispatches.

        seq / t_send_ns are the capture tee's per-command sequence number and
        perf_counter_ns send stamp; they are forwarded only when the service
        supports them (probed once), so an older service keeps working.
        """
        packed_rel = (
            tuple(float(x) for x in rel_dyn)
            if isinstance(rel_dyn, tuple | list)
            else float(rel_dyn)
        )
        packed_vel = None if velocity is None else pack_target(velocity)
        extra: dict[str, int] = {}
        if seq is not None and self.supports_capture:
            extra = {"seq": int(seq), "t_send_ns": int(t_send_ns or 0)}
        self._call(
            "move_joints",
            pack_target(target),
            packed_rel,
            asynchronous,
            packed_vel,
            watchdog=asynchronous,
            **extra,
        )

    def join_motion(self, timeout_s: float = 30.0) -> bool:
        """Wait for the active motion to finish; True if it did."""
        # join can legitimately take long — bypass the staleness watchdog.
        if self._supervised:
            return self._call("join_motion", timeout_s, watchdog=False)
        return self._ensure_conn().root.join_motion(timeout_s)

    def stop(self) -> None:
        """Jerk-limited stop of the current motion."""
        if self._supervised:
            self._call("stop", watchdog=False)
        else:
            self._ensure_conn().root.stop()

    def recover(self) -> None:
        """Recover the robot from reflex/errors.

        Lifecycle call, not a loop-rate one: libfranka's automatic error recovery takes
        ~25 ms (every homing in the 2026-09-10 sessions logged "recover round trip 24-29 ms
        over the 20 ms budget (1/3 in a row)"), so it is exempt from the staleness watchdog
        like connect/server_info.
        """
        self._call("recover", watchdog=False)

    def read_status(self) -> tuple[str, str]:
        """Return (robot_mode, current_errors) as strings — reflex diagnostics."""
        return self._call("read_status", watchdog=False)

    def set_joint_impedance(self, k_theta: Sequence[float]) -> None:
        """Set firmware internal joint impedance (7 values, Nm/rad; idle only)."""
        self._call("set_joint_impedance", pack_target(k_theta), watchdog=False)

    def gripper_connect(self, port: str = "/dev/ttyUSB0") -> None:
        """Connect (and activate if needed) the Robotiq gripper on the NUC."""
        self._call("gripper_connect", port, watchdog=False)

    def gripper_move(self, frac: float, speed: int = 128, force: int = 25) -> None:
        """Command gripper fraction 0 (open) - 1 (closed); non-blocking.

        Watchdog-exempt: gripper serial writes can spike a few ms and a late
        gripper command is not a safety issue (the arm path stays guarded).
        """
        self._call("gripper_move", float(frac), int(speed), int(force), watchdog=False)

    def gripper_state(self) -> tuple[float, float]:
        """Return (position_frac, is_grasped) — served from the NUC-side 20 Hz cache."""
        return self._call("gripper_state", watchdog=False)

    def flange_pose_of(self, q: Sequence[float]) -> tuple[float, ...]:
        """Base->flange pose (16 floats, row-major 4x4) for arbitrary q (robot model FK)."""
        return self._call("flange_pose_of", pack_target(q), watchdog=False)

    def set_joint_velocity_limit(self, limits: Sequence[float]) -> None:
        """Set absolute per-joint velocity limits [rad/s] (Ruckig-enforced)."""
        self._call("set_joint_velocity_limit", pack_target(limits), watchdog=False)

    def set_collision_behavior(
        self, torque_thresholds: Sequence[float], force_thresholds: Sequence[float]
    ) -> None:
        """Set contact reflex thresholds (7 joint torques [Nm], 6 wrench values)."""
        self._call(
            "set_collision_behavior",
            tuple(float(v) for v in torque_thresholds),
            tuple(float(v) for v in force_thresholds),
            watchdog=False,
        )
