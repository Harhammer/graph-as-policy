"""Jerk-limited joint reference toward GaP's latest target, stepped once per bridge tick.

Why it exists (2026-11-06, first ``--execute`` run): the NUC service runs ``--controller
tracking`` — asynchronous ``move_joints`` only moves the reference of a 1 kHz joint
impedance loop, with no interpolation (``rel_dyn`` is ignored; the optional reference
shaper is off). It was built for teleop, which never steps more than 0.075 rad per tick.
GaP sends whole-move targets (0.4-0.8 rad); fed straight in, each one was a torque step
that made the arm jump, reflex, auto-recover and jump again.

So the bridge plans the motion itself with Ruckig — time-optimal under velocity,
acceleration and jerk limits, phase-synchronised where possible so all joints arrive
together on a straight joint-space line — and streams one small step per tick (0.8 rad at
0.7 rad/s: 2.1 s, at most 0.014 rad per 20 ms step).
"""

from __future__ import annotations

import numpy as np


class JointReference:
    """Ruckig-planned reference; ``set_target`` may change the goal at any time.

    Args:
        dt: Tick period [s] (one ``step`` per tick).
        max_velocity: Per-joint [rad/s].
        max_acceleration: Per-joint [rad/s^2].
        max_jerk: Per-joint [rad/s^3].
    """

    def __init__(self, dt: float, max_velocity, max_acceleration, max_jerk) -> None:
        import ruckig

        self._ruckig = ruckig
        self._otg = ruckig.Ruckig(7, dt)
        self._inp = ruckig.InputParameter(7)
        self._out = ruckig.OutputParameter(7)
        self._inp.max_velocity = [float(v) for v in max_velocity]
        self._inp.max_acceleration = [float(v) for v in max_acceleration]
        self._inp.max_jerk = [float(v) for v in max_jerk]
        self._inp.synchronization = ruckig.Synchronization.Phase
        self.active = False
        self.initialised = False

    def reset(self, q) -> None:
        """Start at rest at ``q`` (the measured pose), holding it."""
        q = [float(v) for v in q]
        self._inp.current_position = q
        self._inp.current_velocity = [0.0] * 7
        self._inp.current_acceleration = [0.0] * 7
        self._inp.target_position = q
        self._inp.target_velocity = [0.0] * 7
        self._inp.target_acceleration = [0.0] * 7
        self.active = False
        self.initialised = True

    def set_target(self, q) -> None:
        """New goal, planned from the current reference state (motion stays continuous)."""
        if not self.initialised:
            raise RuntimeError("JointReference.reset() first")
        self._inp.target_position = [float(v) for v in q]
        self.active = True

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance one tick; returns (position, velocity). Sets ``active`` False on arrival."""
        result = self._otg.update(self._inp, self._out)
        if result not in (self._ruckig.Result.Working, self._ruckig.Result.Finished):
            raise RuntimeError(f"Ruckig failed: {result}")
        position = np.array(self._out.new_position, dtype=np.float64)
        velocity = np.array(self._out.new_velocity, dtype=np.float64)
        self._out.pass_to_input(self._inp)
        if result == self._ruckig.Result.Finished:
            self.active = False
        return position, velocity

    @property
    def position(self) -> np.ndarray:
        """Current reference position."""
        return np.array(self._inp.current_position, dtype=np.float64)
