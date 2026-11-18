"""Real-hardware gripper calls block until the fingers arrive or stop, not a fixed time.

A fake clock advances 20 ms per poll; the "gripper" reports a scripted position per time.
"""

from __future__ import annotations

import pytest

from gap.connector import core
from gap.connector.core import Connector


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(core.time, "monotonic", c.monotonic)
    monkeypatch.setattr(core.time, "sleep", c.sleep)
    return c


def _connector(clock: Clock, position_at) -> Connector:
    conn = object.__new__(Connector)
    conn.is_real = True
    conn.get_gripper_fraction = lambda arm_id=0: float(position_at(clock.t))
    conn.set_gripper = lambda frac, arm_id=0: None
    return conn


def ramp(start, end, t_start, duration, stop_at=None):
    """Position over time: holds ``start`` until the command lands at ``t_start``, then
    moves linearly to ``end`` over ``duration`` (or stops early at ``stop_at``)."""

    def at(t):
        if t < t_start:
            return start
        frac = min(1.0, (t - t_start) / duration)
        pos = start + (end - start) * frac
        if stop_at is not None:
            pos = max(pos, stop_at) if end < start else min(pos, stop_at)
        return round(pos * 255) / 255  # Robotiq resolution

    return at


def test_close_returns_when_the_target_is_reached(clock):
    conn = _connector(clock, ramp(1.0, 0.45, t_start=0.2, duration=0.8))
    out = conn.close_gripper(fraction=0.55)
    assert out["position"] == pytest.approx(0.45, abs=0.02)
    assert 0.9 <= clock.t <= 1.1, "returns on arrival, not after a fixed settle"


def test_close_on_a_part_returns_once_the_fingers_stop(clock):
    conn = _connector(clock, ramp(1.0, 0.0, t_start=0.2, duration=1.5, stop_at=0.5))
    out = conn.close_gripper(fraction=1.0)
    assert out["position"] == pytest.approx(0.5, abs=0.01)
    stopped_t = 0.2 + 1.5 * 0.5
    assert stopped_t + 0.25 <= clock.t <= stopped_t + 0.45  # ~0.3 s of no motion


def test_command_latency_is_not_mistaken_for_a_stop(clock):
    """The reading stays at the old position for 0.6 s before the fingers move."""
    conn = _connector(clock, ramp(1.0, 0.45, t_start=0.6, duration=0.8))
    out = conn.close_gripper(fraction=0.55)
    assert out["position"] == pytest.approx(0.45, abs=0.02)


def test_open_when_already_open_returns_at_once(clock):
    conn = _connector(clock, lambda t: 1.0)
    assert conn.open_gripper()["position"] == 1.0
    assert clock.t == 0.0


def test_a_gripper_that_never_moves_returns_after_a_second(clock):
    """Dry run: the bridge sends nothing, the reading never changes."""
    conn = _connector(clock, lambda t: 1.0)
    conn.close_gripper(fraction=0.55)
    assert 1.0 <= clock.t <= 1.05


def test_timeout_caps_a_slow_gripper(clock):
    conn = _connector(clock, ramp(1.0, 0.0, t_start=0.0, duration=20.0))
    conn.close_gripper(fraction=1.0, timeout_s=2.0)
    assert 2.0 <= clock.t <= 2.05


def test_a_gripper_that_must_open_but_never_moves_raises(clock):
    """2026-11-10: the Robotiq lost power (camera remount) and sat unactivated at 75 %
    closed; every command was ignored and a whole run went on as if it had opened."""
    from gap_core.errors import ToolError

    conn = _connector(clock, lambda t: 0.2477)
    with pytest.raises(ToolError, match="did not move"):
        conn.open_gripper()
    assert 1.0 <= clock.t <= 1.05
    with pytest.raises(ToolError, match="not responding"):
        conn.close_gripper(fraction=0.55)  # target 0.45 open: it has to open to get there


def test_closing_on_a_held_part_that_does_not_move_is_fine(clock):
    """Closing again on a part already held: the fingers can't move, nothing is wrong."""
    conn = _connector(clock, lambda t: 0.45)
    assert conn.close_gripper(fraction=1.0)["position"] == pytest.approx(0.45)
