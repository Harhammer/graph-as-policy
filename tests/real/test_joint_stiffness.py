"""robot.set_joint_stiffness over the real wire: GaP sends the request, a client
standing in for the z3 bridge applies it and echoes ``left.impedance_seq``."""

from __future__ import annotations

import pytest
from gap_core.errors import ToolError

from gap.connector.real import real

from .conftest import ClientFeeder, FakeRRClient, free_port, full_frame

SOFT = [100.0, 80.0, 100.0, 100.0, 30.0, 50.0, 30.0]


class StiffnessClient(ClientFeeder):
    """Echoes the last joint-stiffness request it saw (``ack=False``: a client that
    never confirms — an old bridge, or an arm that never comes to rest)."""

    def __init__(self, port: int, ack: bool = True, **kw):
        self.ack = ack
        self.applied = 0
        self.seen: list[list[float]] = []
        super().__init__(port, **kw)

    def _run(self) -> None:
        import time

        try:
            client = FakeRRClient(self.port)
        except TimeoutError:
            return
        try:
            while not self._stop.is_set():
                frame = full_frame(**self.frame_kwargs)
                frame["left"]["impedance_seq"] = self.applied
                left = client.send_request(frame).get(b"left") or {}
                seq = int(left.get(b"impedance_seq", 0))
                if self.ack and seq > self.applied:
                    self.seen.append([float(v) for v in left.get(b"joint_impedance", [])])
                    self.applied = seq
                time.sleep(1 / 30)
        except (ConnectionError, OSError):
            pass
        finally:
            client.close()


def _connect(ack: bool):
    port = free_port()
    feeder = StiffnessClient(port, ack=ack, cam_key="camera_top")
    conn = real(
        "franka", rr_autostart=False, port=port, cameras=["robot0_robotview"],
        wait_timeout_s=20.0,
    )
    # The republisher only runs once GaP has an arm target (as after any graph move).
    conn.move_to_joints(list(conn.env._current_joints), max_steps=0)
    return conn, feeder


def test_request_is_sent_and_waits_for_the_echo():
    conn, feeder = _connect(ack=True)
    try:
        assert conn.set_joint_stiffness(SOFT, timeout_s=5.0) == {"applied": True}
        assert conn.set_joint_stiffness(None, timeout_s=5.0) == {"applied": True}
        assert feeder.seen == [SOFT, []], "None travels as [] = the client's startup values"
    finally:
        conn.close()
        feeder.stop()


def test_unconfirmed_request_raises():
    conn, feeder = _connect(ack=False)
    try:
        with pytest.raises(ToolError, match="did not confirm"):
            conn.set_joint_stiffness(SOFT, timeout_s=0.5)
    finally:
        conn.close()
        feeder.stop()


def test_bad_values_are_refused_before_sending():
    conn, feeder = _connect(ack=True)
    try:
        with pytest.raises(ToolError, match="7 values"):
            conn.set_joint_stiffness([100.0] * 6)
        with pytest.raises(ToolError, match="7 values"):
            conn.set_joint_stiffness([100.0] * 6 + [-1.0])
        assert feeder.seen == []
    finally:
        conn.close()
        feeder.stop()
