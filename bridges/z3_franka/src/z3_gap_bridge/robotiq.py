"""Robotiq 2F-85 over Modbus RTU, driven from the machine the RS-485 adapter is plugged into.

``RobotiqGripper`` is the NUC service's ``_RobotiqGripper`` copied verbatim from oktonex
``packages/nuc/src/okto_nuc/franky_service.py`` @ c83e4ed9 (renamed only). Fix it there.
Constructing it ACTIVATES an unactivated gripper, which closes and opens the fingers;
``read_state`` below reads without ever activating.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

SLAVE_ID = 9


def read_state(port: str) -> tuple[bool, float, float]:
    """(activated, position_frac 0 open..1 closed, is_grasped) — read-only, never activates."""
    import minimalmodbus

    inst = minimalmodbus.Instrument(port, SLAVE_ID)
    inst.serial.baudrate = 115200
    inst.serial.timeout = 0.2
    try:
        regs = inst.read_registers(2000, 3, functioncode=4)
    finally:
        inst.serial.close()
    b = []
    for r in regs:
        b += [r >> 8, r & 0xFF]
    frac = (b[4] - RobotiqGripper.POS_OPEN) / (RobotiqGripper.POS_CLOSED - RobotiqGripper.POS_OPEN)
    activated = (b[0] >> 4) & 3 == 3
    return activated, min(1.0, max(0.0, float(frac))), float(((b[0] >> 6) & 3) in (1, 2))


class RobotiqGripper:
    """Minimal Robotiq 2F-85 driver (Modbus RTU over /dev/ttyUSB0, slave 9).

    Register map (probed 2026-08-12): write FC16 @1000 [action, pos, speed|force],
    read FC04 @2000 -> [status, rsv, fault, pr_echo, position, current].
    Position: 3 = fully open, 225 = fully closed (mechanical stops, this finger set).
    """

    POS_OPEN = 3.0
    POS_CLOSED = 225.0

    def __init__(self, port: str) -> None:
        import minimalmodbus

        self._inst = minimalmodbus.Instrument(port, 9)
        self._inst.serial.baudrate = 115200
        self._inst.serial.timeout = 0.2
        self._io_lock = threading.Lock()
        status = self._status()
        if (status[0] >> 4) & 3 != 3:  # gSTA != activated
            log.info("activating gripper (auto-calibration close/open cycle)")
            with self._io_lock:
                self._inst.write_registers(1000, [0x0000, 0x0000, 0x0000])
                time.sleep(0.1)
                self._inst.write_registers(1000, [0x0100, 0x0000, 0x0000])
            for _ in range(50):
                time.sleep(0.2)
                if (self._status()[0] >> 4) & 3 == 3:
                    break
            else:
                raise RuntimeError("gripper activation timed out")
        log.info("gripper ready (pos=%d)", self._status()[4])
        # Background poller: serial reads (~2-5 ms, occasionally spiky) stay off
        # the RPC hot path — gripper_state() serves this cache instantly.
        self._cache = self.state_direct()
        self._poll_stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True, name="gripper-poll")
        self._poller.start()

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                self._cache = self.state_direct()
            except Exception:
                log.exception("gripper poll failed")
            time.sleep(0.05)

    def _status(self) -> list:
        with self._io_lock:
            regs = self._inst.read_registers(2000, 3, functioncode=4)
        out = []
        for r in regs:
            out += [r >> 8, r & 0xFF]
        return out

    def move(self, frac: float, speed: int, force: int) -> None:
        """Command position as fraction 0.0 (open) - 1.0 (closed); non-blocking."""
        frac = min(1.0, max(0.0, float(frac)))
        pos = round(self.POS_OPEN + frac * (self.POS_CLOSED - self.POS_OPEN))
        with self._io_lock:
            self._inst.write_registers(
                1000, [0x0900, pos & 0xFF, ((speed & 0xFF) << 8) | (force & 0xFF)]
            )

    def state(self) -> tuple:
        """Return the cached (position_frac, is_grasped) — updated at 20 Hz."""
        return self._cache

    def state_direct(self) -> tuple:
        """Read (position_frac, is_grasped) from the bus — is_grasped = gOBJ 1/2."""
        b = self._status()
        frac = (b[4] - self.POS_OPEN) / (self.POS_CLOSED - self.POS_OPEN)
        g_obj = (b[0] >> 6) & 3
        return (min(1.0, max(0.0, float(frac))), float(g_obj in (1, 2)))


class LocalGripper:
    """The Robotiq on a serial port of THIS machine (the bridge's default).

    ``probe``/``state`` before ``connect`` read the bus without activating; ``connect``
    activates if needed (fingers close and open) and starts the 20 Hz state poller.
    """

    #: Stable path of the cell's Robotiq RS-485 adapter (ttyUSB numbering is not).
    DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAASNFTR-if00-port0"

    def __init__(self, port: str = DEFAULT_PORT, read_period_s: float = 0.2) -> None:
        self.port = port
        self._gripper: RobotiqGripper | None = None
        self._read_period_s = read_period_s
        self._read_t = 0.0
        self._read_cache = (0.0, 0.0)

    def connect(self) -> None:
        """Open, activate if needed, start polling."""
        self._gripper = RobotiqGripper(self.port)

    def probe(self) -> bool:
        """True when the gripper answers and is already activated (read-only)."""
        activated, closed, grasped = read_state(self.port)
        self._read_cache = (closed, grasped)
        return activated

    def state(self) -> tuple[float, float]:
        """(closed_frac, is_grasped); read-only and rate-limited before ``connect``."""
        if self._gripper is not None:
            return self._gripper.state()
        if time.monotonic() - self._read_t >= self._read_period_s:
            self._read_t = time.monotonic()
            _, closed, grasped = read_state(self.port)
            self._read_cache = (closed, grasped)
        return self._read_cache

    def move(self, closed_frac: float, speed: int, force: int) -> None:
        """Command 0 (open) .. 1 (closed); non-blocking."""
        if self._gripper is None:
            raise RuntimeError("gripper not connected")
        self._gripper.move(closed_frac, speed, force)

    def close(self) -> None:
        """Stop the poller and release the port."""
        if self._gripper is not None:
            self._gripper._poll_stop.set()
            self._gripper._poller.join(timeout=1.0)
            self._gripper._inst.serial.close()
            self._gripper = None


class ServiceGripper:
    """The Robotiq on the NUC, through the franky service's own copy of the driver."""

    def __init__(self, client, port: str = "/dev/ttyUSB0") -> None:
        self._client = client
        self.port = port

    def connect(self) -> None:
        """Activate on the NUC if needed."""
        self._client.gripper_connect(self.port)

    def probe(self) -> bool:
        """True when the service already has the gripper connected (read-only)."""
        try:
            self._client.gripper_state()
        except Exception:
            return False
        return True

    def state(self) -> tuple[float, float]:
        """(closed_frac, is_grasped) from the service's 20 Hz cache."""
        return self._client.gripper_state()

    def move(self, closed_frac: float, speed: int, force: int) -> None:
        """Command 0 (open) .. 1 (closed); non-blocking."""
        self._client.gripper_move(closed_frac, speed=speed, force=force)

    def close(self) -> None:
        """Nothing to release; the service owns the port."""
