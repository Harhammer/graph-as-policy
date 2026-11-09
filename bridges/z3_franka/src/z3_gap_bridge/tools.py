"""Hand-test tools: drive the Robotiq alone, and stand in for GaP to test the bridge loop.

``z3-gap-gripper``      status / activate / open / close / move FRACTION
``z3-gap-fake-server``  a GaP stand-in that only ever replies with GaP's seed action,
                          so the bridge's gates stay shut and nothing moves; it prints
                          what the bridge streams (joints, gripper, camera rate, pose)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from z3_gap_bridge.wire import decode_msg, encode_msg, field

log = logging.getLogger(__name__)

#: ``FrankaRealEnv``'s pre-seeded hold command (gap/envs/franka_real_env.py).
GAP_SEED_JOINTS = (0.0256, -0.5002, -0.0217, -2.374, -0.0109, 1.874, -2.346)


@dataclass
class GripperTest:
    """Exercise the Robotiq by itself. Everything but `status` can move the fingers."""

    command: Literal["status", "activate", "open", "close", "move"] = "status"
    fraction: float = 0.5
    """For `move`: 0 = open, 1 = closed (the service's convention, not GaP's)."""
    via: Literal["nuc", "workstation"] = "nuc"
    port: str | None = None
    """Default: /dev/ttyUSB0 on the NUC, the adapter's by-id path on the workstation."""
    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    speed: float = 0.5
    force: float = 0.1
    watch_s: float = 2.5
    """How long to print the position after a command."""


def gripper_test(cfg: GripperTest) -> int:
    """Run one gripper command and print the resulting state."""
    from z3_gap_bridge.robotiq import LocalGripper, ServiceGripper, read_state

    if cfg.via == "workstation":
        gripper = LocalGripper(cfg.port or LocalGripper.DEFAULT_PORT)
    else:
        from z3_gap_bridge.franky_client import FrankyClient

        client = FrankyClient(cfg.nuc_host, cfg.service_port)
        gripper = ServiceGripper(client, cfg.port or "/dev/ttyUSB0")

    if cfg.command == "status":
        if cfg.via == "workstation":
            activated, closed, grasped = read_state(gripper.port)
            print(f"activated {activated}  closed {closed:.2f}  grasped {bool(grasped)}")
        elif gripper.probe():
            closed, grasped = gripper.state()
            print(f"connected on the NUC  closed {closed:.2f}  grasped {bool(grasped)}")
        else:
            print("not connected on the NUC service yet — run `activate` (fingers will cycle)")
        return 0

    print("activating if needed (an unactivated gripper closes and opens once)...")
    gripper.connect()
    target = {"activate": None, "open": 0.0, "close": 1.0, "move": cfg.fraction}[cfg.command]
    if target is not None:
        gripper.move(target, round(cfg.speed * 255), round(cfg.force * 255))
    t_end = time.monotonic() + cfg.watch_s
    while time.monotonic() < t_end:
        closed, grasped = gripper.state()
        print(f"  closed {closed:.2f}  grasped {bool(grasped)}")
        time.sleep(0.25)
    gripper.close()
    return 0


@dataclass
class FakeServer:
    """Stand in for `gap run --real franka`: accept the bridge, hold the arm, print the stream."""

    host: str = "127.0.0.1"
    port: int = 9000
    save_frame: str | None = None
    """Write the first camera frame the bridge sends (rgb, depth, K, pose) to this .npz."""
    test_move_rad: float = 0.0
    """With --execute on the bridge: after test_hold_s, turn test_joint by this much, hold,
    and turn it back — a small first check of the streamed motion. 0 = never move."""
    test_joint: int = 6
    """0-indexed; 6 is the last wrist joint (rotates the gripper in place)."""
    test_hold_s: float = 3.0


def fake_server(cfg: FakeServer) -> int:
    """Serve GaP's wire protocol, replying only with GaP's seed (nothing ever moves)."""
    import socket
    import struct

    header = struct.Struct("!I")
    seed = {"timestamp": 0.0, "left": {"joint_pos": list(GAP_SEED_JOINTS), "gripper": 1.0}}

    def recv_exactly(conn: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge disconnected")
            buf += chunk
        return bytes(buf)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((cfg.host, cfg.port))
    except OSError as exc:
        srv.close()
        print(
            f"{cfg.host}:{cfg.port} is taken ({exc.strerror}) — another fake server or a "
            f"`gap run` is already listening. Find it with: ss -ltnp 'sport = :{cfg.port}'"
        )
        return 1
    srv.listen(1)
    print(f"fake GaP listening on {cfg.host}:{cfg.port} — start the bridge now (Ctrl-C to stop)")
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("bridge connected; replying with GaP's seed only — the bridge must hold the arm")
    if abs(cfg.test_move_rad) > 0.3:
        raise SystemExit("test_move_rad is limited to 0.3 rad")
    n = cams = 0
    t0 = time.monotonic()
    t_start = time.monotonic()
    q_first = None
    saved = False
    try:
        while True:
            (length,) = header.unpack(recv_exactly(conn, header.size))
            obs = decode_msg(recv_exactly(conn, length))
            reply = seed
            if cfg.test_move_rad:
                if q_first is None:
                    q_first = np.asarray(field(field(obs, "left"), "joint_pos"))[:7].astype(float)
                    print(
                        f"test move: joint {cfg.test_joint} by {cfg.test_move_rad:+.3f} rad "
                        f"after {cfg.test_hold_s:.0f} s, back after another {cfg.test_hold_s:.0f} s"
                    )
                phase = (time.monotonic() - t_start) / cfg.test_hold_s
                if phase >= 1.0:
                    target = q_first.copy()
                    if phase < 2.0:
                        target[cfg.test_joint] += cfg.test_move_rad
                    reply = {**seed, "left": {"joint_pos": target.tolist(), "gripper": 1.0}}
            payload = encode_msg({**reply, "timestamp": time.time()})
            conn.sendall(header.pack(len(payload)) + payload)
            n += 1
            joints = np.asarray(field(field(obs, "left"), "joint_pos"))
            cam = field(obs, "camera_top")
            if cam is not None:
                cams += 1
                if cfg.save_frame and not saved:
                    np.savez_compressed(
                        cfg.save_frame,
                        rgb=field(field(cam, "images"), "rgb"),
                        depth_m=field(cam, "depth_data"),
                        intrinsics=field(
                            field(field(cam, "intrinsics"), "left"), "intrinsics_matrix"
                        ),
                        pose_mat=field(cam, "pose_mat"),
                    )
                    saved = True
                    print(f"saved first camera frame to {cfg.save_frame}")
                last_pose = np.asarray(field(cam, "pose_mat"))
            dt = time.monotonic() - t0
            if dt >= 1.0:
                line = (
                    f"{n / dt:5.1f} Hz  cam {cams / dt:4.1f} Hz  "
                    f"q {np.round(joints[:7], 3)}  gripper open {joints[7]:.2f}"
                )
                if cams:
                    line += f"  camera xyz {np.round(last_pose[:3, 3], 3)}"
                print(line)
                n = cams = 0
                t0 = time.monotonic()
    except (ConnectionError, KeyboardInterrupt) as exc:
        print(f"stopped: {exc or 'Ctrl-C'}")
    finally:
        conn.close()
        srv.close()
    return 0


def gripper_main() -> None:
    """Console entry point: ``z3-gap-gripper``."""
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(gripper_test(tyro.cli(GripperTest)))


def fake_server_main() -> None:
    """Console entry point: ``z3-gap-fake-server``."""
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(fake_server(tyro.cli(FakeServer)))


@dataclass
class JointsConfig:
    """Print the arm's joint angles (read-only), ready to paste into a graph's POSES."""

    watch: bool = False
    """Keep printing twice a second until Ctrl-C (for hand-guiding to a pose)."""
    record: bool = False
    """Hand-guide, press Enter, type a name: prints ready-to-paste POSES and EEF_POSES
    lines for z3_graphs/poses_demo/build.py, and the whole EEF_POSES block on Ctrl-C."""
    tool_length_m: float = 0.217
    """Flange -> fingertip, for the printed tip position (z3 pivot calibration)."""
    kinematics: Path | None = None
    """The bridge's --kinematics directory: print joints WITH its zero offsets added, so
    that recorded POSES / EEF_POSES are in the model GaP works in when the bridge runs
    with the same --kinematics. Without it, record for a bridge without it."""
    nuc_host: str = "192.168.99.100"
    service_port: int = 18861
    robot_ip: str = "192.168.98.121"


def gap_tool_pose(q) -> tuple[np.ndarray, np.ndarray]:
    """GaP's end-effector pose for joints ``q``: (xyz [m], roll/pitch/yaw [deg]) in the base
    frame, exactly what ``robot.get_ee_pose`` reports and ``robot.go_to_pose`` aims for.

    Reads GaP's own tool settings (GAP_FRANKA_TCP_FLANGE_M, default 0,0,0.157 in the flange
    frame; GAP_FRANKA_TCP_ROT_Z_DEG, default 45 relative to panda_hand, which is the flange
    turned -45 deg about z). Set them the same way as for ``gap run``. rpy is scipy's
    extrinsic "xyz" (R = Rz(yaw) Ry(pitch) Rx(roll)), as build.py's EEF_POSES expects.
    """
    import math
    import os

    from z3_gap_bridge.panda_fk import flange_pose

    raw = os.environ.get("GAP_FRANKA_TCP_FLANGE_M", "").strip()
    tcp = np.array([float(v) for v in raw.split(",")] if raw else [0.0, 0.0, 0.157])
    rot_z = math.radians(float(os.environ.get("GAP_FRANKA_TCP_ROT_Z_DEG", "45")) - 45.0)
    c, s = math.cos(rot_z), math.sin(rot_z)
    flange = flange_pose(np.asarray(q, dtype=float))
    rot = flange[:3, :3] @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    xyz = flange[:3, 3] + flange[:3, :3] @ tcp
    rpy = np.degrees(
        [
            math.atan2(rot[2, 1], rot[2, 2]),
            math.asin(-max(-1.0, min(1.0, rot[2, 0]))),
            math.atan2(rot[1, 0], rot[0, 0]),
        ]
    )
    return xyz, rpy


def _eef_line(name: str, q) -> str:
    xyz, rpy = gap_tool_pose(q)
    return (
        f'"{name}": {{"xyz": [{", ".join(f"{v:.4f}" for v in xyz)}], '
        f'"rpy_deg": [{", ".join(f"{v:.1f}" for v in rpy)}]}},'
    )


def _record_poses(client, dq=None) -> int:
    """Enter -> name -> one POSES and one EEF_POSES line; Ctrl-C/empty name prints the block."""
    recorded: list[tuple[str, np.ndarray]] = []
    dq = np.zeros(7) if dq is None else dq
    print("Hand-guide the arm, then press Enter to record (Ctrl-C to finish).")
    try:
        while True:
            input()
            q1 = np.asarray(client.read_q())
            time.sleep(0.3)
            q = np.asarray(client.read_q())
            if np.abs(q - q1).max() > 2e-3:
                print("  the arm is still moving — let go of it, wait a second, Enter again")
                continue
            q = q + dq
            name = input("name (empty = finish): ").strip()
            if not name:
                break
            recorded.append((name, q))
            print(f'  POSES:     "{name}": [{", ".join(f"{v:.3f}" for v in q)}],')
            print(f"  EEF_POSES: {_eef_line(name, q)}")
            print("next pose: hand-guide, then Enter")
    except (KeyboardInterrupt, EOFError):
        pass
    if recorded:
        print("\nEEF_POSES = {")
        for name, q in recorded:
            print(f"    {_eef_line(name, q)}")
        print("}")
    return 0


def print_joints(cfg: JointsConfig) -> int:
    """One line per reading: joint list [rad], the fingertip position [m, base frame], and
    GaP's end-effector pose (xyz + rpy, ready for build.py's EEF_POSES)."""
    from z3_gap_bridge.franky_client import FrankyClient
    from z3_gap_bridge.panda_fk import tool_tip_position

    dq = np.zeros(7)
    if cfg.kinematics is not None:
        from z3_gap_bridge.bridge import load_kinematics

        offsets, _ = load_kinematics(cfg.kinematics)
        dq = np.radians(offsets)
        print(f"joint zero offsets {list(offsets)} deg added (bridge --kinematics {cfg.kinematics})")
    else:
        print(
            "NOTE: poses in the ROBOT'S OWN model. If the bridge runs with --kinematics DIR, "
            "rerun with the same --kinematics DIR, or the poses land in the wrong place "
            "(z3 cell since 2026-11-09: ~13 mm too high, 5 mm off)."
        )
    client = FrankyClient(cfg.nuc_host, cfg.service_port, stale_threshold_s=0.1, stale_run_s=0.5)
    client.connect_robot(cfg.robot_ip)
    if cfg.record:
        try:
            return _record_poses(client, dq)
        finally:
            client.close()
    try:
        while True:
            q = np.asarray(client.read_q()) + dq
            tip = tool_tip_position(q, cfg.tool_length_m)
            xyz, rpy = gap_tool_pose(q)
            print(
                f"[{', '.join(f'{v:.3f}' for v in q)}]   "
                f"fingertip x {tip[0]:+.3f} y {tip[1]:+.3f} z {tip[2]:+.3f} m   "
                f"GaP eef xyz [{', '.join(f'{v:.4f}' for v in xyz)}] "
                f"rpy [{', '.join(f'{v:.1f}' for v in rpy)}]"
            )
            if not cfg.watch:
                return 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        client.close()


def joints_main() -> None:
    """Console entry point: ``z3-gap-joints``."""
    import tyro

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(print_joints(tyro.cli(JointsConfig)))
