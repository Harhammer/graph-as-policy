"""Bridge behaviour against fakes: no robot, camera or GaP needed."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from z3_gap_bridge.bridge import Bridge, BridgeConfig, ChangeGate, load_flange_t_camera
from z3_gap_bridge.panda_fk import flange_pose, tool_tip_position
from z3_gap_bridge.realsense_client import RealSenseFrame, depth_to_metres

Q_START = np.array([0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0])  # DROID reset pose
GAP_SEED = [0.0256, -0.5002, -0.0217, -2.374, -0.0109, 1.874, -2.346]  # FrankaRealEnv's seed


class FakeFranky:
    def __init__(self, q=Q_START) -> None:
        self.q = np.asarray(q, dtype=float)
        self.calls: list[tuple] = []
        self.fail_move = False
        self.fail_impedance = 0
        self.controller = "position"
        self.status = ("RobotMode.Idle", "[]")

    def control_config(self):
        return {"controller": self.controller}

    def connect_robot(self, ip):
        self.calls.append(("connect_robot", ip))

    def claim_control(self):
        self.calls.append(("claim_control",))

    def read_q(self):
        return tuple(self.q)

    def flange_pose_of(self, q):
        return tuple(flange_pose(q).reshape(-1))

    def set_joint_impedance(self, k):
        if self.fail_impedance:
            self.fail_impedance -= 1
            raise RuntimeError("command not possible in the current mode (Move)")
        self.calls.append(("set_joint_impedance", tuple(k)))

    def set_collision_behavior(self, t, f):
        self.calls.append(("set_collision_behavior",))

    def set_joint_velocity_limit(self, v):
        self.calls.append(("set_joint_velocity_limit",))

    def move_joints(self, target, rel_dyn, velocity=None):
        if self.fail_move:
            raise RuntimeError("Reflex: cartesian_reflex\nmore")
        self.calls.append(("move_joints", target))
        self.q = np.asarray(target, dtype=float)  # a perfectly tracking arm

    def read_status(self):
        return self.status

    def begin_warmup(self):
        pass

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        pass

    def moves(self):
        return [c for c in self.calls if c[0] in ("move_joints", "gripper_move")]


class FakeGripper:
    """LocalGripper/ServiceGripper surface; logs into the franky fake's call list."""

    def __init__(self, calls: list, closed: float = 0.0, activated: bool = True) -> None:
        self.calls = calls
        self.closed = closed
        self.activated = activated

    def connect(self):
        self.calls.append(("gripper_connect",))
        self.activated = True

    def probe(self):
        self.calls.append(("gripper_probe",))
        return self.activated

    def state(self):
        return (self.closed, 0.0)

    def move(self, closed_frac, speed, force):
        self.calls.append(("gripper_move", closed_frac))

    def close(self):
        pass


class FakeCamera:
    def __init__(self) -> None:
        self.n = 0

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self, max_age_s=None):
        self.n += 1
        return RealSenseFrame(
            rgb=np.zeros((4, 6, 3), np.uint8),
            depth_m=np.full((4, 6), 0.3, np.float32),
            intrinsics=np.array([[400.0, 0, 3], [0, 400.0, 2], [0, 0, 1]]),
            frame_number=self.n,
            hw_timestamp_ms=0.0,
            timestamp_domain="Global Time",
            host_time_s=0.0,
        )


class FakeWire:
    """Replays scripted GaP actions and records the observations sent."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = list(actions)
        self.sent: list[dict] = []

    def connect(self, wait_s):
        pass

    def exchange(self, obs):
        self.sent.append(obs)
        return self.actions.pop(0) if self.actions else {}

    def close(self):
        pass


def gap_action(joints, gripper=1.0, seq=None, impedance=None, impedance_seq=None) -> dict:
    """An action as GaP's server sends it, after raw=True decoding (bytes keys).
    ``seq``: GaP's gripper_seq (count of real gripper commands); None = older GaP.
    ``impedance`` / ``impedance_seq``: a joint-stiffness request ([] = startup values)."""
    left = {b"joint_pos": [float(v) for v in joints], b"gripper": float(gripper)}
    if seq is not None:
        left[b"gripper_seq"] = int(seq)
    if impedance_seq is not None:
        left[b"joint_impedance"] = list(impedance or [])
        left[b"impedance_seq"] = int(impedance_seq)
    return {b"timestamp": 0.0, b"left": left}


def make_bridge(actions, *, execute=True, franky=None, gripper_closed=0.0, **cfg):
    franky = franky or FakeFranky()
    wire = FakeWire(actions)
    cfg.setdefault("camera_hz", 1e6)
    gripper = FakeGripper(franky.calls, closed=gripper_closed)
    bridge = Bridge(
        BridgeConfig(execute=execute, **cfg), franky, FakeCamera(), wire, np.eye(4), gripper
    )
    bridge.connect()
    return bridge, franky, wire


def run_ticks(bridge, n):
    for _ in range(n):
        bridge.tick()


def _streamed(franky):
    return [np.asarray(c[1]) for c in franky.calls if c[0] == "move_joints"]


def test_gap_seed_never_reaches_the_arm():
    target = Q_START + 0.1
    bridge, franky, _ = make_bridge(
        [gap_action(GAP_SEED)] * 5 + [gap_action(target)] * 200, motion="stream"
    )
    run_ticks(bridge, 5)
    assert not _streamed(franky)
    run_ticks(bridge, 200)
    steps = _streamed(franky)
    assert len(steps) > 10, "a 0.1 rad move must be streamed in many small steps"
    np.testing.assert_allclose(steps[-1], target, atol=1e-6)
    assert bridge.dispatch_count == 1, "re-stamped identical targets plan once"


def test_execute_claims_supervised_control_before_connecting():
    _, franky, _ = make_bridge([])
    names = [c[0] for c in franky.calls]
    assert names.index("claim_control") < names.index("connect_robot")


def test_a_large_target_is_never_sent_as_one_jump():
    """The 2026-11-06 failure: 0.4-0.8 rad targets went straight into the tracker."""
    target = Q_START.copy()
    target[0] += 0.8
    bridge, franky, _ = make_bridge(
        [gap_action(GAP_SEED)] + [gap_action(target)] * 400, motion="stream"
    )
    run_ticks(bridge, 401)
    steps = np.array([Q_START] + _streamed(franky))
    per_tick = np.abs(np.diff(steps, axis=0)).max()
    v_limit = BridgeConfig().max_joint_velocity[0] * BridgeConfig().dynamics[0]
    assert per_tick <= v_limit / BridgeConfig().loop_hz + 1e-9, per_tick
    np.testing.assert_allclose(steps[-1], target, atol=1e-6)
    assert len(steps) / BridgeConfig().loop_hz < 3.5, "must arrive within GaP's ~3.5 s wait"


def test_gripper_gate_does_not_release_the_arm_seed():
    # GaP's open_gripper changes only the gripper; joint_pos is still the seed (GaP home).
    bridge, franky, _ = make_bridge(
        [gap_action(GAP_SEED, 1.0), gap_action(GAP_SEED, 0.0), gap_action(GAP_SEED, 0.0)]
    )
    run_ticks(bridge, 3)
    assert not [c for c in franky.calls if c[0] == "move_joints"]
    assert ("gripper_move", 1.0) in franky.calls  # GaP 0 = closed -> service 1 = closed


def test_first_open_equal_to_the_seed_is_sent():
    """2026-11-09: the graph's first step was "open" (1.0, same as GaP's seed); the value
    gate dropped it and a closed gripper stayed closed."""
    bridge, franky, _ = make_bridge(
        [gap_action(GAP_SEED, 1.0, seq=0)] * 3 + [gap_action(GAP_SEED, 1.0, seq=1)] * 3,
        gripper_closed=0.8,
    )
    run_ticks(bridge, 3)
    assert not franky.moves(), "the seed itself must not move the fingers"
    run_ticks(bridge, 3)
    assert franky.moves() == [("gripper_move", 0.0)], "open once, not re-sent every tick"


def test_a_repeated_command_is_resent_even_within_the_deadband():
    actions = [gap_action(GAP_SEED, 1.0, seq=0), gap_action(GAP_SEED, 1.0, seq=1)]
    actions += [gap_action(GAP_SEED, 0.45, seq=2), gap_action(GAP_SEED, 0.45, seq=3)]
    bridge, franky, _ = make_bridge(actions)
    run_ticks(bridge, 4)
    assert franky.moves() == [
        ("gripper_move", 0.0),
        ("gripper_move", pytest.approx(0.55)),
        ("gripper_move", pytest.approx(0.55)),
    ]


def test_gripper_polarity_both_ways():
    bridge, franky, wire = make_bridge(
        [gap_action(GAP_SEED, 1.0), gap_action(GAP_SEED, 0.7)], gripper_closed=0.25
    )
    run_ticks(bridge, 2)
    # Robotiq 0.25 closed -> GaP 0.75 open; GaP 0.7 open -> Robotiq 0.3 closed.
    assert wire.sent[0]["left"]["joint_pos"][7] == pytest.approx(0.75)
    assert franky.moves() == [("gripper_move", pytest.approx(0.3))]


def test_dry_run_sends_nothing_and_never_activates_the_gripper():
    actions = [gap_action(GAP_SEED, 1.0), gap_action(Q_START + 0.2, 0.0)] * 3
    bridge, franky, _ = make_bridge(actions, execute=False)
    run_ticks(bridge, 6)
    names = {c[0] for c in franky.calls}
    assert names <= {"connect_robot", "gripper_probe"}, names


def test_dry_run_reports_an_unactivated_gripper_as_open():
    franky = FakeFranky()
    wire = FakeWire([{}])
    gripper = FakeGripper(franky.calls, closed=0.8, activated=False)
    bridge = Bridge(BridgeConfig(camera_hz=1e6), franky, FakeCamera(), wire, np.eye(4), gripper)
    bridge.connect()
    bridge.tick()
    assert wire.sent[0]["left"]["joint_pos"][7] == pytest.approx(1.0)
    assert ("gripper_connect",) not in franky.calls


def test_floor_guard_clamps_onto_the_floor():
    cfg = BridgeConfig()  # the z3 cell's floor pair
    q_down = Q_START.copy()
    q_down[1], q_down[3] = 0.9, -2.0  # shoulder forward, elbow bent: tip far below the floor
    assert tool_tip_position(q_down, cfg.tool_length_m)[2] < cfg.z_floor_m
    assert tool_tip_position(Q_START, cfg.tool_length_m)[2] > cfg.z_floor_m
    bridge, franky, _ = make_bridge([gap_action(GAP_SEED)] + [gap_action(q_down)] * 400)
    run_ticks(bridge, 401)
    z = tool_tip_position(_streamed(franky)[-1], cfg.tool_length_m)[2]
    assert cfg.z_floor_m <= z < cfg.z_floor_m + 0.005
    assert bridge.floor_clamp_count == 1


def test_reflex_latches_and_stops_forwarding():
    franky = FakeFranky()
    franky.fail_move = True
    actions = [gap_action(GAP_SEED), gap_action(Q_START + 0.1), gap_action(Q_START + 0.2, 0.0)]
    bridge, franky, wire = make_bridge(actions, franky=franky)
    run_ticks(bridge, 3)
    assert bridge.fault is not None and "cartesian_reflex" in bridge.fault
    assert not franky.moves()
    assert len(wire.sent) == 3, "state keeps streaming after a fault"


def test_camera_payload_matches_gap_wire_format_and_pose():
    t = np.eye(4)
    t[:3, 3] = [0.05, 0.0, 0.02]
    franky = FakeFranky()
    wire = FakeWire([{}])
    bridge = Bridge(BridgeConfig(camera_hz=1e6), franky, FakeCamera(), wire, t)
    bridge.connect()
    bridge.tick()
    cam = wire.sent[0]["camera_top"]
    assert cam["images"]["rgb"].dtype == np.uint8
    assert cam["depth_data"].dtype == np.float32
    assert cam["intrinsics"]["left"]["intrinsics_matrix"].shape == (3, 3)
    np.testing.assert_allclose(cam["pose_mat"], flange_pose(Q_START) @ t, atol=1e-6)


def test_fk_mismatch_refuses_to_start():
    franky = FakeFranky()

    def off_by_1cm(q):
        pose = flange_pose(q)
        pose[2, 3] += 0.01
        return tuple(pose.reshape(-1))

    franky.flange_pose_of = off_by_1cm
    with pytest.raises(RuntimeError, match="FK mismatch"):
        Bridge(BridgeConfig(), franky, FakeCamera(), FakeWire([]), np.eye(4)).connect()


def test_change_gate():
    gate = ChangeGate()
    assert not gate.update([1.0, 2.0])
    assert not gate.update([1.0, 2.0])
    assert gate.update([1.0, 2.1])
    assert gate.update([1.0, 2.0]), "stays open"


def test_flange_t_camera_validation(tmp_path):
    good = np.eye(4)
    good[:3, 3] = [0.05, 0.07, 0.0]
    np.save(tmp_path / "ok.npy", good)
    np.testing.assert_allclose(load_flange_t_camera(tmp_path / "ok.npy"), good)
    bad = good.copy()
    bad[0, 0] = 2.0
    np.save(tmp_path / "bad.npy", bad)
    with pytest.raises(ValueError, match="rotation"):
        load_flange_t_camera(tmp_path / "bad.npy")


def test_flange_fk_known_pose():
    # Panda at q = 0: flange at x = 0.088, z = 0.333 + 0.316 + 0.384 - 0.107.
    np.testing.assert_allclose(flange_pose(np.zeros(7))[:3, 3], [0.088, 0.0, 0.926], atol=1e-9)


def test_depth_to_metres_uses_the_device_scale():
    raw = np.array([[0, 1000, 65535]], dtype=np.uint16)
    np.testing.assert_allclose(depth_to_metres(raw, 1e-4), [[0.0, 0.1, 6.5535]], rtol=1e-6)
    with pytest.raises(ValueError):
        depth_to_metres(raw, 0.0)


def test_run_exits_cleanly_when_gap_disconnects():
    bridge, franky, wire = make_bridge([gap_action(GAP_SEED)])

    def boom(obs):
        raise ConnectionError("GaP closed the connection")

    wire.exchange = boom
    t = threading.Thread(target=bridge.run)
    t.start()
    t.join(timeout=2)
    assert not t.is_alive()


def test_a_single_stalled_read_skips_a_tick_but_a_sustained_stall_raises(monkeypatch):
    from z3_gap_bridge.franky_client import FrankyStaleError

    bridge, franky, wire = make_bridge([gap_action(GAP_SEED)] * 10, max_stale_s=0.2)
    real_read = franky.read_q
    stalls = {"n": 1}

    def flaky_read():
        if stalls["n"] > 0:
            stalls["n"] -= 1
            raise FrankyStaleError("read_q round trip took 72.5 ms")
        return real_read()

    franky.read_q = flaky_read
    bridge.tick()  # stalled: skipped, no exception
    assert bridge.stale_ticks == 1 and len(wire.sent) == 0
    bridge.tick()  # recovered
    assert len(wire.sent) == 1

    stalls["n"] = 10**6
    clock = {"t": 0.0}
    monkeypatch.setattr("z3_gap_bridge.bridge.time.monotonic", lambda: clock["t"])
    bridge.tick()
    clock["t"] = 0.5  # stale for longer than max_stale_s
    with pytest.raises(FrankyStaleError):
        bridge.tick()


def test_observations_carry_a_wall_clock_timestamp():
    import time

    bridge, _, wire = make_bridge([{}])
    before = time.time()
    bridge.tick()
    assert before <= wire.sent[0]["timestamp"] <= time.time()


def test_nuc_motion_sends_one_command_per_target():
    target = Q_START + 0.3
    bridge, franky, _ = make_bridge([gap_action(GAP_SEED)] + [gap_action(target)] * 20)
    run_ticks(bridge, 21)
    (only,) = _streamed(franky)
    np.testing.assert_allclose(only, target)  # the NUC's Ruckig interpolates


def test_nuc_motion_refuses_the_tracking_controller():
    franky = FakeFranky()
    franky.controller = "tracking"
    with pytest.raises(RuntimeError, match="POSITION controller"):
        make_bridge([], franky=franky)
    assert not _streamed(franky)


def test_a_reflex_during_a_move_latches_via_status_polling():
    bridge, franky, wire = make_bridge(
        [gap_action(GAP_SEED), gap_action(Q_START + 0.2)] + [gap_action(Q_START - 0.2)] * 5,
        status_poll_s=0.0,
    )
    run_ticks(bridge, 2)
    assert bridge.fault is None and len(_streamed(franky)) == 1
    franky.status = ("RobotMode.Reflex", '["joint_motion_generator_velocity_limits_violation"]')
    run_ticks(bridge, 5)
    assert bridge.fault is not None and "Reflex" in bridge.fault
    assert len(_streamed(franky)) == 1, "nothing is sent after the reflex"


def test_dry_run_never_polls_or_claims():
    bridge, franky, _ = make_bridge([gap_action(GAP_SEED)] * 3, execute=False)
    run_ticks(bridge, 3)
    assert {c[0] for c in franky.calls} <= {"connect_robot", "gripper_probe"}


SOFT = [100.0, 80.0, 100.0, 100.0, 30.0, 50.0, 30.0]


def _impedance_calls(franky):
    return [c[1] for c in franky.calls if c[0] == "set_joint_impedance"]


def _acks(wire):
    return [obs["left"]["impedance_seq"] for obs in wire.sent]


def test_stiffness_change_waits_for_a_still_arm_then_is_acknowledged():
    target = Q_START + 0.05
    actions = [gap_action(GAP_SEED), gap_action(target)] + [
        gap_action(target, impedance=SOFT, impedance_seq=1)
    ] * 15
    bridge, franky, wire = make_bridge(actions)
    startup = len(_impedance_calls(franky))  # connect() sets the configured k_theta
    run_ticks(bridge, 2)  # the move is dispatched: the arm is not still
    run_ticks(bridge, 5)
    assert len(_impedance_calls(franky)) == startup, "not applied before the arm is still"
    run_ticks(bridge, 10)
    assert _impedance_calls(franky)[startup:] == [tuple(SOFT)]
    assert _acks(wire)[-1] == 1 and _acks(wire)[0] == 0


def test_empty_request_restores_the_startup_stiffness():
    actions = [gap_action(GAP_SEED, impedance=[], impedance_seq=1)] * 15
    bridge, franky, wire = make_bridge(actions, k_theta=(1000.0,) * 7)
    run_ticks(bridge, 15)
    assert _impedance_calls(franky) == [(1000.0,) * 7, (1000.0,) * 7]
    assert _acks(wire)[-1] == 1


def test_a_rejected_change_is_retried_once_the_arm_is_still_again():
    actions = [gap_action(GAP_SEED, impedance=SOFT, impedance_seq=1)] * 30
    bridge, franky, wire = make_bridge(actions)
    franky.fail_impedance = 1  # after connect(): the first request is refused (still moving)
    run_ticks(bridge, 12)
    assert _acks(wire)[-1] == 0, "not acknowledged while refused"
    run_ticks(bridge, 18)
    assert _impedance_calls(franky)[-1] == tuple(SOFT)
    assert _acks(wire)[-1] == 1


def test_dry_run_acknowledges_a_stiffness_change_without_touching_the_robot():
    actions = [gap_action(GAP_SEED, impedance=SOFT, impedance_seq=1)] * 3
    bridge, franky, wire = make_bridge(actions, execute=False)
    run_ticks(bridge, 3)
    assert _impedance_calls(franky) == []
    assert _acks(wire)[-1] == 1


# -- joint zero offsets (--kinematics) -------------------------------------------------

DQ_DEG = (0.0, 1.77, 0.0, 0.0, 0.99, 0.52, 0.0)
DQ = np.radians(DQ_DEG)


def test_offsets_are_added_to_readings_and_removed_from_commands():
    target_model = Q_START + DQ + 0.3  # what GaP wants, in the corrected model
    bridge, franky, wire = make_bridge(
        [gap_action(GAP_SEED)] + [gap_action(target_model)] * 5, joint_offsets_deg=DQ_DEG
    )
    run_ticks(bridge, 6)
    np.testing.assert_allclose(wire.sent[0]["left"]["joint_pos"][:7], Q_START + DQ, atol=1e-6)
    (only,) = _streamed(franky)
    np.testing.assert_allclose(only, Q_START + 0.3, atol=1e-9)  # the robot's own angles
    # the arm now reads where GaP asked, in GaP's model
    run_ticks(bridge, 1)
    np.testing.assert_allclose(wire.sent[-1]["left"]["joint_pos"][:7], target_model, atol=1e-6)


def test_offsets_joint_limits_are_the_robots():
    from z3_gap_bridge.bridge import JOINT_POS_MAX

    target = Q_START + DQ
    target[1] = JOINT_POS_MAX[1] + DQ[1] + 0.1  # beyond the robot's limit, in the model
    bridge, franky, _ = make_bridge([gap_action(GAP_SEED)] + [gap_action(target)] * 2, joint_offsets_deg=DQ_DEG)
    run_ticks(bridge, 3)
    (only,) = _streamed(franky)
    assert only[1] == pytest.approx(JOINT_POS_MAX[1])


def test_offsets_camera_pose_uses_the_corrected_model():
    t = np.eye(4)
    t[:3, 3] = [0.05, 0.0, 0.02]
    wire = FakeWire([{}])
    bridge = Bridge(BridgeConfig(camera_hz=1e6, joint_offsets_deg=DQ_DEG), FakeFranky(), FakeCamera(), wire, t)
    bridge.connect()
    bridge.tick()
    np.testing.assert_allclose(wire.sent[0]["camera_top"]["pose_mat"], flange_pose(Q_START + DQ) @ t, atol=1e-6)


def test_offsets_streamed_references_are_in_the_robots_frame():
    franky = FakeFranky()
    franky.controller = "tracking"
    target_model = Q_START + DQ + 0.2
    bridge, franky, _ = make_bridge(
        [gap_action(GAP_SEED)] + [gap_action(target_model)] * 400, franky=franky,
        motion="stream", joint_offsets_deg=DQ_DEG,
    )
    run_ticks(bridge, 401)
    refs = _streamed(franky)
    assert len(refs) > 10
    np.testing.assert_allclose(refs[-1], Q_START + 0.2, atol=1e-3)
    assert np.abs(np.diff(np.array(refs), axis=0)).max() < 0.02  # still small steps


def test_offsets_floor_guard_clamps_in_the_corrected_model():
    """A GaP target whose tool tip (in the corrected model, which is what GaP's IK and the
    camera use) is below the floor is clamped onto it, and the robot receives the clamped
    angles in its own frame."""
    from z3_gap_bridge.panda_fk import tool_tip_position

    cfg = dict(joint_offsets_deg=DQ_DEG, z_floor_m=0.02)
    q_model = Q_START + DQ
    low = q_model.copy()
    while tool_tip_position(low, 0.220)[2] > 0.0:
        low[1] += 0.002  # lean forward until the tip is 2 cm below the floor
    bridge, franky, _ = make_bridge([gap_action(GAP_SEED)] + [gap_action(low)] * 2, **cfg)
    run_ticks(bridge, 3)
    assert bridge.floor_clamp_count == 1
    (sent,) = _streamed(franky)
    model_sent = sent + DQ
    assert tool_tip_position(model_sent, 0.220)[2] >= 0.02 - 1e-3
    assert tool_tip_position(model_sent, 0.220)[2] < tool_tip_position(q_model, 0.220)[2]


def test_load_kinematics_reads_a_handeye_result(tmp_path):
    import json

    from z3_gap_bridge.bridge import load_kinematics

    np.save(tmp_path / "flange_T_camera.npy", np.eye(4))
    (tmp_path / "report.json").write_text(json.dumps({"joint_offsets_deg": list(DQ_DEG)}))
    offsets, x = load_kinematics(tmp_path)
    assert offsets == DQ_DEG and x.shape == (4, 4)
    (tmp_path / "report.json").write_text(json.dumps({"joint_offsets_deg": [0, 9, 0, 0, 0, 0, 0]}))
    with pytest.raises(ValueError, match="not a zero offset"):
        load_kinematics(tmp_path)
