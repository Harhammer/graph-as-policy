"""robot.insert_twist against a toy compliant plant: no robot, no IK backend.

The plant: commanded tool height z_cmd, measured height = max(z_cmd, block) + sag, where
``block`` is the tooth tops (contact) unless the commanded yaw sits in an aligned window,
then the seat. That is the gear insertion in one dimension: free fall to contact, stuck
until the twist lines the teeth up, then down to the seat.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

import pytest
from gap_core.errors import ToolError

from gap.connector.core import Connector, _twist_pattern

START_Z = 0.150
CONTACT_Z = 0.140
SEAT_Z = 0.130


START_XY = (0.43, 0.02)


class Plant:
    """``droop_xy``: a soft arm's sideways droop (measured = commanded + droop)."""

    def __init__(self, aligned=lambda yaw: abs(yaw - 3.0) < 0.6, sag=-0.0004, droop_xy=(0.0, 0.0)):
        self.aligned = aligned
        self.sag = sag
        self.droop_xy = np.asarray(droop_xy, float)
        self.z_cmd = START_Z - sag  # the arm rests at START_Z
        self.xy_cmd = np.asarray(START_XY, float) - self.droop_xy
        self.yaw = 0.0
        self.commands: list[tuple[float, float]] = []
        self.xy_commands: list[np.ndarray] = []
        self.gripper: list[float] = []

    def measured(self) -> float:
        block = SEAT_Z if self.aligned(self.yaw) else CONTACT_Z
        return max(self.z_cmd, block) + self.sag

    def measured_xy(self) -> np.ndarray:
        return self.xy_cmd + self.droop_xy


def _connector(plant: Plant) -> Connector:
    conn = object.__new__(Connector)
    conn.is_real = True
    conn.control_freq = 50.0
    conn._tcp_offset = (0.0, 0.0, -0.157)

    def get_ee_pose(arm_id=0):
        x, y = plant.measured_xy()
        return {
            "position": {"x": float(x), "y": float(y), "z": plant.measured()},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},  # fingers down
        }

    def solve_ik(pose, tcp, arm_id=0):
        q = pose["rotation"]
        # fingers-down base (w=0,x=1) composed with a tool-z yaw: x=cos(y/2), y=sin(y/2)
        yaw = math.degrees(2 * math.atan2(-q["y"], q["x"]))
        yaw = (yaw + 180.0) % 360.0 - 180.0  # q and -q are the same rotation
        p = pose["position"]
        return [p["x"], p["y"], p["z"], yaw]

    def move_to_joints(joints, **_):
        x, y, plant.z_cmd, plant.yaw = joints
        plant.xy_cmd = np.array([x, y])
        plant.commands.append((joints[2], joints[3]))
        plant.xy_commands.append(np.array([x, y]))

    conn.get_ee_pose = get_ee_pose
    conn._solve_ik = solve_ik
    conn.move_to_joints = move_to_joints
    conn.set_gripper = lambda frac, arm_id=0: plant.gripper.append(frac)
    # the fingers arrive at once: the relax step's wait returns immediately
    conn.get_gripper_fraction = lambda arm_id=0: plant.gripper[-1] if plant.gripper else 0.45
    conn._quiet_motion = contextlib.nullcontext
    return conn


def _insert(plant: Plant, **kw):
    kw.setdefault("dwell_s", 0.0)
    return _connector(plant).insert_twist(depth_m=START_Z - SEAT_Z, **kw)


def test_pattern_sweeps_out_and_back():
    assert _twist_pattern(2.0, 1.0, 1) == [1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]
    assert len(_twist_pattern(5.0, 1.0, 3)) == 60


def test_twists_until_the_teeth_align_then_seats():
    plant = Plant()
    out = _insert(plant)
    assert out["inserted"] is True
    assert out["contact_z"] == pytest.approx(CONTACT_Z + plant.sag)
    assert out["yaw_deg"] == pytest.approx(3.0)
    assert out["twist_steps"] == 3
    assert out["final_z"] <= SEAT_Z + 0.0015
    # never pushed more than push_m (+ one descend step) past the contact point
    deepest_while_stuck = min(z for z, yaw in plant.commands if abs(yaw - 3.0) >= 0.6)
    assert deepest_while_stuck >= CONTACT_Z - 0.003 - 0.001 - 1e-9
    # push released once seated: the last command is the seat height, not below it
    assert plant.commands[-1][0] == pytest.approx(SEAT_Z - plant.sag)


def test_already_aligned_goes_straight_in_without_twisting():
    plant = Plant(aligned=lambda yaw: True)
    out = _insert(plant)
    assert out["inserted"] is True and out["twist_steps"] == 0 and out["contact_z"] is None


def test_finds_negative_yaw():
    out = _insert(Plant(aligned=lambda yaw: abs(yaw + 4.0) < 0.6))
    assert out["inserted"] is True and out["yaw_deg"] == pytest.approx(-4.0)


def test_never_aligns_raises_and_holds_without_pulling_out():
    plant = Plant(aligned=lambda yaw: False)
    with pytest.raises(ToolError, match="not seated after 32 twist steps"):
        _insert(plant, max_cycles=1, twist_deg=8.0, twist_step_deg=1.0)
    z_last, _ = plant.commands[-1]
    assert z_last == pytest.approx(CONTACT_Z), "holds at the contact height, no push, no lift"


def test_relaxes_the_grip_at_first_contact_only():
    plant = Plant()
    _insert(plant, relax_close_fraction=0.45)
    assert plant.gripper == [pytest.approx(0.55)]


def test_rejects_non_positive_depth():
    with pytest.raises(ToolError):
        _connector(Plant()).insert_twist(depth_m=0.0)


def test_start_pose_compensates_the_soft_arms_droop():
    """The arm droops 2.3 mm sideways after the switch to soft: with start_pose the tool
    still goes down the intended line (measured x/y = start x/y), without it it would
    start from the drooped spot."""
    droop = (-0.0023, 0.0003)
    plant = Plant(droop_xy=droop)
    plant.xy_cmd = np.asarray(START_XY) + 0.0  # commanded the column's top ...
    start = {"position": {"x": START_XY[0], "y": START_XY[1], "z": START_Z},
             "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}
    plant.z_cmd = START_Z - plant.sag  # ... and drooped sideways from it
    out = _insert(plant, start_pose=start)
    assert out["inserted"] is True
    for xy in plant.xy_commands:
        np.testing.assert_allclose(xy + np.asarray(droop), START_XY, atol=1e-6)
    np.testing.assert_allclose(plant.measured_xy(), START_XY, atol=1e-6)


def test_start_pose_far_from_the_arm_is_refused():
    plant = Plant()
    far = {"position": {"x": START_XY[0] + 0.02, "y": START_XY[1], "z": START_Z},
           "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}
    with pytest.raises(ToolError, match="not at the column's top"):
        _insert(plant, start_pose=far)
    assert plant.commands == []


def test_a_squeezed_but_unseated_rotor_is_not_called_seated():
    """Pressing compresses a stuck rotor by up to 1 mm (fixture, teeth, gripper): the old
    check saw that dip as seated. Now the push is released and the height re-measured."""
    plant = Plant(aligned=lambda yaw: abs(yaw - 4.0) < 0.6, sag=0.0)  # blocked heights are real
    stuck = SEAT_Z + 0.0013  # a planet on top of the teeth, 1.3 mm short
    squish = 0.001

    def measured():
        if plant.aligned(plant.yaw):
            block = SEAT_Z
        else:
            block = stuck
        if plant.z_cmd >= block:
            return plant.z_cmd + plant.sag
        # pushing below the block compresses things: 0.17 mm per mm of push, as measured on
        # 2026-11-09 (a 3 mm push dipped a stuck rotor ~0.5 mm), up to `squish`
        return block - min(squish, 0.17 * (block - plant.z_cmd)) + plant.sag

    plant.measured = measured
    out = _insert(plant, seated_tol_m=0.0008)
    assert out["inserted"] is True
    assert out["yaw_deg"] == pytest.approx(4.0), "kept twisting until really aligned"
    assert out["false_seats"] >= 1
    assert out["final_z"] <= SEAT_Z + 0.0008


def _dragging_plant():
    """The sun on free-spinning planets: while it presses on them (commanded below the
    contact) they turn with it, so the relative phase only changes when it turns lifted.
    Meshes when the relative phase is within 0.6 deg of 4 deg."""
    plant = Plant(sag=0.0)
    state = {"rel": 0.0, "last_yaw": 0.0}
    stuck = SEAT_Z + 0.008

    def on_command(z_cmd, yaw):
        pressing = z_cmd < stuck and abs(state["rel"] - 4.0) >= 0.6
        if not pressing:
            state["rel"] += yaw - state["last_yaw"]  # turns freely: phase changes
        state["last_yaw"] = yaw

    def measured():
        block = SEAT_Z if abs(state["rel"] - 4.0) < 0.6 else stuck
        return max(plant.z_cmd, block)

    plant.measured = measured
    return plant, on_command


def _insert_with_hook(plant, on_command, **kw):
    conn = _connector(plant)
    move = conn.move_to_joints

    def hooked(joints, **k):
        on_command(joints[2], joints[3])
        move(joints, **k)
    conn.move_to_joints = hooked
    kw.setdefault("dwell_s", 0.0)
    return conn.insert_twist(depth_m=START_Z - SEAT_Z, **kw)


def test_twisting_under_load_never_meshes_when_the_planets_turn_along():
    plant, hook = _dragging_plant()
    with pytest.raises(ToolError, match="not seated"):
        _insert_with_hook(plant, hook, twist_deg=13.0, twist_step_deg=1.0, max_cycles=1)


def test_lifting_to_twist_meshes():
    plant, hook = _dragging_plant()
    out = _insert_with_hook(plant, hook, twist_deg=13.0, twist_step_deg=1.0, max_cycles=1,
                            lift_to_twist_m=0.001)
    assert out["inserted"] is True and out["yaw_deg"] == pytest.approx(4.0)


def test_a_stall_inside_the_accepted_band_counts_as_inserted():
    """The sun stops where it physically ends, above the planned seat: accepted when within
    stall_ok_m, held and failed when above it."""
    plant = Plant(aligned=lambda yaw: False, sag=0.0)
    out = _insert(plant, max_cycles=1, twist_deg=3.0, stall_ok_m=0.011)  # contact is 10 mm above seat
    assert out["inserted"] is True and out["stalled"] is True
    assert out["final_z"] == pytest.approx(CONTACT_Z, abs=1e-6)
    plant = Plant(aligned=lambda yaw: False, sag=0.0)
    with pytest.raises(ToolError, match="not seated"):
        _insert(plant, max_cycles=1, twist_deg=3.0, stall_ok_m=0.005)


def test_small_unload_meshes_planets_that_roll_under_load():
    """The rotor's planets roll on the ring teeth while pressed (phase kept); unloading by
    0.5 mm for each turn lets them slide into phase. Same drag model as for the sun."""
    plant, hook = _dragging_plant()
    out = _insert_with_hook(plant, hook, twist_deg=9.5, twist_step_deg=1.0, max_cycles=2,
                            lift_to_twist_m=0.0005, push_m=0.002)
    assert out["inserted"] is True and abs(out["yaw_deg"] - 4.0) < 0.6  # inside the mesh window


def test_a_stall_inside_the_band_is_accepted_at_once_without_twisting():
    plant = Plant(aligned=lambda yaw: False, sag=0.0)  # stops 10 mm above the planned end
    out = _insert(plant, max_cycles=1, twist_deg=13.0, stall_ok_m=0.011, stall_accept_at_contact=True)
    assert out["inserted"] is True and out["stalled"] is True
    assert out["twist_steps"] == 0, "no twisting once it has stopped inside the band"
    assert all(yaw == 0.0 for _, yaw in plant.commands)


def test_a_stall_above_the_band_still_twists():
    plant = Plant(aligned=lambda yaw: abs(yaw - 3.0) < 0.6, sag=0.0)  # stops 10 mm up, meshes at 3 deg
    out = _insert(plant, max_cycles=1, twist_deg=5.0, stall_ok_m=0.005, stall_accept_at_contact=True)
    assert out["inserted"] is True and out["twist_steps"] > 0 and out.get("stalled") is not True


def test_droop_is_measured_after_it_has_developed():
    """After the switch to soft the arm sags over ~0.3 s. Measured at once, the droop would
    be ~0 and the tool would go down the drooped line; it must wait and compensate it all."""
    import time as _time

    full = np.array([-0.0023, 0.0003])
    t0 = _time.monotonic()
    plant = Plant(sag=0.0)
    plant.xy_cmd = np.asarray(START_XY, float)
    plant.droop_xy = np.zeros(2)

    def measured_xy():
        frac = min(1.0, (_time.monotonic() - t0) / 0.3)  # the sag develops
        return plant.xy_cmd + full * frac

    plant.measured_xy = measured_xy
    start = {"position": {"x": START_XY[0], "y": START_XY[1], "z": START_Z},
             "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}
    out = _insert(plant, start_pose=start, settle_window_s=0.15)
    assert out["inserted"] is True
    assert out["droop_mm"] == pytest.approx(np.hypot(*full) * 1e3, abs=0.05)
    np.testing.assert_allclose(plant.measured_xy(), START_XY, atol=2e-5)  # on the intended line


def test_a_part_on_the_rim_is_found_by_the_sideways_search():
    """The sun 1 mm off the bore's axis rests on its rim; the ring search moves it sideways
    until it drops in, then it is accepted at its end stop."""
    from gap.connector.core import _search_rings

    plant = Plant(sag=0.0)
    hole = np.asarray(START_XY) + np.array([-0.001, 0.0003])  # the bore's real axis
    rim = SEAT_Z + 0.013  # stuck on the rim, above the accepted band

    def measured():
        inside = np.hypot(*(plant.xy_cmd - hole)) < 0.00025
        return max(plant.z_cmd, SEAT_Z if inside else rim)

    plant.measured = measured
    out = _insert(plant, stall_ok_m=0.010, stall_accept_at_contact=True,
                  search_radius_m=0.0015, search_step_m=0.0003, max_cycles=1, twist_deg=3.0)
    assert out["inserted"] is True
    assert np.hypot(*(plant.xy_cmd - hole)) < 0.00025, "ended over the bore"
    assert len(_search_rings(0.0015, 0.0003)) > 50


def test_without_search_the_part_on_the_rim_fails():
    plant = Plant(sag=0.0)
    hole = np.asarray(START_XY) + np.array([-0.001, 0.0003])

    def measured():
        inside = np.hypot(*(plant.xy_cmd - hole)) < 0.00025
        return max(plant.z_cmd, SEAT_Z if inside else SEAT_Z + 0.013)

    plant.measured = measured
    with pytest.raises(ToolError, match="not seated"):
        _insert(plant, stall_ok_m=0.010, stall_accept_at_contact=True, max_cycles=1, twist_deg=3.0)
