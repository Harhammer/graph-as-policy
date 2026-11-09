#!/usr/bin/env python
"""The smallest real-robot graph: move to fixed poses, open and close the gripper.

No perception, no LLM. Joint poses (POSES, "move" steps) go straight to the arm (through
the z3 bridge and the NUC's Ruckig position controller); end-effector poses (EEF_POSES,
"pose" steps) go through GaP's IK first. Build + validate, no robot needed:

    cd ~/graph-as-policy && .venv/bin/python z3_graphs/poses_demo/build.py

then run it like any graph (bridge in the other terminal; dry run first):

    .venv/bin/gap run z3_graphs/poses_demo/graph --real franka --no-rr-autostart

Edit the poses below. Record one by hand-guiding the arm there and running
``uv run z3-gap-joints --kinematics ~/z3_handeye/combined_kinfit --record`` in
bridges/z3_franka — it prints POSES and EEF_POSES lines to paste. The ``--kinematics`` must
match the bridge's: the poses here are in the corrected kinematics (2026-11-09), a pose
recorded without it is in the robot's own model and lands ~13 mm too high / 5 mm off.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

# Joint angles [rad], joints 1-7. START is the pose the arm stood in for the 2026-11-06
# motion test; the others are small offsets from it. Replace them with your own.
POSES: dict[str, list[float]] = {
    "start": [-0.175, -0.498, 0.064, -2.572, -0.006, 2.098, -0.068],
    "on_top_off_gears": [-0.121, -0.052, -0.107, -2.567, -0.008, 2.510, -0.176],  
    "at_the_gears": [-0.107, 0.023, -0.117, -2.589, -0.009, 2.616, -0.182], 
}

# End-effector poses: the gripper's tool point in the robot base frame (x forward, z up),
# xyz in metres and roll/pitch/yaw in degrees about the base's fixed x, y, z axes
# (scipy "xyz"). Roll 180 = fingers pointing straight down. GaP's IK solves for the joints,
# so the arm can take a different elbow path than a recorded joint pose would.
# Convert recorded joint poses with joints_to_eef.py (same FK and tool point as the IK).
EEF_POSES: dict[str, dict[str, list[float]]] = {
    # In the CORRECTED kinematics (bridge --kinematics ~/z3_handeye/combined_kinfit): xyz
    # converted 2026-11-09 with convert_poses.py from the poses taught against the robot's
    # own model (the conversion showed those were tilted ~1.2 deg without anyone knowing).
    # [180, 0, 0] is truly vertical in this model (the table normal is within 0.5 deg of it).
    # The rotor grasp itself comes from the camera (vision_grasp: vertical, x/y detected,
    # z = at_gears' z); these are the look / transit poses.
    "start": {"xyz": [0.3898, -0.0461, 0.2637], "rpy_deg": [180, 0, 0]},
    "above_gears": {"xyz": [0.4274, -0.096, 0.1354], "rpy_deg": [180, 0, 0]},
    "at_gears": {"xyz": [0.4257, -0.0967, 0.0848], "rpy_deg": [180, 0, 0]},
    # Sun gear look pose: like look_gearbox (camera ~18 cm above the part, part centred),
    # over SUN_NOMINAL_XY (0.317, -0.064).
    "look_sun": {"xyz": [0.332, -0.064, 0.105], "rpy_deg": [180, 0, 0]},
    # Gearbox look pose: hand-guided 2026-11-09 to (0.4431, 0.0278, 0.0986) (camera 16.5 cm
    # above the top face, gearbox centred in the image) and raised 3 cm for clearance.
    "look_gearbox": {"xyz": [0.4431, 0.0278, 0.1286], "rpy_deg": [180, 0, 0]},
}

# The gearbox column: the rotor SEATED in the gearbox, recorded by hand-guiding it in from
# above_gearbox with the rotation locked (`z3-gap-joints --kinematics
# ~/z3_handeye/combined_kinfit --record`). The lock is an impedance: seating the rotor
# tilts the tool by ~0.5 deg, and the rotor hangs ~6 cm below the TCP point, so each
# recording is corrected to a vertical tool about the rotor (0.5 mm per 0.5 deg). Two
# seatings on 2026-11-09 gave (0.4310, 0.0265, 0.0983) and (0.4309, 0.0274, 0.0979); the
# mean is used (the difference is the rotor's play). The orientation is the graph's exact
# vertical, never the recorded one. The other heights are derived, so the insertion goes
# straight down along one line.
SEATED = {"xyz": [0.4310, 0.0270, 0.0981], "rpy_deg": [180, 0, 0]}
# How much deeper than at_gears the jaws close on the rotor [m]. At at_gears' height the
# fingertips (0.217 m) end ~2.5 mm below the rotor's top: only the rim was held and the rotor
# pivoted in the jaws during the insertion (2026-11-09). The rotor stands 19 mm proud of its
# fixture, but the jaws ride down with it into the gearbox: seated, their tips must stay above
# the gearbox's top face (33.9 mm) — 4.2 mm clearance at 0, 2.2 mm at 0.002. The rotor then
# sits that much higher in the jaws, so the whole column below moves down by the same amount
# (SEATED above is the rotor seated as recorded with the old grasp).
GRASP_DEEPER_M = 0.002
# With ("vision_gearbox",) in STEPS, SEATED's x/y is only where the search starts and what
# the detection is compared with: the camera measured the gearbox axis at (0.4281, 0.0277)
# on 2026-11-09, 2.9 mm off in x — the hand-seated rotor had slid along the jaw faces (the
# jaws hold it only across them). The column's heights stay as taught.
COLUMN_ABOVE_SEATED_M = {
    "in_gearbox": 0.0,
    "touching_gearbox": 0.0036,  # teeth first touch: 3.6 mm of engagement
    "pre_insert": 0.0152,  # where the insertion starts (soft arm, 1 mm steps)
    "above_gearbox": 0.0378,  # clear of the gearbox for the transfer
}
for _name, _dz in COLUMN_ABOVE_SEATED_M.items():
    EEF_POSES[_name] = {
        "xyz": [SEATED["xyz"][0], SEATED["xyz"][1], round(SEATED["xyz"][2] + _dz - GRASP_DEEPER_M, 4)],
        "rpy_deg": list(SEATED["rpy_deg"]),
    }

#: The sequence: ("move", joint pose name), ("pose", EEF pose name), ("open",),
#: ("close",) to close fully, or ("close", 0.5) to close only that fraction of the way
#: (0 = stay open, 1 = fully closed), ("wait", 2.0) to pause that many seconds,
#: ("insert", "in_gearbox") to push straight down from the previous pose step to that EEF
#: pose, twisting about the tool axis while the teeth stick (robot.insert_twist, INSERT),
#: ("stiffness", "soft") to switch the arm's joint stiffness to STIFFNESS["soft"] (applied
#: once the arm is at rest), ("stiffness", "startup") for the bridge's --k-theta again,
#: ("capture",) / ("capture", "label") to save the wrist camera frame to CAPTURE_DIR,
#: ("vision_grasp",) to find the rotor with the wrist camera from the current pose and move
#: onto it (VISION): above it, jaws across two empty screw holes, then down to grasp height.
STEPS: list[tuple[str, ...]] = [
    ("open",),
    ("pose", "start"),
    # find the gearbox first, gripper empty: the column below then sits on its real axis
    ("pose", "look_gearbox"),
    ("wait", .5),
    ("vision_gearbox",),
    ("pose", "above_gears"),
    ("wait", .5),
    ("vision_grasp",),
    ("close", 0.55),
    #("wait", 1.0),
   # ("wait", 2.0),
  #  ("wait", 1.0),
    ("pose", "above_gears"),
    ("pose", "above_gearbox"),
    ("pose", "pre_insert"),
    ("wait", .5),
    ("capture", "held"),  # the held rotor seen from the wrist: is it tilted in the jaws?
    ("stiffness", "soft"),
    ("insert", "in_gearbox"),
    ("open",),
    #("wait", 1.0),
    ("pose", "above_gearbox"),
    ("stiffness", "startup"),
    # --- the sun gear: find it, measure the seated rotor, grasp the sun as it stands, insert ---
    ("pose", "start"),
    ("find_sun",),  # wide look from start: where is it roughly? then over to it
    ("vision_sun",),
    ("plan_sun",),  # no camera: the rotor's top = the housing's top from vision_gearbox
    ("grasp_sun",),
    ("insert_sun",),
    ("pose", "start"),
    ("open",),
]

# On the real arm, robot.move_to_joints' defaults (tolerance 0.01 rad, max_steps 120 ~ 1.2 s)
# would send the next pose before a 2 s move arrives, and the wrist settles ~13 mrad short
# (measured 2026-11-06). 0.03 rad / 600 steps (~6 s) waits for arrival.
MOVE = {"tolerance": 0.03, "max_steps": 600}
GRIPPER_SETTLE_STEPS = 75  # ~1.5 s: the Robotiq's full stroke at the bridge's speed
PAUSE_S = 0  # pause inserted after every step (except explicit waits); 0 = none

# robot.insert_twist settings. The arm is soft (the bridge's joint impedance), so pushing
# push_m below the stuck height gives a gentle press; start small and raise it if the planets
# don't drop in. The twist only has to jog the free-spinning planets into mesh.
# Joint stiffness profiles [Nm/rad, joints 1-7] for ("stiffness", name) steps. "startup" =
# whatever the bridge was started with (--k-theta). "soft" is the bridge's default, the
# oktonex teleop values for this cell: compliant enough for insert_twist's gentle press.
STIFFNESS: dict[str, list[float] | None] = {
    "startup": None,
    "soft": [300.0, 200.0, 300.0, 300.0, 50.0, 100.0, 50.0],
}

# ("vision_grasp",) settings. Look from above_gears (rotor ~15 cm from the camera, ~0.34
# mm/px), held still: put ("wait", 0.5) before it. scripts/rotor_vision.py has the method.
VISION = {
    "nominal": "at_gears",  # the taught grasp: its x/y is where the rotor should be, z = grasp z
    "above": "above_gears",  # approach height over the detected rotor
    # rotor top surface [m] in the corrected kinematics (0.0275 from the 2026-11-09 multi-view
    # check; it was 0.039 in the robot's own model). Re-measured from depth anyway. ~6 cm below
    # at_gears' z because GaP's TCP is the stock 0.157 m while the custom fingertips reach
    # further (the bridge uses 0.217) — taught poses carry that offset consistently.
    "z_top_guess": 0.0275,
    # The direction the jaws close at rpy [180, 0, 0], in the base frame: 0 = along base x
    # (front-back), 90 = along base y (left-right). 90 read from the 2026-11-09 captures: the
    # fingertips sit left/right in the wrist image, whose horizontal axis is base -y.
    "jaw_axis_deg": 90.0,
    "max_offset_m": 0.008,  # further from nominal than the fixture play = misdetection: stop
    "grasp_deeper_m": GRASP_DEEPER_M,  # see GRASP_DEEPER_M (moves the gearbox column too)
    # Added to every detected centre [m]. Zero since the bridge runs the corrected kinematics
    # (2026-11-09); before that, [-0.0026, 0.0038] patched the arm-model error for the one
    # look pose. `tune_rotor.py --reference <frames>` re-measures it (rotor sitting where
    # at_gears grasps it centred); it should stay within a millimetre now.
    "correction_xy": [0.0, 0.0],
    # The bridge's --kinematics result (camera in the flange frame), for the multi-view check.
    "flange_t_camera": "~/z3_handeye/combined_kinfit/flange_T_camera.npy",
    "rotor_top_z": 0.0275,
}
CAPTURE_DIR = "~/z3_rotor_images"  # ("capture",) frames; detections go to <dir>/detections

# ("vision_gearbox",): from look_gearbox (gripper empty), find the gearbox axis
# (scripts/gearbox_vision.py: the 4 corner holes on their 41.0 mm square, checked against
# the 38-tooth ring) and put the insertion column on it. Later ("pose", <column pose>) and
# ("insert", ...) steps then use the detected column instead of EEF_POSES.
GEARBOX_COLUMN = ("above_gearbox", "pre_insert", "touching_gearbox", "in_gearbox")
# The sun gear (no fixture): looked for around SUN_NOMINAL_XY, grasped as it stands with the
# jaws on its top SUN["grip_m"], inserted into the seated rotor's centre bore in two moves
# (scripts/plan_sun_insert.py): held, twisting while it meshes with the planets, until the
# fingertips are clearance_m above the rotor's top; then released and pressed flush with
# the closed fingertips, wiggling a little — dropped 2026-11-09: the sun never sits flush, so
# after the held move the gripper just opens and the arm leaves. The held move starts
# start_gap_m above contact: more than the soft arm's ~4.4 mm droop.
SUN_NOMINAL_XY = (0.317, -0.064)  # where it stood on 2026-11-09: preferred if several candidates
# ("find_sun",) looks from the start pose and ignores these (x, y, radius) areas: the rotor's
# fixture (looks sun-like from afar once empty) and the gearbox.
SUN_SEARCH_EXCLUDE = [[0.4257, -0.0967, 0.03], [0.4310, 0.0270, 0.04]]
SUN = {
    "grip_m": 0.004,
    "close_fraction": 0.85,  # Ø17.6 mm: the Robotiq stops on it well before this
    "clearance_m": 0.002,  # fingertips above the rotor's top at the end of the held move
    "start_gap_m": 0.008,
    "press_gap_m": 0.008,
}
# 14 teeth: 25.7 deg pitch, so +-13 deg covers any phase. Twisted while pressing (lifting 1 mm
# for every step was tried 2026-11-09: jerky and no better). Where the sun stops is accepted
# when it is inside the bore (plan_sun's insert_stall_ok_m), see robot.insert_twist stall_ok_m.
SUN_INSERT = {"push_m": 0.003, "twist_deg": 13.0, "twist_step_deg": 1.5, "max_cycles": 1,
              "seated_tol_m": 0.0008,
              # it sinks in to its end stop on its own: accept that stop right away; twist
              # only when it stops above the accepted band (on the rim / the planets' tops)
              "stall_accept_at_contact": True,
              # stopped on the bore's rim (0.5 mm clearance; 2026-11-09 17:24 it sat ~1 mm off):
              # search sideways in rings out to 1.5 mm, 0.3 mm apart, until it drops in
              "search_radius_m": 0.0015, "search_step_m": 0.0003}

GEARBOX_VISION = {
    "z_top_guess": 0.034,  # top face [m], corrected kinematics; re-measured from depth
    "max_offset_m": 0.006,  # further from SEATED than this = misdetection or moved: stop
    "correction_xy": [0.0, 0.0],  # held rotor vs tool axis; 0 with the vision grasp
}

INSERT = {
    # The settings of the runs that worked on 2026-11-09 (16:32, 16:57). Tried and reverted the
    # same day: a +-9.5 deg twist (a caught rotor tilts further) and unloading 0.5 mm per twist
    # step. Failures with these settings came from the rotor's body catching the ring's rim
    # (0.5 mm clearance per side; the rotor hangs ~3 deg tilted in the jaws), not from the twist.
    "push_m": 0.003,  # how far past the stuck height to aim (sets the press force)
    "twist_deg": 5.0,  # sweep 0 -> +5 -> -5 -> 0 deg about the tool axis while stuck
    "twist_step_deg": 1.0,
    "max_cycles": 3,  # sweeps before giving up (the arm then holds and the graph stops)
    # seated = within this of in_gearbox's height. The teeth engage only 3.6 mm (touching
    # 0.1139 -> seated 0.1103), so the 1.5 mm default would call it seated 2.1 mm in.
    "seated_tol_m": 0.0008,
    "relax_close_fraction": -1.0,  # e.g. 0.45 to loosen the 0.55 grasp at contact; -1 = keep
}


_MULTIVIEW = False


def args_multiview() -> bool:
    return _MULTIVIEW


def eef_pose(spec: dict[str, list[float]]) -> dict:
    """{"xyz", "rpy_deg"} (or {"xyz", "quat_wxyz"}) -> GaP's Se3Pose dict."""
    from scipy.spatial.transform import Rotation

    x, y, z = (float(v) for v in spec["xyz"])
    if "quat_wxyz" in spec:
        w, qx, qy, qz = (float(v) for v in spec["quat_wxyz"])
    else:
        qx, qy, qz, w = Rotation.from_euler("xyz", spec["rpy_deg"], degrees=True).as_quat()
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": float(w), "x": float(qx), "y": float(qy), "z": float(qz)},
    }


def _insert_depth(before: list[tuple[str, ...]], seated: str) -> float:
    """Depth of an insert step: the last pose step's height minus the seated pose's height."""
    starts = [step[1] for step in before if step[0] == "pose"]
    if not starts:
        raise ValueError(f"insert {seated!r} needs a ('pose', ...) step before it")
    if seated not in EEF_POSES:
        raise ValueError(f"insert: record EEF_POSES[{seated!r}] first (the seated pose)")
    a, b = EEF_POSES[starts[-1]], EEF_POSES[seated]
    dxy = ((a["xyz"][0] - b["xyz"][0]) ** 2 + (a["xyz"][1] - b["xyz"][1]) ** 2) ** 0.5
    drpy = max(abs(float(u) - float(v)) for u, v in zip(a["rpy_deg"], b["rpy_deg"], strict=True))
    if dxy > 0.001 or drpy > 0.2:
        raise ValueError(
            f"insert goes straight down: {starts[-1]!r} and {seated!r} must share x/y "
            f"(off by {dxy * 1e3:.1f} mm) and rpy_deg (off by {drpy:.2f} deg)"
        )
    depth = a["xyz"][2] - b["xyz"][2]
    if not 0.0 < depth < 0.1:
        raise ValueError(f"insert depth {depth:.4f} m from {starts[-1]!r} looks wrong")
    return depth


def build() -> Workflow:
    """One subgraph that walks STEPS in order, wrapped in a workflow with two ends."""
    sg = Subgraph(name="poses_sg")
    previous = "START"
    gearbox_node = None  # set by a ("vision_gearbox",) step
    sun_node = plan_node = find_node = None  # set by ("vision_sun",) / ("plan_sun",) / ("find_sun",)

    def chain(prefix: str, nodes: list[tuple[str, dict]]) -> tuple[str, str]:
        """Add nodes prefix_<suffix> in order, edges between them; (first, last) names."""
        names = []
        for suffix, spec in nodes:
            n = f"{prefix}_{suffix}"
            sg.add_node(n, **spec)
            if names:
                sg.add_edge(names[-1], n)
            names.append(n)
        return names[0], names[-1]

    def go(ref: str) -> dict:
        return {"type": "tool", "tool": "robot.go_to_pose", "inputs": {"pose": Ref(ref), **MOVE}}

    def stiff(profile: str) -> dict:
        k = STIFFNESS[profile]
        return {"type": "tool", "tool": "robot.set_joint_stiffness", "inputs": {} if k is None else {"k_theta": k}}

    def gripper(close_fraction: float | None) -> dict:
        if close_fraction is None:
            return {"type": "tool", "tool": "robot.open_gripper", "inputs": {"settle_steps": GRIPPER_SETTLE_STEPS}}
        return {"type": "tool", "tool": "robot.close_gripper",
                "inputs": {"settle_steps": GRIPPER_SETTLE_STEPS, "fraction": close_fraction}}
    for i, step in enumerate(STEPS):
        name = f"{i:02d}_{'_'.join(str(s).replace('.', 'p') for s in step)}"
        first = last = name  # a step that adds several nodes chains them first -> last
        column_ref = (
            (lambda pose_name: Ref(f"{gearbox_node}.{pose_name}"))
            if gearbox_node else None
        )
        if step[0] == "wait":
            sg.add_node(name, type="tool", tool="robot.wait", inputs={"seconds": float(step[1])})
        elif step[0] == "move":
            sg.add_node(
                name,
                type="tool",
                tool="robot.move_to_joints",
                inputs={"joint_config": POSES[step[1]], **MOVE},
            )
        elif step[0] == "pose":
            detected = column_ref is not None and step[1] in GEARBOX_COLUMN
            sg.add_node(
                name,
                type="tool",
                tool="robot.go_to_pose",
                inputs={"pose": column_ref(step[1]) if detected else eef_pose(EEF_POSES[step[1]]), **MOVE},
            )
        elif step[0] == "vision_gearbox":
            gearbox_node = name
            sg.add_node(
                name,
                type="script",
                script="scripts/detect_gearbox.py",
                inputs={
                    "nominal_xy": SEATED["xyz"][:2],
                    "column_z": {n: EEF_POSES[n]["xyz"][2] for n in GEARBOX_COLUMN},
                    "z_top_guess": GEARBOX_VISION["z_top_guess"],
                    "max_offset_m": GEARBOX_VISION["max_offset_m"],
                    "correction_xy": GEARBOX_VISION["correction_xy"],
                    "save_dir": f"{CAPTURE_DIR}/gearbox",
                },
            )
        elif step[0] == "insert":
            start_name = [s_[1] for s_ in STEPS[:i] if s_[0] == "pose"][-1]
            sg.add_node(
                name,
                type="tool",
                tool="robot.insert_twist",
                inputs={
                    "depth_m": _insert_depth(STEPS[:i], step[1]),
                    # the column's top: insert_twist compensates the soft arm's droop from it
                    "start_pose": (
                        column_ref(start_name)
                        if column_ref is not None and start_name in GEARBOX_COLUMN
                        else eef_pose(EEF_POSES[start_name])
                    ),
                    **INSERT,
                },
            )
        elif step[0] == "stiffness":
            sg.add_node(
                name,
                type="tool",
                tool="robot.set_joint_stiffness",
                inputs={} if STIFFNESS[step[1]] is None else {"k_theta": STIFFNESS[step[1]]},
            )
        elif step[0] == "capture":
            sg.add_node(
                name,
                type="script",
                script="scripts/capture_frame.py",
                inputs={"out_dir": CAPTURE_DIR + ("/multiview" if args_multiview() else ""),
                        "label": step[1] if len(step) > 1 else ""},
            )
        elif step[0] == "vision_grasp":
            if VISION["jaw_axis_deg"] is None:
                raise ValueError("vision_grasp: set VISION['jaw_axis_deg'] first (see its comment)")
            nominal, above = EEF_POSES[VISION["nominal"]], EEF_POSES[VISION["above"]]
            first, last = f"{name}_detect", f"{name}_descend"
            sg.add_node(
                first,
                type="script",
                script="scripts/detect_rotor.py",
                inputs={
                    "nominal_xy": nominal["xyz"][:2],
                    "z_top_guess": VISION["z_top_guess"],
                    "grasp_z": nominal["xyz"][2] - VISION["grasp_deeper_m"],
                    "above_z": above["xyz"][2],
                    "jaw_axis_deg": float(VISION["jaw_axis_deg"]),
                    "max_offset_m": VISION["max_offset_m"],
                    "save_dir": f"{CAPTURE_DIR}/detections",
                    "correction_xy": VISION["correction_xy"],
                },
            )
            sg.add_node(
                f"{name}_above",
                type="tool",
                tool="robot.go_to_pose",
                inputs={"pose": Ref(f"{first}.above_pose"), **MOVE},
            )
            sg.add_node(
                last,
                type="tool",
                tool="robot.go_to_pose",
                inputs={"pose": Ref(f"{first}.grasp_pose"), **MOVE},
            )
            sg.add_edge(first, f"{name}_above")
            sg.add_edge(f"{name}_above", last)
        elif step[0] == "find_sun":
            find_node = f"{name}_search"
            first, last = chain(name, [
                ("search", {"type": "script", "script": "scripts/find_sun.py", "inputs": {
                    "prefer_xy": list(SUN_NOMINAL_XY), "exclude": SUN_SEARCH_EXCLUDE,
                    "save_dir": f"{CAPTURE_DIR}/sun"}}),
                ("look", go(f"{find_node}.look_pose")),
            ])
        elif step[0] == "vision_sun":
            sun_node = name
            sg.add_node(name, type="script", script="scripts/detect_sun.py", inputs={
                "nominal_xy": Ref(f"{find_node}.center_xy") if find_node else list(SUN_NOMINAL_XY),
                "max_offset_m": 0.012 if find_node else 0.03,
                "grip_m": SUN["grip_m"], "save_dir": f"{CAPTURE_DIR}/sun",
            })
        elif step[0] == "plan_sun":
            if gearbox_node is None or sun_node is None:
                raise ValueError("plan_sun needs ('vision_gearbox',) and ('vision_sun',) before it")
            plan_node = name
            sg.add_node(name, type="script", script="scripts/plan_sun_insert.py", inputs={
                "center_xy": Ref(f"{gearbox_node}.center_xy"),
                "housing_top_z": Ref(f"{gearbox_node}.housing_top_z"),
                "sun_height_m": Ref(f"{sun_node}.height_m"),
                "grip_m": SUN["grip_m"], "clearance_m": SUN["clearance_m"],
                "start_gap_m": SUN["start_gap_m"], "press_gap_m": SUN["press_gap_m"],
            })
        elif step[0] == "grasp_sun":
            if sun_node is None:
                raise ValueError("grasp_sun needs ('vision_sun',) before it")
            first, last = chain(name, [
                ("above", go(f"{sun_node}.above_pose")),
                ("descend", go(f"{sun_node}.grasp_pose")),
                ("close", gripper(SUN["close_fraction"])),
                ("lift", go(f"{sun_node}.above_pose")),
            ])
        elif step[0] == "insert_sun":
            if plan_node is None:
                raise ValueError("insert_sun needs ('plan_sun',) before it")
            first, last = chain(name, [
                ("above", go(f"{plan_node}.above_rotor")),
                ("pre", go(f"{plan_node}.pre")),
                ("soft", stiff("soft")),
                ("insert", {"type": "tool", "tool": "robot.insert_twist", "inputs": {
                    "depth_m": Ref(f"{plan_node}.insert_depth_m"),
                    "stall_ok_m": Ref(f"{plan_node}.insert_stall_ok_m"),
                    "start_pose": Ref(f"{plan_node}.pre"), **SUN_INSERT}}),
                # let go and leave it where it stopped: it never sits flush (2026-11-09), and
                # pressing it further only fights its end stop
                ("release", gripper(None)),
                ("stiff", stiff("startup")),
                ("up", go(f"{plan_node}.above_rotor")),
            ])
        elif step[0] == "open":
            sg.add_node(
                name,
                type="tool",
                tool="robot.open_gripper",
                inputs={"settle_steps": GRIPPER_SETTLE_STEPS},
            )
        elif step[0] == "close":
            sg.add_node(
                name,
                type="tool",
                tool="robot.close_gripper",
                inputs={
                    "settle_steps": GRIPPER_SETTLE_STEPS,
                    "fraction": float(step[1]) if len(step) > 1 else 1.0,
                },
            )
        else:
            raise ValueError(f"unknown step {step!r}")
        sg.add_edge(previous, first)
        previous = last
        if PAUSE_S > 0 and step[0] != "wait" and i < len(STEPS) - 1:
            pause = f"{name}_pause"
            sg.add_node(pause, type="tool", tool="robot.wait", inputs={"seconds": PAUSE_S})
            sg.add_edge(previous, pause)
            previous = pause
    sg.add_exit("finished")  # success marker
    sg.set_on_error("failed")  # any tool error -> "failed"
    sg.add_edge(previous, "finished")
    sg.add_edge("finished", "END")

    wf = Workflow(name="z3_poses_demo", description="Move through fixed joint poses, open/close.")
    wf.add_subgraph(sg)
    wf.add_node("poses", type="subgraph", ref="poses_sg")
    wf.add_node("done", type="end", status="success")
    wf.add_node("abort", type="end", status="failure")  # nothing moves after an error
    wf.add_edge("START", "poses")
    wf.add_conditional_edges("poses", {"finished": "done", "failed": "abort"}, router_field="exit")
    return wf


def multiview_poses(
    tilt_deg: float = 25.0, distance_m: float = 0.21
) -> dict[str, dict[str, list[float]]]:
    """Look poses for the multi-view rotor check: the camera ``distance_m`` from the rotor,
    straight above it and tilted ``tilt_deg`` toward +x, -x, +y, -y, optical axis on the
    rotor, image x axis along base -y (as in the top-down look). Each is converted to the
    GaP tool pose (TCP 0.157 m on the flange z axis, tool orientation = flange orientation)
    through the bridge's flange_T_camera."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    X = np.load(Path(VISION["flange_t_camera"]).expanduser())
    nominal = EEF_POSES[VISION["nominal"]]["xyz"]
    rotor = np.array([nominal[0], nominal[1], VISION["rotor_top_z"]])
    t = np.radians(tilt_deg)
    views = {
        "mv_top": np.array([0.0, 0.0, 1.0]),
        "mv_px": np.array([np.sin(t), 0.0, np.cos(t)]),
        "mv_mx": np.array([-np.sin(t), 0.0, np.cos(t)]),
        "mv_py": np.array([0.0, np.sin(t), np.cos(t)]),
        "mv_my": np.array([0.0, -np.sin(t), np.cos(t)]),
    }
    poses = {}
    for name, direction in views.items():
        cam_pos = rotor + distance_m * direction
        z_axis = (rotor - cam_pos) / np.linalg.norm(rotor - cam_pos)  # optical axis
        x_ref = np.array([0.0, -1.0, 0.0])
        x_axis = x_ref - z_axis * (x_ref @ z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        cam = np.eye(4)
        cam[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        cam[:3, 3] = cam_pos
        flange = cam @ np.linalg.inv(X)
        tcp = flange @ np.array([0.0, 0.0, 0.157, 1.0])
        tips = flange @ np.array([0.0, 0.0, 0.217, 1.0])
        if tips[2] < rotor[2] + 0.02:  # at least 2 cm above the rotor's top surface
            raise ValueError(f"{name}: fingertips would be at z={tips[2]:.3f} m — too low")
        qx, qy, qz, qw = Rotation.from_matrix(flange[:3, :3]).as_quat()
        poses[name] = {"xyz": [round(float(v), 4) for v in tcp[:3]],
                       "quat_wxyz": [round(float(v), 6) for v in (qw, qx, qy, qz)]}
    return poses


def multiview_steps() -> list[tuple[str, ...]]:
    """Multi-view rotor check: capture the rotor from 5 look poses without touching it.
    Analyse with multiview_check.py. The rotor must stay put during the run."""
    EEF_POSES.update(multiview_poses())
    steps: list[tuple[str, ...]] = [("open",), ("pose", "start"), ("pose", "above_gears")]
    for name in ("mv_top", "mv_px", "mv_mx", "mv_py", "mv_my"):
        steps += [("pose", name), ("wait", 0.7), ("capture", name)]
    steps += [("pose", "above_gears"), ("pose", "start")]
    return steps


def sun_steps() -> list[tuple[str, ...]]:
    """Only the sun gear stage, for a rotor that is already seated in the gearbox."""
    return [
        ("open",), ("stiffness", "startup"), ("pose", "start"),
        ("pose", "look_gearbox"), ("wait", .5), ("vision_gearbox",),
        ("pose", "start"), ("find_sun",), ("vision_sun",), ("plan_sun",),
        ("grasp_sun",), ("insert_sun",), ("pose", "start"),
    ]


def capture_steps(n: int) -> list[tuple[str, ...]]:
    """Dataset session: look from above_gears and save n frames. Run with --step and move
    or turn the rotor in its fixture (and vary the light) at each prompt."""
    return [("open",), ("pose", "above_gears"), ("wait", 0.5)] + [
        ("capture", f"{k:02d}") for k in range(n)
    ]


def main(argv: list[str] | None = None) -> int:
    """``build.py`` builds graph/ from STEPS; ``build.py --capture 15`` builds capture_graph/
    (open, above_gears, 15 frame captures) for collecting rotor images."""
    import argparse

    global STEPS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", type=int, default=0, metavar="N",
                    help="build capture_graph/ with N frame captures instead")
    ap.add_argument("--sun", action="store_true",
                    help="build sun_graph/: only the sun gear stage (rotor already seated)")
    ap.add_argument("--multiview", action="store_true",
                    help="build multiview_graph/: look at the rotor from 5 poses and save each frame")
    args = ap.parse_args(argv)
    here = Path(__file__).resolve().parent
    out = here / "graph"
    if args.capture > 0:
        STEPS = capture_steps(args.capture)
        out = here / "capture_graph"
    elif args.sun:
        STEPS = sun_steps()
        out = here / "sun_graph"
    elif args.multiview:
        global _MULTIVIEW
        _MULTIVIEW = True
        STEPS = multiview_steps()
        out = here / "multiview_graph"
    out.mkdir(exist_ok=True)
    # script nodes load from <graph>/scripts: copy the sources there before validating
    shutil.copytree(here / "scripts", out / "scripts", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    build().save(out / "workflow.json")  # validated here, with the runtime's own validator
    print(f"built and validated {out}")
    for i, step in enumerate(STEPS):
        table = {"move": POSES, "pose": EEF_POSES, "insert": EEF_POSES,
                 "stiffness": STIFFNESS}.get(step[0], {})
        detail = table.get(step[1], "") if len(step) > 1 else ""
        if step[0] in ("pose", "insert") and step[1] in GEARBOX_COLUMN and ("vision_gearbox",) in STEPS[:i]:
            detail = f"x/y from the camera (vision_gearbox), z {EEF_POSES[step[1]]['xyz'][2]}"
        print(f"  {i:02d} {' '.join(map(str, step))}  {detail}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
