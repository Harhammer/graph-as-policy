#!/usr/bin/env python
"""Multi-view rotor check: does the rotor's detected centre agree between look poses?

    .venv/bin/python z3_graphs/poses_demo/build.py --multiview
    .venv/bin/gap run z3_graphs/poses_demo/multiview_graph --real franka --no-rr-autostart
    .venv/bin/python z3_graphs/poses_demo/multiview_check.py ~/z3_rotor_images/multiview/*.npz

The rotor does not move between the views, so a correct camera pose puts its centre at
the same base-frame point from every view. Each frame is evaluated twice: with the camera
pose the bridge sent (the nominal robot model + its hand-eye result) and with a corrected
model (joint zero offsets + the jointly fitted camera transform, from a
`z3-gap-handeye --fit-joints` result directory, default ~/z3_handeye/combined_kinfit).
The spread of the centres under each model is that model's error in this workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "bridges" / "z3_franka" / "src"))
import build  # noqa: E402
from rotor_vision import RotorNotFound, detect_rotor  # noqa: E402
from z3_gap_bridge.panda_fk import flange_pose  # noqa: E402


def load_model(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads((directory / "report.json").read_text())
    dq = np.radians(np.asarray(report["joint_offsets_deg"], float))
    X = np.load(directory / "flange_T_camera.npy")
    return dq, X


def main(argv: list[str]) -> int:
    model_dir = Path("~/z3_handeye/combined_kinfit").expanduser()
    paths = []
    for a in argv:
        if a.startswith("--model="):
            model_dir = Path(a.split("=", 1)[1]).expanduser()
        else:
            paths.append(Path(a))
    paths = [p for p in paths if p.suffix == ".npz"]
    if len(paths) < 2:
        raise SystemExit("need at least two frames")
    dq, X = load_model(model_dir)
    taught = np.asarray(build.EEF_POSES[build.VISION["nominal"]]["xyz"][:2], float)
    z_guess = build.VISION["z_top_guess"]
    rows = []
    for p in paths:
        d = np.load(p)
        q = d["joints"][:7]
        T_nom = d["T_wc"]
        T_cor = flange_pose(q + dq) @ X
        row = {"name": p.stem}
        for tag, T in (("nominal", T_nom), ("corrected", T_cor)):
            try:
                det = detect_rotor(d["rgb"], d["depth"], d["K"], T, tuple(taught), z_guess)
                row[tag] = (det.center_xy.copy(), det.z_top)
            except RotorNotFound as exc:
                row[tag] = None
                print(f"{p.name} [{tag}]: NOT FOUND {exc}")
        rows.append(row)
    print(f"model: {model_dir}  offsets {np.round(np.degrees(dq), 2).tolist()} deg\n")
    for tag in ("nominal", "corrected"):
        found = [(r["name"], *r[tag]) for r in rows if r.get(tag) is not None]
        if len(found) < 2:
            continue
        c = np.array([f[1] for f in found])
        z = np.array([f[2] for f in found])
        mean = c.mean(axis=0)
        print(f"{tag}:")
        for name, cxy, zt in found:
            d = (cxy - mean) * 1e3
            print(f"  {name:28s} centre ({cxy[0]:.4f}, {cxy[1]:.4f}) z_top {zt:.4f}   "
                  f"vs mean ({d[0]:+.1f}, {d[1]:+.1f}) mm")
        spread = np.hypot(*(c - mean).T) * 1e3
        taught_note = (f";  mean centre is {np.hypot(*(mean - taught)) * 1e3:.1f} mm from the taught grasp"
                       if tag == "nominal" else "  (the taught pose moves too in this model: no comparison)")
        print(f"  -> xy spread {np.sqrt(np.mean(spread**2)):.2f} mm rms / {spread.max():.2f} max,"
              f" z_top spread {z.std() * 1e3:.2f} mm{taught_note}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
