#!/usr/bin/env python
"""Run the rotor detector on saved frames (capture / detect_rotor .npz) — no robot.

    .venv/bin/python z3_graphs/poses_demo/tune_rotor.py ~/z3_rotor_images/*.npz

Prints one line per frame and writes <frame>_tuned.png (the annotated camera image) and
<frame>_steps.png (every detection step: top view + outline fit, depth heights, brightness
along the bolt circle, hole centroids -> centre, screw-or-empty per hole).
Uses build.py's VISION settings (nominal x/y, top height guess, jaw axis, correction).
Centres are printed in the robot frame, i.e. with VISION["correction_xy"] applied.

    tune_rotor.py --reference <frames>

measures the correction instead: for frames where the rotor sits exactly where the taught
grasp (VISION["nominal"]) holds it centred, it prints taught - median(camera centre).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
import build  # noqa: E402
from rotor_vision import RotorNotFound, detect_rotor, draw_detection, draw_steps  # noqa: E402


def main(paths: list[str]) -> int:
    reference = "--reference" in paths
    paths = [p for p in paths if p != "--reference"]
    taught = np.asarray(build.EEF_POSES[build.VISION["nominal"]]["xyz"][:2], float)
    corr = np.asarray(build.VISION.get("correction_xy") or [0.0, 0.0], float)
    nominal = tuple(taught - corr)  # where the camera sees the taught position
    jaw = float(build.VISION["jaw_axis_deg"] or 0.0)
    found = 0
    raw_centres = []
    for p in map(Path, paths):
        if p.stem.endswith(("_tuned", "_detection", "_steps", "_popup")):
            continue
        d = np.load(p)
        depth = d["depth"] if d["depth"].size else None
        trace: dict = {}
        try:
            det = detect_rotor(d["rgb"], depth, d["K"], d["T_wc"], nominal,
                               build.VISION["z_top_guess"], trace=trace)
        except RotorNotFound as exc:
            print(f"{p.name}: NOT FOUND  {exc}")
            img = draw_detection(d["rgb"], d["K"], d["T_wc"], None, note=f"NOT FOUND: {exc}")
            steps = draw_steps(d["rgb"], d["K"], d["T_wc"], None, trace, jaw, error=str(exc))
            cv2.imwrite(str(p.with_name(p.stem + "_steps.png")),
                        cv2.cvtColor(steps, cv2.COLOR_RGB2BGR))
        else:
            found += 1
            raw_centres.append(det.center_xy)
            c = det.center_xy + corr
            off = np.hypot(*(c - taught)) * 1e3
            print(
                f"{p.name}: centre ({c[0]:.4f}, {c[1]:.4f}) "
                f"{off:4.1f} mm from nominal  z_top {det.z_top:.4f}  r {det.radius_m * 1e3:.2f} mm  "
                f"contrast {det.hole_contrast:.2f}  screws {'even' if det.screws_even else 'odd '} "
                f"score {det.screw_score:+5.1f} (depth {det.cue_scores['depth']:+.1f}, "
                f"gray {det.cue_scores['gray']:+.1f})  yaw {det.grasp_yaw_deg(jaw):+5.1f}"
                + (f"  WARN {'; '.join(det.warnings)}" if det.warnings else "")
            )
            img = draw_detection(d["rgb"], d["K"], d["T_wc"], det, jaw)
            steps = draw_steps(d["rgb"], d["K"], d["T_wc"], det, trace, jaw)
            cv2.imwrite(str(p.with_name(p.stem + "_steps.png")),
                        cv2.cvtColor(steps, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(p.with_name(p.stem + "_tuned.png")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    frames = [p for p in paths if not Path(p).stem.endswith(("_tuned", "_detection", "_steps",
                                                             "_popup"))]
    print(f"{found}/{len(frames)} found")
    if reference and raw_centres:
        raw = np.asarray(raw_centres)
        med = np.median(raw, axis=0)
        spread = np.max(np.hypot(*(raw - med).T)) * 1e3
        new = taught - med
        print(f"\nreference: camera centre median ({med[0]:.5f}, {med[1]:.5f}), spread "
              f"{spread:.2f} mm over {len(raw)} frames")
        print(f'VISION["correction_xy"] = [{new[0]:.4f}, {new[1]:.4f}]   '
              f"(now [{corr[0]:.4f}, {corr[1]:.4f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
