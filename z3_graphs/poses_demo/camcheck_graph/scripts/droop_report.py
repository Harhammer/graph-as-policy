"""Droop test: start a session file, and summarise one (or compare two).

In the graph: ``begin`` creates ~/z3_droop/<timestamp>[_note].jsonl, ``run`` (the last node)
prints the table. Offline, to compare e.g. two Desk load settings:

    python z3_graphs/poses_demo/scripts/droop_report.py ~/z3_droop/A.jsonl [~/z3_droop/B.jsonl]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np


class Output(TypedDict):
    session: str


def begin(note: str = "", out_dir: str = "~/z3_droop") -> str:
    d = Path(out_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d_%H%M%S") + (f"_{note}" if note else "")
    path = d / f"{stem}.jsonl"
    path.touch()
    return str(path)


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).expanduser().read_text().splitlines() if line.strip()]


def summarise(recs: list[dict]) -> dict:
    """{(label, phase): {n, droop_mm, tip_mm, deg, dz_mm, torque_nm: (mean, std) ...}}"""
    out: dict = {}
    for key in dict.fromkeys((r["label"], r["phase"]) for r in recs):
        rs = [r for r in recs if (r["label"], r["phase"]) == key]

        def ms(vals):
            a = np.asarray(vals, float)
            return a.mean(axis=0), (a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(a.mean(axis=0)))

        s = {"n": len(rs), "droop_mm": ms([r["droop_mm"] for r in rs]),
             "tip_mm": ms([r["tip_droop_mm"] for r in rs]), "deg": ms([r["droop_deg"] for r in rs]),
             "tip_dp_mm": ms([r["tip_dp_mm"] for r in rs])}
        if all("torque_nm" in r for r in rs):
            s["torque_nm"] = ms([r["torque_nm"] for r in rs])
        out[key] = s
    return out


def table(recs: list[dict], title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'pose':<14}{'phase':<6}{'n':>2}  {'tool mm':>11}  {'tip mm':>11}  {'tilt deg':>12}  "
                 f"{'tip dx/dy/dz mm':>20}  torque j1..j7 [Nm]")
    for (label, phase), s in summarise(recs).items():
        dxyz = "/".join(f"{v:+.2f}" for v in s["tip_dp_mm"][0])
        tq = " ".join(f"{v:+.2f}" for v in s["torque_nm"][0]) if "torque_nm" in s else ""
        lines.append(
            f"{label:<14}{phase:<6}{s['n']:>2}  {s['droop_mm'][0]:5.2f}±{s['droop_mm'][1]:4.2f}  "
            f"{s['tip_mm'][0]:5.2f}±{s['tip_mm'][1]:4.2f}  {s['deg'][0]:6.3f}±{s['deg'][1]:5.3f}  "
            f"{dxyz:>20}  {tq}"
        )
    return "\n".join(lines)


def run(ctx, session: str) -> Output:
    print(table(load(session), f"droop test {session}"))
    return {"session": session}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for p in argv:
        print(table(load(p), str(p)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
