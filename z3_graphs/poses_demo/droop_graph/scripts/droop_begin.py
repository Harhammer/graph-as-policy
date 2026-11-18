"""Droop test: create the session file the measure_droop nodes append to."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from gap import NodeContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from droop_report import begin  # noqa: E402


class Output(TypedDict):
    session: str


def run(ctx: NodeContext, note: str = "", out_dir: str = "~/z3_droop") -> Output:
    path = begin(note, out_dir)
    print(f"[droop_begin] session {path}")
    return {"session": path}
