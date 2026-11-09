"""``gap run --step``: pause before each node, Enter / c / q."""

from __future__ import annotations

import json
from pathlib import Path

from gap_core.tools import ToolRegistry

import gap
from gap.cli.run import _StepPrompter

WORKFLOW = {
    "version": 3,
    "meta": {},
    "nodes": {
        "first": {"type": "tool", "tool": "stub.const", "inputs": {"value": 21}},
        "second": {"type": "tool", "tool": "stub.double", "inputs": {"x": {"$ref": "first.value"}}},
        "third": {"type": "tool", "tool": "stub.double", "inputs": {"x": {"$ref": "second.value"}}},
        "done": {"type": "end", "status": "success"},
    },
    "edges": [["START", "first"], ["first", "second"], ["second", "third"], ["third", "done"]],
}


def _run(tmp_path: Path, answers: list[str]):
    (tmp_path / "workflow.json").write_text(json.dumps(WORKFLOW))
    ran: list[str] = []
    reg = ToolRegistry()
    reg.register_callable(
        "stub.const", lambda value: ran.append("const") or {"value": value}, summary="c"
    )
    reg.register_callable(
        "stub.double", lambda x: ran.append(f"double {x}") or {"value": 2 * x}, summary="d"
    )
    prompts: list[str] = []
    printed: list[str] = []
    replies = iter(answers)

    def fake_input(prompt: str) -> str:
        prompts.append(f"{len(ran)} ran")  # how many tools had run when we were asked
        return next(replies)

    from gap.runtime.executor import WorkflowExecutor

    ex = WorkflowExecutor(
        tmp_path, tool_registry=reg, before_node=_StepPrompter(fake_input, printed.append)
    )
    try:
        ex.execute()
        error = None
    except Exception as exc:  # noqa: BLE001
        error = exc
    return ex, ran, prompts, printed, error


def test_pauses_before_every_node_with_resolved_inputs(tmp_path):
    ex, ran, prompts, printed, error = _run(tmp_path, ["", "", ""])
    assert error is None and ex.exit_status == "success"
    assert prompts == ["0 ran", "1 ran", "2 ran"], "asked before each node, not after"
    assert ran == ["const", "double 21", "double 42"]
    assert any('"x": 21' in line for line in printed), "shows the resolved $ref input"


def test_c_runs_the_rest_without_pausing(tmp_path):
    ex, ran, prompts, _, _ = _run(tmp_path, ["", "c"])
    assert ex.exit_status == "success"
    assert len(prompts) == 2 and ran == ["const", "double 21", "double 42"]


def test_q_aborts_before_the_node_runs(tmp_path):
    _, ran, _, _, error = _run(tmp_path, ["", "q"])
    assert ran == ["const"], "the node at the q prompt must not run"
    assert error is not None and "aborted by user before second" in str(error)


def test_q_through_gap_execute_is_a_clean_failure(tmp_path):
    """gap.execute() (what `gap run` calls) turns the abort into a failed result, no crash."""
    wf = {
        "version": 3,
        "meta": {},
        "nodes": {
            "first": {"type": "tool", "tool": "skills.list", "inputs": {}},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "first"], ["first", "done"]],
    }
    (tmp_path / "workflow.json").write_text(json.dumps(wf))
    result = gap.execute(
        tmp_path, None, before_node=_StepPrompter(lambda _p: "q", lambda _s: None)
    )
    assert not result.success
    assert "aborted by user before first" in str(result.error)
