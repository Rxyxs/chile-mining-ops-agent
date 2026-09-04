"""Tests for the eval harness's scoring logic (src/eval_harness.py), using
the same scripted-mock-client pattern as tests/test_agent.py. These prove
extract_numbers/check_groundedness/run_eval are correct -- they do NOT
measure real model behavior, since no real model is involved. See
src/eval_harness.py's module docstring for why that distinction matters.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.eval_harness import EvalQuery, check_groundedness, extract_numbers, run_eval


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_input: dict, tool_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)


def _response(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def test_extract_numbers_finds_plain_and_formatted_numbers():
    numbers = extract_numbers("Recovery was 88.1% on 5 batches, 2,000,000 CLP requested.")
    assert "88.1" in numbers
    assert "5" in numbers
    assert "2,000,000" in numbers


def test_extract_numbers_empty_for_no_digits():
    assert extract_numbers("no numbers here at all") == set()


def test_check_groundedness_flags_number_not_in_any_tool_result():
    tool_calls = [{"tool": "get_flotation_summary", "input": {}, "result": {"avg_recovery_pct": 88.1}}]
    result = check_groundedness("Recovery was 88.1%, up from 75% last month.", tool_calls)

    assert result["is_grounded"] is False
    assert "75" in result["ungrounded_numbers"]
    assert "88.1" not in result["ungrounded_numbers"]


def test_check_groundedness_all_numbers_present_in_tool_results():
    tool_calls = [{"tool": "get_flotation_summary", "input": {}, "result": {"avg_recovery_pct": 88.1, "n_batches": 5}}]
    result = check_groundedness("Across 5 batches, recovery averaged 88.1%.", tool_calls)

    assert result["is_grounded"] is True
    assert result["ungrounded_numbers"] == []


def test_check_groundedness_no_tool_calls_and_no_numbers_is_grounded():
    result = check_groundedness("I don't have a tool for that.", [])
    assert result["is_grounded"] is True


def test_run_eval_scores_correct_tool_selection_and_groundedness():
    eval_queries = (
        EvalQuery("flotation query", ("get_flotation_summary",)),
        EvalQuery("out of scope query", ()),
    )

    # Query 1: correct tool, grounded answer.
    # Query 2: no tool expected and none called, but the model fabricates a number.
    responses = [
        _response("tool_use", [_tool_use_block("get_flotation_summary", {"month": "2025-09"})]),
        _response("end_turn", [_text_block("Recovery averaged 88.1%.")]),
        _response("end_turn", [_text_block("I estimate it's around 42% but I'm not sure.")]),
    ]
    client = MagicMock()
    client.messages.create.side_effect = responses

    fake_flotation_tool = MagicMock(return_value={"avg_recovery_pct": 88.1})
    import src.eval_harness as eh

    original_registry = eh.TOOL_REGISTRY
    eh.TOOL_REGISTRY = {"get_flotation_summary": fake_flotation_tool}
    try:
        summary = run_eval(client, eval_queries)
    finally:
        eh.TOOL_REGISTRY = original_registry

    assert summary["n_queries"] == 2
    assert summary["tool_selection_accuracy"] == 1.0  # both queries got the (non-)tool right
    assert summary["groundedness_rate"] == 0.5  # query 2's "42%" is fabricated

    q1, q2 = summary["results"]
    assert q1["tool_selection_correct"] is True
    assert q1["is_grounded"] is True
    assert q2["tool_selection_correct"] is True
    assert q2["is_grounded"] is False
    assert "42" in q2["ungrounded_numbers"]


def test_run_eval_flags_wrong_tool_selection():
    eval_queries = (EvalQuery("flotation query", ("get_maintenance_alerts",)),)

    responses = [
        _response("tool_use", [_tool_use_block("get_flotation_summary", {"month": "2025-09"})]),
        _response("end_turn", [_text_block("Recovery averaged 88.1%.")]),
    ]
    client = MagicMock()
    client.messages.create.side_effect = responses

    fake_flotation_tool = MagicMock(return_value={"avg_recovery_pct": 88.1})
    import src.eval_harness as eh

    original_registry = eh.TOOL_REGISTRY
    eh.TOOL_REGISTRY = {"get_flotation_summary": fake_flotation_tool}
    try:
        summary = run_eval(client, eval_queries)
    finally:
        eh.TOOL_REGISTRY = original_registry

    assert summary["tool_selection_accuracy"] == 0.0
    assert summary["results"][0]["actual_tools"] == ["get_flotation_summary"]
    assert summary["results"][0]["expected_tools"] == ["get_maintenance_alerts"]
