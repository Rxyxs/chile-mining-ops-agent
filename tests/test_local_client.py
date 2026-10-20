import json
from types import SimpleNamespace

from src.eval_harness import check_arguments, check_groundedness, wilson_ci
from src.eval_queries import EVAL_QUERIES
from src.local_client import ToolCallParsingClient, parse_tool_calls


def test_parse_tool_calls_reads_qwen_blocks_including_doubled_braces():
    content = '<tool_call>\n{{"name": "get_flotation_summary", "arguments": {"month": "2017-06"}}}\n</tool_call>'
    text, calls = parse_tool_calls(content)
    assert text is None
    assert len(calls) == 1
    assert calls[0].function.name == "get_flotation_summary"
    assert json.loads(calls[0].function.arguments) == {"month": "2017-06"}


def test_parse_tool_calls_reads_several_blocks_and_skips_broken_ones():
    content = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
               '<tool_call>not json</tool_call>'
               '<tool_call>{"name": "b", "arguments": {"days": 7}}</tool_call>')
    _, calls = parse_tool_calls(content)
    assert [c.function.name for c in calls] == ["a", "b"]


def test_plain_answer_passes_through_unchanged():
    assert parse_tool_calls("Lima.") == ("Lima.", [])


def test_client_wrapper_fills_tool_calls():
    raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='<tool_call>{"name": "get_maintenance_alerts", "arguments": {"days": 7}}</tool_call>', tool_calls=None))])
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: raw)))
    message = ToolCallParsingClient(inner).chat.completions.create(model="x").choices[0].message
    assert message.tool_calls[0].function.name == "get_maintenance_alerts"


def test_check_arguments_requires_expected_values():
    calls = [{"tool": "get_procurement_summary", "input": {"status": "delivered", "category": "mining_machinery"}}]
    assert check_arguments({"get_procurement_summary": {"status": "delivered"}}, calls)
    assert not check_arguments({"get_procurement_summary": {"status": "pending"}}, calls)
    assert not check_arguments({"get_flotation_summary": {"month": "2017-06"}}, calls)


def test_groundedness_accepts_numbers_from_the_question():
    calls = [{"tool": "get_maintenance_alerts", "input": {"days": 30}, "result": {"n_events": 0}}]
    assert check_groundedness("En los últimos 30 días hubo 0 eventos.", calls, "¿Alertas en 30 días?")["is_grounded"]


def test_eval_set_has_100_distinct_queries_with_expected_args():
    assert len(EVAL_QUERIES) == 100
    for q in EVAL_QUERIES:
        assert set(q.expected_args) <= set(q.expected_tools)


def test_wilson_ci_brackets_the_rate():
    lo, hi = wilson_ci(80, 100)
    assert lo < 0.8 < hi


def test_parse_tool_calls_tolerates_unbalanced_closing_braces():
    content = '<tool_call> {{"name": "get_procurement_summary", "arguments": {"status": "pending"}} }} </tool_call>'
    _, calls = parse_tool_calls(content)
    assert json.loads(calls[0].function.arguments) == {"status": "pending"}
