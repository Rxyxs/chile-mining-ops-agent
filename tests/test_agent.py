"""Tests for the tool-use agent loop, using a fake Anthropic client so no
real API call (and no ANTHROPIC_API_KEY) is required."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agent import MiningOpsAgent


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_input: dict, tool_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)


def _response(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def test_agent_calls_tool_and_returns_final_text():
    fake_tool = MagicMock(return_value={"n_batches": 5, "avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}
    schemas = [{"name": "get_flotation_summary", "description": "x", "input_schema": {}}]

    client = MagicMock()
    tool_use_response = _response(
        "tool_use",
        [_tool_use_block("get_flotation_summary", {"month": "2025-09"}, "toolu_abc")],
    )
    final_response = _response("end_turn", [_text_block("Recovery averaged 88.1%.")])
    client.messages.create.side_effect = [tool_use_response, final_response]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=schemas)
    answer = agent.run("What was the flotation summary for September?")

    assert answer == "Recovery averaged 88.1%."
    fake_tool.assert_called_once_with(month="2025-09")
    assert client.messages.create.call_count == 2

    # Verify the tool_result message sent back on the second call.
    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    tool_result_message = messages[-1]
    assert tool_result_message["role"] == "user"
    tool_result_block = tool_result_message["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "toolu_abc"
    assert tool_result_block["is_error"] is False
    assert "avg_recovery_pct" in tool_result_block["content"]


def test_agent_reports_tool_error_without_crashing():
    def failing_tool(**kwargs):
        raise RuntimeError("warehouse offline")

    registry = {"get_flotation_summary": failing_tool}
    schemas = [{"name": "get_flotation_summary", "description": "x", "input_schema": {}}]

    client = MagicMock()
    tool_use_response = _response(
        "tool_use",
        [_tool_use_block("get_flotation_summary", {"month": "2025-09"}, "toolu_err")],
    )
    final_response = _response("end_turn", [_text_block("The warehouse tool failed.")])
    client.messages.create.side_effect = [tool_use_response, final_response]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=schemas)
    answer = agent.run("Give me the flotation summary.")

    assert answer == "The warehouse tool failed."
    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    tool_result_block = second_call_kwargs["messages"][-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "warehouse offline" in tool_result_block["content"]


def test_agent_handles_unknown_tool_name_gracefully():
    registry: dict = {}
    schemas: list = []

    client = MagicMock()
    tool_use_response = _response(
        "tool_use", [_tool_use_block("nonexistent_tool", {}, "toolu_x")]
    )
    final_response = _response("end_turn", [_text_block("I could not find that tool.")])
    client.messages.create.side_effect = [tool_use_response, final_response]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=schemas)
    answer = agent.run("Do something unsupported.")

    assert answer == "I could not find that tool."


def test_agent_returns_text_directly_when_no_tool_use():
    client = MagicMock()
    client.messages.create.return_value = _response(
        "end_turn", [_text_block("Hello, how can I help?")]
    )

    agent = MiningOpsAgent(client=client, tool_registry={}, tool_schemas=[])
    answer = agent.run("hi")

    assert answer == "Hello, how can I help?"
    assert client.messages.create.call_count == 1


def test_chat_remembers_context_across_turns():
    # NOTE: MiningOpsAgent mutates its `messages` list in place (see _run_loop's
    # docstring) so that `chat`'s growing `self.history` reflects tool_use/
    # tool_result turns as they happen. That means MagicMock's `call_args`,
    # which stores a *reference* to that same list rather than a copy, would
    # show its FINAL state (after every later mutation) no matter which past
    # call you inspect it from -- a mocking artifact, not something the real
    # Anthropic SDK exhibits (it serializes the request over HTTP at call
    # time). A custom side_effect that snapshots `list(messages)` on entry
    # sidesteps that and lets this test check what was actually sent.
    fake_tool = MagicMock(return_value={"n_batches": 5, "avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}
    schemas = [{"name": "get_flotation_summary", "description": "x", "input_schema": {}}]

    responses = [
        _response("tool_use", [_tool_use_block("get_flotation_summary", {"month": "2025-09"}, "toolu_1")]),
        _response("end_turn", [_text_block("September recovery averaged 88.1%.")]),
        _response("end_turn", [_text_block("August was not asked about; only September data was retrieved.")]),
    ]
    sent_messages_snapshots: list[list[dict]] = []

    def _create(**kwargs):
        sent_messages_snapshots.append(list(kwargs["messages"]))
        return responses[len(sent_messages_snapshots) - 1]

    client = MagicMock()
    client.messages.create.side_effect = _create

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=schemas)
    first = agent.chat("What was the flotation summary for September?")
    second = agent.chat("And what about August?")

    assert first == "September recovery averaged 88.1%."
    assert second == "August was not asked about; only September data was retrieved."
    assert len(sent_messages_snapshots) == 3

    # The third call (second turn's first request) must include the full prior
    # exchange -- the follow-up question alone would lose the September context.
    third_call_messages = sent_messages_snapshots[2]
    assert third_call_messages[0]["content"] == "What was the flotation summary for September?"
    assert third_call_messages[-1]["content"] == "And what about August?"
    assert len(third_call_messages) == 5  # user, assistant(tool_use), user(tool_result), assistant(text), user(Q2)


def test_chat_reset_clears_history():
    sent_messages_snapshots: list[list[dict]] = []

    def _create(**kwargs):
        sent_messages_snapshots.append(list(kwargs["messages"]))
        return _response("end_turn", [_text_block("hi")])

    client = MagicMock()
    client.messages.create.side_effect = _create

    agent = MiningOpsAgent(client=client, tool_registry={}, tool_schemas=[])
    agent.chat("first message")
    assert len(agent.history) == 2

    agent.reset()
    assert agent.history == []

    agent.chat("fresh start")
    # After reset, the request should carry only the new message, not the old one.
    last_call_messages = sent_messages_snapshots[-1]
    assert len(last_call_messages) == 1
    assert last_call_messages[0]["content"] == "fresh start"


def test_run_does_not_touch_history():
    fake_tool = MagicMock(return_value={"avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}
    schemas = [{"name": "get_flotation_summary", "description": "x", "input_schema": {}}]

    client = MagicMock()
    client.messages.create.side_effect = [
        _response("tool_use", [_tool_use_block("get_flotation_summary", {"month": "2025-09"}, "toolu_1")]),
        _response("end_turn", [_text_block("88.1%.")]),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=schemas)
    agent.run("What was the flotation summary for September?")

    assert agent.history == []


def test_agent_stops_after_max_iterations():
    client = MagicMock()
    # Always asks for a tool -- would loop forever without the bound.
    client.messages.create.return_value = _response(
        "tool_use", [_tool_use_block("noop_tool", {}, "toolu_loop")]
    )
    registry = {"noop_tool": MagicMock(return_value={"ok": True})}

    agent = MiningOpsAgent(
        client=client, tool_registry=registry, tool_schemas=[], max_iterations=3
    )
    answer = agent.run("loop forever")

    assert "maximum number of tool-use steps" in answer
    assert client.messages.create.call_count == 3
