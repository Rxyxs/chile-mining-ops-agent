"""Tests for the tool-use agent loop, using a fake OpenAI client so no real
API call (and no OPENAI_API_KEY) is required."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agent import MiningOpsAgent

SCHEMAS = [
    {
        "type": "function",
        "function": {"name": "get_flotation_summary", "description": "x", "parameters": {}},
    }
]


def _tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> SimpleNamespace:
    """Mirrors the wire shape: `arguments` is a JSON *string*, not a dict."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=raw),
    )


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _sent_messages(client: MagicMock, call_index: int) -> list[dict]:
    """The request payload for one call. `_run_loop` builds a fresh
    `[system, *messages]` list per request, so what MagicMock recorded is a
    real snapshot of that call rather than a reference to a list still being
    appended to."""
    return client.chat.completions.create.call_args_list[call_index].kwargs["messages"]


def test_agent_calls_tool_and_returns_final_text():
    fake_tool = MagicMock(return_value={"n_batches": 5, "avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("get_flotation_summary", {"month": "2025-09"}, "call_abc")]),
        _response(content="Recovery averaged 88.1%."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=SCHEMAS)
    answer = agent.run("What was the flotation summary for September?")

    assert answer == "Recovery averaged 88.1%."
    fake_tool.assert_called_once_with(month="2025-09")
    assert client.chat.completions.create.call_count == 2

    # Verify the tool message sent back on the second call.
    messages = _sent_messages(client, 1)
    assert messages[0]["role"] == "system"
    assistant_turn = messages[-2]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["tool_calls"][0]["id"] == "call_abc"
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "get_flotation_summary"

    tool_message = messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_abc"
    assert "avg_recovery_pct" in tool_message["content"]
    assert "error" not in json.loads(tool_message["content"])


def test_agent_reports_tool_error_without_crashing():
    def failing_tool(**kwargs):
        raise RuntimeError("warehouse offline")

    registry = {"get_flotation_summary": failing_tool}

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("get_flotation_summary", {"month": "2025-09"}, "call_err")]),
        _response(content="The warehouse tool failed."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=SCHEMAS)
    answer = agent.run("Give me the flotation summary.")

    assert answer == "The warehouse tool failed."
    tool_message = _sent_messages(client, 1)[-1]
    assert "warehouse offline" in json.loads(tool_message["content"])["error"]


def test_agent_reports_malformed_tool_arguments():
    """Arguments arrive as a JSON string the model can get wrong -- that must
    come back as a tool error, not an exception out of the loop."""
    fake_tool = MagicMock(return_value={"ok": True})
    registry = {"get_flotation_summary": fake_tool}

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("get_flotation_summary", "{not valid json", "call_bad")]),
        _response(content="I could not read those arguments."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=SCHEMAS)
    answer = agent.run("Give me the flotation summary.")

    assert answer == "I could not read those arguments."
    fake_tool.assert_not_called()
    tool_message = _sent_messages(client, 1)[-1]
    assert "not valid JSON" in json.loads(tool_message["content"])["error"]


def test_agent_handles_unknown_tool_name_gracefully():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("nonexistent_tool", {}, "call_x")]),
        _response(content="I could not find that tool."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry={}, tool_schemas=[])
    answer = agent.run("Do something unsupported.")

    assert answer == "I could not find that tool."
    tool_message = _sent_messages(client, 1)[-1]
    assert "Unknown tool" in json.loads(tool_message["content"])["error"]


def test_agent_returns_text_directly_when_no_tool_use():
    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="Hello, how can I help?")

    agent = MiningOpsAgent(client=client, tool_registry={}, tool_schemas=[])
    answer = agent.run("hi")

    assert answer == "Hello, how can I help?"
    assert client.chat.completions.create.call_count == 1


def test_chat_remembers_context_across_turns():
    fake_tool = MagicMock(return_value={"n_batches": 5, "avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("get_flotation_summary", {"month": "2025-09"})]),
        _response(content="September recovery averaged 88.1%."),
        _response(content="August was not asked about; only September data was retrieved."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=SCHEMAS)
    first = agent.chat("What was the flotation summary for September?")
    second = agent.chat("And what about August?")

    assert first == "September recovery averaged 88.1%."
    assert second == "August was not asked about; only September data was retrieved."
    assert client.chat.completions.create.call_count == 3

    # The third call (second turn's first request) must carry the full prior
    # exchange -- the follow-up question alone would lose the September context.
    third = _sent_messages(client, 2)
    # system, user Q1, assistant(tool_calls), tool result, assistant(text), user Q2
    assert len(third) == 6
    assert [m["role"] for m in third] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert third[1]["content"] == "What was the flotation summary for September?"
    assert third[-1]["content"] == "And what about August?"


def test_chat_reset_clears_history():
    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="hi")

    agent = MiningOpsAgent(client=client, tool_registry={}, tool_schemas=[])
    agent.chat("first message")
    assert len(agent.history) == 2

    agent.reset()
    assert agent.history == []

    agent.chat("fresh start")
    # After reset, the request carries only the system prompt and the new
    # message, not the old one.
    last = _sent_messages(client, -1)
    assert len(last) == 2
    assert last[0]["role"] == "system"
    assert last[1]["content"] == "fresh start"


def test_run_does_not_touch_history():
    fake_tool = MagicMock(return_value={"avg_recovery_pct": 88.1})
    registry = {"get_flotation_summary": fake_tool}

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("get_flotation_summary", {"month": "2025-09"})]),
        _response(content="88.1%."),
    ]

    agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=SCHEMAS)
    agent.run("What was the flotation summary for September?")

    assert agent.history == []


def test_agent_stops_after_max_iterations():
    client = MagicMock()
    # Always asks for a tool -- would loop forever without the bound.
    client.chat.completions.create.return_value = _response(
        tool_calls=[_tool_call("noop_tool", {}, "call_loop")]
    )
    registry = {"noop_tool": MagicMock(return_value={"ok": True})}

    agent = MiningOpsAgent(
        client=client, tool_registry=registry, tool_schemas=[], max_iterations=3
    )
    answer = agent.run("loop forever")

    assert "maximum number of tool-use steps" in answer
    assert client.chat.completions.create.call_count == 3
