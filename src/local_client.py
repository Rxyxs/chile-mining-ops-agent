"""Adapter for local models whose server does not parse tool calls.

llama-cpp-python's OpenAI-compatible server renders the tools into Qwen2.5's own chat
template, and the model answers with `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
blocks, but the server returns them as plain `content` with `tool_calls=None`. (Its
`chatml-function-calling` format does fill `tool_calls`, but in testing the model then
never called a tool at all.) This wrapper parses those blocks into the same
`message.tool_calls` shape the OpenAI SDK returns, so `MiningOpsAgent` runs unchanged.
"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def _parse_block(block: str) -> dict | None:
    text = block.strip()
    # The template escapes braces, so the model often writes {{...}} and sometimes
    # leaves extra closing braces: drop the doubled opening brace and read the first
    # complete JSON object, ignoring whatever trails it.
    while text.startswith("{{"):
        text = text[1:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    return payload


def parse_tool_calls(content: str | None) -> tuple[str | None, list[SimpleNamespace]]:
    """Returns (remaining text, tool calls) from a raw completion."""
    if not content or "<tool_call>" not in content:
        return content, []
    calls = []
    for block in _TOOL_CALL_RE.findall(content):
        payload = _parse_block(block)
        if payload is None:
            continue
        arguments = payload.get("arguments", {})
        calls.append(SimpleNamespace(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=SimpleNamespace(
                name=payload["name"],
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
            ),
        ))
    remaining = _TOOL_CALL_RE.sub("", content).strip() or None
    return remaining, calls


class _Completions:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            return response
        text, calls = parse_tool_calls(message.content)
        if not calls:
            return response
        fixed = SimpleNamespace(role="assistant", content=text, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=fixed, finish_reason="tool_calls")])


class ToolCallParsingClient:
    """Wraps an OpenAI-compatible client; only `chat.completions.create` changes."""

    def __init__(self, client: Any) -> None:
        self.chat = SimpleNamespace(completions=_Completions(client.chat.completions))
