"""Tool-use agent loop for mining operations / risk queries.

MiningOpsAgent takes an injected Anthropic client so it can be exercised in
tests with a mock/fake client that never calls the real API. The real CLI
entry point (src/cli.py) injects a genuine `anthropic.Anthropic()` client.
"""
from __future__ import annotations

import json
from typing import Any, Callable

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_ITERATIONS = 5

DEFAULT_SYSTEM_PROMPT = (
    "You are an operations and risk assistant for a Chilean mining company. "
    "Answer questions by calling the tools available to you -- never invent "
    "numbers that a tool could return. If a tool call fails, explain the "
    "failure to the user in plain language."
)


class MiningOpsAgent:
    """Runs a bounded tool-use loop against an Anthropic-compatible client."""

    def __init__(
        self,
        client: Any,
        tool_registry: dict[str, Callable[..., Any]],
        tool_schemas: list[dict],
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry
        self.tool_schemas = tool_schemas
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def _execute_tool(self, name: str, tool_input: dict) -> dict:
        """Executes a registered tool function. Never raises -- exceptions are
        turned into an error payload so the model can see and react to them."""
        func = self.tool_registry.get(name)
        if func is None:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return func(**tool_input)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: never crash the loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    def run(self, user_message: str) -> str:
        """Runs the tool-use loop for a single user message and returns the
        model's final text response."""
        messages: list[dict] = [{"role": "user", "content": user_message}]

        for _ in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.tool_schemas,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                return _extract_text(response)

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result = self._execute_tool(block.name, block.input or {})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                        "is_error": isinstance(result, dict) and "error" in result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return (
            "I reached the maximum number of tool-use steps "
            f"({self.max_iterations}) without a final answer."
        )


def _extract_text(response: Any) -> str:
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts).strip()
