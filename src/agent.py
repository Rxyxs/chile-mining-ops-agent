"""Tool-use agent loop for mining operations / risk queries.

MiningOpsAgent takes an injected OpenAI-compatible client so it can be
exercised in tests with a mock/fake client that never calls the real API. The
real CLI entry point (src/cli.py) injects a genuine `openai.OpenAI()` client.
"""
from __future__ import annotations

import json
from typing import Any, Callable

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_ITERATIONS = 5

DEFAULT_SYSTEM_PROMPT = (
    "You are an operations and risk assistant for a Chilean mining company. "
    "Answer questions by calling the tools available to you -- never invent "
    "numbers that a tool could return. If a tool call fails, explain the "
    "failure to the user in plain language."
)


class MiningOpsAgent:
    """Runs a bounded tool-use loop against an OpenAI-compatible client."""

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
        self.history: list[dict] = []

    def reset(self) -> None:
        """Clears accumulated multi-turn conversation history. Does not affect
        `run`, which never touches `self.history` in the first place."""
        self.history = []

    def chat(self, user_message: str) -> str:
        """Runs one turn of an ongoing multi-turn conversation: unlike `run`,
        this appends to and reads from `self.history`, so a follow-up question
        can refer back to a prior answer or tool result (e.g. "and what about
        August?") without the caller having to resend that context by hand."""
        self.history.append({"role": "user", "content": user_message})
        answer = self._run_loop(self.history)
        self.history.append({"role": "assistant", "content": answer})
        return answer

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
        """Runs the tool-use loop for a single, stateless message and returns
        the model's final text response -- no memory of prior calls, and
        `self.history` is left untouched. See `chat` for a multi-turn
        conversation that remembers context across calls."""
        messages: list[dict] = [{"role": "user", "content": user_message}]
        return self._run_loop(messages)

    def _run_loop(self, messages: list[dict]) -> str:
        """Shared tool-use loop: mutates `messages` in place as tool calls
        happen (so a caller holding a reference, like `chat`'s `self.history`,
        sees the intermediate assistant/tool turns too) and returns the
        model's final text answer.

        The system prompt is prepended per request rather than stored in
        `messages`, so `self.history` stays a pure record of the conversation
        and a prompt change takes effect on the next turn of an existing
        conversation instead of only on a fresh one."""
        for _ in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "system", "content": self.system_prompt}, *messages],
                tools=self.tool_schemas,
            )

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return (message.content or "").strip()

            # Replayed as a plain dict rather than the SDK's own message object:
            # the next request has to carry this turn's tool_calls back verbatim,
            # and a dict is what actually goes over the wire.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                name = call.function.name
                tool_input, parse_error = _parse_arguments(call.function.arguments)
                result = (
                    {"error": parse_error}
                    if parse_error is not None
                    else self._execute_tool(name, tool_input)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result, default=str),
                    }
                )

        return (
            "I reached the maximum number of tool-use steps "
            f"({self.max_iterations}) without a final answer."
        )


def _parse_arguments(raw: Any) -> tuple[dict, str | None]:
    """Tool-call arguments arrive as a JSON *string*, which the model is free
    to get wrong. Returns `(arguments, error)` instead of raising, so a
    malformed payload is fed back to the model as a tool error on the same
    never-crash path as a tool that itself blew up (see `_execute_tool`)."""
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}, "Tool arguments were not valid JSON."
    if not isinstance(parsed, dict):
        return {}, "Tool arguments must be a JSON object."
    return parsed, None
