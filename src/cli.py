"""Command-line entry point for the mining ops agent.

    python -m src.cli "como estuvo la silice en junio de 2017?"   -- one-shot, stateless
    python -m src.cli                                              -- interactive multi-turn chat

Talks to the OpenAI API with OPENAI_API_KEY, or, if LOCAL_LLM_BASE_URL is set, to a
local OpenAI-compatible server (llama-cpp-python serving Qwen2.5) with no key; local
replies go through `src/local_client.ToolCallParsingClient`. The tools themselves work
without any model (see src/tools/), and the agent loop is unit-tested with a mocked
client (see tests/test_agent.py).
"""
from __future__ import annotations

import os
import sys
from typing import Any

from src.agent import MiningOpsAgent
from src.tools import TOOL_REGISTRY, TOOL_SCHEMAS


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    local_url = os.environ.get("LOCAL_LLM_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not local_url:
        print(
            "OPENAI_API_KEY is not set. Set it in your environment to run "
            "the agent against the real OpenAI API, e.g.:\n"
            "  PowerShell:  $env:OPENAI_API_KEY = 'sk-...'\n"
            "  bash:        export OPENAI_API_KEY='sk-...'\n"
            "Or set LOCAL_LLM_BASE_URL (e.g. http://localhost:8011/v1) to use a local model.\n"
            "The tools themselves (src/tools/) work without a key and are "
            "covered by the test suite (`pytest`).",
            file=sys.stderr,
        )
        return 1

    try:
        import openai
    except ImportError:
        print(
            "The `openai` package is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    if local_url:
        from src.local_client import ToolCallParsingClient

        client = ToolCallParsingClient(openai.OpenAI(api_key="not-needed", base_url=local_url, timeout=600))
        agent = MiningOpsAgent(client=client, tool_registry=TOOL_REGISTRY, tool_schemas=TOOL_SCHEMAS,
                               model=os.environ.get("LOCAL_LLM_MODEL", "local"))
    else:
        client = openai.OpenAI(api_key=api_key)
        agent = MiningOpsAgent(client=client, tool_registry=TOOL_REGISTRY, tool_schemas=TOOL_SCHEMAS)

    if argv:
        question = " ".join(argv)
        try:
            answer = agent.run(question)
        except Exception as exc:  # noqa: BLE001
            print(f"Agent run failed: {exc}", file=sys.stderr)
            return 1
        print(answer)
        return 0

    return _run_repl(agent)


def _run_repl(agent: Any, input_fn: Any = input, output_fn: Any = print) -> int:
    """Interactive multi-turn chat loop: each question is answered with
    `agent.chat`, so follow-up questions can refer back to earlier answers.
    `input_fn`/`output_fn` are injected so this can be exercised in tests
    without a real terminal."""
    output_fn("Chile Mining Ops Agent -- interactive mode.")
    output_fn("Type 'reset' to clear conversation memory, 'exit' to quit.\n")

    while True:
        try:
            question = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", "salir"}:
            return 0
        if question.lower() == "reset":
            agent.reset()
            output_fn("(conversation history cleared)")
            continue

        try:
            answer = agent.chat(question)
        except Exception as exc:  # noqa: BLE001
            output_fn(f"Agent run failed: {exc}")
            continue
        output_fn(answer)
        output_fn("")


if __name__ == "__main__":
    raise SystemExit(main())
