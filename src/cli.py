"""Command-line entry point for the mining ops agent.

    python -m src.cli "que tan riesgoso es un cliente con..."   -- one-shot, stateless
    python -m src.cli                                            -- interactive multi-turn chat

Requires a real ANTHROPIC_API_KEY in the environment -- this entry point
talks to the live Anthropic API. The tools themselves work without any API
key (see src/tools/), and the agent loop is unit-tested with a mocked
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

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set. Set it in your environment to run "
            "the agent against the real Anthropic API, e.g.:\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "  bash:        export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "The tools themselves (src/tools/) work without a key and are "
            "covered by the test suite (`pytest`).",
            file=sys.stderr,
        )
        return 1

    try:
        import anthropic
    except ImportError:
        print(
            "The `anthropic` package is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    client = anthropic.Anthropic(api_key=api_key)
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
