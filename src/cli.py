"""Command-line entry point for the mining ops agent.

    python -m src.cli "que tan riesgoso es un cliente con..."

Requires a real ANTHROPIC_API_KEY in the environment -- this entry point
talks to the live Anthropic API. The tools themselves work without any API
key (see src/tools/), and the agent loop is unit-tested with a mocked
client (see tests/test_agent.py).
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python -m src.cli "your question here"', file=sys.stderr)
        return 2

    question = " ".join(argv)

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

    from src.agent import MiningOpsAgent
    from src.tools import TOOL_REGISTRY, TOOL_SCHEMAS

    client = anthropic.Anthropic(api_key=api_key)
    agent = MiningOpsAgent(client=client, tool_registry=TOOL_REGISTRY, tool_schemas=TOOL_SCHEMAS)

    try:
        answer = agent.run(question)
    except Exception as exc:  # noqa: BLE001
        print(f"Agent run failed: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
