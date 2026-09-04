"""Evaluation harness for the mining-ops agent: measures whether the model
calls the right tool(s) for a query, and whether every number in its final
answer is actually grounded in a real tool result rather than invented.

This exists because "the system prompt says never invent numbers" (see
DEFAULT_SYSTEM_PROMPT in src/agent.py) is an instruction, not a guarantee --
the whole point of routing through tool calls instead of free-text answers
is that it becomes possible to check that guarantee mechanically instead of
just trusting the prompt. tests/test_eval_harness.py exercises the scoring
logic (extract_numbers, check_groundedness, run_eval) against small scripted
mock clients, the same pattern tests/test_agent.py already uses, so it needs
no live API key to prove the harness itself is correct.

This module's own __main__ block, by contrast, needs a real OPENAI_API_KEY
-- it measures actual model behavior, which a mock client can't tell you
anything true about. Run it with:

    OPENAI_API_KEY=sk-... python -m src.eval_harness

No live key was available on the machine this harness was built on, so no
real accuracy numbers are claimed anywhere in this repo's README for it --
only that the harness exists and is ready to run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.agent import MiningOpsAgent
from src.tools import TOOL_REGISTRY, TOOL_SCHEMAS

# Requires a digit on both sides of any internal '.'/',' so a trailing comma
# or period (e.g. right after a number in JSON, or at a sentence's end)
# isn't swallowed into the match -- '88.1,' in `{"x": 88.1, "y": 5}` must
# extract as '88.1', not '88.1,', to compare equal with '88.1' from prose.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_tools: tuple[str, ...]
    note: str = ""


EVAL_QUERIES: tuple[EvalQuery, ...] = (
    EvalQuery(
        "Cual fue el resumen de flotacion de septiembre 2025?",
        ("get_flotation_summary",),
    ),
    EvalQuery(
        "Hay alertas de mantenimiento en los ultimos 7 dias?",
        ("get_maintenance_alerts",),
    ),
    EvalQuery(
        "Dame el resumen de compras entregadas este trimestre.",
        ("get_procurement_summary",),
    ),
    EvalQuery(
        "Un cliente de 34 anos, ingreso mensual 850000 CLP, deuda sobre ingreso 0.4, "
        "11 meses en su empleo actual, 1 pago atrasado, pide 2000000 CLP. Que tan "
        "riesgoso es?",
        ("score_credit_risk",),
    ),
    EvalQuery(
        "Algun equipo muestra un patron de mantenimiento anomalo en los ultimos 60 dias?",
        ("check_maintenance_anomalies",),
    ),
    EvalQuery(
        "Dame el resumen de flotacion de agosto 2025 y dime si hay alertas de "
        "mantenimiento activas.",
        ("get_flotation_summary", "get_maintenance_alerts"),
        note="multi-tool query -- both must fire",
    ),
    EvalQuery(
        "Cual es la capital de Peru?",
        (),
        note="out-of-scope for every tool here -- none should fire",
    ),
)


def extract_numbers(text: str) -> set[str]:
    """Extracts numeric substrings from text for a loose groundedness check.
    Deliberately string-based rather than float-parsing, so formatted numbers
    like '88.1%' or '2,000,000' still match without reproducing every
    locale's number formatting -- this trades precision for never crashing
    on a format it doesn't recognize."""
    return set(_NUMBER_RE.findall(text))


def check_groundedness(answer_text: str, tool_calls: list[dict]) -> dict:
    """Every number in the final answer should also appear somewhere in the
    JSON of some tool call's result -- if not, the model may have invented
    it rather than reading it off a real tool result. This is a heuristic,
    not a proof: a model that echoes a number from the user's own question
    (and that number happens not to also appear in any tool result) would be
    flagged as a false positive here. Most of this repo's tools happen to
    echo their own inputs back in the result (e.g. score_credit_risk's
    `input_profile`), which covers the common case, but it's not guaranteed
    for every tool."""
    answer_numbers = extract_numbers(answer_text)
    tool_blob = " ".join(json.dumps(c["result"], default=str) for c in tool_calls)
    tool_numbers = extract_numbers(tool_blob)

    ungrounded = {n for n in answer_numbers if n not in tool_numbers}
    return {
        "answer_numbers": sorted(answer_numbers),
        "ungrounded_numbers": sorted(ungrounded),
        "is_grounded": not ungrounded,
    }


def _tracing_registry(base_registry: dict[str, Callable[..., Any]]) -> tuple[dict, list[dict]]:
    """Wraps every tool in `base_registry` so each real call (and its real
    return value) gets logged, without changing what the tool actually does.
    Returns the wrapped registry plus the (initially empty) list it logs into."""
    calls: list[dict] = []

    def _wrap(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(**kwargs: Any) -> Any:
            result = func(**kwargs)
            calls.append({"tool": name, "input": kwargs, "result": result})
            return result

        return wrapped

    return {name: _wrap(name, func) for name, func in base_registry.items()}, calls


def run_eval(client: Any, eval_queries: tuple[EvalQuery, ...] = EVAL_QUERIES) -> dict:
    """Runs every query in `eval_queries` against a fresh agent (one per
    query, so no cross-query state leaks), using the real tools from
    src/tools -- only the model client is swappable, so this can run
    against either a real `openai.OpenAI()` client or a scripted mock.
    Returns per-query results plus aggregate tool-selection accuracy and
    groundedness rate."""
    results = []
    for eq in eval_queries:
        registry, calls = _tracing_registry(TOOL_REGISTRY)
        agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=TOOL_SCHEMAS)
        answer = agent.run(eq.query)

        actual_tools = tuple(dict.fromkeys(c["tool"] for c in calls))
        tool_selection_correct = set(actual_tools) == set(eq.expected_tools)
        groundedness = check_groundedness(answer, calls)

        results.append(
            {
                "query": eq.query,
                "note": eq.note,
                "expected_tools": list(eq.expected_tools),
                "actual_tools": list(actual_tools),
                "tool_selection_correct": tool_selection_correct,
                "answer": answer,
                **groundedness,
            }
        )

    n = len(results)
    tool_selection_accuracy = (sum(r["tool_selection_correct"] for r in results) / n) if n else 0.0
    groundedness_rate = (sum(r["is_grounded"] for r in results) / n) if n else 0.0

    return {
        "n_queries": n,
        "tool_selection_accuracy": round(tool_selection_accuracy, 4),
        "groundedness_rate": round(groundedness_rate, 4),
        "results": results,
    }


if __name__ == "__main__":
    import os
    import sys

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. This harness measures real model "
            "behavior and refuses to fabricate numbers by running against a "
            "mock client instead -- see tests/test_eval_harness.py for a "
            "mocked demonstration of the scoring logic itself, which is not "
            "the same thing as a real accuracy measurement.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    import openai

    real_client = openai.OpenAI(api_key=api_key)
    summary = run_eval(real_client)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
