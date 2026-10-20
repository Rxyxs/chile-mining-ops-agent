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

This module's own __main__ block runs the 100 queries in `src/eval_queries.py`
against a real model behind any OpenAI-compatible endpoint: OpenAI itself
(`OPENAI_API_KEY`), or a local model served by llama-cpp-python, which needs no key:

    python -m src.eval_harness --base-url http://localhost:8000/v1 --model local \
        --output reports/eval_local.json
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


from src.eval_queries import EVAL_QUERIES, EvalQuery  # noqa: E402,F401 -- re-exported

Z_95 = 1.959963985


def extract_numbers(text: str) -> set[str]:
    """Extracts numeric substrings from text for a loose groundedness check.
    Deliberately string-based rather than float-parsing, so formatted numbers
    like '88.1%' or '2,000,000' still match without reproducing every
    locale's number formatting -- this trades precision for never crashing
    on a format it doesn't recognize."""
    return set(_NUMBER_RE.findall(text))


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson interval for a proportion (stays inside [0, 1] near the edges)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5)
    return (round(max(0.0, (center - margin) / denom), 4), round(min(1.0, (center + margin) / denom), 4))


def check_arguments(expected_args: dict, calls: list[dict]) -> bool:
    """Every expected tool was called at least once with the expected argument values
    (other arguments are free; a tool listed with `{}` must be called with no filter
    among the expected ones, so nothing is checked beyond the call itself)."""
    for tool, expected in expected_args.items():
        matching = [c for c in calls if c["tool"] == tool]
        if not any(all(c["input"].get(k) == v for k, v in expected.items()) for c in matching):
            return False
    return True


def _number_values(token: str) -> set[float]:
    """Every value a formatted number could mean: '83,678' and '83.678' are 83678 in
    a Spanish or English answer, '64,85' is 64.85, '6.514.323,484' is 6514323.484."""
    values = set()
    for decimal, thousands in ((".", ","), (",", ".")):
        parts = token.split(decimal)
        if len(parts) > 2:
            continue
        whole = parts[0].replace(thousands, "")
        if thousands in parts[0] and not all(len(g) == 3 for g in parts[0].split(thousands)[1:]):
            continue
        try:
            values.add(float(whole + ("." + parts[1] if len(parts) == 2 else "")))
        except ValueError:
            pass
    # A bare '83,678' could also be 83.678 with a decimal comma.
    if token.count(",") == 1 and "." not in token:
        values.add(float(token.replace(",", ".")))
    return values


def _matches(value: float, sources: set[float]) -> bool:
    """Equal, or equal once the source is rounded to the answer's precision."""
    for source in sources:
        if source == value:
            return True
        for digits in range(0, 4):
            if round(source, digits) == value:
                return True
    return False


def check_groundedness(
    answer_text: str, tool_calls: list[dict], question: str = "", context: str = ""
) -> dict:
    """Every number in the final answer should also appear in some tool call's
    result -- if not, the model may have invented it. Numbers are compared by
    value, not as strings, so '83,678' matches a result of 83678.0 and '2.05'
    matches 2.045 rounded. Numbers from the user's question and from `context`
    (the tool descriptions, e.g. the 4.14% high-silica cut) count as sources too.
    It checks numbers, not labels: an answer that calls 720 hours "720 days" passes."""
    answer_numbers = extract_numbers(answer_text)
    tool_blob = " ".join(json.dumps(c["result"], default=str) for c in tool_calls)
    source_tokens = extract_numbers(tool_blob) | extract_numbers(question) | extract_numbers(context)
    source_values = {v for t in source_tokens for v in _number_values(t)}

    ungrounded = {n for n in answer_numbers if not any(_matches(v, source_values) for v in _number_values(n))}
    return {
        "answer_numbers": sorted(answer_numbers),
        "ungrounded_numbers": sorted(ungrounded),
        "is_grounded": not ungrounded,
    }


TOOL_CONTEXT = " ".join(json.dumps(s_["function"], ensure_ascii=False) for s_ in TOOL_SCHEMAS)


def _tracing_registry(base_registry: dict[str, Callable[..., Any]]) -> tuple[dict, list[dict]]:
    """Wraps every tool in `base_registry` so each real call (and its real
    return value) gets logged, without changing what the tool actually does.
    Returns the wrapped registry plus the (initially empty) list it logs into."""
    calls: list[dict] = []

    def _wrap(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(**kwargs: Any) -> Any:
            try:
                result = func(**kwargs)
            except Exception as exc:
                calls.append({"tool": name, "input": kwargs, "result": {"error": str(exc)}})
                raise
            calls.append({"tool": name, "input": kwargs, "result": result})
            return result

        return wrapped

    return {name: _wrap(name, func) for name, func in base_registry.items()}, calls


def run_eval(
    client: Any, eval_queries: tuple[EvalQuery, ...] = EVAL_QUERIES, model: str | None = None,
    progress: bool = False,
) -> dict:
    """Runs every query in `eval_queries` against a fresh agent (one per
    query, so no cross-query state leaks), using the real tools from
    src/tools -- only the model client is swappable, so this can run
    against either a real `openai.OpenAI()` client or a scripted mock.
    Returns per-query results plus aggregate tool-selection accuracy and
    groundedness rate."""
    replay = []
    for i, eq in enumerate(eval_queries):
        registry, calls = _tracing_registry(TOOL_REGISTRY)
        kwargs = {"model": model} if model else {}
        agent = MiningOpsAgent(client=client, tool_registry=registry, tool_schemas=TOOL_SCHEMAS, **kwargs)
        try:
            answer = agent.run(eq.query)
        except Exception as exc:  # noqa: BLE001 -- a server error is a result, not a crash
            answer = f"[error] {type(exc).__name__}: {exc}"
        replay.append((eq, answer, calls))
        if progress:
            r = _score(eq, answer, calls)
            print(f"[{i + 1:>3}/{len(eval_queries)}] tools={r['actual_tools']} ok={r['tool_selection_correct']} "
                  f"args={r['arguments_correct']} grounded={r['is_grounded']}", flush=True)
    return _summarize(replay)


def _score(eq: EvalQuery, answer: str, calls: list[dict]) -> dict:
    actual_tools = tuple(dict.fromkeys(c["tool"] for c in calls))
    tool_selection_correct = set(actual_tools) == set(eq.expected_tools)
    arguments_correct = tool_selection_correct and check_arguments(eq.expected_args, calls)
    return {
        "query": eq.query,
        "note": eq.note,
        "expected_tools": list(eq.expected_tools),
        "actual_tools": list(actual_tools),
        "tool_selection_correct": tool_selection_correct,
        "arguments_correct": arguments_correct,
        "tool_calls": [{"tool": c["tool"], "input": c["input"]} for c in calls],
        "answer": answer,
        **check_groundedness(answer, calls, eq.query, TOOL_CONTEXT),
    }


def _summarize(replay: list[tuple[EvalQuery, str, list[dict]]]) -> dict:
    results = [_score(eq, answer, calls) for eq, answer, calls in replay]
    n = len(results)
    tool_hits = sum(r["tool_selection_correct"] for r in results)
    arg_hits = sum(r["arguments_correct"] for r in results)
    grounded = sum(r["is_grounded"] for r in results)

    by_group: dict[str, dict] = {}
    for (eq, _, _), r in zip(replay, results):
        group = "out_of_scope" if not eq.expected_tools else (
            "multi_tool" if len(eq.expected_tools) > 1 else eq.expected_tools[0])
        g = by_group.setdefault(group, {"n": 0, "tool_selection": 0, "arguments": 0, "grounded": 0})
        g["n"] += 1
        g["tool_selection"] += int(r["tool_selection_correct"])
        g["arguments"] += int(r["arguments_correct"])
        g["grounded"] += int(r["is_grounded"])

    return {
        "n_queries": n,
        "tool_selection_accuracy": round(tool_hits / n, 4) if n else 0.0,
        "tool_selection_ci_95": list(wilson_ci(tool_hits, n)),
        "arguments_accuracy": round(arg_hits / n, 4) if n else 0.0,
        "arguments_ci_95": list(wilson_ci(arg_hits, n)),
        "groundedness_rate": round(grounded / n, 4) if n else 0.0,
        "groundedness_ci_95": list(wilson_ci(grounded, n)),
        "by_group": by_group,
        "results": results,
    }


def rescore(path: str) -> dict:
    """Recomputes the scores of a saved run without calling the model again: the logged
    tool calls are re-executed (the tools are deterministic) to recover their results."""
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    by_query = {q.query: q for q in EVAL_QUERIES}
    replay = []
    for r in saved["results"]:
        calls = []
        for c in r["tool_calls"]:
            try:
                result = TOOL_REGISTRY[c["tool"]](**c["input"])
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            calls.append({"tool": c["tool"], "input": c["input"], "result": result})
        replay.append((by_query[r["query"]], r["answer"], calls))
    return _summarize(replay)


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import time

    parser = argparse.ArgumentParser(description="Runs the 100 eval queries against a real model.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", default=None, help="Model name sent to the endpoint.")
    parser.add_argument("--output", default=None, help="Where to write the JSON summary.")
    parser.add_argument("--rescore", default=None, help="Re-score a saved run (no model call) and overwrite it.")
    parser.add_argument("--parse-tool-calls", action="store_true",
                        help="Parse <tool_call> blocks from content (local Qwen via llama-cpp, see src/local_client.py).")
    args = parser.parse_args()

    if args.rescore:
        with open(args.rescore, encoding="utf-8") as f:
            saved = json.load(f)
        summary = {**rescore(args.rescore), "model": saved.get("model"), "elapsed_seconds": saved.get("elapsed_seconds")}
        with open(args.rescore, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, ensure_ascii=False))
        raise SystemExit(0)

    api_key = os.environ.get("OPENAI_API_KEY")
    if args.base_url is None and not api_key:
        print(
            "Set OPENAI_API_KEY, or pass --base-url for a local OpenAI-compatible server. "
            "This harness measures real model behavior and never falls back to a mock.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    import openai

    real_client = openai.OpenAI(api_key=api_key or "not-needed", base_url=args.base_url, timeout=600)
    if args.parse_tool_calls:
        from src.local_client import ToolCallParsingClient

        real_client = ToolCallParsingClient(real_client)
    start = time.perf_counter()
    summary = run_eval(real_client, model=args.model, progress=True)
    summary["model"] = args.model
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, ensure_ascii=False))

