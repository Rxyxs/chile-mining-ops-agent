"""Tests for the CLI entry point: the one-shot mode (existing behavior) and
the new interactive multi-turn REPL. Both are exercised without a real
ANTHROPIC_API_KEY by monkeypatching os.environ and the `anthropic` module
that src.cli imports lazily inside main().
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from src import cli


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> None:
    fake_module = types.SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-tests")


def test_main_without_api_key_fails_cleanly(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = cli.main(["hola"])

    assert exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_main_one_shot_mode_calls_agent_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    fake_client = MagicMock()
    _install_fake_anthropic(monkeypatch, fake_client)

    fake_agent = MagicMock()
    fake_agent.run.return_value = "Recovery averaged 88.1%."
    monkeypatch.setattr(cli, "MiningOpsAgent", MagicMock(return_value=fake_agent))

    exit_code = cli.main(["what", "was", "the", "recovery?"])

    assert exit_code == 0
    fake_agent.run.assert_called_once_with("what was the recovery?")
    fake_agent.chat.assert_not_called()
    assert "Recovery averaged 88.1%." in capsys.readouterr().out


def test_main_with_no_args_launches_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    _install_fake_anthropic(monkeypatch, fake_client)

    called = {}

    def fake_repl(agent, input_fn=input, output_fn=print):
        called["agent"] = agent
        return 0

    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    exit_code = cli.main([])

    assert exit_code == 0
    assert "agent" in called


def test_repl_reset_and_exit_commands() -> None:
    agent = MagicMock()
    agent.chat.return_value = "some answer"

    inputs = iter(["hola", "reset", "exit"])
    outputs: list[str] = []

    exit_code = cli._run_repl(agent, input_fn=lambda _prompt: next(inputs), output_fn=outputs.append)

    assert exit_code == 0
    agent.chat.assert_called_once_with("hola")
    agent.reset.assert_called_once()
    assert any("cleared" in line for line in outputs)


def test_repl_skips_blank_input_and_survives_tool_error() -> None:
    agent = MagicMock()
    agent.chat.side_effect = [RuntimeError("boom"), "recovered answer"]

    inputs = iter(["", "first question", "second question", "exit"])
    outputs: list[str] = []

    exit_code = cli._run_repl(agent, input_fn=lambda _prompt: next(inputs), output_fn=outputs.append)

    assert exit_code == 0
    assert agent.chat.call_count == 2
    assert any("Agent run failed" in line for line in outputs)
    assert any("recovered answer" in line for line in outputs)


def test_repl_eof_exits_cleanly() -> None:
    agent = MagicMock()

    def _raise_eof(_prompt):
        raise EOFError

    exit_code = cli._run_repl(agent, input_fn=_raise_eof, output_fn=lambda _: None)

    assert exit_code == 0
    agent.chat.assert_not_called()
