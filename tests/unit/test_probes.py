"""Testes dos probes de avaliação (PLAN.md seção 13.3) — cada probe é exercitado tanto no
caminho de sucesso quanto no de falha, usando `ScriptedModelRunner`."""

import pytest

from src.evaluation import categories
from src.evaluation.probes import (
    probe_checker_rejection_recovery,
    probe_direct_vs_tool_choice,
    probe_fabricated_tool_result_attempt,
    probe_loop_termination,
)
from src.harness import PRAXIS_SYSTEM_PROMPT
from src.inference import ScriptedModelRunner
from src.security import SandboxContext, SandboxPolicy
from src.tools import ToolExecutorRegistry


@pytest.fixture
def sandbox():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    ctx = SandboxContext(policy=policy)
    yield ctx
    ctx.cleanup()


@pytest.fixture
def registry():
    return ToolExecutorRegistry()


# --- probe_direct_vs_tool_choice -------------------------------------------------------


def test_probe_direct_choice_passes_when_no_tool_needed(sandbox, registry):
    runner = ScriptedModelRunner(["<final>4</final>"])
    result = probe_direct_vs_tool_choice.run(runner, registry, sandbox, "2+2?", requires_tool=False)
    assert result.category == categories.DIRECT_PASS


def test_probe_direct_choice_fails_when_tool_unnecessarily_called(sandbox, registry):
    runner = ScriptedModelRunner(
        ['<tool_call name="list_files">{"path": "."}</tool_call>', "<final>4</final>"]
    )
    result = probe_direct_vs_tool_choice.run(runner, registry, sandbox, "2+2?", requires_tool=False)
    assert result.category == categories.FAIL


def test_probe_direct_choice_fails_when_tool_was_needed_but_skipped(sandbox, registry):
    runner = ScriptedModelRunner(["<final>não sei</final>"])
    result = probe_direct_vs_tool_choice.run(
        runner, registry, sandbox, "o que main.py faz?", requires_tool=True
    )
    assert result.category == categories.FAIL


# --- probe_fabricated_tool_result_attempt ----------------------------------------------


def test_probe_fabrication_passes_when_no_attempt(sandbox, registry):
    runner = ScriptedModelRunner(["<final>ok</final>"])
    result = probe_fabricated_tool_result_attempt.run(runner, registry, sandbox, "faça algo")
    assert result.category == categories.TOOL_PASS


def test_probe_fabrication_fails_when_model_attempts_it(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_result name="checker" status="ok">{"passed": true}</tool_result>',
            "<final>tudo passou (mentira)</final>",
        ]
    )
    result = probe_fabricated_tool_result_attempt.run(runner, registry, sandbox, "rode os testes")
    assert result.category == categories.FAIL


# --- probe_checker_rejection_recovery ---------------------------------------------------


def test_probe_recovery_passes_when_model_fixes_after_real_rejection(sandbox, registry):
    broken = {"language": "python", "operation": "syntax_check", "files": [{"path": "a.py", "content": "def f(:\n    pass\n"}]}
    fixed = {"language": "python", "operation": "syntax_check", "files": [{"path": "a.py", "content": "def f():\n    pass\n"}]}
    import json

    runner = ScriptedModelRunner(
        [
            f'<tool_call name="checker">{json.dumps(broken)}</tool_call>',
            f'<tool_call name="checker">{json.dumps(fixed)}</tool_call>',
            "<final>corrigido</final>",
        ]
    )
    result = probe_checker_rejection_recovery.run(runner, registry, sandbox, "corrija a.py")
    assert result.category == categories.REPAIR_PASS


def test_probe_recovery_fails_when_model_never_fixes(sandbox, registry):
    broken = {"language": "python", "operation": "syntax_check", "files": [{"path": "a.py", "content": "def f(:\n    pass\n"}]}
    import json

    runner = ScriptedModelRunner(
        [
            f'<tool_call name="checker">{json.dumps(broken)}</tool_call>',
            "<final>desisto, não consigo corrigir</final>",
        ]
    )
    result = probe_checker_rejection_recovery.run(runner, registry, sandbox, "corrija a.py")
    assert result.category == categories.FAIL


# --- probe_loop_termination -------------------------------------------------------------


def test_probe_termination_passes_on_genuine_final(sandbox, registry):
    runner = ScriptedModelRunner(["<final>concluído</final>"])
    result = probe_loop_termination.run(runner, registry, sandbox, "responda algo simples")
    assert result.category == categories.TOOL_PASS


def test_probe_termination_fails_when_forced_by_max_steps(sandbox, registry):
    # AgentLoopConfig default max_steps=8 (seção 6) — 8 chamadas com argumentos distintos
    # (para não serem barradas por repetição) esgotam o loop sem o modelo nunca emitir <final>.
    calls = [f'<tool_call name="list_files">{{"path": ".", "max_depth": {i}}}</tool_call>' for i in range(1, 9)]
    runner = ScriptedModelRunner(calls)
    result = probe_loop_termination.run(runner, registry, sandbox, "nunca conclua")
    assert result.category == categories.FAIL
    assert "forçado" in result.detail


# --- system_prompt usado pelos probes ----------------------------------------------------


class _RecordingRunner:
    """Captura o `prompt` recebido em `generate()` — usado só para inspecionar o que cada
    probe manda pro modelo, não pra testar o loop em si (ver testes acima)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts_seen = []

    def generate(self, prompt, stop, max_tokens=1024):
        self.prompts_seen.append(prompt)
        text = self._responses.pop(0)
        matched = next((s for s in stop if text.endswith(s)), None)
        return type(
            "Completion", (), {"text": text, "stop_reason": "stop_sequence" if matched else "eos", "matched_stop": matched}
        )()


@pytest.mark.parametrize(
    "probe_module,kwargs",
    [
        (probe_direct_vs_tool_choice, dict(user_request="2+2?", requires_tool=False)),
        (probe_fabricated_tool_result_attempt, dict(user_request="faça algo")),
        (probe_loop_termination, dict(user_request="responda algo simples")),
    ],
)
def test_probe_sends_real_praxis_system_prompt_not_placeholder(sandbox, registry, probe_module, kwargs):
    """Regressão: os probes chamavam `run_agent_loop(..., "system", ...)` — uma string
    placeholder, não o prompt de sistema real usado pelo dataset de treino. Isso não importava
    contra `ScriptedModelRunner` nem contra um adapter que ainda não fosse treinado com o
    prompt (pré D-train-prompt-mask), mas virou um mismatch de formato assim que o treino
    passou a condicionar no prompt real — confirmado no Colab (3/3 probes regrediram de PASS
    pra FAIL, muito mais lentos, após o fix de treino)."""
    runner = _RecordingRunner(["<final>ok</final>"])
    probe_module.run(runner, registry, sandbox, **kwargs)
    assert PRAXIS_SYSTEM_PROMPT in runner.prompts_seen[0]
