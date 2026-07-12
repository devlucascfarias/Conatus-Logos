"""Testes dos probes de avaliação (PLAN.md seção 13.3) — cada probe é exercitado tanto no
caminho de sucesso quanto no de falha, usando `ScriptedModelRunner`."""

import pytest

from src.evaluation import categories
from src.evaluation.probes import (
    probe_checker_rejection_recovery,
    probe_cross_language,
    probe_direct_vs_tool_choice,
    probe_fabricated_tool_result_attempt,
    probe_frontend_checker_language_choice,
    probe_loop_termination,
    probe_multi_file_edit,
    probe_paraphrase_generalization,
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


# --- probe_paraphrase_generalization ----------------------------------------------------


def test_probe_paraphrase_passes_when_file_really_created(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "hello.py", "content": "print(1)\\n"}</tool_call>',
            '<tool_call name="checker">{"language": "python", "operation": "syntax_check", '
            '"files": [{"path": "hello.py", "content": "print(1)\\n"}]}</tool_call>',
            "<final>arquivo criado e validado com sucesso</final>",
        ]
    )
    result = probe_paraphrase_generalization.run(
        runner, registry, sandbox, "crie hello.py", expected_file="hello.py"
    )
    assert result.category == categories.TOOL_PASS


def test_probe_paraphrase_fails_when_final_claims_success_without_real_file(sandbox, registry):
    """Regressão de D-generalization-gap: um adapter mal generalizado pulava direto para
    <final> alegando sucesso sem nunca chamar <tool_call> nenhum — os outros probes não
    pegavam isso porque não conferem o estado real do workspace."""
    runner = ScriptedModelRunner(["<final>arquivo criado e validado com sucesso</final>"])
    result = probe_paraphrase_generalization.run(
        runner, registry, sandbox, "crie hello.py", expected_file="hello.py"
    )
    assert result.category == categories.FAIL
    assert "não existe" in result.detail


def test_probe_paraphrase_fails_when_forced_by_max_steps(sandbox, registry):
    calls = [f'<tool_call name="list_files">{{"path": ".", "max_depth": {i}}}</tool_call>' for i in range(1, 9)]
    runner = ScriptedModelRunner(calls)
    result = probe_paraphrase_generalization.run(
        runner, registry, sandbox, "crie hello.py", expected_file="hello.py"
    )
    assert result.category == categories.FAIL


# --- probe_cross_language (D-eval-fase-g-probes) ----------------------------------------


def test_probe_cross_language_passes_when_two_languages_touched_and_final_checker_ok(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "report.py", "content": '
            '"import json\\nwith open(\\"report.json\\", \\"w\\") as f:\\n    '
            'json.dump({\\"value\\": 42}, f)\\n"}</tool_call>',
            '<tool_call name="shell">{"binary": "python", "args": ["report.py"]}</tool_call>',
            '<tool_call name="checker">{"language": "node", "operation": "run", "files": '
            '[{"path": "read.mjs", "content": "import { readFileSync } from \\"node:fs\\";\\n'
            'const d = JSON.parse(readFileSync(new URL(\\"./report.json\\", import.meta.url), '
            '\\"utf-8\\"));\\nconsole.log(d.value);\\n"}, '
            '{"path": "report.json", "content": "{\\"value\\": 42}"}], '
            '"entrypoint": "read.mjs"}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_cross_language.run(runner, registry, sandbox, "gere em python e leia em node")
    assert result.category == categories.REPAIR_PASS


def test_probe_cross_language_fails_when_only_one_language_touched(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="checker">{"language": "python", "operation": "syntax_check", '
            '"files": [{"path": "a.py", "content": "print(1)\\n"}]}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_cross_language.run(runner, registry, sandbox, "só python")
    assert result.category == categories.FAIL
    assert "2 linguagens" in result.detail


def test_probe_cross_language_fails_when_final_checker_errors(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="shell">{"binary": "python", "args": ["-c", "print(1)"]}</tool_call>',
            '<tool_call name="checker">{"language": "node", "operation": "run", "files": '
            '[{"path": "bad.mjs", "content": "throw new Error(\\"boom\\");\\n"}], '
            '"entrypoint": "bad.mjs"}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_cross_language.run(runner, registry, sandbox, "cross language que falha")
    assert result.category == categories.FAIL


# --- probe_multi_file_edit (D-eval-fase-g-probes) -----------------------------------------


def test_probe_multi_file_edit_passes_when_two_files_and_final_checker_ok(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "calc.py", "content": '
            '"def absolute(n):\\n    return n if n >= 0 else -n\\n"}</tool_call>',
            '<tool_call name="write_file">{"path": "test_calc.py", "content": '
            '"from calc import absolute\\n\\ndef test_absolute():\\n    '
            'assert absolute(-5) == 5\\n"}</tool_call>',
            '<tool_call name="checker">{"language": "python", "operation": "compile_and_test", '
            '"files": [{"path": "calc.py", "content": "def absolute(n):\\n    '
            'return n if n >= 0 else -n\\n"}, {"path": "test_calc.py", "content": '
            '"from calc import absolute\\n\\ndef test_absolute():\\n    '
            'assert absolute(-5) == 5\\n"}]}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_multi_file_edit.run(runner, registry, sandbox, "crie a função e o teste")
    assert result.category == categories.TOOL_PASS


def test_probe_multi_file_edit_fails_when_only_one_file_touched(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "calc.py", "content": "def f():\\n    pass\\n"}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_multi_file_edit.run(runner, registry, sandbox, "só um arquivo")
    assert result.category == categories.FAIL
    assert "2 arquivos" in result.detail


def test_probe_multi_file_edit_fails_when_final_checker_never_passes(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "a.py", "content": "x = 1\\n"}</tool_call>',
            '<tool_call name="write_file">{"path": "b.py", "content": "y = 2\\n"}</tool_call>',
            '<tool_call name="checker">{"language": "python", "operation": "syntax_check", '
            '"files": [{"path": "a.py", "content": "def f(:\\n"}]}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_multi_file_edit.run(runner, registry, sandbox, "dois arquivos mas checker falha")
    assert result.category == categories.FAIL


# --- probe_frontend_checker_language_choice (D-eval-fase-g-probes) ------------------------


def test_probe_frontend_language_choice_passes_when_correct_language_and_checker_ok(sandbox, registry):
    import json

    scss = "$primary: #3498db;\n\n.card {\n  color: $primary;\n}\n"
    write_args = json.dumps({"path": "theme.scss", "content": scss})
    checker_args = json.dumps(
        {"language": "scss", "operation": "compile", "files": [{"path": "theme.scss", "content": scss}]}
    )
    runner = ScriptedModelRunner(
        [
            f'<tool_call name="write_file">{write_args}</tool_call>',
            f'<tool_call name="checker">{checker_args}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_frontend_checker_language_choice.run(
        runner, registry, sandbox, "crie um tema scss", expected_language="scss"
    )
    assert result.category == categories.TOOL_PASS


def test_probe_frontend_language_choice_fails_when_wrong_language_used(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="checker">{"language": "html", "operation": "run", '
            '"files": [{"path": "a.html", "content": "<html></html>"}], "entrypoint": "a.html"}</tool_call>',
            "<final>ok</final>",
        ]
    )
    result = probe_frontend_checker_language_choice.run(
        runner, registry, sandbox, "crie um tema scss", expected_language="scss"
    )
    assert result.category == categories.FAIL
    assert "scss" in result.detail


def test_probe_frontend_language_choice_fails_when_no_checker_called(sandbox, registry):
    runner = ScriptedModelRunner(["<final>pronto, sem validar nada</final>"])
    result = probe_frontend_checker_language_choice.run(
        runner, registry, sandbox, "crie um tema scss", expected_language="scss"
    )
    assert result.category == categories.FAIL


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
        (probe_paraphrase_generalization, dict(user_request="crie hello.py", expected_file="hello.py")),
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
