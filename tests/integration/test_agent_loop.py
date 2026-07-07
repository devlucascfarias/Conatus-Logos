"""Testes de integração do loop do agente (PLAN.md seção 6, 19.2) com `ScriptedModelRunner` —
cobrem: resposta direta, uma ferramenta, múltiplas ferramentas, correção após erro, chamada
repetida (deve ser barrada), limite de passos atingido."""

import pytest

from src.checker import errors as error_codes
from src.harness import AgentLoopConfig, run_agent_loop
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


def test_direct_answer_no_tool_needed(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            "<think>não preciso de ferramenta para isso</think><final>2 + 2 = 4</final>",
        ]
    )
    result = run_agent_loop("quanto é 2 + 2?", "system", runner, registry, sandbox)
    assert result.public_output.strip() != ""
    assert "4" in result.public_output
    assert result.steps_taken == 1
    assert not result.forced_final


def test_single_tool_call_then_final(sandbox, registry):
    (sandbox.workspace / "main.py").write_text("print('oi')\n", encoding="utf-8")
    runner = ScriptedModelRunner(
        [
            '<think>vou ler o arquivo</think><tool_call name="read_file">{"path": "main.py"}</tool_call>',
            "<think>o arquivo imprime oi</think><final>o arquivo imprime 'oi'</final>",
        ]
    )
    result = run_agent_loop("o que main.py faz?", "system", runner, registry, sandbox)
    assert runner.calls_made == 2
    assert '<tool_result name="read_file" status="ok">' in result.trajectory.raw_text
    assert "oi" in result.public_output


def test_multi_tool_call_sequence(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="write_file">{"path": "a.py", "content": "def add(a, b):\\n    return a + b\\n"}</tool_call>',
            '<tool_call name="checker">{"language": "python", "operation": "syntax_check", "files": [{"path": "a.py", "content": "def add(a, b):\\n    return a + b\\n"}]}</tool_call>',
            "<final>arquivo criado e sintaxe validada com sucesso</final>",
        ]
    )
    result = run_agent_loop("crie a.py com uma função add", "system", runner, registry, sandbox)
    assert runner.calls_made == 3
    assert '<tool_result name="write_file" status="ok">' in result.trajectory.raw_text
    assert '<tool_result name="checker" status="ok">' in result.trajectory.raw_text


def test_model_corrects_after_checker_rejection(sandbox, registry):
    broken_code = "def add(a, b:\\n    return a + b\\n"
    fixed_code = "def add(a, b):\\n    return a + b\\n"
    runner = ScriptedModelRunner(
        [
            f'<tool_call name="checker">{{"language": "python", "operation": "syntax_check", "files": [{{"path": "a.py", "content": "{broken_code}"}}]}}</tool_call>',
            f'<tool_call name="checker">{{"language": "python", "operation": "syntax_check", "files": [{{"path": "a.py", "content": "{fixed_code}"}}]}}</tool_call>',
            "<final>corrigido — a sintaxe agora é válida</final>",
        ]
    )
    result = run_agent_loop("valide e corrija a.py", "system", runner, registry, sandbox)
    assert runner.calls_made == 3
    raw = result.trajectory.raw_text
    first_result_idx = raw.index('<tool_result name="checker" status="error">')
    second_result_idx = raw.index('<tool_result name="checker" status="ok">')
    assert first_result_idx < second_result_idx
    assert error_codes.SYNTAX_ERROR in raw


def test_identical_repeated_tool_call_is_blocked(sandbox, registry):
    (sandbox.workspace / "a.py").write_text("print(1)\n", encoding="utf-8")
    call = '<tool_call name="read_file">{"path": "a.py"}</tool_call>'
    runner = ScriptedModelRunner([call, call, "<final>ok</final>"])
    result = run_agent_loop("leia a.py duas vezes", "system", runner, registry, sandbox)
    assert error_codes.MAX_STEPS_EXCEEDED in result.trajectory.raw_text
    # a segunda chamada idêntica não deve ter re-executado a ferramenta de verdade
    assert result.trajectory.raw_text.count('<tool_result name="read_file" status="ok">') == 1


def test_max_steps_reached_forces_final(sandbox, registry):
    (sandbox.workspace / "a.py").write_text("print(1)\n", encoding="utf-8")
    # Cada chamada usa argumentos diferentes para não ser barrada por repetição —
    # o loop deve ainda assim parar em max_steps e forçar um <final>.
    calls = [
        f'<tool_call name="list_files">{{"path": ".", "max_depth": {i}}}</tool_call>' for i in range(1, 4)
    ]
    runner = ScriptedModelRunner(calls)
    config = AgentLoopConfig(max_steps=3)
    result = run_agent_loop("nunca conclua", "system", runner, registry, sandbox, config=config)
    assert result.forced_final
    assert result.trajectory.forced_final_reason == error_codes.MAX_STEPS_EXCEEDED
    assert result.steps_taken == 3


def test_unsupported_tool_reports_structured_error(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<tool_call name="does_not_exist">{"x": 1}</tool_call>',
            "<final>desisto dessa ferramenta</final>",
        ]
    )
    result = run_agent_loop("chame uma ferramenta inexistente", "system", runner, registry, sandbox)
    assert error_codes.UNSUPPORTED_TOOL in result.trajectory.raw_text


def test_prod_mode_hides_reasoning_and_tool_calls(sandbox, registry):
    runner = ScriptedModelRunner(
        [
            '<think>segredo interno</think><tool_call name="list_files">{"path": "."}</tool_call>',
            "<think>mais segredo</think><final>resposta pública limpa</final>",
        ]
    )
    config = AgentLoopConfig(mode="prod")
    result = run_agent_loop("liste arquivos", "system", runner, registry, sandbox, config=config)
    assert result.public_output.strip() == "resposta pública limpa"
    assert "segredo" not in result.public_output
    assert "<think>" not in result.public_output
