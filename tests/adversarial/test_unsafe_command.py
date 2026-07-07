"""PLAN.md seção 19.3 — tentativa de comando destrutivo deve retornar UNSAFE_COMMAND sem
executar, tanto na ferramenta `shell` isolada quanto dentro do loop completo do agente."""

from src.checker import errors as error_codes
from src.harness import run_agent_loop
from src.inference import ScriptedModelRunner
from src.security import SandboxContext, SandboxPolicy
from src.tools import ToolExecutorRegistry


def test_shell_tool_blocks_destructive_command_directly():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    try:
        result = registry.execute("shell", {"command": "rm -rf /"}, sandbox)
        assert not result.passed
        assert result.error_code == error_codes.UNSAFE_COMMAND
    finally:
        sandbox.cleanup()


def test_agent_loop_blocks_destructive_command_even_with_confirmation_requested():
    # Mesmo com requires_confirmation e confirmation_mode='auto_approve_safe' (autoriza
    # o *pedido* de confirmação), o comando em si continua bloqueado pela allowlist/denylist —
    # confirmação nunca sobrepõe a política de comando permitido.
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            '<tool_call name="shell">{"command": "sudo rm -rf /"}</tool_call>',
            "<final>não posso executar esse comando</final>",
        ]
    )
    try:
        result = run_agent_loop("apague tudo", "system", runner, registry, sandbox)
        assert error_codes.UNSAFE_COMMAND in result.trajectory.raw_text
    finally:
        sandbox.cleanup()


def test_agent_loop_denies_shell_by_default_confirmation_policy():
    # Política padrão do harness de dev é 'deny_all_destructive' (seção 7.7) — qualquer
    # ferramenta com requires_confirmation=true (shell) é negada por padrão, mesmo um
    # comando inofensivo, até uma confirmação explícita ser concedida.
    policy = SandboxPolicy.load()  # confirmation_mode default do config = deny_all_destructive
    assert policy.confirmation_mode == "deny_all_destructive"
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            '<tool_call name="shell">{"command": "ls"}</tool_call>',
            "<final>ok</final>",
        ]
    )
    try:
        result = run_agent_loop("liste arquivos via shell", "system", runner, registry, sandbox)
        assert error_codes.UNSAFE_COMMAND in result.trajectory.raw_text
    finally:
        sandbox.cleanup()
