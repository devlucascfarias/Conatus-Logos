"""PLAN.md seção 19.3 — vazamento de raciocínio interno para dentro de <final> deve ser
sinalizado por verify_consistency (INTERNAL_REASONING_LEAK, seção 10.3), e o modo prod nunca
deve expor esse conteúdo de qualquer forma."""

from src.checker import errors as error_codes
from src.harness import AgentLoopConfig, run_agent_loop
from src.inference import ScriptedModelRunner
from src.security import SandboxContext, SandboxPolicy
from src.tools import ToolExecutorRegistry


def test_final_segment_leaking_raw_tags_is_flagged_by_consistency_check():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            "<final>a resposta é 4 <think>na verdade não tenho certeza</think></final>",
        ]
    )
    try:
        result = run_agent_loop("quanto é 2+2?", "system", runner, registry, sandbox)
        assert error_codes.INTERNAL_REASONING_LEAK in result.consistency_errors
    finally:
        sandbox.cleanup()


def test_clean_final_segment_does_not_trigger_leak_flag():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(["<think>2+2=4</think><final>a resposta é 4</final>"])
    try:
        result = run_agent_loop("quanto é 2+2?", "system", runner, registry, sandbox)
        assert error_codes.INTERNAL_REASONING_LEAK not in result.consistency_errors
    finally:
        sandbox.cleanup()


def test_prod_mode_never_exposes_think_content_regardless_of_leak():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            "<think>segredo que nao deveria vazar</think><final>resposta pública limpa</final>",
        ]
    )
    config = AgentLoopConfig(mode="prod")
    try:
        result = run_agent_loop("pergunta qualquer", "system", runner, registry, sandbox, config=config)
        assert "segredo" not in result.public_output
        assert result.public_output.strip() == "resposta pública limpa"
    finally:
        sandbox.cleanup()
