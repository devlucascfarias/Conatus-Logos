"""`probe_direct_vs_tool_choice` (PLAN.md seção 13.3) — o modelo deve responder direto quando
não precisa de ferramenta, e usar ferramenta quando precisa."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop
from src.parsers.segments import ToolCallSegment

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_direct_vs_tool_choice"


def run(model_runner, tool_registry, sandbox, user_request: str, requires_tool: bool) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno")

    used_tool = any(isinstance(seg, ToolCallSegment) for seg in result.trajectory.segments())

    if requires_tool and not used_tool:
        return ProbeResult(
            PROBE_ID, categories.FAIL, "tarefa exigia ferramenta, mas o modelo respondeu direto"
        )
    if not requires_tool and used_tool:
        return ProbeResult(
            PROBE_ID, categories.FAIL, "tarefa não exigia ferramenta, mas o modelo chamou uma mesmo assim"
        )

    category = categories.TOOL_PASS if used_tool else categories.DIRECT_PASS
    return ProbeResult(PROBE_ID, category, "escolha entre resposta direta e ferramenta foi correta")
