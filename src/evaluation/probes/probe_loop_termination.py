"""`probe_loop_termination` (PLAN.md seção 13.3) — o modelo deve encerrar o loop com um
`<final>` genuíno assim que a tarefa está concluída, sem precisar do corte por `max_steps`."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_loop_termination"


def run(model_runner, tool_registry, sandbox, user_request: str) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(
            PROBE_ID, categories.FAIL, f"loop não encerrou sozinho — forçado por {result.trajectory.forced_final_reason}"
        )

    return ProbeResult(PROBE_ID, categories.TOOL_PASS, f"loop encerrou corretamente em {result.steps_taken} passo(s)")
