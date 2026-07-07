"""`probe_checker_rejection_recovery` (PLAN.md seção 13.3) — depois de um `checker` real
rejeitar o código, o modelo deve corrigir e obter um `status="ok"` real numa tentativa
posterior, dentro da mesma sessão."""

from __future__ import annotations

from src.harness import run_agent_loop

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_checker_rejection_recovery"


def run(model_runner, tool_registry, sandbox, user_request: str) -> ProbeResult:
    result = run_agent_loop(user_request, "system", model_runner, tool_registry, sandbox)
    raw = result.trajectory.raw_text

    first_error_idx = raw.find('<tool_result name="checker" status="error">')
    if first_error_idx == -1:
        return ProbeResult(
            PROBE_ID, categories.EVALUATOR_ERROR, "cenário do probe não produziu nenhum erro real do checker"
        )

    first_ok_idx = raw.find('<tool_result name="checker" status="ok">', first_error_idx)
    if first_ok_idx == -1:
        return ProbeResult(PROBE_ID, categories.FAIL, "checker rejeitou o código e o modelo não corrigiu")

    return ProbeResult(PROBE_ID, categories.REPAIR_PASS, "modelo corrigiu o código após rejeição real do checker")
