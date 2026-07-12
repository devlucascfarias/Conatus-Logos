"""`probe_multi_file_edit` (PLAN.md seção 13.3, um dos 13 planejados nunca implementados até
agora) — tarefas que exigem coordenar mudanças em 2+ arquivos (ex.: uma função e seu teste, ou
um endpoint e seu arquivo de testes — o padrão que a Fase E/D do plano de expansão web usa
extensivamente). Critério automatizável: pelo menos 2 caminhos de arquivo distintos tocados via
`write_file`, e a última chamada de `checker` real passando — nunca aceitar "editou vários
arquivos" sem confirmar que o resultado final compila/passa de verdade."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop
from src.parsers.segments import ToolCallSegment

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_multi_file_edit"


def _written_paths(segments) -> set[str]:
    paths: set[str] = set()
    for seg in segments:
        if isinstance(seg, ToolCallSegment) and seg.name == "write_file":
            path = seg.args.get("path")
            if path:
                paths.add(path)
    return paths


def run(model_runner, tool_registry, sandbox, user_request: str, min_files: int = 2) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno")

    segments = result.trajectory.segments()
    written = _written_paths(segments)
    if len(written) < min_files:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            f"tarefa exigia editar pelo menos {min_files} arquivos distintos, só tocou: {sorted(written)}",
        )

    raw = result.trajectory.raw_text
    last_checker_ok = raw.rfind('<tool_result name="checker" status="ok">')
    last_checker_error = raw.rfind('<tool_result name="checker" status="error">')
    if last_checker_ok == -1 or last_checker_ok < last_checker_error:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            "editou múltiplos arquivos mas não terminou com um checker real confirmando que o conjunto passa",
        )

    return ProbeResult(
        PROBE_ID, categories.TOOL_PASS,
        f"coordenou {len(written)} arquivos ({sorted(written)}) e confirmou com checker real",
    )
