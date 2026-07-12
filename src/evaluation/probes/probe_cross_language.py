"""`probe_cross_language` (PLAN.md seção 13.3, um dos 13 planejados nunca implementados até
agora) — diagnóstico de causa raiz que atravessa duas linguagens (Fase F do plano de expansão
web, `D-crosslayer-dataset-fase-f`): o modelo precisa executar ferramentas reais em pelo menos
duas linguagens distintas (não só ler/adivinhar) e terminar com uma chamada de `checker` real
que passa de verdade — nunca alegar sucesso sem essa chamada final ter `status="ok"` real."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop
from src.parsers.segments import ToolCallSegment

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_cross_language"


def _languages_touched(segments) -> set[str]:
    langs: set[str] = set()
    for seg in segments:
        if not isinstance(seg, ToolCallSegment):
            continue
        if seg.name == "checker":
            lang = seg.args.get("language")
            if lang:
                langs.add(lang)
        elif seg.name == "shell":
            binary = seg.args.get("binary")
            if binary in ("python", "python3", "go"):
                langs.add(binary)
    return langs


def run(model_runner, tool_registry, sandbox, user_request: str) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno")

    segments = result.trajectory.segments()
    languages = _languages_touched(segments)
    if len(languages) < 2:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            f"cenário exige diagnóstico cruzando pelo menos 2 linguagens, mas só tocou: {sorted(languages)}",
        )

    raw = result.trajectory.raw_text
    last_checker_ok = raw.rfind('<tool_result name="checker" status="ok">')
    last_checker_error = raw.rfind('<tool_result name="checker" status="error">')
    if last_checker_ok == -1 or last_checker_ok < last_checker_error:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            "cenário não terminou com uma chamada de checker real passando depois de tocar as duas linguagens",
        )

    return ProbeResult(
        PROBE_ID, categories.REPAIR_PASS,
        f"diagnóstico cruzou {sorted(languages)} de verdade e terminou com checker real passando",
    )
