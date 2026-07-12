"""`probe_frontend_checker_language_choice` — novo probe (não estava nos 13 originais da seção
13.3; nasceu da expansão de frontend/web desta sessão, que ampliou `MVP_LANGUAGES` de 2 pra 7
linguagens de checker: python, go, html, javascript, typescript, scss, node). Tarefas de
frontend têm MÚLTIPLOS checkers plausíveis mas só um correto pro que foi pedido (ex.: SCSS com
`@mixin`/variável precisa do backend `scss`, que compila com dart-sass antes de renderizar — o
backend `html` sozinho não entende sintaxe SCSS). Critério automatizável: a chamada de `checker`
usa a `language` esperada pro cenário E passa de verdade — escolher a linguagem errada não é só
estilo, é uma falha funcional real (o backend errado rejeita ou nem reconhece a sintaxe)."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop
from src.parsers.segments import ToolCallSegment

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_frontend_checker_language_choice"


def run(model_runner, tool_registry, sandbox, user_request: str, expected_language: str) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno")

    segments = result.trajectory.segments()
    checker_calls = [seg for seg in segments if isinstance(seg, ToolCallSegment) and seg.name == "checker"]
    if not checker_calls:
        return ProbeResult(PROBE_ID, categories.FAIL, "tarefa exigia validar com checker, mas nenhuma chamada foi feita")

    last_call = checker_calls[-1]
    used_language = last_call.args.get("language")
    if used_language != expected_language:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            f"escolheu checker language={used_language!r}, mas o cenário exige {expected_language!r} pra essa sintaxe",
        )

    raw = result.trajectory.raw_text
    last_checker_ok = raw.rfind('<tool_result name="checker" status="ok">')
    last_checker_error = raw.rfind('<tool_result name="checker" status="error">')
    if last_checker_ok == -1 or last_checker_ok < last_checker_error:
        return ProbeResult(
            PROBE_ID, categories.FAIL,
            f"escolheu a linguagem certa ({expected_language}) mas o checker não passou de verdade",
        )

    return ProbeResult(
        PROBE_ID, categories.TOOL_PASS,
        f"escolheu checker language={expected_language!r} corretamente e passou de verdade",
    )
