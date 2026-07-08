"""`probe_paraphrase_generalization` (PLAN.md seção 13.3, D-generalization-gap) — o modelo deve
completar a MESMA família de tarefa (criar arquivo + validar) independente de como o pedido é
formulado, não só na frase exata vista no treino.

Nasceu de um bug real: um adapter que passava nos outros 4 probes (frases próximas ao dataset)
alucinava `<final>` de sucesso sem nenhum `<tool_call>` em 5/5 reformulações da mesma tarefa
(ver D-generalization-gap). Os outros probes não pegavam isso porque nenhum verifica o ESTADO
REAL do workspace contra a alegação do `<final>` — só a estrutura da trajetória. Este probe
fecha esse ponto cego: falha se o `<final>` alega sucesso mas o arquivo esperado não existe de
verdade no sandbox (o mesmo tipo de checagem que a seção 10.3 atribui ao Evaluator, aplicada
aqui como um probe determinístico e barato)."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_paraphrase_generalization"


def run(model_runner, tool_registry, sandbox, user_request: str, expected_file: str) -> ProbeResult:
    result = run_agent_loop(user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox)

    if result.forced_final:
        return ProbeResult(
            PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno"
        )

    file_exists = (sandbox.workspace / expected_file).exists()
    if not file_exists:
        return ProbeResult(
            PROBE_ID,
            categories.FAIL,
            f"<final> não foi forçado mas {expected_file} não existe no workspace — "
            "alegação de conclusão sem trabalho real (ver D-generalization-gap)",
        )

    return ProbeResult(
        PROBE_ID, categories.TOOL_PASS, f"tarefa completada de verdade — {expected_file} existe no workspace"
    )
