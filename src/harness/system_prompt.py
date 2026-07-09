"""Prompt de sistema canônico do modelo — fonte única de verdade (PLAN.md seção 8/13).

Usado tanto para gerar o dataset de treino (`scripts/generate_dataset.py`) quanto para
avaliar/rodar o modelo (`src/evaluation/probes/*`, notebook de treino) — divergir esse texto
entre treino e avaliação recria o mesmo tipo de mismatch de formato que D-train-prompt-mask
corrigiu (PLAN.md), só que na ponta da avaliação em vez da ponta do treino.

D-conatus-logos3-identity: identidade trocada de "Praxis" para "Logos-3" (família Conatus) —
ver docs/plan_dataset_expansion_wave2_identity_multiturn.md. O nome do símbolo Python
(`PRAXIS_SYSTEM_PROMPT`) foi mantido por enquanto para não forçar uma renomeação mecânica em
todos os módulos que o importam; é um identificador interno, nunca exposto ao modelo ou ao
usuário."""

from __future__ import annotations

PRAXIS_SYSTEM_PROMPT = (
    "Você é Logos-3, um agente de engenharia de software da família de modelos Conatus. "
    "Raciocine em <think>, use "
    '<tool_call name="..."> quando precisar de uma ferramenta, e responda ao usuário só '
    "dentro de <final>."
)
