"""Prompt de sistema canônico do Praxis — fonte única de verdade (PLAN.md seção 8/13).

Usado tanto para gerar o dataset de treino (`scripts/generate_dataset.py`) quanto para
avaliar/rodar o modelo (`src/evaluation/probes/*`, notebook de treino) — divergir esse texto
entre treino e avaliação recria o mesmo tipo de mismatch de formato que D-train-prompt-mask
corrigiu (PLAN.md), só que na ponta da avaliação em vez da ponta do treino."""

from __future__ import annotations

PRAXIS_SYSTEM_PROMPT = (
    "Você é Praxis, um agente de engenharia de software. Raciocine em <think>, use "
    '<tool_call name="..."> quando precisar de uma ferramenta, e responda ao usuário só '
    "dentro de <final>."
)
