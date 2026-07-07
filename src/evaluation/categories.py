"""Categorias de resultado da avaliação (PLAN.md seção 13.1) — nunca misturar erro do modelo
com erro de infraestrutura do harness/avaliador."""

from __future__ import annotations

DIRECT_PASS = "DIRECT_PASS"
TOOL_PASS = "TOOL_PASS"
REPAIR_PASS = "REPAIR_PASS"
PARTIAL_PASS = "PARTIAL_PASS"
FAIL = "FAIL"
HARNESS_ERROR = "HARNESS_ERROR"
EVALUATOR_ERROR = "EVALUATOR_ERROR"

ALL_CATEGORIES = frozenset(
    {DIRECT_PASS, TOOL_PASS, REPAIR_PASS, PARTIAL_PASS, FAIL, HARNESS_ERROR, EVALUATOR_ERROR}
)

# Usado por relatórios (seção 13.2) para separar "qualidade do modelo" de "infraestrutura".
MODEL_ATTRIBUTABLE = frozenset({DIRECT_PASS, TOOL_PASS, REPAIR_PASS, PARTIAL_PASS, FAIL})
INFRASTRUCTURE_ATTRIBUTABLE = frozenset({HARNESS_ERROR, EVALUATOR_ERROR})
