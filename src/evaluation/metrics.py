"""Agregação de métricas (PLAN.md seção 13.2) a partir de uma lista de `ProbeResult`/execuções
de loop já categorizadas. Esqueleto funcional — a lista completa de métricas (latência, custo
de contexto, desempenho por linguagem/task_type) é preenchida quando houver volume real de
execuções vindas de M6 (adapter treinado + dataset de probes em escala)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .categories import INFRASTRUCTURE_ATTRIBUTABLE
from .probes.base import ProbeResult


def summarize(results: Iterable[ProbeResult]) -> dict:
    results = list(results)
    by_category = Counter(r.category for r in results)
    by_probe = Counter(r.probe_id for r in results)

    total = len(results)
    infra_failures = sum(by_category.get(c, 0) for c in INFRASTRUCTURE_ATTRIBUTABLE)
    model_attributable_total = total - infra_failures

    return {
        "total": total,
        "by_category": dict(by_category),
        "by_probe": dict(by_probe),
        "infrastructure_failure_rate": (infra_failures / total) if total else 0.0,
        "model_attributable_total": model_attributable_total,
    }
