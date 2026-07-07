"""Contrato comum de um probe (PLAN.md seção 13.3) — cada probe roda o loop do agente sobre
um cenário fixo e devolve um veredito automatizável, nunca uma inspeção manual."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    category: str  # uma das constantes de src.evaluation.categories
    detail: str
