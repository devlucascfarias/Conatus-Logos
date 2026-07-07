"""Interface `Model Runner` (PLAN.md seção 5.2) — backend-agnóstica.

Único contrato exigido de qualquer backend: `generate(prompt, stop, max_tokens)`. Isso é
suportado por Transformers, vLLM, llama.cpp e APIs remotas por igual (seção 3.3) — nenhum
outro módulo do harness conhece detalhes de backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str  # "stop_sequence" | "max_tokens" | "eos"
    matched_stop: Optional[str] = None


class ModelRunner(Protocol):
    def generate(self, prompt: str, stop: list[str], max_tokens: int = 1024) -> Completion: ...
