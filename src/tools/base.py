"""Tipo de retorno uniforme de qualquer executor de ferramenta (PLAN.md seção 5.2, 6).

O loop do agente trata todo resultado de ferramenta da mesma forma
(`"ok" if result.passed else "error"`, `result.to_json()` como corpo do `<tool_result>`) —
por isso todo executor em `src/tools/executors/*.py` devolve um `ToolExecutionResult`, nunca
um dict solto ou uma exceção para o caminho de erro esperado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ToolExecutionResult:
    passed: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        if self.passed:
            return dict(self.data)
        body = dict(self.data)
        body["code"] = self.error_code
        body["message"] = self.error_message
        return body
