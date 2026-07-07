"""`Trajectory` — estado acumulado de um turno do agente (PLAN.md seção 5.2/6).

Só o harness escreve `<tool_result>` na trajetória (via `append_tool_result`); o modelo nunca
deveria produzir essa tag (seção 3.2) — é isso que torna `contains_fabricated_tool_result`
(src.parsers) uma checagem válida sobre o texto cru gerado antes de ele ser anexado aqui."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.parsers import Segment, parse_segments


@dataclass
class Trajectory:
    system_prompt: str
    user_request: str
    raw_text: str = ""
    forced_final_reason: Optional[str] = None

    def render_for_model(self) -> str:
        """Texto completo que vira o prompt de continuação para o Model Runner (seção 3.6/6).

        Modela a decisão D2: tool calling multi-etapa é UMA mensagem assistant contínua — o
        harness nunca monta múltiplos turnos de chat com role 'tool', só concatena texto."""
        return f"{self.system_prompt}\n\n[USER]\n{self.user_request}\n\n[ASSISTANT]\n{self.raw_text}"

    def append_raw(self, text: str) -> None:
        self.raw_text += text

    def append_tool_result(self, name: str, status: str, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False)
        self.raw_text += f'<tool_result name="{name}" status="{status}">{payload}</tool_result>'

    def force_final(self, reason: str) -> None:
        """Seção 6 — usado quando `max_steps` é atingido sem o modelo emitir `<final>`."""
        self.forced_final_reason = reason
        self.raw_text += f"<final>Não foi possível concluir a tarefa: {reason}.</final>"

    def segments(self) -> list[Segment]:
        return parse_segments(self.raw_text)
