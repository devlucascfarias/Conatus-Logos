"""`Trajectory` — estado acumulado de um turno do agente (PLAN.md seção 5.2/6).

Só o harness escreve `<tool_result>` na trajetória (via `append_tool_result`); o modelo nunca
deveria produzir essa tag (seção 3.2) — é isso que torna `contains_fabricated_tool_result`
(src.parsers) uma checagem válida sobre o texto cru gerado antes de ele ser anexado aqui.

D-dataset-history-schema (docs/plan_dataset_expansion_wave2_identity_multiturn.md seção 4):
`history` representa turnos JÁ CONCLUÍDOS da mesma sessão — mesmo papel que `agent.Turn`/
`Run(..., history []Turn, ...)` do lado da CLI Go (`cli/internal/agent/agent.go`).
`render_for_model` concatena esses turnos no prompt no MESMO formato `[USER]/[ASSISTANT]` que
a CLI já usa em produção — divergir esse formato entre os dois lados recriaria o tipo de
mismatch treino/inferência que `D-train-prompt-mask` corrigiu uma vez. `history` é só contexto
de entrada: nunca deveria entrar na loss de treino (isso é responsabilidade do pipeline de
treino/collator, ainda não implementada — ver seção 4 do plano)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.parsers import Segment, parse_segments


@dataclass
class HistoryTurn:
    """Um turno já concluído — `user_request` é o pedido, `raw_text` é a trajetória COMPLETA
    gerada pro turno (com `<think>`/`<tool_call>`/`<tool_result>`/`<final>`, não só a resposta
    final), porque é esse o formato que o `[ASSISTANT]` sempre teve durante o treino."""

    user_request: str
    raw_text: str


def render_environment(environment: Optional[dict]) -> str:
    """Bloco de contexto de ambiente (D-prompt-environment-block) — SO, shell, cwd do host,
    injetado pelo harness (não digitado pelo usuário), igual ao `<env>`/`Platform:` que o Claude
    Code e o Codex colocam no contexto. É isso que permite o modelo escolher o CLI certo (ex.:
    PowerShell no Windows) SEM o usuário declarar o SO. Renderizado logo após o system prompt,
    então entra no PREFIXO mascarado do treino (contexto, nunca tokens treinados) — ver
    `_render_prefix` em src/training/data_collator.py. Chaves conhecidas saem em ordem estável;
    ausente/vazio => string vazia, mantendo o formato antigo idêntico para exemplos sem env."""
    if not environment:
        return ""
    lines = []
    for key in ("os", "shell", "cwd"):
        val = environment.get(key)
        if val:
            lines.append(f"{key}: {val}")
    for key in sorted(environment):
        if key not in ("os", "shell", "cwd") and environment.get(key):
            lines.append(f"{key}: {environment[key]}")
    if not lines:
        return ""
    return "\n\n<environment>\n" + "\n".join(lines) + "\n</environment>"


@dataclass
class Trajectory:
    system_prompt: str
    user_request: str
    raw_text: str = ""
    forced_final_reason: Optional[str] = None
    history: list[HistoryTurn] = field(default_factory=list)
    environment: Optional[dict] = None

    def render_for_model(self) -> str:
        """Texto completo que vira o prompt de continuação para o Model Runner (seção 3.6/6).

        Modela a decisão D2: tool calling multi-etapa é UMA mensagem assistant contínua — o
        harness nunca monta múltiplos turnos de chat com role 'tool', só concatena texto.
        Turnos de `history` (se houver) entram ANTES do turno atual, no mesmo formato
        `[USER]/[ASSISTANT]` — espelha `agent.Run`'s `historyPrefix` byte a byte. O bloco de
        `<environment>` (se houver) entra logo após o system prompt, antes do histórico
        (D-prompt-environment-block)."""
        env_block = render_environment(self.environment)
        history_prefix = "".join(
            f"\n\n[USER]\n{turn.user_request}\n\n[ASSISTANT]\n{turn.raw_text}" for turn in self.history
        )
        return f"{self.system_prompt}{env_block}{history_prefix}\n\n[USER]\n{self.user_request}\n\n[ASSISTANT]\n{self.raw_text}"

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

    def to_example_dict(self) -> dict[str, Any]:
        """Corpo de `trajectory` no formato canônico do dataset (`src/dataset/schema.py`,
        `TRAJECTORY_SCHEMA`) — `history` só entra no dict quando não está vazio, pra manter o
        formato de exemplos de turno único idêntico ao que já existia (D-dataset-history-schema:
        "Ausente ou vazio = comportamento de hoje")."""
        body: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "user_request": self.user_request,
            "raw_text": self.raw_text,
        }
        if self.history:
            body["history"] = [{"user_request": t.user_request, "raw_text": t.raw_text} for t in self.history]
        if self.environment:
            body["environment"] = dict(self.environment)
        return body
