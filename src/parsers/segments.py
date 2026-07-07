"""Tipos de segmento da gramática canônica de trajetória (PLAN.md, seção 3.2).

Um turno de assistente é uma sequência ordenada de segmentos. Cada tipo aqui corresponde
a exatamente um dos quatro tipos de tag reconhecidos pela gramática, mais um tipo
`MalformedSegment` para qualquer trecho que viole a gramática (tag não fechada, JSON
inválido dentro de tool_call/tool_result, texto fora de qualquer tag reconhecida).

Nenhuma classe aqui executa nada nem valida contra JSON Schema de ferramenta — isso é
responsabilidade de `src/schemas`. Este módulo só sabe reconhecer a forma sintática.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass(frozen=True)
class ThinkSegment:
    """Corresponde a <think>...</think> — raciocínio interno, nunca exposto em modo prod."""

    text: str
    start: int
    end: int
    kind: str = "think"


@dataclass(frozen=True)
class ToolCallSegment:
    """Corresponde a <tool_call name="...">{json}</tool_call>, gerado pelo modelo."""

    name: str
    args: dict[str, Any]
    raw_body: str
    start: int
    end: int
    kind: str = "tool_call"


@dataclass(frozen=True)
class ToolResultSegment:
    """Corresponde a <tool_result name="..." status="ok|error">{json}</tool_result>.

    Nunca deve ser gerado pelo modelo (seção 3.2) — só o harness escreve isto na trajetória.
    """

    name: str
    status: str
    body: dict[str, Any]
    raw_body: str
    start: int
    end: int
    kind: str = "tool_result"


@dataclass(frozen=True)
class FinalSegment:
    """Corresponde a <final>...</final> — única parte exposta ao usuário em modo prod."""

    text: str
    start: int
    end: int
    kind: str = "final"


@dataclass(frozen=True)
class MalformedSegment:
    """Qualquer violação da gramática: tag não fechada, JSON inválido, texto solto."""

    reason: str  # código estável, ver src/checker/errors.py (ex.: TOOL_CALL_PARSE_ERROR)
    message: str
    start: int
    end: int
    attempted_name: Optional[str] = None
    kind: str = "malformed"


Segment = Union[
    ThinkSegment,
    ToolCallSegment,
    ToolResultSegment,
    FinalSegment,
    MalformedSegment,
]
