"""Parser da gramática canônica de trajetória (PLAN.md, seção 3.2).

    turn         := segment+
    segment      := think_seg | tool_call_seg | tool_result_seg | final_seg
    think_seg    := "<think>" TEXT "</think>"
    tool_call_seg   := '<tool_call name="' TOOL_NAME '">' JSON "</tool_call>"
    tool_result_seg := '<tool_result name="' TOOL_NAME '" status="ok|error">' JSON "</tool_result>"
    final_seg    := "<final>" TEXT "</final>"

Propriedades exigidas pela seção 3.3 e verificadas pelos testes em tests/unit/test_grammar.py:
- Um JSON malformado invalida SÓ o segmento (vira MalformedSegment), não o turno inteiro.
- Uma tag não fechada é detectada e para o parsing ali (o resto pode ainda estar chegando
  via streaming), sem lançar exceção.
- Nenhum `<tool_result>` é tratado como "confiável" só por ter sido reconhecido
  sintaticamente — a confiança vem de QUEM o escreveu (harness), não da gramática; por isso
  `contains_fabricated_tool_result` existe como checagem separada sobre texto bruto gerado
  pelo modelo, antes de qualquer injeção do harness.
"""

from __future__ import annotations

import json
import re

from src.checker import errors as error_codes

from .segments import (
    FinalSegment,
    MalformedSegment,
    Segment,
    ThinkSegment,
    ToolCallSegment,
    ToolResultSegment,
)

_ANY_OPEN = re.compile(
    r"<think>"
    r"|<final>"
    r'|<tool_call name="(?P<tc_name>[^"]*)">'
    r'|<tool_result name="(?P<tr_name>[^"]*)" status="(?P<tr_status>ok|error)">'
)

_CLOSE_TAG = {
    "think": "</think>",
    "final": "</final>",
    "tool_call": "</tool_call>",
    "tool_result": "</tool_result>",
}

# Usadas por harness/checker (seção 10.3), não pelo parser estrutural em si.
_TOOL_RESULT_OPEN_RE = re.compile(r"<tool_result\b")
_INTERNAL_TAG_RE = re.compile(r"</?think>|</?tool_call\b[^>]*>|</?tool_result\b[^>]*>")


def _kind_of(match: re.Match[str]) -> tuple[str, str | None, str | None]:
    if match.group(0) == "<think>":
        return "think", None, None
    if match.group(0) == "<final>":
        return "final", None, None
    if match.group("tc_name") is not None:
        return "tool_call", match.group("tc_name"), None
    return "tool_result", match.group("tr_name"), match.group("tr_status")


def parse_segments(text: str) -> list[Segment]:
    """Converte um texto de trajetória inteiro numa lista ordenada de segmentos.

    Nunca lança exceção por conta de gramática/JSON inválidos — más-formações viram
    `MalformedSegment` na posição em que ocorrem, e o parsing continua depois delas
    (exceto no caso de tag não fechada, que interrompe o parsing porque não há um ponto
    de retomada sintaticamente seguro).
    """
    segments: list[Segment] = []
    pos = 0
    n = len(text)

    while pos < n:
        match = _ANY_OPEN.search(text, pos)
        if match is None:
            trailing = text[pos:]
            if trailing.strip():
                segments.append(
                    MalformedSegment(
                        reason=error_codes.UNRECOGNIZED_CONTENT,
                        message="texto fora de qualquer tag reconhecida",
                        start=pos,
                        end=n,
                    )
                )
            break

        leading = text[pos : match.start()]
        if leading.strip():
            segments.append(
                MalformedSegment(
                    reason=error_codes.UNRECOGNIZED_CONTENT,
                    message="texto fora de qualquer tag reconhecida",
                    start=pos,
                    end=match.start(),
                )
            )

        kind, name, status = _kind_of(match)
        open_end = match.end()
        close_tag = _CLOSE_TAG[kind]
        close_idx = text.find(close_tag, open_end)

        if close_idx == -1:
            segments.append(
                MalformedSegment(
                    reason=error_codes.UNTERMINATED_TAG,
                    message=f"tag <{kind}> aberta sem fechamento correspondente",
                    attempted_name=name,
                    start=match.start(),
                    end=n,
                )
            )
            break

        body = text[open_end:close_idx]
        seg_end = close_idx + len(close_tag)

        if kind == "think":
            segments.append(ThinkSegment(text=body, start=match.start(), end=seg_end))
        elif kind == "final":
            segments.append(FinalSegment(text=body, start=match.start(), end=seg_end))
        elif kind == "tool_call":
            segments.append(_parse_json_segment_as_tool_call(body, name, match.start(), seg_end))
        else:  # tool_result
            segments.append(
                _parse_json_segment_as_tool_result(body, name, status, match.start(), seg_end)
            )

        pos = seg_end

    return segments


def _parse_json_segment_as_tool_call(
    body: str, name: str | None, start: int, end: int
) -> Segment:
    try:
        args = json.loads(body)
        if not isinstance(args, dict):
            raise ValueError("corpo de <tool_call> deve ser um objeto JSON")
    except (json.JSONDecodeError, ValueError) as exc:
        return MalformedSegment(
            reason=error_codes.TOOL_CALL_PARSE_ERROR,
            message=str(exc),
            attempted_name=name,
            start=start,
            end=end,
        )
    return ToolCallSegment(name=name or "", args=args, raw_body=body, start=start, end=end)


def _parse_json_segment_as_tool_result(
    body: str, name: str | None, status: str | None, start: int, end: int
) -> Segment:
    try:
        result_body = json.loads(body)
        if not isinstance(result_body, dict):
            raise ValueError("corpo de <tool_result> deve ser um objeto JSON")
    except (json.JSONDecodeError, ValueError) as exc:
        return MalformedSegment(
            reason=error_codes.TOOL_CALL_PARSE_ERROR,
            message=str(exc),
            attempted_name=name,
            start=start,
            end=end,
        )
    return ToolResultSegment(
        name=name or "",
        status=status or "ok",
        body=result_body,
        raw_body=body,
        start=start,
        end=end,
    )


def parse_last_segment(text: str) -> Segment | None:
    """Retorna o último segmento reconhecido no texto (usado pelo loop do agente, seção 6,
    logo após cada passada de geração do modelo). None se nenhum segmento foi encontrado."""
    segments = parse_segments(text)
    return segments[-1] if segments else None


def is_complete_turn(segments: list[Segment]) -> bool:
    """Um turno só é aceito como completo quando termina em final_seg (seção 3.2)."""
    return bool(segments) and segments[-1].kind == "final"


def contains_fabricated_tool_result(model_generated_text: str) -> bool:
    """True se texto GERADO PELO MODELO (antes de qualquer injeção do harness) contém uma
    tag <tool_result>. Isso é sempre fabricação (TOOL_RESULT_FABRICATION, seção 10.3): o
    parser nunca deveria ter visto essa tag vinda do modelo, já que só o harness a escreve."""
    return bool(_TOOL_RESULT_OPEN_RE.search(model_generated_text))


def final_segment_leaks_internal_tags(final_text: str) -> bool:
    """Heurística determinística de INTERNAL_REASONING_LEAK (seção 10.3): tags internas
    (<think>, <tool_call>, <tool_result>) vazando para dentro do conteúdo público de <final>."""
    return bool(_INTERNAL_TAG_RE.search(final_text))
