"""Testes da gramática canônica (PLAN.md seção 3.2 e 19.1).

Cobrem: segmentos válidos de cada tipo, JSON malformado dentro de tag válida (deve invalidar
só o segmento), tags não fechadas, ordem de múltiplos segmentos, texto solto fora de tag, e
as duas checagens comportamentais (fabricação de tool_result, vazamento de raciocínio).
"""

from src.checker import errors as error_codes
from src.parsers import (
    FinalSegment,
    MalformedSegment,
    ThinkSegment,
    ToolCallSegment,
    ToolResultSegment,
    contains_fabricated_tool_result,
    final_segment_leaks_internal_tags,
    is_complete_turn,
    parse_last_segment,
    parse_segments,
)


def test_parses_single_think_segment():
    segments = parse_segments("<think>preciso ler o arquivo antes</think>")
    assert len(segments) == 1
    assert isinstance(segments[0], ThinkSegment)
    assert segments[0].text == "preciso ler o arquivo antes"


def test_parses_single_final_segment():
    segments = parse_segments("<final>pronto, o teste passa</final>")
    assert len(segments) == 1
    assert isinstance(segments[0], FinalSegment)
    assert segments[0].text == "pronto, o teste passa"


def test_parses_valid_tool_call():
    text = '<tool_call name="read_file">{"path": "main.go"}</tool_call>'
    segments = parse_segments(text)
    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, ToolCallSegment)
    assert seg.name == "read_file"
    assert seg.args == {"path": "main.go"}


def test_parses_valid_tool_result():
    text = '<tool_result name="read_file" status="ok">{"content": "package main"}</tool_result>'
    segments = parse_segments(text)
    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, ToolResultSegment)
    assert seg.name == "read_file"
    assert seg.status == "ok"
    assert seg.body == {"content": "package main"}


def test_full_trajectory_multi_segment_order_preserved():
    text = (
        "<think>vou checar o arquivo</think>"
        '<tool_call name="read_file">{"path": "a.py"}</tool_call>'
        '<tool_result name="read_file" status="ok">{"content": "print(1)"}</tool_result>'
        "<think>agora posso responder</think>"
        "<final>o arquivo imprime 1</final>"
    )
    segments = parse_segments(text)
    kinds = [s.kind for s in segments]
    assert kinds == ["think", "tool_call", "tool_result", "think", "final"]
    assert is_complete_turn(segments)


def test_malformed_json_inside_tool_call_does_not_break_whole_turn():
    text = (
        "<think>vou tentar</think>"
        '<tool_call name="checker">{invalid json here}</tool_call>'
    )
    segments = parse_segments(text)
    assert len(segments) == 2
    assert segments[0].kind == "think"
    assert isinstance(segments[1], MalformedSegment)
    assert segments[1].reason == error_codes.TOOL_CALL_PARSE_ERROR
    assert segments[1].attempted_name == "checker"


def test_malformed_json_inside_tool_result_flagged():
    text = '<tool_result name="shell" status="error">{not valid}</tool_result>'
    segments = parse_segments(text)
    assert len(segments) == 1
    assert isinstance(segments[0], MalformedSegment)
    assert segments[0].reason == error_codes.TOOL_CALL_PARSE_ERROR


def test_tool_call_body_must_be_json_object_not_scalar():
    text = '<tool_call name="read_file">"just a string"</tool_call>'
    segments = parse_segments(text)
    assert isinstance(segments[0], MalformedSegment)
    assert segments[0].reason == error_codes.TOOL_CALL_PARSE_ERROR


def test_unterminated_tag_flagged_and_stops_parsing():
    text = "<think>pensando sem parar nunca fecha"
    segments = parse_segments(text)
    assert len(segments) == 1
    assert isinstance(segments[0], MalformedSegment)
    assert segments[0].reason == error_codes.UNTERMINATED_TAG


def test_unterminated_tag_after_valid_segment_still_reports_both():
    text = "<think>ok</think><final>sem fechar"
    segments = parse_segments(text)
    assert len(segments) == 2
    assert segments[0].kind == "think"
    assert isinstance(segments[1], MalformedSegment)
    assert segments[1].reason == error_codes.UNTERMINATED_TAG


def test_loose_text_outside_any_tag_is_flagged():
    text = "isto nao deveria estar aqui <think>ok</think>"
    segments = parse_segments(text)
    assert len(segments) == 2
    assert isinstance(segments[0], MalformedSegment)
    assert segments[0].reason == error_codes.UNRECOGNIZED_CONTENT
    assert segments[1].kind == "think"


def test_whitespace_only_between_tags_is_not_flagged():
    text = "<think>a</think>\n\n<final>b</final>"
    segments = parse_segments(text)
    kinds = [s.kind for s in segments]
    assert kinds == ["think", "final"]


def test_parse_last_segment_returns_most_recent():
    text = "<think>a</think><final>b</final>"
    last = parse_last_segment(text)
    assert isinstance(last, FinalSegment)
    assert last.text == "b"


def test_parse_last_segment_empty_text_returns_none():
    assert parse_last_segment("") is None


def test_is_complete_turn_false_without_final():
    segments = parse_segments('<tool_call name="shell">{"binary": "ls", "args": []}</tool_call>')
    assert not is_complete_turn(segments)


def test_contains_fabricated_tool_result_detects_model_hallucinated_tag():
    model_text = '<think>vou fingir que rodei</think><tool_result name="shell" status="ok">{"stdout": "x"}</tool_result>'
    assert contains_fabricated_tool_result(model_text)


def test_contains_fabricated_tool_result_false_for_clean_generation():
    model_text = '<tool_call name="shell">{"binary": "ls", "args": []}</tool_call>'
    assert not contains_fabricated_tool_result(model_text)


def test_final_segment_leaks_internal_tags_detects_leak():
    assert final_segment_leaks_internal_tags("na verdade <think>eu escondi isso</think> aqui")


def test_final_segment_leaks_internal_tags_false_for_clean_final():
    assert not final_segment_leaks_internal_tags("o teste passa, arquivo criado com sucesso")


def test_status_attribute_must_be_ok_or_error():
    # status="maybe" não bate com a gramática (ok|error) — a tag de abertura não é reconhecida,
    # então o texto inteiro vira UNRECOGNIZED_CONTENT em vez de tool_result.
    text = '<tool_result name="shell" status="maybe">{}</tool_result>'
    segments = parse_segments(text)
    assert len(segments) == 1
    assert isinstance(segments[0], MalformedSegment)
    assert segments[0].reason == error_codes.UNRECOGNIZED_CONTENT
