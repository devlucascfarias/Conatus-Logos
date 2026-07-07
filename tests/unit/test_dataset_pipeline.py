"""Testes do pipeline de dataset (PLAN.md seção 9). Usa só exemplos Python — o backend Go
depende de uma toolchain que pode não estar instalada no ambiente de CI/dev (ver
tests/unit/test_checker_go.py); os testes daqui não devem depender disso."""

import json

import pytest

from src.dataset.pipeline import (
    classify_execution,
    content_hash,
    dedup,
    execute_when_possible,
    reject_or_accept,
    schema_validate,
    split_assign,
    stats_report,
    structural_validate,
)
from src.schemas import ToolRegistry


def _tool_call(name, args):
    return f'<tool_call name="{name}">{json.dumps(args)}</tool_call>'


def _tool_result(name, status, body):
    return f'<tool_result name="{name}" status="{status}">{json.dumps(body)}</tool_result>'


def _base_metadata(id_, task_type="single_tool_call", **overrides):
    metadata = {
        "id": id_,
        "domain": "test",
        "language": "python",
        "difficulty": "easy",
        "tools_used": ["checker"],
        "num_steps": 1,
        "task_type": task_type,
        "source": "curated_manual",
        "license": "synthetic-no-license-needed",
        "validation_status": "pending",
        "execution_performed": True,
    }
    metadata.update(overrides)
    return metadata


@pytest.fixture
def registry():
    return ToolRegistry.load()


# --- structural_validate ------------------------------------------------------------


def test_well_handled_malformed_call_passes_structural_validate():
    raw_text = (
        "<think>vou tentar</think>"
        '<tool_call name="read_file">{"path": broken}</tool_call>'
        + _tool_result("read_file", "error", {"code": "TOOL_CALL_PARSE_ERROR", "message": "..."})
        + "<final>corrigido</final>"
    )
    example = {"metadata": _base_metadata("t1"), "trajectory": {"raw_text": raw_text}}
    assert structural_validate(example) == []


def test_unhandled_malformed_call_fails_structural_validate():
    raw_text = (
        '<tool_call name="read_file">{"path": broken}</tool_call>'
        "<final>finjo que deu certo</final>"
    )
    example = {"metadata": _base_metadata("t2"), "trajectory": {"raw_text": raw_text}}
    problems = structural_validate(example)
    assert any("más-formação não tratada" in p for p in problems)


def test_missing_final_segment_fails_structural_validate():
    raw_text = "<think>pensando</think>"
    example = {"metadata": _base_metadata("t3"), "trajectory": {"raw_text": raw_text}}
    problems = structural_validate(example)
    assert any("não termina em <final>" in p for p in problems)


def test_invalid_metadata_reported_by_structural_validate():
    example = {
        "metadata": _base_metadata("t4", language="rust"),  # removida do escopo
        "trajectory": {"raw_text": "<final>ok</final>"},
    }
    problems = structural_validate(example)
    assert any("metadata:" in p for p in problems)


# --- schema_validate -----------------------------------------------------------------


def test_well_handled_unsupported_tool_passes_schema_validate(registry):
    raw_text = (
        '<tool_call name="apply_patch">{"path": "a.py", "diff": "..."}</tool_call>'
        + _tool_result("apply_patch", "error", {"code": "UNSUPPORTED_TOOL", "message": "..."})
        + "<final>ferramenta indisponível</final>"
    )
    example = {"metadata": _base_metadata("t5"), "trajectory": {"raw_text": raw_text}}
    assert schema_validate(example, registry) == []


def test_unhandled_invalid_schema_call_fails_schema_validate(registry):
    raw_text = (
        '<tool_call name="checker">{"language": "python"}</tool_call>'  # falta operation/files
        "<final>finjo que rodei</final>"
    )
    example = {"metadata": _base_metadata("t6"), "trajectory": {"raw_text": raw_text}}
    problems = schema_validate(example, registry)
    assert problems


# --- execute_when_possible + reject_or_accept (D11) -----------------------------------


def test_genuine_matching_checker_result_is_accepted():
    files = [{"path": "ok.py", "content": "def f():\n    return 1\n"}]
    raw_text = (
        _tool_call("checker", {"language": "python", "operation": "syntax_check", "files": files})
        + _tool_result("checker", "ok", {"passed": True, "errors": []})
        + "<final>sintaxe válida</final>"
    )
    example = {"metadata": _base_metadata("t7"), "trajectory": {"raw_text": raw_text}}
    comparisons = execute_when_possible(example)
    assert len(comparisons) == 1
    assert comparisons[0].matches

    status, reasons = reject_or_accept([], [], comparisons)
    assert status == "validated"
    assert reasons == []


def test_fabricated_checker_result_is_rejected():
    files = [{"path": "broken.py", "content": "def f(:\n    return 1\n"}]  # sintaxe inválida
    raw_text = (
        _tool_call("checker", {"language": "python", "operation": "syntax_check", "files": files})
        + _tool_result("checker", "ok", {"passed": True, "errors": []})  # MENTIRA
        + "<final>sintaxe válida (mentira)</final>"
    )
    example = {"metadata": _base_metadata("t8"), "trajectory": {"raw_text": raw_text}}
    comparisons = execute_when_possible(example)
    assert not comparisons[0].matches

    status, reasons = reject_or_accept([], [], comparisons)
    assert status == "rejected"
    assert any("TOOL_RESULT_FABRICATION" in r for r in reasons)


def test_classify_execution_tested_beats_static_only():
    files = [{"path": "a.py", "content": "def f():\n    return 1\n"}]
    example_run = {"trajectory": {"raw_text": _tool_call("checker", {"language": "python", "operation": "compile_and_test", "files": files})}}
    comparisons = [
        type("C", (), {"operation": "compile_and_test"})()  # objeto simples só com .operation
    ]
    assert classify_execution(example_run, comparisons) == "tested"


def test_classify_execution_no_checker_calls_but_has_code_is_static_only():
    example = {"trajectory": {"raw_text": _tool_call("write_file", {"path": "a.py", "content": "x = 1"})}}
    assert classify_execution(example, []) == "static_only"


def test_classify_execution_no_code_at_all_is_non_executable():
    example = {"trajectory": {"raw_text": "<final>resposta direta</final>"}}
    assert classify_execution(example, []) == "non_executable_external_dep"


# --- dedup / split_assign / stats_report ----------------------------------------------


def test_dedup_removes_exact_duplicate_by_content_hash():
    example_a = {"metadata": _base_metadata("dup1"), "trajectory": {"raw_text": "<final>x</final>"}}
    example_b = {"metadata": _base_metadata("dup2"), "trajectory": {"raw_text": "<final>x</final>"}}
    unique, duplicates = dedup([example_a, example_b])
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert content_hash(example_a) == content_hash(example_b)


def test_split_assign_groups_by_task_group():
    ex1 = {"metadata": _base_metadata("g1", task_group="grupo-x", split="validation")}
    ex2 = {"metadata": _base_metadata("g2", task_group="grupo-x", split="train")}  # ignorado
    splits = split_assign([ex1, ex2])
    assert ex1 in splits["validation"]
    assert ex2 in splits["validation"]  # mesmo grupo, mesmo split que a primeira ocorrência
    assert ex2 not in splits["train"]


def test_stats_report_counts_by_language_and_task_type():
    ex1 = {"metadata": _base_metadata("s1", language="python", task_type="debugging")}
    ex2 = {"metadata": _base_metadata("s2", language="go", task_type="refactor")}
    stats = stats_report([ex1, ex2])
    assert stats["total"] == 2
    assert stats["by_language"] == {"python": 1, "go": 1}
    assert stats["by_task_type"] == {"debugging": 1, "refactor": 1}
