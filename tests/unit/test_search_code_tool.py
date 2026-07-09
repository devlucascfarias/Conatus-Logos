"""Testes do executor `search_code` (PLAN.md seção 4.6, reativada em D-frontend-pivot-model-swap
— docs/plan_frontend_specialization_wave3.md seção 4, dimensão "reuse-before-create" do think
técnico de frontend)."""

import pytest

from src.security import SandboxContext, SandboxPolicy
from src.tools.executors import search_code_tool


@pytest.fixture
def sandbox():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    ctx = SandboxContext(policy=policy)
    (ctx.workspace / "src").mkdir()
    (ctx.workspace / "src" / "Button.tsx").write_text(
        "export function Button() {\n  return <button className=\"btn-primary\">ok</button>\n}\n",
        encoding="utf-8",
    )
    (ctx.workspace / "src" / "Card.tsx").write_text(
        "export function Card() {\n  return <div className=\"card\">x</div>\n}\n",
        encoding="utf-8",
    )
    yield ctx
    ctx.cleanup()


def test_plain_text_match_finds_hits_across_files(sandbox):
    result = search_code_tool.execute({"pattern": "export function"}, sandbox)
    assert result.passed
    paths = {m["path"] for m in result.data["matches"]}
    assert paths == {"src/Button.tsx", "src/Card.tsx"}


def test_match_reports_correct_line_number(sandbox):
    result = search_code_tool.execute({"pattern": "btn-primary"}, sandbox)
    assert result.passed
    assert len(result.data["matches"]) == 1
    assert result.data["matches"][0]["line"] == 2
    assert result.data["matches"][0]["path"] == "src/Button.tsx"


def test_no_match_returns_empty_list_not_error(sandbox):
    result = search_code_tool.execute({"pattern": "does-not-exist-anywhere"}, sandbox)
    assert result.passed
    assert result.data["matches"] == []


def test_regex_mode_matches_pattern(sandbox):
    result = search_code_tool.execute({"pattern": r"class Name=\"(btn|card)", "regex": True}, sandbox)
    assert result.passed
    assert len(result.data["matches"]) == 0  # className != class Name, garante que regex é real


def test_regex_mode_matches_real_pattern(sandbox):
    result = search_code_tool.execute({"pattern": r"className=\"(btn|card)", "regex": True}, sandbox)
    assert result.passed
    assert len(result.data["matches"]) == 2


def test_invalid_regex_returns_structured_error(sandbox):
    result = search_code_tool.execute({"pattern": "(unclosed", "regex": True}, sandbox)
    assert not result.passed
    assert result.error_code == "TOOL_ARGUMENT_SCHEMA_ERROR"


def test_search_scoped_to_path_argument(sandbox):
    result = search_code_tool.execute({"pattern": "export function", "path": "src/Card.tsx"}, sandbox)
    assert result.passed
    assert len(result.data["matches"]) == 1
    assert result.data["matches"][0]["path"] == "src/Card.tsx"


def test_path_escaping_workspace_is_rejected(sandbox):
    result = search_code_tool.execute({"pattern": "x", "path": "../../etc"}, sandbox)
    assert not result.passed
    assert result.error_code == "FILE_NOT_FOUND"


def test_nonexistent_path_returns_structured_error(sandbox):
    result = search_code_tool.execute({"pattern": "x", "path": "does/not/exist"}, sandbox)
    assert not result.passed
    assert result.error_code == "FILE_NOT_FOUND"


def test_tool_is_enabled_and_wired_in_default_registry(sandbox):
    from src.tools.registry import ToolExecutorRegistry

    registry = ToolExecutorRegistry()
    spec = registry.get_spec("search_code")
    assert spec is not None
    assert spec.enabled is True
    result = registry.execute("search_code", {"pattern": "export function"}, sandbox)
    assert result.passed
    assert len(result.data["matches"]) == 2
