"""Testes de validação de schema por ferramenta (PLAN.md seção 4 e 19.1)."""

import pytest

from src.schemas import ToolRegistry


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.load()


def test_all_mvp_tools_enabled(registry: ToolRegistry):
    enabled_names = {spec.name for spec in registry.all_enabled()}
    assert {"checker", "read_file", "write_file", "list_files", "shell"} <= enabled_names


def test_phase2_tools_disabled_per_D5(registry: ToolRegistry):
    assert registry.get("apply_patch") is None
    assert registry.get("search_code") is None
    assert registry.get("git_diff") is None


def test_unknown_tool_returns_none(registry: ToolRegistry):
    assert registry.get("does_not_exist") is None


# --- checker ---------------------------------------------------------------


def test_checker_valid_args_pass(registry: ToolRegistry):
    args = {
        "language": "python",
        "operation": "compile_and_test",
        "files": [{"path": "main.py", "content": "print(1)"}],
    }
    assert registry.validate_args("checker", args) == []


def test_checker_missing_required_field_fails(registry: ToolRegistry):
    args = {"language": "python", "operation": "compile_and_test"}  # falta 'files'
    errors = registry.validate_args("checker", args)
    assert errors


def test_checker_invalid_language_fails(registry: ToolRegistry):
    args = {
        "language": "rust",  # removida do escopo (seção 18)
        "operation": "compile",
        "files": [{"path": "main.rs", "content": "fn main() {}"}],
    }
    errors = registry.validate_args("checker", args)
    assert errors


def test_checker_invalid_operation_fails(registry: ToolRegistry):
    args = {
        "language": "python",
        "operation": "not_a_real_operation",
        "files": [{"path": "main.py", "content": "print(1)"}],
    }
    assert registry.validate_args("checker", args)


def test_checker_timeout_out_of_range_fails(registry: ToolRegistry):
    args = {
        "language": "go",
        "operation": "compile",
        "files": [{"path": "main.go", "content": "package main"}],
        "timeout_ms": 999999,
    }
    assert registry.validate_args("checker", args)


# --- read_file / write_file / list_files / shell ----------------------------


def test_read_file_requires_path(registry: ToolRegistry):
    assert registry.validate_args("read_file", {}) != []
    assert registry.validate_args("read_file", {"path": "src/main.py"}) == []


def test_read_file_line_range_valid(registry: ToolRegistry):
    args = {"path": "src/main.py", "start_line": 1, "end_line": 10}
    assert registry.validate_args("read_file", args) == []


def test_write_file_requires_path_and_content(registry: ToolRegistry):
    assert registry.validate_args("write_file", {"path": "a.py"}) != []
    assert registry.validate_args("write_file", {"path": "a.py", "content": "x = 1"}) == []


def test_write_file_invalid_mode_fails(registry: ToolRegistry):
    args = {"path": "a.py", "content": "x = 1", "mode": "delete"}
    assert registry.validate_args("write_file", args)


def test_list_files_no_required_fields(registry: ToolRegistry):
    assert registry.validate_args("list_files", {}) == []
    assert registry.validate_args("list_files", {"path": "src", "max_depth": 2}) == []


def test_shell_requires_command(registry: ToolRegistry):
    assert registry.validate_args("shell", {}) != []
    assert registry.validate_args("shell", {"command": "pytest -q"}) == []


def test_shell_timeout_bounds(registry: ToolRegistry):
    assert registry.validate_args("shell", {"command": "ls", "timeout_ms": 50}) != []
    assert registry.validate_args("shell", {"command": "ls", "timeout_ms": 5000}) == []
