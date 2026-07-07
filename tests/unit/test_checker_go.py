"""Testes do backend Go do checker (PLAN.md seção 10, 19.1).

Pulados (não falham) se o binário `go` não estiver disponível no ambiente — o backend em si
já trata essa ausência como MISSING_DEPENDENCY estruturado (ver
test_missing_go_toolchain_reports_structured_error), então cobrimos os dois casos.
"""

import shutil

import pytest

from src.checker import check, errors

_HAS_GO = shutil.which("go") is not None
_requires_go = pytest.mark.skipif(not _HAS_GO, reason="toolchain 'go' não instalada neste ambiente")


@_requires_go
def test_compile_passes_for_valid_go_program():
    result = check(
        language="go",
        operation="compile",
        files=[
            {
                "path": "main.go",
                "content": 'package main\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\nfunc main() {}\n',
            }
        ],
    )
    assert result.passed, result.stderr
    assert result.metadata["language"] == "go"


@_requires_go
def test_compile_fails_for_invalid_go_program():
    result = check(
        language="go",
        operation="compile",
        files=[{"path": "main.go", "content": "package main\n\nfunc main() {\n"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.COMPILATION_ERROR


@_requires_go
def test_compile_and_test_passes_when_tests_pass():
    result = check(
        language="go",
        operation="compile_and_test",
        files=[
            {
                "path": "calc.go",
                "content": "package praxis_check\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
            },
            {
                "path": "calc_test.go",
                "content": (
                    "package praxis_check\n\n"
                    "import \"testing\"\n\n"
                    "func TestAdd(t *testing.T) {\n"
                    "\tif Add(2, 3) != 5 {\n"
                    "\t\tt.Fatal(\"esperado 5\")\n"
                    "\t}\n"
                    "}\n"
                ),
            },
        ],
    )
    assert result.passed, result.stderr


@_requires_go
def test_compile_and_test_reports_test_failure():
    result = check(
        language="go",
        operation="compile_and_test",
        files=[
            {
                "path": "calc.go",
                "content": "package praxis_check\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
            },
            {
                "path": "calc_test.go",
                "content": (
                    "package praxis_check\n\n"
                    "import \"testing\"\n\n"
                    "func TestAdd(t *testing.T) {\n"
                    "\tif Add(2, 3) != 999 {\n"
                    "\t\tt.Fatal(\"esperado 999\")\n"
                    "\t}\n"
                    "}\n"
                ),
            },
        ],
    )
    assert not result.passed
    assert result.errors[0].code == errors.TEST_FAILURE


@_requires_go
def test_run_executes_entrypoint_successfully():
    result = check(
        language="go",
        operation="run",
        files=[
            {
                "path": "main.go",
                "content": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello praxis")\n}\n',
            }
        ],
        entrypoint="main.go",
    )
    assert result.passed, result.stderr
    assert "hello praxis" in result.stdout


@_requires_go
def test_timeout_is_reported_as_timeout_code():
    result = check(
        language="go",
        operation="run",
        files=[
            {
                "path": "main.go",
                "content": (
                    'package main\n\nimport "time"\n\nfunc main() {\n\ttime.Sleep(5 * time.Second)\n}\n'
                ),
            }
        ],
        entrypoint="main.go",
        timeout_ms=200,
    )
    assert not result.passed
    assert result.errors[0].code == errors.TIMEOUT


@_requires_go
def test_lint_runs_go_vet():
    result = check(
        language="go",
        operation="lint",
        files=[{"path": "main.go", "content": "package main\n\nfunc main() {}\n"}],
    )
    assert result.passed, result.stderr


def test_missing_go_toolchain_reports_structured_error(monkeypatch):
    import src.checker.backends.go_backend as go_backend

    monkeypatch.setattr(go_backend, "_go_available", lambda: False)
    result = check(
        language="go",
        operation="compile",
        files=[{"path": "main.go", "content": "package main\n\nfunc main() {}\n"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.MISSING_DEPENDENCY
