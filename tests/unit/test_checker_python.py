"""Testes do backend Python do checker (PLAN.md seção 10, 19.1)."""

from src.checker import check, errors


def test_syntax_check_passes_for_valid_code():
    result = check(
        language="python",
        operation="syntax_check",
        files=[{"path": "main.py", "content": "def add(a, b):\n    return a + b\n"}],
    )
    assert result.passed
    assert result.errors == []
    assert result.metadata["language"] == "python"


def test_syntax_check_fails_for_invalid_code():
    result = check(
        language="python",
        operation="syntax_check",
        files=[{"path": "main.py", "content": "def add(a, b:\n    return a + b\n"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.SYNTAX_ERROR
    assert result.errors[0].file is not None


def test_compile_and_test_passes_when_tests_pass():
    result = check(
        language="python",
        operation="compile_and_test",
        files=[
            {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
            {
                "path": "test_calc.py",
                "content": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            },
        ],
    )
    assert result.passed, result.stderr


def test_compile_and_test_reports_test_failure_not_compilation_error():
    result = check(
        language="python",
        operation="compile_and_test",
        files=[
            {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
            {
                "path": "test_calc.py",
                "content": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 999\n",
            },
        ],
    )
    assert not result.passed
    assert result.errors[0].code == errors.TEST_FAILURE


def test_compile_and_test_short_circuits_on_syntax_error():
    # Se o código nem compila, o checker não deve tentar rodar pytest sobre ele — o erro
    # relatado tem que ser SYNTAX_ERROR, não uma falha de coleta do pytest.
    result = check(
        language="python",
        operation="compile_and_test",
        files=[{"path": "calc.py", "content": "def add(a, b:\n    return a + b\n"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.SYNTAX_ERROR


def test_run_executes_entrypoint_successfully():
    result = check(
        language="python",
        operation="run",
        files=[{"path": "main.py", "content": "print('hello praxis')\n"}],
        entrypoint="main.py",
    )
    assert result.passed
    assert "hello praxis" in result.stdout


def test_run_reports_runtime_error():
    result = check(
        language="python",
        operation="run",
        files=[{"path": "main.py", "content": "raise ValueError('boom')\n"}],
        entrypoint="main.py",
    )
    assert not result.passed
    assert result.errors[0].code == errors.RUNTIME_ERROR


def test_run_without_entrypoint_is_incomplete_solution():
    result = check(language="python", operation="run", files=[{"path": "main.py", "content": "pass"}])
    assert not result.passed
    assert result.errors[0].code == errors.INCOMPLETE_SOLUTION


def test_timeout_is_reported_as_timeout_code():
    result = check(
        language="python",
        operation="run",
        files=[{"path": "main.py", "content": "import time\ntime.sleep(5)\n"}],
        entrypoint="main.py",
        timeout_ms=200,
    )
    assert not result.passed
    assert result.errors[0].code == errors.TIMEOUT


def test_missing_dependency_reported_correctly():
    result = check(
        language="python",
        operation="compile_and_test",
        files=[
            {
                "path": "test_missing.py",
                "content": "import totally_nonexistent_package_xyz\n\ndef test_x():\n    assert True\n",
            }
        ],
    )
    assert not result.passed
    assert result.errors[0].code == errors.MISSING_DEPENDENCY


def test_unsupported_language_returns_structured_error_not_exception():
    result = check(language="rust", operation="compile", files=[{"path": "main.rs", "content": "fn main() {}"}])
    assert not result.passed
    assert result.errors[0].code == errors.UNSUPPORTED_LANGUAGE
