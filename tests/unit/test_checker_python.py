"""Testes do backend Python do checker (PLAN.md seção 10, 19.1)."""

from src.checker import check, errors
from src.checker.backends.python_backend import _strip_plugin_warnings


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


def test_strip_plugin_warnings_removes_deprecation_warning_block():
    """D-checker-pytest-plugin-warning-leak (achado real, diferente de D-checker-path-leak):
    ao regenerar exemplos de dataset de verdade, um aviso de depreciação do plugin
    pytest-asyncio (instalado só nesta máquina de dev, não é dependência do projeto) vazou o
    caminho absoluto do pacote instalado (`.../site-packages/pytest_asyncio/plugin.py`) — texto
    real capturado do `stderr` do pytest, reproduzido aqui como fixture pra não depender de
    qual plugin está instalado em cada máquina/CI que rodar este teste."""
    stderr = (
        'C:\\Users\\devuser\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages'
        '\\pytest_asyncio\\plugin.py:247: PytestDeprecationWarning: The configuration option '
        '"asyncio_default_fixture_loop_scope" is unset.\n'
        "The event loop scope for asynchronous fixtures will default to the fixture caching "
        "scope.\n"
        "\n"
        "  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))\n"
    )
    cleaned = _strip_plugin_warnings(stderr)
    assert cleaned == ""


def test_strip_plugin_warnings_keeps_unrelated_text_untouched():
    text = "1 passed in 0.02s\n"
    assert _strip_plugin_warnings(text) == text


def test_run_error_never_leaks_absolute_temp_dir_path():
    """D-checker-path-leak (achado real, docs/PLAN.md): 20 exemplos do dataset (`aug-debug-*`)
    tinham o caminho absoluto da máquina de geração (`C:\\Users\\<usuário>\\AppData\\Local\\Temp\\
    praxis_checker_py_<hash>\\main.py`) vazado dentro do traceback devolvido pelo checker,
    porque o backend nunca sanitizava stdout/stderr antes de retornar. O `<final>` do exemplo
    reproduzia esse traceback verbatim — ou seja, o modelo aprenderia a ecoar o path local de
    quem gerou o dataset. `errors[0].message`/`stderr` precisam conter só o path RELATIVO
    (`main.py`), nunca o diretório temporário completo."""
    result = check(
        language="python",
        operation="run",
        files=[{"path": "main.py", "content": "x = 1.0\ny = 0.0\nprint(x / y)\n"}],
        entrypoint="main.py",
    )
    assert not result.passed
    assert 'File "main.py"' in result.errors[0].message
    # "praxis_checker_py_" é o prefixo literal do próprio backend (src/checker/backends/
    # python_backend.py) para o TemporaryDirectory — presente em qualquer SO, ao contrário de
    # nomes de pasta específicos de plataforma (AppData é só Windows).
    assert "praxis_checker_py_" not in result.errors[0].message
    assert "praxis_checker_py_" not in result.stderr


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
