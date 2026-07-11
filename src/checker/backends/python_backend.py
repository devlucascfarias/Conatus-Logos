"""Backend do checker para Python (PLAN.md, seção 10.4): `py_compile` (sintaxe) + `pytest`
(teste). Registrado em `src.checker.core` sob a chave "python".
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files, run_command

_TRACEBACK_LOCATION_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')

# D-checker-pytest-plugin-warning-leak (achado real, não a mesma causa de D-checker-path-leak):
# avisos de depreciação de PLUGINS instalados na máquina de dev (ex.: pytest-asyncio, que nem é
# dependência declarada do projeto) vêm no formato padrão do módulo `warnings` do Python —
# "<caminho absoluto do pacote instalado>:<linha>: <NomeDoWarning>: <mensagem>\n\n  <linha
# fonte>\n" — e esse caminho aponta pro site-packages da máquina que rodou o checker (vaza nome
# de usuário/SO), não pro código do usuário. Confirmado real rodando `compile_and_test` de
# verdade num ambiente com pytest-asyncio instalado: aparece tanto quando os testes passam
# quanto quando falham, sempre em `stderr`, nunca junto do relatório de verdade do pytest (que
# fica inteiro em `stdout`) — então removê-lo nunca perde diagnóstico real, só ruído do
# ambiente local.
_PYTEST_WARNING_BLOCK_RE = re.compile(r"^\S+\.py:\d+: \w+Warning: .*?\n(?:.*\n)*?\n(?:  .*\n)?", re.MULTILINE)


def _strip_plugin_warnings(text: str) -> str:
    return _PYTEST_WARNING_BLOCK_RE.sub("", text)


def _extract_location(stderr: str, base_dir: Path) -> tuple[Optional[str], Optional[int]]:
    matches = list(_TRACEBACK_LOCATION_RE.finditer(stderr))
    if not matches:
        return None, None
    last = matches[-1]
    file_path = last.group("file")
    try:
        rel = str(Path(file_path).resolve().relative_to(base_dir.resolve()))
    except ValueError:
        rel = file_path
    return rel, int(last.group("line"))


def _syntax_check(base_dir: Path, files: list[CheckFile], timeout_ms: int) -> CheckResult:
    found_errors: list[CheckError] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    total_duration_ms = 0

    py_files = [f for f in files if f.path.endswith(".py")]
    if not py_files:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.UNSUPPORTED_LANGUAGE, message="nenhum arquivo .py em 'files'")],
            stdout="",
            stderr="",
            metadata={"language": "python", "duration_ms": 0},
        )

    for f in py_files:
        result = run_command([sys.executable, "-m", "py_compile", f.path], base_dir, timeout_ms)
        total_duration_ms += result.duration_ms
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        if result.timed_out:
            found_errors.append(
                CheckError(code=errors.TIMEOUT, message=f"timeout ao compilar {f.path}", file=f.path)
            )
            continue
        if result.returncode != 0:
            file_, line_ = _extract_location(result.stderr, base_dir)
            found_errors.append(
                CheckError(
                    code=errors.SYNTAX_ERROR,
                    message=result.stderr.strip() or "erro de sintaxe",
                    file=file_ or f.path,
                    line=line_,
                )
            )

    return CheckResult(
        passed=not found_errors,
        errors=found_errors,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
        metadata={"language": "python", "duration_ms": total_duration_ms},
    )


def _run_pytest(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command([sys.executable, "-m", "pytest", "-q", "--no-header"], base_dir, timeout_ms)
    metadata = {"language": "python", "duration_ms": result.duration_ms}
    stderr = _strip_plugin_warnings(result.stderr)

    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao rodar pytest")],
            stdout=result.stdout,
            stderr=stderr,
            metadata=metadata,
        )

    combined = result.stdout + stderr
    passed = result.returncode == 0
    found_errors: list[CheckError] = []
    if not passed:
        if "ModuleNotFoundError" in combined or "No module named" in combined:
            code = errors.MISSING_DEPENDENCY
        elif result.returncode == 1:
            code = errors.TEST_FAILURE
        else:
            code = errors.RUNTIME_ERROR
        found_errors.append(CheckError(code=code, message=combined.strip()[-2000:]))

    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=stderr, metadata=metadata
    )


def _run_entrypoint(base_dir: Path, entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not entrypoint:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INCOMPLETE_SOLUTION, message="operation=run requer 'entrypoint'")],
            stdout="",
            stderr="",
            metadata={"language": "python", "duration_ms": 0},
        )

    result = run_command([sys.executable, entrypoint], base_dir, timeout_ms)
    metadata = {"language": "python", "duration_ms": result.duration_ms}

    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message=f"timeout ao executar {entrypoint}")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )

    passed = result.returncode == 0
    found_errors: list[CheckError] = []
    if not passed:
        file_, line_ = _extract_location(result.stderr, base_dir)
        found_errors.append(
            CheckError(
                code=errors.RUNTIME_ERROR,
                message=result.stderr.strip()[-2000:] or "erro em tempo de execução",
                file=file_ or entrypoint,
                line=line_,
            )
        )

    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata
    )


def _lint(base_dir: Path, timeout_ms: int) -> CheckResult:
    if shutil.which("ruff") is None:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.MISSING_DEPENDENCY, message="ruff não disponível no ambiente")],
            stdout="",
            stderr="",
            metadata={"language": "python", "duration_ms": 0},
        )

    result = run_command(["ruff", "check", "."], base_dir, timeout_ms)
    passed = result.returncode == 0
    found_errors = (
        [] if passed else [CheckError(code=errors.SYNTAX_ERROR, message=result.stdout.strip()[-2000:])]
    )
    return CheckResult(
        passed=passed,
        errors=found_errors,
        stdout=result.stdout,
        stderr=result.stderr,
        metadata={"language": "python", "duration_ms": result.duration_ms},
    )


def check_python(
    operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int
) -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="praxis_checker_py_") as tmp:
        base_dir = Path(tmp)
        materialize_files(base_dir, files)

        if operation in ("syntax_check", "compile"):
            return _syntax_check(base_dir, files, timeout_ms)
        if operation == "compile_and_test":
            syntax_result = _syntax_check(base_dir, files, timeout_ms)
            if not syntax_result.passed:
                return syntax_result
            return _run_pytest(base_dir, timeout_ms)
        if operation == "run":
            return _run_entrypoint(base_dir, entrypoint, timeout_ms)
        if operation == "lint":
            return _lint(base_dir, timeout_ms)

        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
            stdout="",
            stderr="",
            metadata={"language": "python", "duration_ms": 0},
        )


register_backend("python", check_python)
