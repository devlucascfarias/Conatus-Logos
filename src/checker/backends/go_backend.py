"""Backend do checker para Go (PLAN.md, seção 10.4): `go build` + `go vet` + `go test`.

Se o binário `go` não estiver disponível no ambiente, retorna MISSING_DEPENDENCY de forma
estruturada em vez de lançar exceção — permite que o resto do harness/testes funcione mesmo
em ambientes sem toolchain Go instalada.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files, run_command

_GO_ERROR_RE = re.compile(r"(?P<file>[^\s:]+\.go):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.+)")
_GO_MOD_TEMPLATE = "module praxis_check\n\ngo 1.21\n"


def _go_available() -> bool:
    return shutil.which("go") is not None


def _unavailable_result() -> CheckResult:
    return CheckResult(
        passed=False,
        errors=[
            CheckError(code=errors.MISSING_DEPENDENCY, message="toolchain 'go' não disponível no ambiente")
        ],
        stdout="",
        stderr="",
        metadata={"language": "go", "duration_ms": 0},
    )


def _extract_errors(stderr: str) -> list[CheckError]:
    found = [
        CheckError(
            code=errors.COMPILATION_ERROR,
            message=m.group("msg").strip(),
            file=m.group("file"),
            line=int(m.group("line")),
        )
        for m in _GO_ERROR_RE.finditer(stderr)
    ]
    if not found and stderr.strip():
        found.append(CheckError(code=errors.COMPILATION_ERROR, message=stderr.strip()[-2000:]))
    return found


def _ensure_go_mod(base_dir: Path) -> None:
    go_mod = base_dir / "go.mod"
    if not go_mod.exists():
        go_mod.write_text(_GO_MOD_TEMPLATE, encoding="utf-8")


def _build(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command(["go", "build", "./..."], base_dir, timeout_ms)
    metadata = {"language": "go", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao compilar")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )
    passed = result.returncode == 0
    if passed:
        found_errors: list[CheckError] = []
    elif "missing go.sum entry" in result.stderr or "cannot find module" in result.stderr:
        found_errors = [CheckError(code=errors.MISSING_DEPENDENCY, message=result.stderr.strip()[-2000:])]
    else:
        found_errors = _extract_errors(result.stderr)
    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata
    )


def _test(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command(["go", "test", "./..."], base_dir, timeout_ms)
    metadata = {"language": "go", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao rodar testes")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )
    passed = result.returncode == 0
    found_errors: list[CheckError] = []
    combined = result.stdout + result.stderr
    if not passed:
        if "FAIL" in combined and "build failed" not in combined:
            found_errors.append(CheckError(code=errors.TEST_FAILURE, message=combined.strip()[-2000:]))
        else:
            found_errors.append(CheckError(code=errors.RUNTIME_ERROR, message=combined.strip()[-2000:]))
    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata
    )


def _run(base_dir: Path, entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not entrypoint:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INCOMPLETE_SOLUTION, message="operation=run requer 'entrypoint'")],
            stdout="",
            stderr="",
            metadata={"language": "go", "duration_ms": 0},
        )
    result = run_command(["go", "run", entrypoint], base_dir, timeout_ms)
    metadata = {"language": "go", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message=f"timeout ao executar {entrypoint}")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )
    passed = result.returncode == 0
    if passed:
        found_errors: list[CheckError] = []
    elif ".go:" in result.stderr:
        found_errors = _extract_errors(result.stderr)
    else:
        found_errors = [CheckError(code=errors.RUNTIME_ERROR, message=result.stderr.strip()[-2000:])]
    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata
    )


def _lint(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command(["go", "vet", "./..."], base_dir, timeout_ms)
    metadata = {"language": "go", "duration_ms": result.duration_ms}
    passed = result.returncode == 0
    found_errors = [] if passed else _extract_errors(result.stderr)
    return CheckResult(
        passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata
    )


def check_go(
    operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int
) -> CheckResult:
    if not _go_available():
        return _unavailable_result()

    with tempfile.TemporaryDirectory(prefix="praxis_checker_go_") as tmp:
        base_dir = Path(tmp)
        materialize_files(base_dir, files)
        _ensure_go_mod(base_dir)

        if operation in ("syntax_check", "compile"):
            return _build(base_dir, timeout_ms)
        if operation == "compile_and_test":
            build_result = _build(base_dir, timeout_ms)
            if not build_result.passed:
                return build_result
            return _test(base_dir, timeout_ms)
        if operation == "run":
            return _run(base_dir, entrypoint, timeout_ms)
        if operation == "lint":
            return _lint(base_dir, timeout_ms)

        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
            stdout="",
            stderr="",
            metadata={"language": "go", "duration_ms": 0},
        )


register_backend("go", check_go)
