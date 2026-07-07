"""Núcleo do checker (PLAN.md, seção 10) — camada independente do modelo.

Nunca confia em texto gerado pelo modelo: sempre reexecuta/recompila a partir dos arquivos
reais passados em `files` (D11). É a MESMA implementação usada em três contextos (D3):
ferramenta `checker` em tempo de execução do agente, validador offline do pipeline de
dataset (seção 9) e motor de métricas do avaliador (seção 13) — nunca uma cópia paralela.

Backends por linguagem se registram via `register_backend(language, fn)`; o MVP registra
"python" e "go" (seção 10.4). Uma linguagem sem backend registrado retorna
`UNSUPPORTED_LANGUAGE` de forma estruturada, nunca lança exceção.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import errors


@dataclass(frozen=True)
class CheckFile:
    path: str
    content: str


@dataclass(frozen=True)
class CheckError:
    code: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.file is not None:
            out["file"] = self.file
        if self.line is not None:
            out["line"] = self.line
        return out


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    errors: list[CheckError]
    stdout: str
    stderr: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": [e.to_json() for e in self.errors],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": self.metadata,
        }


# operation, files, entrypoint, timeout_ms -> CheckResult
BackendFn = Callable[[str, list[CheckFile], Optional[str], int], CheckResult]

_BACKENDS: dict[str, BackendFn] = {}


def register_backend(language: str, fn: BackendFn) -> None:
    _BACKENDS[language] = fn


def registered_languages() -> list[str]:
    return sorted(_BACKENDS.keys())


def check(
    language: str,
    operation: str,
    files: list[dict[str, Any]],
    entrypoint: Optional[str] = None,
    timeout_ms: int = 15000,
) -> CheckResult:
    """Ponto de entrada único do checker — contrato de saída seção 10.1."""
    backend = _BACKENDS.get(language)
    if backend is None:
        return CheckResult(
            passed=False,
            errors=[
                CheckError(
                    code=errors.UNSUPPORTED_LANGUAGE,
                    message=f"nenhum backend de checker registrado para '{language}'",
                )
            ],
            stdout="",
            stderr="",
            metadata={"language": language, "duration_ms": 0},
        )

    check_files = [CheckFile(path=f["path"], content=f["content"]) for f in files]
    start = time.monotonic()
    try:
        return backend(operation, check_files, entrypoint, timeout_ms)
    except Exception as exc:  # falha da infraestrutura do checker/sandbox, não do código do modelo
        duration_ms = int((time.monotonic() - start) * 1000)
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.SANDBOX_ERROR, message=str(exc))],
            stdout="",
            stderr="",
            metadata={"language": language, "duration_ms": duration_ms},
        )
