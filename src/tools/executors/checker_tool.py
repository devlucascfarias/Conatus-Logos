"""Executor da ferramenta `checker` (seção 4.1) — delega inteiramente a `src.checker.check`,
que nunca confia em texto do modelo (D11)."""

from __future__ import annotations

from typing import Any

from src.checker import check as run_check

from ..base import ToolExecutionResult


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    result = run_check(
        language=args["language"],
        operation=args["operation"],
        files=args["files"],
        entrypoint=args.get("entrypoint"),
        timeout_ms=args.get("timeout_ms", 15000),
    )
    if result.passed:
        return ToolExecutionResult(passed=True, data=result.to_json())

    first_error = result.errors[0] if result.errors else None
    return ToolExecutionResult(
        passed=False,
        data=result.to_json(),
        error_code=first_error.code if first_error else "UNKNOWN",
        error_message=first_error.message if first_error else "checker falhou sem detalhe",
    )
