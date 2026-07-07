"""Executor da ferramenta `shell` (seção 4.5) — nunca executa comando fora da allowlist; toda
a política vem de `sandbox.run_shell` (seção 7.5)."""

from __future__ import annotations

from typing import Any

from src.checker import errors

from ..base import ToolExecutionResult


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    result = sandbox.run_shell(
        args["binary"], args=args.get("args", []), cwd=args.get("cwd", "."), timeout_ms=args.get("timeout_ms")
    )

    if not result.allowed:
        return ToolExecutionResult(passed=False, error_code=errors.UNSAFE_COMMAND, error_message=result.denial_reason)

    data = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

    if result.timed_out:
        return ToolExecutionResult(
            passed=False, data=data, error_code=errors.TIMEOUT, error_message="comando excedeu o timeout"
        )

    if result.returncode == 0:
        return ToolExecutionResult(passed=True, data=data)

    return ToolExecutionResult(
        passed=False,
        data=data,
        error_code=errors.RUNTIME_ERROR,
        error_message=f"comando saiu com código {result.returncode}",
    )
