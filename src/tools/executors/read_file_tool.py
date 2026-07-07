"""Executor da ferramenta `read_file` (seção 4.2) — lê por intervalo de linhas, sempre
relativo ao workspace do sandbox (seção 7.4)."""

from __future__ import annotations

from typing import Any

from src.checker import errors

from ..base import ToolExecutionResult


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    path = sandbox.resolve_path(args["path"])
    if path is None:
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"caminho escapa do workspace: {args['path']}",
        )
    if not path.exists() or not path.is_file():
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"arquivo não encontrado no workspace: {args['path']}",
        )

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start_line = args.get("start_line", 1)
    end_line = min(args.get("end_line", len(lines)), len(lines))
    selected = "".join(lines[start_line - 1 : end_line])

    return ToolExecutionResult(
        passed=True,
        data={"content": selected, "start_line": start_line, "end_line": end_line, "total_lines": len(lines)},
    )
