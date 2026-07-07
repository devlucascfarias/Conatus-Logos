"""Executor da ferramenta `write_file` (seção 4.3) — cria ou substitui, sempre relativo ao
workspace do sandbox (seção 7.4). `apply_patch` por diff fica para fase 2 (D5)."""

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

    mode = args.get("mode", "overwrite")
    if mode == "create" and path.exists():
        return ToolExecutionResult(
            passed=False,
            error_code=errors.INCOMPLETE_SOLUTION,
            error_message=f"arquivo já existe e mode='create' não permite sobrescrever: {args['path']}",
        )

    content = args["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return ToolExecutionResult(
        passed=True,
        data={"path": args["path"], "bytes_written": len(content.encode("utf-8")), "mode": mode},
    )
