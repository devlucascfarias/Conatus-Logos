"""Executor da ferramenta `list_files` (seção 4.4) — inspeciona a estrutura do workspace sem
inserir o repositório inteiro no contexto."""

from __future__ import annotations

from typing import Any

from src.checker import errors

from ..base import ToolExecutionResult


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    base = sandbox.resolve_path(args.get("path", "."))
    if base is None:
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"caminho escapa do workspace: {args.get('path', '.')}",
        )
    if not base.exists() or not base.is_dir():
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"diretório não encontrado no workspace: {args.get('path', '.')}",
        )

    glob_pattern = args.get("glob") or "**/*"
    max_depth = args.get("max_depth", 3)

    found: list[str] = []
    for candidate in base.glob(glob_pattern):
        try:
            rel = candidate.relative_to(sandbox.workspace)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        found.append(str(rel).replace("\\", "/"))

    return ToolExecutionResult(passed=True, data={"files": sorted(found)})
