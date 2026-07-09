"""Executor da ferramenta `search_code` (seção 4.6) — busca por texto/regex em arquivos do
workspace, tipo `ctrl+shift+f`. Reativada para o pivô de frontend (D-frontend-pivot-model-swap,
docs/plan_frontend_specialization_wave3.md seção 4): usada no dimension "reuse-before-create" do
`<think>` — checar se já existe um componente antes de criar um novo, evitando código morto.

Implementação em Python puro (sem depender de `grep`/`rg` no PATH) para ficar OS-independente,
mesmo espírito do `shell.json` v2 (seção 4.5)."""

from __future__ import annotations

import re
from typing import Any

from src.checker import errors

from ..base import ToolExecutionResult

_MAX_MATCHES = 200
_MAX_FILE_BYTES = 2_000_000


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    pattern_text = args["pattern"]
    use_regex = args.get("regex", False)
    base = sandbox.resolve_path(args.get("path", "."))
    if base is None:
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"caminho escapa do workspace: {args.get('path', '.')}",
        )
    if not base.exists():
        return ToolExecutionResult(
            passed=False,
            error_code=errors.FILE_NOT_FOUND,
            error_message=f"caminho não encontrado no workspace: {args.get('path', '.')}",
        )

    try:
        matcher = re.compile(pattern_text) if use_regex else None
    except re.error as exc:
        return ToolExecutionResult(
            passed=False,
            error_code=errors.TOOL_ARGUMENT_SCHEMA_ERROR,
            error_message=f"regex inválida em 'pattern': {exc}",
        )

    candidates = [base] if base.is_file() else sorted(base.rglob("*"))
    matches: list[dict[str, Any]] = []
    truncated = False

    for candidate in candidates:
        if not candidate.is_file() or candidate.stat().st_size > _MAX_FILE_BYTES:
            continue
        try:
            rel = candidate.relative_to(sandbox.workspace)
        except ValueError:
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            hit = matcher.search(line) if matcher else (pattern_text in line)
            if not hit:
                continue
            if len(matches) >= _MAX_MATCHES:
                truncated = True
                break
            matches.append(
                {"path": str(rel).replace("\\", "/"), "line": line_number, "text": line.strip()}
            )
        if truncated:
            break

    return ToolExecutionResult(passed=True, data={"matches": matches, "truncated": truncated})
