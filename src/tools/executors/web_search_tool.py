"""Executor da ferramenta `web_search` (seção 4.6, 5.4, D14) — usa o `SearchBackend` ativo
(`configs/search_backend.yaml`); nunca instancia `OllamaSearchBackend`/`MockSearchBackend`
diretamente, sempre via `build_search_backend` (factory)."""

from __future__ import annotations

from typing import Any, Optional

from src.checker import errors
from src.search import SearchBackend, build_search_backend

from ..base import ToolExecutionResult

_cached_backend: Optional[SearchBackend] = None


def _get_backend() -> SearchBackend:
    global _cached_backend
    if _cached_backend is None:
        _cached_backend = build_search_backend()
    return _cached_backend


def reset_cached_backend() -> None:
    """Usado por testes para forçar reconstrução do backend após trocar config/env."""
    global _cached_backend
    _cached_backend = None


def execute(args: dict[str, Any], sandbox) -> ToolExecutionResult:
    try:
        backend = _get_backend()
        results = backend.search(args["query"], max_results=args.get("max_results", 5))
    except Exception as exc:  # falha de rede/config — não é comportamento incorreto do modelo
        return ToolExecutionResult(passed=False, error_code=errors.SANDBOX_ERROR, error_message=str(exc))

    return ToolExecutionResult(passed=True, data={"results": [r.to_json() for r in results]})
