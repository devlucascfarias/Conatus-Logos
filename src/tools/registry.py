"""Registro de EXECUTORES de ferramenta (seção 5.2) — compõe com `src.schemas.ToolRegistry`,
que continua sendo a fonte única de verdade sobre quais ferramentas existem/estão habilitadas
nesta fase (D10). Este módulo só decide QUAL FUNÇÃO roda quando uma chamada válida chega."""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.checker import errors
from src.schemas import ToolRegistry

from .base import ToolExecutionResult
from .executors import (
    checker_tool,
    list_files_tool,
    read_file_tool,
    search_code_tool,
    shell_tool,
    web_search_tool,
    write_file_tool,
)

ExecutorFn = Callable[[dict[str, Any], Any], ToolExecutionResult]

_DEFAULT_EXECUTORS: dict[str, ExecutorFn] = {
    "checker": checker_tool.execute,
    "read_file": read_file_tool.execute,
    "write_file": write_file_tool.execute,
    "list_files": list_files_tool.execute,
    "shell": shell_tool.execute,
    "search_code": search_code_tool.execute,
    "web_search": web_search_tool.execute,
}


class ToolExecutorRegistry:
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        executors: Optional[dict[str, ExecutorFn]] = None,
    ):
        self._tool_registry = tool_registry or ToolRegistry.load()
        self._executors = executors if executors is not None else dict(_DEFAULT_EXECUTORS)

    def get_spec(self, name: str):
        return self._tool_registry.get(name)

    def validate_args(self, name: str, args: dict[str, Any]) -> list[str]:
        return self._tool_registry.validate_args(name, args)

    def execute(self, name: str, args: dict[str, Any], sandbox) -> ToolExecutionResult:
        spec = self.get_spec(name)
        if spec is None:
            return ToolExecutionResult(
                passed=False,
                error_code=errors.UNSUPPORTED_TOOL,
                error_message=f"ferramenta desconhecida/desabilitada nesta fase: {name}",
            )
        executor = self._executors.get(name)
        if executor is None:
            return ToolExecutionResult(
                passed=False,
                error_code=errors.UNSUPPORTED_TOOL,
                error_message=f"nenhum executor implementado para: {name}",
            )
        return executor(args, sandbox)
