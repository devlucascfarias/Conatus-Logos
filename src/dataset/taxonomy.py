"""Taxonomia do dataset (PLAN.md seção 8.1) — vocabulário fechado de `task_type`, usado para
validar metadados e para o relatório de estatísticas (seção 9, estágio 11)."""

from __future__ import annotations

TASK_TYPES = frozenset(
    {
        "direct_answer",
        "single_tool_call",
        "multi_tool_call",
        "invalid_call_then_correction",
        "checker_rejects_code",
        "model_fixes_after_error",
        "compiles_successfully",
        "test_fails",
        "test_passes",
        "tool_unavailable",
        "insufficient_information",
        "ambiguous_request",
        "forbidden_operation",
        "multi_file_read_and_edit",
        "refactor",
        "debugging",
        "test_authoring",
        "complexity_analysis",
        "code_explanation",
        "language_migration",
        "project_configuration",
        "documentation_usage",
    }
)

# MVP = Python + Go (D4). "html" entrou depois (D-checker-html-backend, ver docs/PLAN.md) — tem
# backend de checker real registrado (src/checker/backends/frontend_backend.py), não é mais uma
# linguagem "de schema só", mesmo sem exemplos de HTML no dataset atual. Outras linguagens do
# enum do checker (seção 4.1) continuam sem backend, não devem aparecer no dataset.
MVP_LANGUAGES = frozenset({"python", "go", "html"})

DIFFICULTIES = frozenset({"easy", "medium", "hard"})

SPLITS = frozenset({"train", "validation", "probes", "benchmark", "adversarial"})

EXECUTION_CLASSIFICATIONS = frozenset(
    {
        "static_only",
        "interpreted",
        "compiled",
        "tested",
        "output_compared",
        "non_executable_external_dep",
    }
)
