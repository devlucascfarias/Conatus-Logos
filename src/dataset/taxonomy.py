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
        # D-taxonomy-backfill-gaps (achado real, docs/PLAN.md): estes 11 valores já eram usados
        # por 760 exemplos reais (com scripts geradores dedicados e propósito comportamental
        # documentado em docs/PLAN.md — Gaps 2b/4/5/6, expansão OOP/erro/padrões/módulos/shell)
        # desde 2026-07-08, mas nunca tinham sido adicionados a este enum fechado — cada um
        # cunhado numa leva de expansão sem atualizar a taxonomia junto, então `structural_validate`
        # rejeitava esses exemplos silenciosamente até serem tocados por outro trabalho e o gap
        # ser notado. Nenhum é sinônimo de um valor já existente (confirmado lendo o histórico
        # de cada um em docs/PLAN.md antes de decidir) — cada um cobre um comportamento
        # genuinamente distinto dos demais.
        "class_implementation",
        "error_handling",
        "python_pattern",
        "realistic_module",
        "shell_command",
        "checker_infra_unavailable",
        "tool_call_json_recovery",
        "multi_file_root_cause_diagnosis",
        "hypothesis_revision_after_failure",
        "no_bug_found_report",
        "own_bug_diagnosis",
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
