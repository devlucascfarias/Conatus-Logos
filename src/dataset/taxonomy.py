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
# enum do checker (seção 4.1) continuam sem backend, não devem CLAIMAR suporte real do checker
# no dataset — continua valendo, ver `LANGUAGE_METADATA_VALUES` abaixo pra por que isso não é a
# mesma coisa que "quais valores metadata.language pode ter".
MVP_LANGUAGES = frozenset({"python", "go", "html"})

# D-language-metadata-widen (achado real, docs/PLAN.md): `metadata.language` só é consumido por
# `stats_report` (`src/dataset/pipeline.py`, bucket `by_language`, puro relatório) — NUNCA decide
# dispatch do checker (isso sempre vem do argumento `language` do próprio `tool_call`,
# independente do metadata) nem é lido em mais nenhum outro lugar do código (confirmado por
# grep). Por isso alargar este vocabulário NÃO reabre o risco que `MVP_LANGUAGES` protege
# (afirmar suporte real de checker pra uma linguagem sem backend) — os dois problemas são
# ortogonais. Dois grupos de exemplos legítimos, achados via D-taxonomy-backfill-gaps, usavam
# valores fora do enum: (1) `checker_infra_unavailable` (30 exemplos) escreve código JavaScript/
# TypeScript de verdade e chama `checker(language=...)` DE PROPÓSITO pra receber
# `UNSUPPORTED_LANGUAGE` real — o ponto do exemplo é a ausência de backend, então
# `metadata.language` refletir a linguagem real do artefato (`javascript`/`typescript`) é
# preciso, não uma alegação de suporte; (2) `shell_command` (90 exemplos) nem passa pelo
# `checker` — roda comandos de shell reais via a ferramenta `shell`, um domínio de execução
# diferente de "linguagem de programação verificada", mas ainda precisa de algum valor no campo
# obrigatório `language`. `LANGUAGE_METADATA_VALUES` é o vocabulário usado pelo SCHEMA
# (`src/dataset/schema.py`) — mais amplo que `MVP_LANGUAGES` de propósito.
LANGUAGE_METADATA_VALUES = MVP_LANGUAGES | frozenset({"shell", "javascript", "typescript"})

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
