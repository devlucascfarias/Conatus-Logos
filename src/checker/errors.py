"""Códigos de erro estáveis do checker (PLAN.md, seção 10.2).

Usado pelo checker, pelo parser (para más-formações de tool_call/tool_result) e pelo harness
(para violações comportamentais como fabricação de tool_result e vazamento de raciocínio).
Nenhum outro módulo deve declarar strings de código de erro soltas — sempre referenciar
uma constante daqui, para que o conjunto de códigos permaneça a fonte única de verdade.
"""

from __future__ import annotations

TOOL_CALL_PARSE_ERROR = "TOOL_CALL_PARSE_ERROR"
TOOL_ARGUMENT_SCHEMA_ERROR = "TOOL_ARGUMENT_SCHEMA_ERROR"
UNSUPPORTED_TOOL = "UNSUPPORTED_TOOL"
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
SYNTAX_ERROR = "SYNTAX_ERROR"
COMPILATION_ERROR = "COMPILATION_ERROR"
RUNTIME_ERROR = "RUNTIME_ERROR"
TEST_FAILURE = "TEST_FAILURE"
TIMEOUT = "TIMEOUT"
MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
INVALID_OUTPUT_FORMAT = "INVALID_OUTPUT_FORMAT"
NONEXISTENT_API = "NONEXISTENT_API"
EXPLANATION_CODE_MISMATCH = "EXPLANATION_CODE_MISMATCH"
TOOL_RESULT_FABRICATION = "TOOL_RESULT_FABRICATION"
INTERNAL_REASONING_LEAK = "INTERNAL_REASONING_LEAK"
MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"
UNSAFE_COMMAND = "UNSAFE_COMMAND"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
PATCH_APPLY_ERROR = "PATCH_APPLY_ERROR"
INCOMPLETE_SOLUTION = "INCOMPLETE_SOLUTION"
SANDBOX_ERROR = "SANDBOX_ERROR"
RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
# Complemento estrutural (não estava na tabela original, mas necessário para texto fora de tag)
UNRECOGNIZED_CONTENT = "UNRECOGNIZED_CONTENT"
UNTERMINATED_TAG = "UNTERMINATED_TAG"
# D-segment-smuggling: `parse_last_segment` só olhava o ÚLTIMO segmento de uma geração — se o
# modelo produzisse conteúdo extra (tool_call mal-formado, sintaxe nunca treinada, tool_call
# real não executado) ANTES de um <final> aparentemente limpo na MESMA geração, esse conteúdo
# era descartado em silêncio e o <final> era aceito como sucesso genuíno. Achado real: adapter
# L4 gerou dois `<tool_call>` (nomes de ferramenta inexistentes, sintaxe self-closing nunca
# vista no treino) seguidos de um `<final>` alegando sucesso — nenhuma ferramenta foi executada
# de verdade. Este código marca exatamente esse caso.
SEGMENT_SMUGGLING = "SEGMENT_SMUGGLING"

ALL_CODES = frozenset(
    {
        TOOL_CALL_PARSE_ERROR,
        TOOL_ARGUMENT_SCHEMA_ERROR,
        UNSUPPORTED_TOOL,
        UNSUPPORTED_LANGUAGE,
        SYNTAX_ERROR,
        COMPILATION_ERROR,
        RUNTIME_ERROR,
        TEST_FAILURE,
        TIMEOUT,
        MISSING_DEPENDENCY,
        INVALID_OUTPUT_FORMAT,
        NONEXISTENT_API,
        EXPLANATION_CODE_MISMATCH,
        TOOL_RESULT_FABRICATION,
        INTERNAL_REASONING_LEAK,
        MAX_STEPS_EXCEEDED,
        UNSAFE_COMMAND,
        FILE_NOT_FOUND,
        PATCH_APPLY_ERROR,
        INCOMPLETE_SOLUTION,
        SANDBOX_ERROR,
        RESOURCE_LIMIT_EXCEEDED,
        UNRECOGNIZED_CONTENT,
        UNTERMINATED_TAG,
        SEGMENT_SMUGGLING,
    }
)
