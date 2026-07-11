"""Schema de metadados obrigatórios por exemplo do dataset (PLAN.md seção 8.2).

Um exemplo bruto é um JSON com dois campos de topo:
  - "metadata": os campos da seção 8.2 (id, domain, language, ...).
  - "trajectory": um dict com "system_prompt", "user_request" e "raw_text" (o texto na
    gramática canônica da seção 3.2 — o que teria sido gerado pelo modelo + injetado pelo
    harness num turno completo), mais um campo opcional "history" (D-dataset-history-schema,
    docs/plan_dataset_expansion_wave2_identity_multiturn.md seção 4) — lista de
    {"user_request", "raw_text"} representando turnos ANTERIORES já concluídos da mesma
    sessão, no mesmo formato de `agent.Turn` do lado da CLI Go. Ausente ou vazio = turno único
    (comportamento de sempre). `user_request`/`raw_text` de topo continuam sendo o turno
    ATUAL — o único que efetivamente entra na loss de treino; `history` é só contexto.

`validate_metadata`/`validate_trajectory` são usados pelo estágio `structural_validate` do
pipeline (seção 9)."""

from __future__ import annotations

from typing import Any

import jsonschema

from .taxonomy import DIFFICULTIES, EXECUTION_CLASSIFICATIONS, LANGUAGE_METADATA_VALUES, TASK_TYPES

METADATA_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "id",
        "domain",
        "language",
        "difficulty",
        "tools_used",
        "num_steps",
        "task_type",
        "source",
        "license",
        "validation_status",
        "execution_performed",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "domain": {"type": "string", "minLength": 1},
        "language": {"enum": sorted(LANGUAGE_METADATA_VALUES)},
        "difficulty": {"enum": sorted(DIFFICULTIES)},
        "tools_used": {"type": "array", "items": {"type": "string"}},
        "num_steps": {"type": "integer", "minimum": 0},
        "task_type": {"enum": sorted(TASK_TYPES)},
        "source": {"type": "string", "minLength": 1},
        "license": {"type": "string", "minLength": 1},
        "validation_status": {"enum": ["validated", "rejected", "pending"]},
        "execution_performed": {"type": "boolean"},
        "checker_used": {"type": "string"},
        "expected_result": {"type": "object"},
        "known_risks": {"type": "array", "items": {"type": "string"}},
        "execution_classification": {"enum": sorted(EXECUTION_CLASSIFICATIONS)},
        "split": {"enum": ["train", "validation", "probes", "benchmark", "adversarial"]},
        "task_group": {
            "type": "string",
            "description": "Seção 8.3: variações da mesma tarefa-base compartilham task_group "
            "e ficam sempre no mesmo split. Se ausente, cada exemplo é seu próprio grupo (id).",
        },
    },
}

_VALIDATOR = jsonschema.Draft202012Validator(METADATA_SCHEMA)


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    return [error.message for error in _VALIDATOR.iter_errors(metadata)]


TRAJECTORY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    # Só "raw_text" é exigido aqui — é o único campo que `structural_validate` sempre leu
    # direto do dict (`example["trajectory"]["raw_text"]"`), então exigi-lo só torna esse
    # KeyError implícito num erro de validação claro, sem reduzir compatibilidade. NÃO exige
    # "system_prompt"/"user_request" porque testes existentes de `structural_validate`
    # (tests/unit/test_dataset_pipeline.py) e alguns estágios do pipeline constroem
    # trajetórias parciais só com "raw_text" para testar a gramática isoladamente.
    "required": ["raw_text"],
    "properties": {
        "system_prompt": {"type": "string", "minLength": 1},
        "user_request": {"type": "string", "minLength": 1},
        "raw_text": {"type": "string"},
        "history": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["user_request", "raw_text"],
                "properties": {
                    "user_request": {"type": "string", "minLength": 1},
                    "raw_text": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

_TRAJECTORY_VALIDATOR = jsonschema.Draft202012Validator(TRAJECTORY_SCHEMA)


def validate_trajectory(trajectory: dict[str, Any]) -> list[str]:
    return [error.message for error in _TRAJECTORY_VALIDATOR.iter_errors(trajectory)]
