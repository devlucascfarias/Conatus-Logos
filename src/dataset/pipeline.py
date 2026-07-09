"""Pipeline de validação e geração de dados (PLAN.md seção 9) — os 11 estágios descritos lá,
implementados sobre os módulos já existentes (gramática, schemas de ferramenta, checker).

O ponto central (D11, seção 10): nenhum exemplo é aceito só por "parecer plausível". Todo
`<tool_call name="checker">` dentro de uma trajetória é RE-EXECUTADO de verdade
(`execute_when_possible`) e o `<tool_result>` declarado é comparado contra o resultado real —
uma divergência é motivo de rejeição (`TOOL_RESULT_FABRICATION`), mesmo que o exemplo pareça
sintaticamente perfeito.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.checker import check as run_check
from src.parsers import parse_segments
from src.parsers.segments import MalformedSegment, ToolCallSegment, ToolResultSegment
from src.schemas import ToolRegistry

from .schema import validate_metadata, validate_trajectory
from .taxonomy import EXECUTION_CLASSIFICATIONS  # noqa: F401  (reexport de conveniência)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_OPERATION_TO_CLASS = {
    "syntax_check": "interpreted",
    "compile": "compiled",
    "compile_and_test": "tested",
    "run": "interpreted",
    "lint": "static_only",
}
_CLASS_PRIORITY = ["tested", "compiled", "interpreted", "static_only"]


# --- 1. ingest ---------------------------------------------------------------


def ingest(raw_dir: Path | str) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    examples = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            example = json.load(f)
        example["_source_file"] = path.name
        examples.append(example)
    return examples


# --- 2. normalize --------------------------------------------------------------


def normalize(example: dict[str, Any]) -> dict[str, Any]:
    traj = example["trajectory"]
    raw_text = traj["raw_text"].replace("\r\n", "\n")
    raw_text = "\n".join(line.rstrip() for line in raw_text.split("\n"))
    raw_text = unicodedata.normalize("NFC", raw_text)
    normalized = dict(example)
    normalized["trajectory"] = {**traj, "raw_text": raw_text}
    return normalized


# --- 3. structural_validate -----------------------------------------------------


_ALWAYS_FATAL_MALFORMED_REASONS = {"UNTERMINATED_TAG", "UNRECOGNIZED_CONTENT"}


def structural_validate(example: dict[str, Any]) -> list[str]:
    """Seção 9, estágio 3 — gramática bem formada + ordem válida.

    Um `<tool_call>` malformado NÃO é automaticamente fatal: exemplos do tipo
    'invalid_call_then_correction' (seção 8.1) legitimamente contêm uma chamada malformada
    seguida do `<tool_result status="error">` que o harness teria realmente produzido —
    isso ensina recuperação, não é uma trajetória quebrada. O que É sempre fatal:
    (a) tag não fechada / texto solto fora de tag (sinal de dado truncado/corrompido), e
    (b) um segmento malformado cujo próximo segmento NÃO é o tool_result de erro
        correspondente (ou seja, a "más-formação" não foi reconhecida/tratada na trajetória).
    """
    problems = []
    metadata_problems = validate_metadata(example.get("metadata", {}))
    problems.extend(f"metadata: {p}" for p in metadata_problems)

    trajectory_problems = validate_trajectory(example.get("trajectory", {}))
    problems.extend(f"trajectory: {p}" for p in trajectory_problems)
    if trajectory_problems:
        # Sem system_prompt/user_request/raw_text bem formados não dá pra parsear segmentos
        # com segurança (ex.: raw_text pode nem existir) — devolve só os problemas de schema.
        return problems

    segments = parse_segments(example["trajectory"]["raw_text"])
    if not segments:
        problems.append("trajetória vazia — nenhum segmento reconhecido pela gramática")
        return problems

    for i, seg in enumerate(segments):
        if not isinstance(seg, MalformedSegment):
            continue
        if seg.reason in _ALWAYS_FATAL_MALFORMED_REASONS:
            problems.append(f"segmento malformado fatal ({seg.reason}): {seg.message}")
            continue
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        handled = (
            isinstance(next_seg, ToolResultSegment)
            and next_seg.status == "error"
            and next_seg.body.get("code") == seg.reason
        )
        if not handled:
            problems.append(
                f"segmento malformado ({seg.reason}) não seguido do tool_result de erro "
                "correspondente — más-formação não tratada na trajetória"
            )

    if segments[-1].kind != "final":
        problems.append("trajetória não termina em <final> — turno incompleto (seção 3.2)")

    return problems


# --- 4. schema_validate ----------------------------------------------------------


def schema_validate(example: dict[str, Any], tool_registry: ToolRegistry) -> list[str]:
    """Seção 9, estágio 4. Mesmo princípio de `structural_validate`: uma chamada para uma
    ferramenta desconhecida/desabilitada ou com argumentos inválidos só é rejeitada se a
    trajetória NÃO reagir corretamente a isso (ex.: task_type='tool_unavailable' legitimamente
    contém uma chamada assim seguida do tool_result de erro correspondente)."""
    problems = []
    segments = parse_segments(example["trajectory"]["raw_text"])

    for i, seg in enumerate(segments):
        if not isinstance(seg, ToolCallSegment):
            continue

        spec = tool_registry.get(seg.name)
        if spec is None:
            errors = [f"ferramenta desconhecida/desabilitada nesta fase: {seg.name}"]
            expected_code = "UNSUPPORTED_TOOL"
        else:
            errors = tool_registry.validate_args(seg.name, seg.args)
            expected_code = "TOOL_ARGUMENT_SCHEMA_ERROR"

        if not errors:
            continue

        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        handled = (
            isinstance(next_seg, ToolResultSegment)
            and next_seg.status == "error"
            and next_seg.body.get("code") == expected_code
        )
        if not handled:
            problems.extend(f"tool_call '{seg.name}': {e}" for e in errors)

    return problems


# --- 5/6. extract_code_blocks + execute_when_possible ---------------------------


@dataclass
class CheckerCallComparison:
    segment_index: int
    operation: str
    declared_status: str | None
    real_status: str
    matches: bool
    real_result: dict[str, Any] = field(default_factory=dict)


def execute_when_possible(example: dict[str, Any]) -> list[CheckerCallComparison]:
    """Re-executa de verdade todo tool_call de `checker` na trajetória e compara com o
    `<tool_result>` declarado logo em seguida — a essência de D11 aplicada ao dataset."""
    segments = parse_segments(example["trajectory"]["raw_text"])
    comparisons: list[CheckerCallComparison] = []

    for i, seg in enumerate(segments):
        if not (isinstance(seg, ToolCallSegment) and seg.name == "checker"):
            continue

        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        declared_status = next_seg.status if isinstance(next_seg, ToolResultSegment) else None

        real = run_check(
            language=seg.args["language"],
            operation=seg.args["operation"],
            files=seg.args["files"],
            entrypoint=seg.args.get("entrypoint"),
            timeout_ms=seg.args.get("timeout_ms", 15000),
        )
        real_status = "ok" if real.passed else "error"

        comparisons.append(
            CheckerCallComparison(
                segment_index=i,
                operation=seg.args["operation"],
                declared_status=declared_status,
                real_status=real_status,
                matches=declared_status == real_status,
                real_result=real.to_json(),
            )
        )

    return comparisons


# --- 7. classify_execution -------------------------------------------------------


def classify_execution(example: dict[str, Any], comparisons: list[CheckerCallComparison]) -> str:
    if not comparisons:
        raw_text = example["trajectory"]["raw_text"]
        if '<tool_call name="write_file"' in raw_text or '<tool_call name="read_file"' in raw_text:
            return "static_only"
        return "non_executable_external_dep"

    classes = {_OPERATION_TO_CLASS.get(c.operation, "static_only") for c in comparisons}
    for candidate in _CLASS_PRIORITY:
        if candidate in classes:
            return candidate
    return "static_only"


# --- 8. reject_or_accept ---------------------------------------------------------


def reject_or_accept(
    structural_problems: list[str],
    schema_problems: list[str],
    comparisons: list[CheckerCallComparison],
) -> tuple[str, list[str]]:
    reasons = list(structural_problems) + list(schema_problems)
    for c in comparisons:
        if not c.matches:
            reasons.append(
                f"tool_result declarado (status={c.declared_status!r}) diverge da execução "
                f"REAL do checker (status={c.real_status!r}) no segmento {c.segment_index} "
                "— possível TOOL_RESULT_FABRICATION"
            )
    status = "rejected" if reasons else "validated"
    return status, reasons


# --- 9. dedup ----------------------------------------------------------------------


def content_hash(example: dict[str, Any]) -> str:
    normalized_text = example["trajectory"]["raw_text"].strip()
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def dedup(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for example in examples:
        h = content_hash(example)
        if h in seen:
            duplicates.append(example)
        else:
            seen[h] = example
    return list(seen.values()), duplicates


# --- 10. split_assign ----------------------------------------------------------------


def split_assign(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Seção 8.3 — split por grupo de tarefa-base, nunca por exemplo individual: todo exemplo
    cujo `task_group` já apareceu num split recebe o MESMO split (primeira ocorrência decide)."""
    groups: dict[str, str] = {}
    result: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "probes": [],
        "benchmark": [],
        "adversarial": [],
    }
    for example in examples:
        metadata = example["metadata"]
        group = metadata.get("task_group", metadata["id"])
        declared_split = metadata.get("split", "train")
        assigned_split = groups.setdefault(group, declared_split)
        result[assigned_split].append(example)
    return result


# --- 11. stats_report -----------------------------------------------------------------


def stats_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, int] = {}
    by_task_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_validation_status: dict[str, int] = {}

    for example in examples:
        metadata = example["metadata"]
        by_language[metadata["language"]] = by_language.get(metadata["language"], 0) + 1
        by_task_type[metadata["task_type"]] = by_task_type.get(metadata["task_type"], 0) + 1
        by_difficulty[metadata["difficulty"]] = by_difficulty.get(metadata["difficulty"], 0) + 1
        status = metadata.get("validation_status", "pending")
        by_validation_status[status] = by_validation_status.get(status, 0) + 1

    total = len(examples)
    rejected = by_validation_status.get("rejected", 0)
    return {
        "total": total,
        "by_language": by_language,
        "by_task_type": by_task_type,
        "by_difficulty": by_difficulty,
        "by_validation_status": by_validation_status,
        "rejection_rate": (rejected / total) if total else 0.0,
    }


# --- Orquestrador ---------------------------------------------------------------------


@dataclass
class PipelineReport:
    validated: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    duplicates_removed: int
    splits: dict[str, list[dict[str, Any]]]
    stats: dict[str, Any]


def run_pipeline(raw_dir: Path | str, tool_registry: ToolRegistry | None = None) -> PipelineReport:
    tool_registry = tool_registry or ToolRegistry.load()

    examples = ingest(raw_dir)
    examples = [normalize(e) for e in examples]

    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for example in examples:
        structural_problems = structural_validate(example)
        schema_problems = schema_validate(example, tool_registry) if not structural_problems else []
        comparisons = execute_when_possible(example) if not structural_problems and not schema_problems else []

        status, reasons = reject_or_accept(structural_problems, schema_problems, comparisons)
        example["metadata"]["validation_status"] = status
        example["metadata"]["execution_classification"] = classify_execution(example, comparisons)
        example["_rejection_reasons"] = reasons

        if status == "validated":
            validated.append(example)
        else:
            rejected.append(example)

    deduped_validated, duplicates = dedup(validated)
    splits = split_assign(deduped_validated)
    stats = stats_report(deduped_validated + rejected)

    return PipelineReport(
        validated=deduped_validated,
        rejected=rejected,
        duplicates_removed=len(duplicates),
        splits=splits,
        stats=stats,
    )


def write_pipeline_outputs(report: PipelineReport, data_dir: Path | str = REPO_ROOT / "data") -> None:
    data_dir = Path(data_dir)

    for example in report.validated:
        out_path = data_dir / "validated" / example["_source_file"]
        _write_example(out_path, example)

    for example in report.rejected:
        out_path = data_dir / "rejected" / example["_source_file"]
        _write_example(out_path, example)

    for split_name, split_examples in report.splits.items():
        split_dir = data_dir / split_name
        for example in split_examples:
            _write_example(split_dir / example["_source_file"], example)

    with (data_dir / "validated" / "_stats.json").open("w", encoding="utf-8") as f:
        json.dump(report.stats, f, ensure_ascii=False, indent=2)


def _write_example(path: Path, example: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in example.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
