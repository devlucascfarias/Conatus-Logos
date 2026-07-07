#!/usr/bin/env python
"""Estágios 2-11 do pipeline de dataset (PLAN.md seção 9): normaliza, valida estrutura/schema,
re-executa checker de verdade, classifica, aceita/rejeita, deduplica, separa splits e produz
estatísticas. Lê de `data/raw/`, escreve em `data/validated/`, `data/rejected/` e nos splits
(`data/train/`, `data/validation/`, `data/probes/`, `data/benchmark/`, `data/adversarial/`).

Uso:
    python scripts/validate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import run_pipeline, write_pipeline_outputs  # noqa: E402


def main() -> None:
    raw_dir = REPO_ROOT / "data" / "raw"
    report = run_pipeline(raw_dir)
    write_pipeline_outputs(report, data_dir=REPO_ROOT / "data")

    print(f"Validados: {len(report.validated)} | Rejeitados: {len(report.rejected)} | "
          f"Duplicados removidos: {report.duplicates_removed}")
    print("Splits:", {name: len(examples) for name, examples in report.splits.items()})
    print("Estatísticas:", report.stats)

    if report.rejected:
        print("\nMotivos de rejeição:")
        for example in report.rejected:
            print(f"  - {example['metadata']['id']}: {example['_rejection_reasons']}")


if __name__ == "__main__":
    main()
