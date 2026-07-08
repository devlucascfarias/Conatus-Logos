#!/usr/bin/env python
"""Checagem grosseira de similaridade entre variantes de frase da mesma tarefa-base
(docs/plan_dataset_expansion_oop_shell.md, seção 4 — AVISO CRÍTICO). Agrupa exemplos por
tarefa-base (prefixo antes de "-var<N>"), extrai só o texto de prosa (`<think>`/`<final>`,
sem `<tool_call>`/`<tool_result>`) e calcula similaridade par a par com difflib. Reporta pares
acima do limiar como possível repetição de template.

Uso:
    python scripts/check_phrasing_similarity.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "train"
SIMILARITY_THRESHOLD = 0.90

_PREFIXES = (
    "gen-oop-", "gen-error-", "gen-pattern-", "gen-module-", "gen-ensearch-", "gen-shell-",
    "gen-checkerinfra-", "gen-jsonrecovery-", "gen-multifile-", "gen-trapword-", "gen-hyprevision-",
)

_TAG_RE = re.compile(r"<tool_call[^>]*>.*?</tool_call>|<tool_result[^>]*>.*?</tool_result>", re.DOTALL)
_STRIP_TAGS_RE = re.compile(r"</?(?:think|final)>")


def _prose_only(raw_text: str) -> str:
    without_tools = _TAG_RE.sub(" ", raw_text)
    return _STRIP_TAGS_RE.sub(" ", without_tools).strip()


def _base_id(example_id: str) -> str:
    return re.sub(r"-var\d+$", "", example_id)


def main() -> int:
    groups = defaultdict(list)
    for path in sorted(DATA_DIR.glob("*.json")):
        if not path.name.startswith(_PREFIXES):
            continue
        example = json.loads(path.read_text(encoding="utf-8"))
        example_id = example["metadata"]["id"]
        base = _base_id(example_id)
        prose = _prose_only(example["trajectory"]["raw_text"])
        request = example["trajectory"]["user_request"]
        groups[base].append((example_id, request, prose))

    total_pairs_checked = 0
    flagged = []
    for base, items in sorted(groups.items()):
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                id_a, req_a, prose_a = items[i]
                id_b, req_b, prose_b = items[j]
                total_pairs_checked += 1
                prose_ratio = SequenceMatcher(None, prose_a, prose_b).ratio()
                request_ratio = SequenceMatcher(None, req_a, req_b).ratio()
                if prose_ratio > SIMILARITY_THRESHOLD or request_ratio > SIMILARITY_THRESHOLD:
                    flagged.append((base, id_a, id_b, prose_ratio, request_ratio))

    print(f"Grupos verificados: {len(groups)}")
    print(f"Pares comparados: {total_pairs_checked}")
    print(f"Pares sinalizados (similaridade > {SIMILARITY_THRESHOLD}): {len(flagged)}")
    if flagged:
        print("\nDetalhe dos pares sinalizados:")
        for base, id_a, id_b, prose_ratio, request_ratio in flagged:
            print(f"  [{base}] {id_a} vs {id_b}: prosa={prose_ratio:.3f} pedido={request_ratio:.3f}")
        return 1
    print("\nNenhum par acima do limiar — diversidade de frase OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
