#!/usr/bin/env python
"""Checagem de vazamento de detalhe interno nos <final> das 6 novas categorias geradas hoje
(docs/plan_dataset_expansion_oop_shell.md, seção 5, passo 6)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "train"

_PREFIXES = ("gen-oop-", "gen-error-", "gen-pattern-", "gen-module-", "gen-ensearch-", "gen-shell-")
_PATTERNS = ["PRAXIS_OLLAMA", "PLAN.md", "AppData", "Pichau", "Desktop\\Logos", "C:\\Users", "Traceback"]


def main() -> None:
    leaks = []
    checked = 0
    for path in DATA_DIR.glob("*.json"):
        if not path.name.startswith(_PREFIXES):
            continue
        checked += 1
        example = json.loads(path.read_text(encoding="utf-8"))
        raw = example["trajectory"]["raw_text"]
        for final_text in re.findall(r"<final>(.*?)</final>", raw, re.DOTALL):
            for pattern in _PATTERNS:
                if pattern in final_text:
                    leaks.append((path.name, pattern))
    print(f"arquivos verificados: {checked}")
    print(f"vazamentos encontrados: {leaks}")


if __name__ == "__main__":
    main()
