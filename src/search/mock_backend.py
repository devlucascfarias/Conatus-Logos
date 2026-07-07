"""Backend de busca mock (seção 5.4) — corpus fixo/curado em `data/fixtures/web_search/`, sem
rede. Usado para geração determinística de dataset/probes e quando `provider: mock`.

Cada fixture é um arquivo `<slug_da_query>.json` contendo uma lista de objetos
`{"title", "url", "snippet"}` — coletados uma vez via `OllamaSearchBackend` e congelados
(seção 5.4: reprodutibilidade vs. busca ao vivo)."""

from __future__ import annotations

import json
from pathlib import Path

from .base import SearchResult


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip().lower()).strip("_")


class MockSearchBackend:
    def __init__(self, fixtures_path: str | Path):
        self._fixtures_path = Path(fixtures_path)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        fixture_file = self._fixtures_path / f"{slugify(query)}.json"
        if not fixture_file.exists():
            return []
        with fixture_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            SearchResult(title=item["title"], url=item["url"], snippet=item["snippet"])
            for item in raw[:max_results]
        ]
