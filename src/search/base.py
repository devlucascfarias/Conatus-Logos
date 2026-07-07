"""Interface `SearchBackend` (PLAN.md seção 5.4, D14) — trocar de motor de busca é mudar
`configs/search_backend.yaml`, nunca código chamador."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_json(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...
