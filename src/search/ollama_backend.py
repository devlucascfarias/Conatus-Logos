"""Backend de busca real via API de busca da Ollama (D14, seção 5.4).

Endpoint e payload exatos devem ser confirmados contra a documentação vigente da Ollama no
momento de uso em produção (API relativamente nova); esta implementação assume:

    POST {endpoint}
    Header: "Authorization: Bearer {api_key}"
    Body:   {"query": query, "max_results": max_results}
    Resposta esperada: {"results": [{"title", "url", "snippet"|"content"}, ...]}

A chave NUNCA é lida de um valor hardcoded — sempre de uma variável de ambiente cujo NOME vem
de `configs/search_backend.yaml` (`api_key_env`). Falha explicitamente (RuntimeError) se a
variável não estiver definida, em vez de silenciosamente rodar sem autenticação.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from .base import SearchResult


class OllamaSearchBackend:
    def __init__(self, endpoint: str, api_key_env: str, timeout_ms: int = 8000, max_results: int = 5):
        self._endpoint = endpoint
        if api_key_env not in os.environ:
            raise RuntimeError(
                f"variável de ambiente '{api_key_env}' não definida — a chave da API de busca "
                "da Ollama nunca é lida de um valor hardcoded (seções 5.4/7.8 do PLAN.md). "
                f"Defina {api_key_env} no ambiente antes de usar provider='ollama'."
            )
        self._api_key = os.environ[api_key_env]
        self._timeout_s = timeout_ms / 1000
        self._default_max_results = max_results

    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        effective_max = max_results or self._default_max_results
        response = requests.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"query": query, "max_results": effective_max},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results", payload) if isinstance(payload, dict) else payload
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", item.get("content", "")),
            )
            for item in raw_results[:effective_max]
        ]
