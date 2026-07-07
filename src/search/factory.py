"""Factory — único ponto de decisão sobre qual motor de busca está ativo (seção 5.4, D14).

Nenhum chamador (harness, gerador de dataset, avaliador) deve instanciar `OllamaSearchBackend`
ou `MockSearchBackend` diretamente — sempre via `build_search_backend`, para que trocar de
motor de busca seja só uma mudança em `configs/search_backend.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import SearchBackend
from .mock_backend import MockSearchBackend
from .ollama_backend import OllamaSearchBackend

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "search_backend.yaml"


class UnsupportedSearchProvider(Exception):
    pass


def build_search_backend(config_path: Path | str = DEFAULT_CONFIG_PATH) -> SearchBackend:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["search_backend"]

    provider = raw["provider"]
    if provider == "mock":
        cfg = raw["mock"]
        return MockSearchBackend(fixtures_path=REPO_ROOT / cfg["fixtures_path"])
    if provider == "ollama":
        cfg = raw["ollama"]
        return OllamaSearchBackend(
            endpoint=cfg["endpoint"],
            api_key_env=cfg["api_key_env"],
            timeout_ms=cfg["timeout_ms"],
            max_results=cfg["max_results"],
        )
    raise UnsupportedSearchProvider(f"provider desconhecido em search_backend.yaml: {provider!r}")
