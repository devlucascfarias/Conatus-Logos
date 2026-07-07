"""Testes do módulo de busca (PLAN.md seção 5.4, D14): MockSearchBackend, factory
parametrizada e a exigência de que OllamaSearchBackend nunca rode sem uma chave real."""

import pytest

from src.search import (
    MockSearchBackend,
    OllamaSearchBackend,
    UnsupportedSearchProvider,
    build_search_backend,
)
from src.search.factory import REPO_ROOT

_FIXTURES_DIR = REPO_ROOT / "data" / "fixtures" / "web_search"


def test_mock_backend_returns_fixture_results():
    backend = MockSearchBackend(fixtures_path=_FIXTURES_DIR)
    results = backend.search("python list comprehension", max_results=5)
    assert len(results) == 2
    assert results[0].title.startswith("List comprehensions")
    assert results[0].url.startswith("https://docs.python.org")


def test_mock_backend_respects_max_results():
    backend = MockSearchBackend(fixtures_path=_FIXTURES_DIR)
    results = backend.search("python list comprehension", max_results=1)
    assert len(results) == 1


def test_mock_backend_missing_fixture_returns_empty_list():
    backend = MockSearchBackend(fixtures_path=_FIXTURES_DIR)
    assert backend.search("query sem fixture nenhuma") == []


def test_ollama_backend_refuses_to_run_without_api_key_env(monkeypatch):
    monkeypatch.delenv("PRAXIS_OLLAMA_SEARCH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PRAXIS_OLLAMA_SEARCH_API_KEY"):
        OllamaSearchBackend(
            endpoint="https://ollama.com/api/web_search",
            api_key_env="PRAXIS_OLLAMA_SEARCH_API_KEY",
        )


def test_ollama_backend_never_reads_hardcoded_key(monkeypatch):
    monkeypatch.setenv("PRAXIS_OLLAMA_SEARCH_API_KEY", "test-key-from-env-only")
    backend = OllamaSearchBackend(
        endpoint="https://ollama.com/api/web_search",
        api_key_env="PRAXIS_OLLAMA_SEARCH_API_KEY",
    )
    assert backend._api_key == "test-key-from-env-only"


def test_factory_builds_mock_backend_from_explicit_config(tmp_path):
    config_path = tmp_path / "search_backend.yaml"
    config_path.write_text(
        f"search_backend:\n  provider: mock\n  mock:\n    fixtures_path: {_FIXTURES_DIR.as_posix()!r}\n",
        encoding="utf-8",
    )
    backend = build_search_backend(config_path)
    assert isinstance(backend, MockSearchBackend)
    assert backend.search("python list comprehension")


def test_factory_rejects_unknown_provider(tmp_path):
    config_path = tmp_path / "search_backend.yaml"
    config_path.write_text("search_backend:\n  provider: bing\n", encoding="utf-8")
    with pytest.raises(UnsupportedSearchProvider):
        build_search_backend(config_path)
