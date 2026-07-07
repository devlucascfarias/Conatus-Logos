from .base import SearchBackend, SearchResult
from .factory import DEFAULT_CONFIG_PATH, UnsupportedSearchProvider, build_search_backend
from .mock_backend import MockSearchBackend
from .ollama_backend import OllamaSearchBackend

__all__ = [
    "SearchBackend",
    "SearchResult",
    "MockSearchBackend",
    "OllamaSearchBackend",
    "build_search_backend",
    "UnsupportedSearchProvider",
    "DEFAULT_CONFIG_PATH",
]
