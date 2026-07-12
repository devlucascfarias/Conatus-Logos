"""Testes do backend SCSS do checker (D-checker-scss-backend, ver docs/PLAN.md).

Pulados (não falham) se `node`/`npm` não estiverem no PATH ou se o projeto-template não tiver
`node_modules` instalado (`npm install`, ver README.md) — o backend em si trata as duas ausências
como MISSING_DEPENDENCY estruturado.
"""

import shutil
from pathlib import Path

import pytest

from src.checker import check

_HAS_NODE = shutil.which("node") is not None and shutil.which("npm") is not None
_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "checker_templates" / "react_three_fiber"
_HAS_TEMPLATE = (_TEMPLATE_DIR / "node_modules").is_dir()
_requires_sass = pytest.mark.skipif(
    not (_HAS_NODE and _HAS_TEMPLATE),
    reason="node/npm ausente ou node_modules do template não instalado (ver README.md)",
)

_SCSS = (
    "$primary: #0071e3;\n"
    "$radius: 980px;\n"
    "$sp: 1rem;\n\n"
    "@mixin pill($bg) {\n"
    "  display: inline-block;\n"
    "  padding: $sp ($sp * 2);\n"
    "  background: $bg;\n"
    "  border-radius: $radius;\n"
    "  color: #fff;\n"
    "  border: none;\n"
    "}\n\n"
    ".card {\n"
    "  background: #f5f5f7;\n"
    "  border-radius: 1rem;\n"
    "  padding: $sp * 2;\n"
    "  h2 { margin: 0 0 $sp; color: #1d1d1f; }\n"
    "  .btn { @include pill($primary); cursor: pointer; }\n"
    "  &:hover { box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08); }\n"
    "}\n"
)

_HTML = (
    '<!doctype html><html lang="pt-br"><head><meta charset="utf-8">'
    '<link rel="stylesheet" href="styles.css"></head>'
    '<body><div class="card"><h2>Olá</h2><button class="btn">Começar</button></div></body></html>'
)


@_requires_sass
def test_compile_passes_for_valid_scss():
    result = check(
        language="scss",
        operation="compile",
        files=[{"path": "styles.scss", "content": _SCSS}],
        timeout_ms=60000,
    )
    assert result.passed, result.stderr
    assert result.metadata["language"] == "scss"


@_requires_sass
def test_compile_fails_for_undefined_variable():
    result = check(
        language="scss",
        operation="compile",
        files=[{"path": "bad.scss", "content": ".x { color: $naoexiste; }"}],
        timeout_ms=60000,
    )
    assert not result.passed
    assert result.errors[0].code == "COMPILATION_ERROR"


@_requires_sass
def test_run_compiles_and_renders_in_browser():
    result = check(
        language="scss",
        operation="run",
        files=[{"path": "styles.scss", "content": _SCSS}, {"path": "index.html", "content": _HTML}],
        entrypoint="index.html",
        timeout_ms=60000,
    )
    assert result.passed, result.to_json()
    assert result.errors == []


@_requires_sass
def test_run_requires_html_to_render():
    result = check(
        language="scss",
        operation="run",
        files=[{"path": "styles.scss", "content": _SCSS}],
        timeout_ms=60000,
    )
    assert not result.passed
    assert result.errors[0].code == "INCOMPLETE_SOLUTION"


def test_scss_without_scss_file_is_unsupported():
    result = check(
        language="scss",
        operation="compile",
        files=[{"path": "styles.css", "content": ".x { color: red; }"}],
    )
    assert not result.passed
    assert result.errors[0].code == "UNSUPPORTED_LANGUAGE"


def test_scss_registered_as_language():
    from src.checker.core import registered_languages

    assert "scss" in registered_languages()
