"""Testes do backend `html` do checker (docs/plan_frontend_specialization_wave3.md seção 3,
camada 1: "renderiza sem erro de console/JS").

Pulados (não falham) se o Playwright/Chromium não estiverem disponíveis no ambiente — o backend
em si já trata essa ausência como MISSING_DEPENDENCY estruturado (ver
test_missing_playwright_reports_structured_error_when_unavailable indiretamente via mensagem)."""

from __future__ import annotations

import pytest

from src.checker import check, errors


def _playwright_chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


_HAS_CHROMIUM = _playwright_chromium_available()
_requires_chromium = pytest.mark.skipif(
    not _HAS_CHROMIUM, reason="playwright/chromium não disponível neste ambiente"
)


@_requires_chromium
def test_run_passes_for_clean_html_page():
    result = check(
        language="html",
        operation="run",
        files=[{"path": "index.html", "content": "<html><body><h1>ok</h1></body></html>"}],
    )
    assert result.passed, result.stderr
    assert result.metadata["language"] == "html"
    assert result.errors == []


@_requires_chromium
def test_run_fails_on_console_error_from_uncaught_exception():
    result = check(
        language="html",
        operation="run",
        files=[
            {
                "path": "index.html",
                "content": "<html><body><script>undefinedFunctionCall()</script></body></html>",
            }
        ],
    )
    assert not result.passed
    assert result.errors
    assert result.errors[0].code == errors.RUNTIME_ERROR
    assert "undefinedFunctionCall" in result.errors[0].message


@_requires_chromium
def test_run_fails_on_explicit_console_error():
    result = check(
        language="html",
        operation="run",
        files=[
            {
                "path": "index.html",
                "content": '<html><body><script>console.error("componente quebrado")</script></body></html>',
            }
        ],
    )
    assert not result.passed
    assert any("componente quebrado" in e.message for e in result.errors)


@_requires_chromium
def test_console_warning_does_not_fail_the_check():
    result = check(
        language="html",
        operation="run",
        files=[
            {
                "path": "index.html",
                "content": '<html><body><script>console.warn("só um aviso")</script></body></html>',
            }
        ],
    )
    assert result.passed
    assert "só um aviso" in result.metadata["console_warnings"][0]


@_requires_chromium
def test_multi_file_project_uses_index_html_as_default_entrypoint():
    result = check(
        language="html",
        operation="run",
        files=[
            {"path": "index.html", "content": "<html><body><h1>home</h1></body></html>"},
            {"path": "about.html", "content": "<html><body><script>boom()</script></body></html>"},
        ],
    )
    assert result.passed


@_requires_chromium
def test_explicit_entrypoint_is_respected():
    result = check(
        language="html",
        operation="run",
        files=[
            {"path": "index.html", "content": "<html><body><h1>home</h1></body></html>"},
            {"path": "about.html", "content": "<html><body><script>boom()</script></body></html>"},
        ],
        entrypoint="about.html",
    )
    assert not result.passed


def test_no_html_file_returns_failure_not_exception():
    result = check(
        language="html",
        operation="run",
        files=[{"path": "styles.css", "content": "body { color: red; }"}],
    )
    assert not result.passed


def test_lint_operation_is_structured_not_implemented():
    # Camada 3 (acessibilidade via axe-core) ainda não existe neste projeto — o backend deve
    # falhar de forma estruturada (MISSING_DEPENDENCY), nunca fingir que rodou uma checagem.
    result = check(
        language="html",
        operation="lint",
        files=[{"path": "index.html", "content": "<html><body></body></html>"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.MISSING_DEPENDENCY


def test_unknown_operation_returns_structured_error():
    result = check(
        language="html",
        operation="not_a_real_operation",
        files=[{"path": "index.html", "content": "<html></html>"}],
    )
    assert not result.passed
    assert result.errors[0].code == errors.INVALID_OUTPUT_FORMAT
