"""Backend do checker para frontend estático (PLAN.md seção 10.4 + docs/
plan_frontend_specialization_wave3.md seção 3): fase 1 do pivô de frontend, HTML/CSS/JS sem
build step. Registrado em `src.checker.core` sob a chave "html".

Camada 1 do design em camadas do checker de frontend (seção 3 do plano): renderiza a página de
verdade num Chromium headless (Playwright) e falha se houver erro de console ou exceção JS não
tratada — objetivo, sem julgamento de estética (isso fica para `scripts/render_frontend_preview.py`
+ inspeção visual, nunca fingido como métrica automática).

`render_and_capture` é a função reutilizável: tanto o wrapper `check_html` (chamado pela
ferramenta `checker` em tempo de agente/dataset) quanto scripts offline de geração/curadoria de
dataset a chamam diretamente — arquitetura independente das ferramentas MCP `preview_*` do
próprio Claude (essas são de sessão, não invocáveis por um script batch offline)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files


@dataclass(frozen=True)
class RenderResult:
    passed: bool
    console_errors: list[str] = field(default_factory=list)
    console_warnings: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    duration_ms: int = 0


def _pick_entrypoint(base_dir: Path, files: list[CheckFile], entrypoint: Optional[str]) -> Optional[Path]:
    if entrypoint:
        candidate = base_dir / entrypoint
        return candidate if candidate.exists() else None
    html_files = [f for f in files if f.path.endswith(".html")]
    if not html_files:
        return None
    preferred = next((f for f in html_files if Path(f.path).name == "index.html"), html_files[0])
    return base_dir / preferred.path


def render_and_capture(
    files: list[CheckFile],
    entrypoint: Optional[str],
    timeout_ms: int,
    screenshot_path: Optional[Path] = None,
) -> RenderResult:
    """Renderiza `entrypoint` (ou o primeiro/`.html`/`index.html`) num Chromium headless real,
    captura console.error/console.warn e exceções JS não tratadas. Levanta `RuntimeError` se o
    Playwright ou o binário do Chromium não estiverem disponíveis — quem chama decide como isso
    vira MISSING_DEPENDENCY (checker) ou uma falha direta (script offline)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(f"playwright não instalado: {exc}") from exc

    import time

    with tempfile.TemporaryDirectory(prefix="praxis_checker_html_") as tmp:
        base_dir = Path(tmp)
        materialize_files(base_dir, files)
        entry = _pick_entrypoint(base_dir, files, entrypoint)
        if entry is None:
            return RenderResult(passed=False, console_errors=["nenhum arquivo .html encontrado em 'files'"])

        console_errors: list[str] = []
        console_warnings: list[str] = []
        page_errors: list[str] = []
        start = time.monotonic()

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                raise RuntimeError(f"chromium não disponível (rode 'playwright install chromium'): {exc}") from exc
            try:
                page = browser.new_page()
                page.on(
                    "console",
                    lambda msg: (
                        console_errors.append(msg.text)
                        if msg.type == "error"
                        else console_warnings.append(msg.text) if msg.type == "warning" else None
                    ),
                )
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                try:
                    page.goto(entry.as_uri(), timeout=timeout_ms)
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except Exception as exc:
                    page_errors.append(f"falha ao carregar a página: {exc}")
                if screenshot_path is not None:
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshot_path), full_page=True)
            finally:
                browser.close()

        duration_ms = int((time.monotonic() - start) * 1000)
        passed = not console_errors and not page_errors
        return RenderResult(
            passed=passed,
            console_errors=console_errors,
            console_warnings=console_warnings,
            page_errors=page_errors,
            screenshot_path=str(screenshot_path) if screenshot_path is not None else None,
            duration_ms=duration_ms,
        )


def _unavailable_result(reason: str) -> CheckResult:
    return CheckResult(
        passed=False,
        errors=[CheckError(code=errors.MISSING_DEPENDENCY, message=reason)],
        stdout="",
        stderr="",
        metadata={"language": "html", "duration_ms": 0},
    )


def _render_check(files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    try:
        result = render_and_capture(files, entrypoint, timeout_ms)
    except RuntimeError as exc:
        return _unavailable_result(str(exc))

    found_errors = [CheckError(code=errors.RUNTIME_ERROR, message=m) for m in result.console_errors]
    found_errors += [CheckError(code=errors.RUNTIME_ERROR, message=m) for m in result.page_errors]

    return CheckResult(
        passed=result.passed,
        errors=found_errors,
        stdout="",
        stderr="\n".join(result.console_errors + result.page_errors),
        metadata={
            "language": "html",
            "duration_ms": result.duration_ms,
            "console_warnings": result.console_warnings,
        },
    )


def check_html(
    operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int
) -> CheckResult:
    if operation in ("run", "syntax_check", "compile", "compile_and_test"):
        return _render_check(files, entrypoint, timeout_ms)
    if operation == "lint":
        # Camada 3 (acessibilidade real via axe-core) fica pra depois — precisa do pacote
        # axe-core-python (não instalado ainda). Falhar de forma estruturada, nunca fingir.
        return _unavailable_result("checagem de acessibilidade (axe-core) ainda não implementada")

    return CheckResult(
        passed=False,
        errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
        stdout="",
        stderr="",
        metadata={"language": "html", "duration_ms": 0},
    )


register_backend("html", check_html)
