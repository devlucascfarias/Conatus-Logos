"""Backend do checker para JavaScript/TypeScript, incluindo React + Vite + React Three Fiber
(D-checker-node-backend, ver docs/PLAN.md). Registrado em `src.checker.core` sob "javascript" e
"typescript".

Reaproveita um projeto-template pré-instalado (`checker_templates/react_three_fiber/` — só
`package.json`/lockfile versionados, `node_modules` NÃO versionado, precisa de `npm install`
manual uma vez) via junção de diretório NTFS (não exige admin no Windows) pro `node_modules` de
cada verificação — evita rodar `npm install` a cada chamada do checker, mesmo princípio do cache
de módulo global que já torna `go build ./...` rápido no backend Go.

D11: nunca finge validar o que não validou. `lint` (eslint) ainda não implementado, retorna
`MISSING_DEPENDENCY` estruturado em vez de fingir sucesso."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files, run_command

TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "checker_templates" / "react_three_fiber"
)

# Config sempre copiada do template quando o agente não traz a própria.
_CONFIG_SCAFFOLD = ("vite.config.ts", "tsconfig.json", "package.json")
# Bootstrap = par ligado (index.html referencia src/main.tsx). Copiado SÓ quando o agente não
# traz o próprio entry point — se o agente fornece index.html ou algum src/main.*, ele está
# bootstrapando à própria maneira (ex.: Three.js vanilla sem React), e copiar o main.tsx do
# template criaria um arquivo órfão importando um `./App` inexistente, quebrando o `tsc`.
_BOOTSTRAP_SCAFFOLD = ("index.html", "src/main.tsx")
_ENTRY_MARKERS = ("index.html", "src/main.ts", "src/main.tsx", "src/main.jsx", "src/main.js")
_TEST_SUFFIXES = (".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx")


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _template_ready() -> bool:
    return (TEMPLATE_DIR / "node_modules").is_dir()


def _unavailable_result(reason: str) -> CheckResult:
    return CheckResult(
        passed=False,
        errors=[CheckError(code=errors.MISSING_DEPENDENCY, message=reason)],
        stdout="",
        stderr="",
        metadata={"language": "typescript", "duration_ms": 0},
    )


def _link_node_modules(base_dir: Path) -> None:
    """Junção NTFS (não exige admin, diferente de symlink) de `base_dir/node_modules` pro
    `node_modules` já instalado no projeto-template."""
    target = base_dir / "node_modules"
    source = TEMPLATE_DIR / "node_modules"
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f'New-Item -ItemType Junction -Path "{target}" -Target "{source}" -Force | Out-Null',
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )


def _copy_template_file(name: str, base_dir: Path) -> None:
    dest = base_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((TEMPLATE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")


def _prepare_project(files: list[CheckFile], base_dir: Path) -> None:
    # Scaffolding do template só é usado quando o conjunto de arquivos do checker não já traz sua
    # própria versão — o agente pode sobrescrever qualquer um.
    provided_paths = {f.path for f in files}
    for name in _CONFIG_SCAFFOLD:
        if name not in provided_paths and (TEMPLATE_DIR / name).exists():
            _copy_template_file(name, base_dir)
    agent_has_own_entry = any(marker in provided_paths for marker in _ENTRY_MARKERS)
    if not agent_has_own_entry:
        for name in _BOOTSTRAP_SCAFFOLD:
            if (TEMPLATE_DIR / name).exists():
                _copy_template_file(name, base_dir)
    materialize_files(base_dir, files)
    _link_node_modules(base_dir)


def _bin_path(base_dir: Path, name: str) -> Path:
    return base_dir / "node_modules" / ".bin" / f"{name}.cmd"


def _typecheck(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command([str(_bin_path(base_dir, "tsc")), "--noEmit"], base_dir, timeout_ms)
    metadata = {"language": "typescript", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao rodar tsc --noEmit")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )
    passed = result.returncode == 0
    combined = (result.stdout + result.stderr).strip()
    found_errors = [] if passed else [CheckError(code=errors.COMPILATION_ERROR, message=combined[-4000:])]
    return CheckResult(passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata)


def _test(base_dir: Path, timeout_ms: int) -> CheckResult:
    result = run_command(
        [str(_bin_path(base_dir, "vitest")), "run", "--environment", "jsdom"], base_dir, timeout_ms
    )
    metadata = {"language": "typescript", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao rodar vitest")],
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
        )
    passed = result.returncode == 0
    combined = (result.stdout + result.stderr).strip()
    found_errors = [] if passed else [CheckError(code=errors.TEST_FAILURE, message=combined[-4000:])]
    return CheckResult(passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata)


def _serve_and_render(dist_dir: Path, entrypoint: str, timeout_ms: int):
    """Serve `dist_dir` por HTTP local (não `file://`) e renderiza de verdade num Chromium
    headless — scripts `type="module"` (o que todo build do Vite produz) são bloqueados por CORS
    quando carregados via `file://`, mesma restrição que um navegador real aplicaria; só HTTP
    real contorna isso, não é uma limitação artificial deste checker."""
    import functools
    import http.server
    import threading
    import time

    from playwright.sync_api import sync_playwright

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    console_errors: list[str] = []
    console_warnings: list[str] = []
    page_errors: list[str] = []
    start = time.monotonic()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
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
                    page.goto(f"http://127.0.0.1:{port}/{entrypoint}", timeout=timeout_ms)
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except Exception as exc:  # noqa: BLE001
                    page_errors.append(f"falha ao carregar a página: {exc}")
            finally:
                browser.close()
    finally:
        httpd.shutdown()

    duration_ms = int((time.monotonic() - start) * 1000)
    passed = not console_errors and not page_errors
    return passed, console_errors, console_warnings, page_errors, duration_ms


def _run(base_dir: Path, entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    build_result = run_command(
        [str(_bin_path(base_dir, "vite")), "build", "--outDir", "dist"], base_dir, timeout_ms
    )
    metadata = {"language": "typescript", "duration_ms": build_result.duration_ms}
    if build_result.timed_out:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.TIMEOUT, message="timeout ao rodar vite build")],
            stdout=build_result.stdout,
            stderr=build_result.stderr,
            metadata=metadata,
        )
    if build_result.returncode != 0:
        combined = (build_result.stdout + build_result.stderr).strip()
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.COMPILATION_ERROR, message=combined[-4000:])],
            stdout=build_result.stdout,
            stderr=build_result.stderr,
            metadata=metadata,
        )

    dist_dir = base_dir / "dist"
    if not (dist_dir / (entrypoint or "index.html")).exists():
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.FILE_NOT_FOUND, message=f"'{entrypoint or 'index.html'}' não existe em dist/ após o build")],
            stdout=build_result.stdout,
            stderr=build_result.stderr,
            metadata=metadata,
        )

    try:
        passed, console_errors, console_warnings, page_errors, render_ms = _serve_and_render(
            dist_dir, entrypoint or "index.html", timeout_ms
        )
    except ImportError as exc:
        return _unavailable_result(f"playwright não instalado: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _unavailable_result(f"chromium não disponível (rode 'playwright install chromium'): {exc}")

    found_errors = [CheckError(code=errors.RUNTIME_ERROR, message=m) for m in console_errors + page_errors]
    return CheckResult(
        passed=passed,
        errors=found_errors,
        stdout=build_result.stdout,
        stderr="\n".join(console_errors + page_errors),
        metadata={
            "language": "typescript",
            "duration_ms": build_result.duration_ms + render_ms,
            "console_warnings": console_warnings,
        },
    )


def check_node(operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not _node_available():
        return _unavailable_result("toolchain 'node'/'npm' não disponível no ambiente")
    if not _template_ready():
        return _unavailable_result(
            "projeto-template do checker Node (checker_templates/react_three_fiber) não tem "
            "node_modules instalado — rode 'npm install' lá antes de usar este backend"
        )

    with tempfile.TemporaryDirectory(prefix="praxis_checker_node_") as tmp:
        base_dir = Path(tmp)
        _prepare_project(files, base_dir)

        if operation in ("syntax_check", "compile"):
            return _typecheck(base_dir, timeout_ms)
        if operation == "compile_and_test":
            type_result = _typecheck(base_dir, timeout_ms)
            if not type_result.passed:
                return type_result
            has_tests = any(f.path.endswith(_TEST_SUFFIXES) for f in files)
            if not has_tests:
                return type_result
            return _test(base_dir, timeout_ms)
        if operation == "run":
            return _run(base_dir, entrypoint, timeout_ms)
        if operation == "lint":
            return _unavailable_result("lint (eslint) ainda não implementado para javascript/typescript")

        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
            stdout="",
            stderr="",
            metadata={"language": "typescript", "duration_ms": 0},
        )


register_backend("javascript", check_node)
register_backend("typescript", check_node)
