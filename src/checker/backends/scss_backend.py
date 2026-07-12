"""Backend do checker para SCSS (D-checker-scss-backend, ver docs/PLAN.md). Registrado em
`src.checker.core` sob "scss".

Design deliberado pra NÃO ficar preso a SCSS: o `.scss` é compilado pra `.css` de verdade (dart-
sass, via o `sass` do projeto-template Node), e a operação `run` renderiza o `.css` resultante no
MESMO Chromium headless que o backend `html` já usa pra CSS puro (`render_and_capture`). Assim
CSS e SCSS compartilham o mesmo oráculo visual: `.css` valida direto pelo backend `html`, `.scss`
compila-e-valida por aqui, os dois terminam renderizados no navegador real.

Reaproveita a junção NTFS do `node_modules` do template (mesma infra de `node_backend`) pra achar
o binário `sass`, sem `npm install` por chamada. D11: `lint` (stylelint) ainda não implementado,
retorna `MISSING_DEPENDENCY` estruturado em vez de fingir.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files, run_command
from .frontend_backend import render_and_capture
from .node_backend import _bin_path, _link_node_modules, _node_available, _template_ready


def _unavailable(reason: str) -> CheckResult:
    return CheckResult(
        passed=False,
        errors=[CheckError(code=errors.MISSING_DEPENDENCY, message=reason)],
        stdout="",
        stderr="",
        metadata={"language": "scss", "duration_ms": 0},
    )


def _compile_all_scss(
    base_dir: Path, files: list[CheckFile], timeout_ms: int
) -> tuple[list[CheckFile], list[CheckError], int]:
    """Compila cada `.scss` pra um `.css` irmão (mesmo nome). Devolve os CheckFile de CSS gerados,
    os erros de compilação (se houver) e a duração total."""
    compiled: list[CheckFile] = []
    compile_errors: list[CheckError] = []
    total_ms = 0
    sass_bin = _bin_path(base_dir, "sass")
    for f in files:
        if not f.path.endswith(".scss"):
            continue
        css_path = f.path[:-5] + ".css"
        result = run_command(
            [str(sass_bin), "--no-source-map", "--style=expanded", f.path, css_path],
            base_dir,
            timeout_ms,
        )
        total_ms += result.duration_ms
        if result.timed_out:
            compile_errors.append(CheckError(code=errors.TIMEOUT, message=f"timeout ao compilar {f.path}"))
            continue
        if result.returncode != 0:
            combined = (result.stdout + result.stderr).strip()
            compile_errors.append(CheckError(code=errors.COMPILATION_ERROR, message=f"{f.path}: {combined[-2000:]}"))
            continue
        css_file = base_dir / css_path
        if css_file.exists():
            compiled.append(
                CheckFile(path=css_path.replace("\\", "/"), content=css_file.read_text(encoding="utf-8"))
            )
    return compiled, compile_errors, total_ms


def check_scss(operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not any(f.path.endswith(".scss") for f in files):
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.UNSUPPORTED_LANGUAGE, message="nenhum arquivo .scss em 'files'")],
            stdout="",
            stderr="",
            metadata={"language": "scss", "duration_ms": 0},
        )
    if operation == "lint":
        return _unavailable("lint (stylelint) ainda não implementado para scss")
    if operation not in ("syntax_check", "compile", "compile_and_test", "run"):
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
            stdout="",
            stderr="",
            metadata={"language": "scss", "duration_ms": 0},
        )
    if not _node_available():
        return _unavailable("toolchain 'node'/'npm' não disponível no ambiente")
    if not _template_ready():
        return _unavailable(
            "projeto-template do checker (checker_templates/react_three_fiber) não tem node_modules "
            "instalado — rode 'npm install' lá antes de usar este backend (D-checker-scss-backend)"
        )

    with tempfile.TemporaryDirectory(prefix="praxis_checker_scss_") as tmp:
        base_dir = Path(tmp)
        materialize_files(base_dir, files)
        _link_node_modules(base_dir)

        compiled, compile_errors, compile_ms = _compile_all_scss(base_dir, files, timeout_ms)
        if compile_errors:
            return CheckResult(
                passed=False,
                errors=compile_errors,
                stdout="",
                stderr="\n".join(e.message for e in compile_errors),
                metadata={"language": "scss", "duration_ms": compile_ms},
            )

        # syntax_check/compile/compile_and_test: compilou sem erro já é sucesso.
        if operation != "run":
            return CheckResult(
                passed=True,
                errors=[],
                stdout="",
                stderr="",
                metadata={"language": "scss", "duration_ms": compile_ms},
            )

        # run: renderiza o CSS compilado no mesmo Chromium do backend html (CSS e SCSS
        # compartilham o oráculo visual). Precisa de um .html (o entrypoint) que linke o .css gerado.
        render_files = [f for f in files if not f.path.endswith(".scss")] + compiled
        if not any(f.path.endswith(".html") for f in render_files):
            return CheckResult(
                passed=False,
                errors=[
                    CheckError(
                        code=errors.INCOMPLETE_SOLUTION,
                        message="operation=run precisa de um .html que linke o CSS compilado do SCSS",
                    )
                ],
                stdout="",
                stderr="",
                metadata={"language": "scss", "duration_ms": compile_ms},
            )
        try:
            render = render_and_capture(render_files, entrypoint, timeout_ms)
        except RuntimeError as exc:
            return _unavailable(str(exc))

        found_errors = [
            CheckError(code=errors.RUNTIME_ERROR, message=m) for m in render.console_errors + render.page_errors
        ]
        return CheckResult(
            passed=render.passed,
            errors=found_errors,
            stdout="",
            stderr="\n".join(render.console_errors + render.page_errors),
            metadata={
                "language": "scss",
                "duration_ms": compile_ms + render.duration_ms,
                "console_warnings": render.console_warnings,
            },
        )


register_backend("scss", check_scss)
