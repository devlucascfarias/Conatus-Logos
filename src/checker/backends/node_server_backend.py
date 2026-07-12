"""Backend do checker pra scripts Node.js server-side / CLI (D-checker-node-server-backend, ver
docs/PLAN.md). Registrado em `src.checker.core` sob "node" — distinto de "javascript"/"typescript"
(`node_backend.py`, orientado a React/Vite/browser). Aqui não há build nem renderização: o
código roda como um processo Node de verdade.

Testar um servidor HTTP (Express) não exige bind de porta nem subprocess separado: `supertest`
(já no projeto-template) envolve o app Express e despacha requisições reais pelo pipeline de
rotas/middleware de verdade, sem mock — é o jeito idiomático de testar Express em produção.
Então `compile_and_test` já cobre "subir servidor, bater endpoint" reaproveitando `vitest`
(mesmo mecanismo de `node_backend.py`). `run` executa um script (CLI/uso direto) via `node`
de verdade, capturando stdout/stderr/exit code reais."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from .. import errors
from ..core import CheckError, CheckFile, CheckResult, register_backend
from ._subprocess_utils import materialize_files, run_command
from .node_backend import _bin_path, _link_node_modules, _node_available, _template_ready, _test, lint as _lint

_TEST_SUFFIXES = (".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.tsx", ".spec.js")


def _unavailable(reason: str) -> CheckResult:
    return CheckResult(
        passed=False,
        errors=[CheckError(code=errors.MISSING_DEPENDENCY, message=reason)],
        stdout="",
        stderr="",
        metadata={"language": "node", "duration_ms": 0},
    )


def _syntax_check(base_dir: Path, files: list[CheckFile], timeout_ms: int) -> CheckResult:
    has_ts = any(f.path.endswith((".ts", ".tsx")) for f in files)
    if has_ts:
        # Sem tsconfig.json (server-side não usa o do template, orientado a React/Vite) e sem
        # arquivos passados explicitamente, `tsc` não sabe o que compilar e imprime a TELA DE
        # AJUDA (saindo com código 1) em vez de um erro real — achado real ao testar, não
        # hipótese. Corrigido passando os .ts/.tsx explicitamente na linha de comando.
        ts_files = [f.path for f in files if f.path.endswith((".ts", ".tsx"))]
        result = run_command(
            [
                str(_bin_path(base_dir, "tsc")), "--noEmit", "--allowJs",
                "--moduleResolution", "bundler", "--module", "esnext", "--target", "es2020",
                "--allowImportingTsExtensions",  # permite `import ... from './x.ts'` (ESM real)
                # Sem --strict (desligado por padrão sem tsconfig), o estreitamento de tipo por
                # NEGAÇÃO booleana (`if (!result.ok)`) numa union discriminada não funciona
                # corretamente — só `=== false` estreitava; achado real testando, reproduzido
                # isoladamente antes de decidir a flag. --strict bate com o tsconfig do template
                # (D-checker-node-backend) de qualquer forma, é o padrão profissional.
                "--strict",
                *ts_files,
            ],
            base_dir,
            timeout_ms,
        )
    else:
        node_bin = "node"
        import shutil

        node_path = shutil.which(node_bin)
        js_files = [f.path for f in files if f.path.endswith(".js") and not f.path.endswith(tuple(_TEST_SUFFIXES))]
        if not js_files:
            return CheckResult(
                passed=False,
                errors=[CheckError(code=errors.UNSUPPORTED_LANGUAGE, message="nenhum arquivo .js/.ts em 'files'")],
                stdout="",
                stderr="",
                metadata={"language": "node", "duration_ms": 0},
            )
        combined_out, combined_err, total_ms, failed = "", "", 0, None
        for path in js_files:
            r = run_command([node_path, "--check", path], base_dir, timeout_ms)
            total_ms += r.duration_ms
            combined_out += r.stdout
            combined_err += r.stderr
            if r.returncode != 0:
                failed = r
                break
        if failed is not None:
            return CheckResult(
                passed=False,
                errors=[CheckError(code=errors.SYNTAX_ERROR, message=combined_err.strip()[-2000:])],
                stdout=combined_out,
                stderr=combined_err,
                metadata={"language": "node", "duration_ms": total_ms},
            )
        return CheckResult(passed=True, errors=[], stdout=combined_out, stderr=combined_err, metadata={"language": "node", "duration_ms": total_ms})

    metadata = {"language": "node", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(passed=False, errors=[CheckError(code=errors.TIMEOUT, message="timeout ao checar tipos")], stdout=result.stdout, stderr=result.stderr, metadata=metadata)
    passed = result.returncode == 0
    combined = (result.stdout + result.stderr).strip()
    found_errors = [] if passed else [CheckError(code=errors.COMPILATION_ERROR, message=combined[-2000:])]
    return CheckResult(passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata)


def _run(base_dir: Path, entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not entrypoint:
        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INCOMPLETE_SOLUTION, message="operation=run requer 'entrypoint'")],
            stdout="",
            stderr="",
            metadata={"language": "node", "duration_ms": 0},
        )
    import shutil

    node_path = shutil.which("node")
    result = run_command([node_path, entrypoint], base_dir, timeout_ms)
    metadata = {"language": "node", "duration_ms": result.duration_ms}
    if result.timed_out:
        return CheckResult(passed=False, errors=[CheckError(code=errors.TIMEOUT, message=f"timeout ao executar {entrypoint}")], stdout=result.stdout, stderr=result.stderr, metadata=metadata)
    passed = result.returncode == 0
    found_errors = [] if passed else [CheckError(code=errors.RUNTIME_ERROR, message=result.stderr.strip()[-2000:] or f"processo saiu com código {result.returncode}")]
    return CheckResult(passed=passed, errors=found_errors, stdout=result.stdout, stderr=result.stderr, metadata=metadata)


def check_node_server(operation: str, files: list[CheckFile], entrypoint: Optional[str], timeout_ms: int) -> CheckResult:
    if not _node_available():
        return _unavailable("toolchain 'node'/'npm' não disponível no ambiente")
    if not _template_ready():
        return _unavailable(
            "projeto-template do checker (checker_templates/react_three_fiber) não tem node_modules "
            "instalado — rode 'npm install' lá antes de usar este backend (D-checker-node-server-backend)"
        )
    with tempfile.TemporaryDirectory(prefix="praxis_checker_nodesrv_") as tmp:
        base_dir = Path(tmp)
        materialize_files(base_dir, files)
        _link_node_modules(base_dir)

        if operation == "lint":
            return _lint(base_dir, files, timeout_ms, language="node")
        if operation in ("syntax_check", "compile"):
            return _syntax_check(base_dir, files, timeout_ms)
        if operation == "compile_and_test":
            type_result = _syntax_check(base_dir, files, timeout_ms)
            if not type_result.passed:
                return type_result
            has_tests = any(f.path.endswith(_TEST_SUFFIXES) for f in files)
            if not has_tests:
                return type_result
            return _test(base_dir, timeout_ms)
        if operation == "run":
            return _run(base_dir, entrypoint, timeout_ms)

        return CheckResult(
            passed=False,
            errors=[CheckError(code=errors.INVALID_OUTPUT_FORMAT, message=f"operação desconhecida: {operation}")],
            stdout="",
            stderr="",
            metadata={"language": "node", "duration_ms": 0},
        )


register_backend("node", check_node_server)
