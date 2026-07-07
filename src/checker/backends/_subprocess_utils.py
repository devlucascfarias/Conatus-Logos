"""Utilitário de execução de subprocess compartilhado pelos backends do checker.

Isolado aqui (não em src/security) porque o checker roda compiladores/interpretadores
confiáveis do próprio ambiente de desenvolvimento (não código arbitrário do usuário final) —
o sandbox de segurança pesado (seção 7) é aplicado pela ferramenta `shell`/`Tool Executor`,
não pelo checker. Ainda assim, timeout e truncamento de saída são sempre aplicados aqui.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 65536  # mesmo teto de configs/sandbox_policy.yaml (seção 7.3)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int


def _truncate(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n...[truncado]"


def run_command(cmd: list[str], cwd: Path, timeout_ms: int) -> CommandResult:
    import time

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return CommandResult(
            returncode=proc.returncode,
            stdout=_truncate(proc.stdout),
            stderr=_truncate(proc.stderr),
            timed_out=False,
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            returncode=-1,
            stdout=_truncate(stdout),
            stderr=_truncate(stderr),
            timed_out=True,
            duration_ms=duration_ms,
        )


def materialize_files(base_dir: Path, files: list) -> None:
    """Escreve CheckFile(path, content) em disco sob base_dir, criando subdiretórios."""
    for f in files:
        dest = base_dir / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.content, encoding="utf-8")
