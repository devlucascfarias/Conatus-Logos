"""Sandbox de execução (PLAN.md seção 7 — D6: subprocess isolado + diretório de trabalho
descartável, não Docker). Nenhuma ferramenta deve chamar `subprocess` diretamente fora daqui —
ver seção 5.2 (Tool Executor sempre passa por sandbox).

Limitação conhecida (seção 17): este dev environment roda em Windows, onde `resource.setrlimit`
(limite de memória, seção 7.3) não existe — o limite é aplicado via `preexec_fn` só quando o
processo roda em POSIX/Linux (ambiente real do Colab). Em Windows o limite de memória vira
no-op documentado, não um erro silencioso.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .policy import DEFAULT_POLICY_PATH, SandboxPolicy

# Variáveis mecânicas exigidas para o shell funcionar no Windows (não são segredos — cmd.exe
# não inicializa corretamente sem elas). Fora do escopo da allowlist de segredos (seção 7.8).
_WINDOWS_MECHANICAL_ENV = ("SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR")


@dataclass(frozen=True)
class ShellExecutionResult:
    allowed: bool
    denial_reason: Optional[str]
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int


def _rlimit_preexec_fn(max_memory_bytes: int):
    def _apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        except Exception:
            pass

    return _apply


class SandboxContext:
    """Um workspace efêmero por sessão do agente (seção 7.2, 7.10)."""

    def __init__(self, policy: Optional[SandboxPolicy] = None, workspace_root: Optional[Path] = None):
        self._policy = policy or SandboxPolicy.load(DEFAULT_POLICY_PATH)
        self._owns_workspace = workspace_root is None
        self._workspace = Path(workspace_root) if workspace_root else Path(tempfile.mkdtemp(prefix="praxis_ws_"))
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    def resolve_path(self, relative_path: str) -> Optional[Path]:
        """Resolve caminho relativo ao workspace; None se escapar dele (seção 7.4)."""
        candidate = (self._workspace / relative_path).resolve()
        try:
            candidate.relative_to(self._workspace.resolve())
        except ValueError:
            return None
        return candidate

    def confirmation_allowed(self, spec, args: dict) -> bool:
        """Seção 7.7 — decide se uma ferramenta com requires_confirmation=true prossegue."""
        if not spec.requires_confirmation:
            return True
        return self._policy.allow_confirmation()

    def run_shell(
        self, command: str, cwd: Optional[str] = None, timeout_ms: Optional[int] = None
    ) -> ShellExecutionResult:
        allowed, reason = self._policy.is_command_allowed(command)
        if not allowed:
            return ShellExecutionResult(
                allowed=False,
                denial_reason=reason,
                returncode=None,
                stdout="",
                stderr="",
                timed_out=False,
                duration_ms=0,
            )

        effective_cwd = self.resolve_path(cwd or ".")
        if effective_cwd is None:
            return ShellExecutionResult(
                allowed=False,
                denial_reason="cwd escapa do workspace",
                returncode=None,
                stdout="",
                stderr="",
                timed_out=False,
                duration_ms=0,
            )

        env = {k: v for k, v in os.environ.items() if k in self._policy.allowed_env_passthrough}
        if os.name == "nt":
            for key in _WINDOWS_MECHANICAL_ENV:
                if key in os.environ:
                    env[key] = os.environ[key]

        timeout_s = (timeout_ms or self._policy.default_timeout_ms) / 1000
        preexec_fn = _rlimit_preexec_fn(self._policy.max_memory_bytes) if os.name != "nt" else None

        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(effective_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=preexec_fn,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            return ShellExecutionResult(
                allowed=True,
                denial_reason=None,
                returncode=proc.returncode,
                stdout=self._truncate(proc.stdout),
                stderr=self._truncate(proc.stderr),
                timed_out=False,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            return ShellExecutionResult(
                allowed=True,
                denial_reason=None,
                returncode=-1,
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                timed_out=True,
                duration_ms=duration_ms,
            )

    def _truncate(self, text: str) -> str:
        max_bytes = self._policy.max_output_bytes
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n...[truncado]"

    def cleanup(self) -> None:
        """Seção 7.10 — limpar ambiente ao final da sessão, não só ao final do processo."""
        if self._owns_workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)

    def __enter__(self) -> "SandboxContext":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()
