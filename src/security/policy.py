"""Política de segurança do sandbox (PLAN.md seção 7), carregada de
`configs/sandbox_policy.yaml`. Nenhum valor de allowlist/denylist/limite de recurso deve ser
hardcoded fora deste módulo — qualquer mudança de política é uma mudança de config, não de
código (mesmo princípio de D10 aplicado a segurança)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POLICY_PATH = REPO_ROOT / "configs" / "sandbox_policy.yaml"


@dataclass(frozen=True)
class SandboxPolicy:
    default_timeout_ms: int
    checker_timeout_ms: int
    max_output_bytes: int
    max_memory_bytes: int
    allowlist_binaries: frozenset
    git_allowed_subcommands: frozenset
    denylist_patterns: tuple
    shell_network_enabled: bool
    package_install_enabled: bool
    web_search_allowed_hosts: frozenset
    confirmation_mode: str
    allowed_env_passthrough: frozenset
    redaction_patterns: tuple

    @classmethod
    def load(cls, path: Path | str = DEFAULT_POLICY_PATH) -> "SandboxPolicy":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            default_timeout_ms=raw["resource_limits"]["default_timeout_ms"],
            checker_timeout_ms=raw["resource_limits"]["checker_compile_and_test_timeout_ms"],
            max_output_bytes=raw["resource_limits"]["max_output_bytes"],
            max_memory_bytes=raw["resource_limits"]["max_memory_bytes"],
            allowlist_binaries=frozenset(raw["shell"]["allowlist_binaries"]),
            git_allowed_subcommands=frozenset(raw["shell"]["git_allowed_subcommands"]),
            denylist_patterns=tuple(raw["shell"]["denylist_patterns"]),
            shell_network_enabled=raw["network"]["shell_network_enabled"],
            package_install_enabled=raw["network"]["package_install_enabled"],
            web_search_allowed_hosts=frozenset(raw["network"]["web_search_allowed_hosts"]),
            confirmation_mode=raw["confirmation"]["mode"],
            allowed_env_passthrough=frozenset(raw["secrets"]["allowed_env_passthrough"]),
            redaction_patterns=tuple(raw["secrets"]["redaction_patterns"]),
        )

    def with_confirmation_mode(self, mode: str) -> "SandboxPolicy":
        """Cópia da política com outro modo de confirmação — útil para testes/dev
        (seção 7.7: 'auto_approve_safe' | 'deny_all_destructive')."""
        return SandboxPolicy(
            default_timeout_ms=self.default_timeout_ms,
            checker_timeout_ms=self.checker_timeout_ms,
            max_output_bytes=self.max_output_bytes,
            max_memory_bytes=self.max_memory_bytes,
            allowlist_binaries=self.allowlist_binaries,
            git_allowed_subcommands=self.git_allowed_subcommands,
            denylist_patterns=self.denylist_patterns,
            shell_network_enabled=self.shell_network_enabled,
            package_install_enabled=self.package_install_enabled,
            web_search_allowed_hosts=self.web_search_allowed_hosts,
            confirmation_mode=mode,
            allowed_env_passthrough=self.allowed_env_passthrough,
            redaction_patterns=self.redaction_patterns,
        )

    def is_command_allowed(self, command: str) -> tuple:
        """Retorna (allowed: bool, reason_if_denied: str | None) — seção 7.5.

        Denylist tem prioridade sobre allowlist: um padrão bloqueado nunca passa, mesmo que
        o binário base esteja na allowlist (ex.: "git push" — git está na allowlist, mas o
        subcomando é bloqueado explicitamente)."""
        stripped = command.strip()
        if not stripped:
            return False, "comando vazio"

        lowered = stripped.lower()
        for pattern in self.denylist_patterns:
            if pattern.lower() in lowered:
                return False, f"padrão bloqueado por denylist: {pattern!r}"

        first_token = stripped.split()[0].strip('"\'')
        # Compara tanto o nome completo (POSIX/Colab: "python") quanto sem extensão
        # (Windows dev: "python.exe" -> stem "python") contra a mesma entrada da allowlist.
        binary_full = Path(first_token).name
        binary_stem = Path(first_token).stem
        if binary_full in self.allowlist_binaries:
            binary = binary_full
        elif binary_stem in self.allowlist_binaries:
            binary = binary_stem
        else:
            return False, f"binário fora da allowlist: {binary_full!r}"

        if binary == "git":
            tokens = stripped.split()
            subcommand = tokens[1] if len(tokens) > 1 else None
            if subcommand not in self.git_allowed_subcommands:
                return False, f"subcomando git não permitido (somente leitura): {subcommand!r}"

        return True, None

    def allow_confirmation(self) -> bool:
        """Seção 7.7 — no harness de desenvolvimento, confirmação é uma política configurável.
        Só operações que exigem confirmação e cujo modo é 'auto_approve_safe' prosseguem."""
        return self.confirmation_mode == "auto_approve_safe"

    def redact(self, text: str) -> str:
        redacted = text
        for pattern in self.redaction_patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted)
        return redacted
