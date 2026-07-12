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
    shell_interpreters: frozenset
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
            shell_interpreters=frozenset(raw["shell"].get("shell_interpreters", [])),
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
            shell_interpreters=self.shell_interpreters,
            git_allowed_subcommands=self.git_allowed_subcommands,
            denylist_patterns=self.denylist_patterns,
            shell_network_enabled=self.shell_network_enabled,
            package_install_enabled=self.package_install_enabled,
            web_search_allowed_hosts=self.web_search_allowed_hosts,
            confirmation_mode=mode,
            allowed_env_passthrough=self.allowed_env_passthrough,
            redaction_patterns=self.redaction_patterns,
        )

    def is_command_allowed(self, binary: str, args: tuple = ()) -> tuple:
        """Retorna (allowed: bool, reason_if_denied: str | None) — seção 7.5.

        D-shell-v2: `binary`/`args` chegam já estruturados (argv), nunca uma string de shell
        — a ferramenta `shell` não passa mais por `shell=True` (seção 5.5/D-shell-v2), o que
        elimina a dependência de qual interpretador de shell existe no SO do host e fecha uma
        classe inteira de risco de injeção via `;`, `&&`, `|`, `` ` ``, `$()` (não há shell
        nenhum interpretando metacaracteres). A denylist continua funcionando por substring,
        mas contra uma reconstrução textual de `binary + args` só para fins de checagem —
        nunca é essa string que é executada.

        Denylist tem prioridade sobre allowlist: um padrão bloqueado nunca passa, mesmo que
        o binário base esteja na allowlist (ex.: "git push" — git está na allowlist, mas o
        subcomando é bloqueado explicitamente)."""
        binary = (binary or "").strip()
        if not binary:
            return False, "binário vazio"

        joined = " ".join([binary, *[str(a) for a in args]]).lower()
        for pattern in self.denylist_patterns:
            if pattern.lower() in joined:
                return False, f"padrão bloqueado por denylist: {pattern!r}"

        # Compara tanto o nome completo (POSIX/Colab: "python") quanto sem extensão
        # (Windows dev: "python.exe" -> stem "python") contra a mesma entrada da allowlist.
        binary_full = Path(binary).name
        binary_stem = Path(binary).stem
        if binary_full in self.allowlist_binaries:
            resolved_binary = binary_full
        elif binary_stem in self.allowlist_binaries:
            resolved_binary = binary_stem
        else:
            return False, f"binário fora da allowlist: {binary_full!r}"

        if resolved_binary == "git":
            subcommand = args[0] if args else None
            if subcommand not in self.git_allowed_subcommands:
                return False, f"subcomando git não permitido (somente leitura): {subcommand!r}"

        # Validação extra por interpretador (D-shell-crossplatform-hardening): a denylist é
        # substring sobre texto claro — um comando codificado em base64 (`powershell
        # -EncodedCommand <b64>`, `pwsh -e <b64>`) driblaria TODA a denylist porque os verbos
        # destrutivos não aparecem em texto. Bloqueio essas flags de bypass casando o ARG
        # inteiro (não substring frouxo), case-insensitive, só para binários interpretadores.
        if resolved_binary in self.shell_interpreters:
            bypass_flags = {"-e", "-ec", "-enc", "-encodedcommand", "-encoded", "/e"}
            for a in args:
                token = str(a).strip().lower()
                if token in bypass_flags:
                    return False, (
                        f"flag de comando codificado bloqueada em interpretador "
                        f"{resolved_binary!r}: {a!r} (driblaria a denylist)"
                    )

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
