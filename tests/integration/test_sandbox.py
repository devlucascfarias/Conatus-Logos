"""Testes de sandbox (PLAN.md seção 19.2): comando fora da allowlist bloqueado, timeout
respeitado, workspace não permite escape de diretório."""

import sys

import pytest

from src.security import SandboxContext, SandboxPolicy


@pytest.fixture
def sandbox():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    ctx = SandboxContext(policy=policy)
    yield ctx
    ctx.cleanup()


def test_command_outside_allowlist_is_blocked(sandbox):
    result = sandbox.run_shell("curl https://example.com")
    assert not result.allowed
    assert "denylist" in result.denial_reason or "allowlist" in result.denial_reason


def test_denylisted_pattern_blocked_even_with_allowlisted_binary_prefix(sandbox):
    result = sandbox.run_shell("git push origin main")
    assert not result.allowed


def test_denylisted_destructive_command_blocked(sandbox):
    result = sandbox.run_shell("rm -rf /")
    assert not result.allowed


def test_allowlisted_command_executes(sandbox):
    result = sandbox.run_shell(f'"{sys.executable}" -c "print(1+1)"')
    assert result.allowed
    assert result.returncode == 0
    assert "2" in result.stdout


def test_timeout_is_respected(sandbox):
    result = sandbox.run_shell(
        f'"{sys.executable}" -c "import time; time.sleep(5)"', timeout_ms=300
    )
    assert result.allowed
    assert result.timed_out


def test_workspace_escape_via_dotdot_is_rejected(sandbox):
    resolved = sandbox.resolve_path("../../etc/passwd")
    assert resolved is None


def test_workspace_escape_via_absolute_path_is_rejected(sandbox):
    resolved = sandbox.resolve_path("C:/Windows/System32")
    assert resolved is None


def test_path_inside_workspace_resolves(sandbox):
    resolved = sandbox.resolve_path("subdir/file.txt")
    assert resolved is not None
    assert sandbox.workspace in resolved.parents or resolved.parent == sandbox.workspace


def test_output_is_truncated_beyond_max_bytes(sandbox):
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    small_limit_policy = SandboxPolicy(
        default_timeout_ms=policy.default_timeout_ms,
        checker_timeout_ms=policy.checker_timeout_ms,
        max_output_bytes=100,
        max_memory_bytes=policy.max_memory_bytes,
        allowlist_binaries=policy.allowlist_binaries,
        git_allowed_subcommands=policy.git_allowed_subcommands,
        denylist_patterns=policy.denylist_patterns,
        shell_network_enabled=policy.shell_network_enabled,
        package_install_enabled=policy.package_install_enabled,
        web_search_allowed_hosts=policy.web_search_allowed_hosts,
        confirmation_mode=policy.confirmation_mode,
        allowed_env_passthrough=policy.allowed_env_passthrough,
        redaction_patterns=policy.redaction_patterns,
    )
    ctx = SandboxContext(policy=small_limit_policy)
    try:
        result = ctx.run_shell(f'"{sys.executable}" -c "print(\'x\' * 10000)"')
        assert len(result.stdout.encode("utf-8")) <= 200  # 100 bytes + marcador de truncamento
        assert "truncado" in result.stdout
    finally:
        ctx.cleanup()
