"""PLAN.md seção 19.3 — tentativa de comando destrutivo deve retornar UNSAFE_COMMAND sem
executar, tanto na ferramenta `shell` isolada quanto dentro do loop completo do agente."""

from src.checker import errors as error_codes
from src.harness import run_agent_loop
from src.inference import ScriptedModelRunner
from src.security import SandboxContext, SandboxPolicy
from src.tools import ToolExecutorRegistry


def test_shell_tool_blocks_destructive_command_directly():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    try:
        result = registry.execute("shell", {"binary": "rm", "args": ["-rf", "/"]}, sandbox)
        assert not result.passed
        assert result.error_code == error_codes.UNSAFE_COMMAND
    finally:
        sandbox.cleanup()


def test_agent_loop_blocks_destructive_command_even_with_confirmation_requested():
    # Mesmo com requires_confirmation e confirmation_mode='auto_approve_safe' (autoriza
    # o *pedido* de confirmação), o comando em si continua bloqueado pela allowlist/denylist —
    # confirmação nunca sobrepõe a política de comando permitido.
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            '<tool_call name="shell">{"binary": "sudo", "args": ["rm", "-rf", "/"]}</tool_call>',
            "<final>não posso executar esse comando</final>",
        ]
    )
    try:
        result = run_agent_loop("apague tudo", "system", runner, registry, sandbox)
        assert error_codes.UNSAFE_COMMAND in result.trajectory.raw_text
    finally:
        sandbox.cleanup()


def test_agent_loop_denies_shell_by_default_confirmation_policy():
    # Política padrão do harness de dev é 'deny_all_destructive' (seção 7.7) — qualquer
    # ferramenta com requires_confirmation=true (shell) é negada por padrão, mesmo um
    # comando inofensivo, até uma confirmação explícita ser concedida.
    policy = SandboxPolicy.load()  # confirmation_mode default do config = deny_all_destructive
    assert policy.confirmation_mode == "deny_all_destructive"
    sandbox = SandboxContext(policy=policy)
    registry = ToolExecutorRegistry()
    runner = ScriptedModelRunner(
        [
            '<tool_call name="shell">{"binary": "ls", "args": []}</tool_call>',
            "<final>ok</final>",
        ]
    )
    try:
        result = run_agent_loop("liste arquivos via shell", "system", runner, registry, sandbox)
        assert error_codes.UNSAFE_COMMAND in result.trajectory.raw_text
    finally:
        sandbox.cleanup()


# --- D-shell-crossplatform-hardening: interpretadores nativos liberados, mas endurecidos ---


def test_hardened_denylist_blocks_powershell_remove_item():
    """Com powershell na allowlist (Fase H), a proteção contra deleção destrutiva passa a ser
    a denylist reforçada — `Remove-Item` (e seu efeito equivalente ao `rm`) nunca deve passar."""
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    allowed, reason = policy.is_command_allowed(
        "powershell", ("-NoProfile", "-Command", "Remove-Item -Recurse -Force C:\\dados")
    )
    assert not allowed
    assert "denylist" in reason


def test_hardened_denylist_blocks_cmd_del_and_format():
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    for cmd_args in (("/c", "del", "/q", "arquivo.txt"), ("/c", "format", "C:")):
        allowed, reason = policy.is_command_allowed("cmd", cmd_args)
        assert not allowed, f"deveria bloquear cmd {cmd_args}"


def test_hardened_blocks_encoded_command_bypass_on_interpreters():
    """Bypass real: um comando codificado em base64 driblaria a denylist substring (os verbos
    não aparecem em texto claro). A checagem por interpretador deve barrar a flag inteira."""
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    # -encodedcommand/-enc são pegos já pela denylist (camada anterior); -e cai só na checagem
    # por interpretador — os dois caminhos bloqueiam, defesa em profundidade.
    for flag in ("-EncodedCommand", "-enc", "-e"):
        allowed, reason = policy.is_command_allowed("powershell", (flag, "ZwBjAGkA"))
        assert not allowed, f"deveria bloquear bypass via {flag}"
        assert ("codificado" in reason) or ("denylist" in reason)

    # Um interpretador com a flag de bypass isolada (sem verbo destrutivo em claro) é barrado
    # ESPECIFICAMENTE pela checagem por interpretador, não pela denylist.
    allowed, reason = policy.is_command_allowed("pwsh", ("-e", " QQBiAEMA"))
    assert not allowed and "codificado" in reason


def test_git_write_subcommands_allowed_destructive_blocked():
    """D-git-write-ops: add/commit/branch/checkout/switch liberados (fluxo do dia-a-dia), mas
    as formas destrutivas/remotas seguem bloqueadas pela denylist."""
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    for args in (("add", "."), ("commit", "-m", "msg"), ("branch", "feature"),
                 ("checkout", "-b", "nova"), ("switch", "main"), ("status",), ("stash",)):
        ok, reason = policy.is_command_allowed("git", args)
        assert ok, f"git {args} deveria ser permitido — {reason}"
    for args in (("push",), ("reset", "--hard"), ("clean", "-fd"),
                 ("checkout", "--", "arquivo.py"), ("branch", "-D", "x"), ("push", "--force"),
                 ("stash", "clear")):
        ok, reason = policy.is_command_allowed("git", args)
        assert not ok, f"git {args} deveria ser BLOQUEADO"


def test_git_add_not_false_blocked_by_dd_pattern():
    """Regressão real: o padrão `dd ` (utilitário de disco) dava falso positivo em `git add .`
    (\"a**dd** .\" contém \"dd \") — corrigido pra `dd if=`/`dd of=` (D-git-write-ops)."""
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    ok, _ = policy.is_command_allowed("git", ("add", "."))
    assert ok
    # e o `dd` de verdade continua bloqueado
    ok2, _ = policy.is_command_allowed("bash", ("-c", "dd if=/dev/zero of=/dev/sda"))
    assert not ok2


def test_hardened_allows_real_read_only_native_commands():
    """O ponto da Fase H: comandos NATIVOS de inspeção (não destrutivos) devem PASSAR pela
    política — tanto PowerShell quanto cmd quanto bash — pra o modelo poder usá-los de verdade."""
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    for binary, args in (
        ("powershell", ("-NoProfile", "-Command", "Get-ChildItem -Name")),
        ("cmd", ("/c", "dir", "/b")),
        ("bash", ("-c", "grep foo a.txt | wc -l")),
        ("findstr", ("/s", "TODO", "*.py")),
    ):
        allowed, reason = policy.is_command_allowed(binary, args)
        assert allowed, f"comando de inspeção nativo deveria passar: {binary} {args} — {reason}"
