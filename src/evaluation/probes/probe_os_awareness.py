"""`probe_os_awareness` (D-prompt-environment-block) — o modelo deve escolher o CLI NATIVO do
SO indicado no bloco `<environment>` SEM o usuário declarar o SO (igual Claude Code/Codex). O
cenário injeta um ambiente (ex.: os=Windows) e um pedido de CLI neutro de SO; o critério é que a
ferramenta `shell` usada bata com a família de comandos daquele SO — PowerShell/cmd no Windows,
bash/sh no Linux/macOS. Escolher a família errada é falha funcional real (o comando nem roda no
SO alvo)."""

from __future__ import annotations

from src.harness import PRAXIS_SYSTEM_PROMPT, run_agent_loop
from src.parsers.segments import ToolCallSegment

from .. import categories
from .base import ProbeResult

PROBE_ID = "probe_os_awareness"

_WINDOWS_BINARIES = {"powershell", "pwsh", "cmd"}
_POSIX_BINARIES = {"bash", "sh", "zsh"}


def run(model_runner, tool_registry, sandbox, user_request: str, environment: dict, expect_family: str) -> ProbeResult:
    result = run_agent_loop(
        user_request, PRAXIS_SYSTEM_PROMPT, model_runner, tool_registry, sandbox, environment=environment
    )
    if result.forced_final:
        return ProbeResult(PROBE_ID, categories.FAIL, "loop atingiu max_steps sem <final> genuíno")

    shell_calls = [
        seg for seg in result.trajectory.segments()
        if isinstance(seg, ToolCallSegment) and seg.name == "shell"
    ]
    if not shell_calls:
        return ProbeResult(PROBE_ID, categories.FAIL, "cenário de CLI, mas o modelo não chamou a ferramenta shell")

    binary = (shell_calls[-1].args.get("binary") or "").lower()
    is_windows = binary in _WINDOWS_BINARIES
    is_posix = binary in _POSIX_BINARIES

    if expect_family == "windows" and not is_windows:
        return ProbeResult(PROBE_ID, categories.FAIL, f"ambiente é Windows, mas usou binário não-Windows: {binary!r}")
    if expect_family == "posix" and not is_posix:
        return ProbeResult(PROBE_ID, categories.FAIL, f"ambiente é POSIX, mas usou binário não-POSIX: {binary!r}")

    return ProbeResult(
        PROBE_ID, categories.TOOL_PASS,
        f"escolheu o CLI nativo certo ({binary!r}) a partir do bloco <environment>, sem o usuário declarar o SO",
    )
