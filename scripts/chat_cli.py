#!/usr/bin/env python
"""Chat interativo na CLI — diferente de `run_eval.py` (bateria de cenários FIXOS, sem
digitação livre), aqui você digita o pedido na hora e vê a resposta real do modelo, incluindo
as ferramentas que ele chama de verdade (write_file/read_file/checker/shell/...) contra um
sandbox real, igual o harness usa em treino/avaliação — nunca uma simulação.

Reaproveita o loop canônico (`run_agent_loop`) sem alterá-lo: cada turno roda de ponta a
ponta, e o script só formata o resultado pra leitura humana depois que o turno termina (sem
streaming passo-a-passo, que exigiria mudar o loop em si).

Uso:
    python scripts/chat_cli.py --adapter outputs/adapter --base-model ibm-granite/granite-4.1-8b --device cuda
    python scripts/chat_cli.py --base-model ibm-granite/granite-4.1-8b --device cpu   # sem adapter, smoke-test
    python scripts/chat_cli.py --adapter outputs/adapter --device cuda --quiet         # só mostra <final>

Comandos dentro da conversa: "sair"/"exit"/"quit" encerra; "novo"/"reset" começa uma conversa
nova (limpa histórico E sandbox); "workspace" mostra o caminho do sandbox atual (pra você ver
os arquivos criados pelo modelo).
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness import AgentLoopConfig, HistoryTurn, PRAXIS_SYSTEM_PROMPT, run_agent_loop  # noqa: E402
from src.parsers.segments import FinalSegment, MalformedSegment, ThinkSegment, ToolCallSegment, ToolResultSegment  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402


def _detect_environment(override_os: str | None, override_shell: str | None, override_cwd: str | None) -> dict:
    """Mesmo bloco `<environment>` que o dataset ensina o modelo a ler (D-prompt-environment-
    block) — detectado de verdade da máquina que roda este script, não perguntado ao usuário."""
    system = platform.system()
    os_label = {"Windows": "Windows", "Linux": "Ubuntu Linux", "Darwin": "macOS"}.get(system, system)
    shell_guess = {"Windows": "powershell", "Linux": "bash", "Darwin": "zsh"}.get(system, "bash")
    return {
        "os": override_os or os_label,
        "shell": override_shell or shell_guess,
        "cwd": override_cwd or str(Path.cwd()),
    }


def _build_runner(base_model: str, adapter_path: str | None, device: str):
    from src.inference.transformers_runner import TransformersModelRunner

    label = f"{base_model}" + (f" + adapter {adapter_path}" if adapter_path else " (sem adapter)")
    print(f"Carregando {label} (device={device})... isso pode levar alguns minutos.", flush=True)
    runner = TransformersModelRunner(base_model, adapter_path=adapter_path, device=device)
    print("Modelo carregado.\n", flush=True)
    return runner


def _fresh_sandbox(policy) -> SandboxContext:
    return SandboxContext(policy=policy)


_SEGMENT_LABELS = {
    ThinkSegment: "think",
    ToolCallSegment: "tool_call",
    ToolResultSegment: "tool_result",
    FinalSegment: "final",
    MalformedSegment: "malformed",
}


def _print_turn_trace(trajectory, turn_start_len: int, quiet: bool) -> str:
    """Formata só os segmentos gerados NESTE turno (a partir de `turn_start_len`, o tamanho de
    raw_text antes do turno começar) — sem misturar com o histórico de turnos anteriores.
    Devolve o texto de <final> (ou aviso, se não houve)."""
    from src.parsers import parse_segments

    new_raw = trajectory.raw_text[turn_start_len:]
    segments = parse_segments(new_raw)
    final_text = None

    for seg in segments:
        kind = _SEGMENT_LABELS.get(type(seg), "?")
        if kind == "think":
            if not quiet:
                print(f"  [pensando] {seg.text.strip()}")
        elif kind == "tool_call":
            if not quiet:
                print(f"  [ferramenta] {seg.name}({seg.args})")
        elif kind == "tool_result":
            if not quiet:
                status_tag = "ok" if seg.status == "ok" else "ERRO"
                print(f"    -> [{status_tag}] {seg.body}")
        elif kind == "final":
            final_text = seg.text.strip()
        elif kind == "malformed":
            if not quiet:
                print(f"  [malformado: {seg.reason}] {seg.message}")

    return final_text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=str, default=None, help="Caminho do adapter LoRA (opcional — omitido testa o modelo base cru, ou um modelo já mesclado apontado em --base-model)")
    parser.add_argument("--base-model", type=str, default="ibm-granite/granite-4.1-8b", help="Modelo base ou caminho local de um modelo mesclado")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' em GPU real; 'cpu' funciona mas é lento")
    parser.add_argument("--quiet", action="store_true", help="Mostra só a resposta final (<final>), sem think/tool_call/tool_result")
    parser.add_argument("--max-steps", type=int, default=8, help="Limite de passos (tool_call) por turno antes de forçar <final>")
    parser.add_argument("--max-tokens-per-step", type=int, default=512, help="Teto de tokens gerados por passo")
    parser.add_argument("--env-os", type=str, default=None, help="Sobrepõe o SO detectado automaticamente (ex.: 'Windows', 'Ubuntu Linux', 'macOS')")
    parser.add_argument("--env-shell", type=str, default=None, help="Sobrepõe o shell detectado (ex.: 'powershell', 'bash')")
    parser.add_argument("--no-environment", action="store_true", help="Não injeta bloco <environment> — o modelo fica sem saber o SO do host")
    args = parser.parse_args()

    runner = _build_runner(args.base_model, args.adapter, args.device)
    environment = None if args.no_environment else _detect_environment(args.env_os, args.env_shell, None)
    if environment:
        print(f"Ambiente detectado: os={environment['os']} shell={environment['shell']} cwd={environment['cwd']}\n")

    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    registry = ToolExecutorRegistry()
    config = AgentLoopConfig(max_steps=args.max_steps, max_tokens_per_step=args.max_tokens_per_step)

    sandbox = _fresh_sandbox(policy)
    history: list[HistoryTurn] = []

    print("Conversa iniciada. Digite seu pedido (ou 'sair', 'novo', 'workspace').\n")

    try:
        while True:
            try:
                user_input = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                break
            if user_input.lower() in ("novo", "reset"):
                sandbox.cleanup()
                sandbox = _fresh_sandbox(policy)
                history = []
                print("[conversa e workspace reiniciados]\n")
                continue
            if user_input.lower() == "workspace":
                print(f"[workspace atual: {sandbox.workspace}]\n")
                continue

            print("Logos-3:", flush=True)
            try:
                result = run_agent_loop(
                    user_request=user_input,
                    system_prompt=PRAXIS_SYSTEM_PROMPT,
                    model_runner=runner,
                    tool_registry=registry,
                    sandbox=sandbox,
                    config=config,
                    environment=environment,
                    history=history,
                )
            except Exception as exc:  # noqa: BLE001 — turno ruim não deve derrubar a sessão inteira
                print(f"  [erro inesperado no turno: {type(exc).__name__}: {exc}]\n")
                continue

            # `result.trajectory.raw_text` é só o turno ATUAL (o `history` passado acima entra
            # à parte, em `result.trajectory.history` — não duplicado dentro de `raw_text`).
            turn_raw = result.trajectory.raw_text
            final_text = _print_turn_trace(result.trajectory, 0, args.quiet)

            if result.forced_final:
                print("  [aviso: limite de passos atingido antes de um <final> genuíno]")
            if not args.quiet:
                print()
            print(f"Logos-3 (resposta): {final_text or '(sem resposta final)'}")
            print(f"[passos: {result.steps_taken} | forçado: {result.forced_final}]\n")

            history.append(HistoryTurn(user_request=user_input, raw_text=turn_raw))
    finally:
        sandbox.cleanup()
        print("Sessão encerrada, workspace limpo.")


if __name__ == "__main__":
    main()
