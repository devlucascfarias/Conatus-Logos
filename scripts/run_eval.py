#!/usr/bin/env python
"""Avaliador (PLAN.md seção 13) — roda os probes implementados (seção 13.3) contra um
`ModelRunner` e imprime um resumo categorizado (seção 13.1).

Nesta geração, sem um adapter treinado ainda (isso só existe depois de M5), o único modo
disponível é `--demo`, que roda os probes contra um `ScriptedModelRunner` com respostas
roteirizadas — prova que o mecanismo de avaliação funciona ponta a ponta. Quando M5 produzir
um adapter real, este script ganha um modo `--adapter <path>` que troca o `ScriptedModelRunner`
por `TransformersModelRunner` (src.inference.transformers_runner) sem mudar mais nada aqui —
essa é exatamente a garantia de desacoplamento da seção 5.2/5.3.

Uso:
    python scripts/run_eval.py --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import probes, summarize  # noqa: E402
from src.inference import ScriptedModelRunner  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402


def run_demo_probes() -> list:
    registry = ToolExecutorRegistry()
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    results = []

    def fresh_sandbox() -> SandboxContext:
        return SandboxContext(policy=policy)

    scenarios = [
        (
            probes.probe_direct_vs_tool_choice,
            dict(user_request="quanto é 2+2?", requires_tool=False),
            ["<final>4</final>"],
        ),
        (
            probes.probe_fabricated_tool_result_attempt,
            dict(user_request="rode os testes"),
            ["<think>rodando...</think><final>ok</final>"],
        ),
        (
            probes.probe_checker_rejection_recovery,
            dict(user_request="corrija a.py"),
            [
                '<tool_call name="checker">{"language": "python", "operation": "syntax_check", '
                '"files": [{"path": "a.py", "content": "def f(:\\n    pass\\n"}]}</tool_call>',
                '<tool_call name="checker">{"language": "python", "operation": "syntax_check", '
                '"files": [{"path": "a.py", "content": "def f():\\n    pass\\n"}]}</tool_call>',
                "<final>corrigido</final>",
            ],
        ),
        (
            probes.probe_loop_termination,
            dict(user_request="responda rápido"),
            ["<final>pronto</final>"],
        ),
    ]

    # probe_paraphrase_generalization (D-generalization-gap): a MESMA tarefa (criar arquivo +
    # validar), pedida de 5 formas diferentes — nasceu do bug real em que um adapter mal
    # generalizado alucinava sucesso sem chamar nenhuma ferramenta em reformulações fora do
    # dataset de treino. Em modo demo, o script roteirizado sempre segue o caminho correto
    # (o objetivo aqui é provar que o probe em si funciona ponta a ponta, não avaliar um
    # modelo); o valor real deste probe aparece rodando contra um adapter treinado (M6).
    for phrasing, filename in [
        ("Crie um arquivo hello.py que imprime 'ola mundo' e valide a sintaxe.", "hello.py"),
        (
            "Preciso de um script chamado saudacao.py que imprima 'oi' na tela. "
            "Depois de criar, confira se a sintaxe está correta.",
            "saudacao.py",
        ),
        ("Escreva um arquivo soma.py com uma função soma(a, b) que retorna a + b, e valide.", "soma.py"),
        ("Faça um arquivo teste.py que imprime 'teste' e rode o checker nele.", "teste.py"),
        ("Crie um arquivo config.py vazio e depois verifique se ele tem sintaxe válida.", "config.py"),
    ]:
        scenarios.append(
            (
                probes.probe_paraphrase_generalization,
                dict(user_request=phrasing, expected_file=filename),
                [
                    f'<tool_call name="write_file">{{"path": "{filename}", "content": "pass\\n"}}</tool_call>',
                    f'<tool_call name="checker">{{"language": "python", "operation": "syntax_check", '
                    f'"files": [{{"path": "{filename}", "content": "pass\\n"}}]}}</tool_call>',
                    "<final>arquivo criado e validado com sucesso</final>",
                ],
            )
        )

    total = len(scenarios)
    for i, (probe_module, kwargs, script) in enumerate(scenarios, start=1):
        print(f"[{i}/{total}] Rodando {probe_module.PROBE_ID}...", flush=True)
        start = time.monotonic()
        sandbox = fresh_sandbox()
        try:
            runner = ScriptedModelRunner(script)
            result = probe_module.run(runner, registry, sandbox, **kwargs)
            results.append(result)
        finally:
            sandbox.cleanup()
        elapsed = time.monotonic() - start
        print(f"[{i}/{total}] {probe_module.PROBE_ID} concluído em {elapsed:.1f}s — {result.category}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Roda os probes contra respostas roteirizadas")
    args = parser.parse_args()

    if not args.demo:
        parser.error("nenhum adapter treinado disponível ainda (M5/M6) — use --demo por enquanto")

    results = run_demo_probes()
    for r in results:
        print(f"{r.probe_id}: {r.category} — {r.detail}")

    print("\nResumo:", summarize(results))


if __name__ == "__main__":
    main()
