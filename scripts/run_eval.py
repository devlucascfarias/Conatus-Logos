#!/usr/bin/env python
"""Avaliador (PLAN.md seção 13) — roda os probes implementados (seção 13.3) contra um
`ModelRunner` e imprime um resumo categorizado (seção 13.1).

Dois modos:
- `--demo`: roda os probes contra um `ScriptedModelRunner` com respostas roteirizadas — prova
  que o MECANISMO de avaliação funciona ponta a ponta, não avalia qualidade de modelo nenhum.
- `--adapter <path>`: roda os MESMOS cenários dos probes (mesmo `user_request`/kwargs de cada
  um, só sem o roteiro) contra um `TransformersModelRunner` real (`src.inference.
  transformers_runner`) — troca só o `model_runner` passado a `probe_module.run(...)`, o resto
  do probe (execução real de ferramentas contra o sandbox, critério de aprovação) não muda nada,
  a garantia de desacoplamento da seção 5.2/5.3. Sem GPU (`--device cpu`) funciona como
  smoke-test de wiring — lento e não reflete qualidade real do adapter, mas prova que o caminho
  de código funciona antes de rodar em GPU de verdade (Colab/Kaggle/L4).

Uso:
    python scripts/run_eval.py --demo
    python scripts/run_eval.py --adapter outputs/adapter --base-model ibm-granite/granite-4.1-8b --device cuda
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


def _build_scenarios() -> list:
    """Lista (probe_module, kwargs, script) — `script` só é usado em `--demo`
    (`ScriptedModelRunner`); `--adapter` reaproveita só `probe_module`/`kwargs`."""
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
        # probe_cross_language (D-eval-fase-g-probes): mesmo padrão da Fase F do dataset
        # (D-crosslayer-dataset-fase-f) — python roda de verdade via shell, node roda de
        # verdade via checker sobre o JSON que o python realmente produziria.
        (
            probes.probe_cross_language,
            dict(user_request="gere o relatório em python e leia o valor em node"),
            [
                '<tool_call name="write_file">{"path": "report.py", "content": '
                '"import json\\nwith open(\\"report.json\\", \\"w\\") as f:\\n    '
                'json.dump({\\"value\\": 42}, f)\\nprint(\\"ok\\")\\n"}</tool_call>',
                '<tool_call name="shell">{"binary": "python", "args": ["report.py"]}</tool_call>',
                '<tool_call name="checker">{"language": "node", "operation": "run", "files": '
                '[{"path": "read_report.mjs", "content": "import { readFileSync } from '
                '\\"node:fs\\";\\nconst d = JSON.parse(readFileSync(new URL(\\"./report.json\\", '
                'import.meta.url), \\"utf-8\\"));\\nconsole.log(d.value);\\n"}, '
                '{"path": "report.json", "content": "{\\"value\\": 42}"}], '
                '"entrypoint": "read_report.mjs"}</tool_call>',
                "<final>relatório gerado em python e lido corretamente em node</final>",
            ],
        ),
        # probe_multi_file_edit (D-eval-fase-g-probes): função + teste, dois arquivos
        # coordenados, confirmados com pytest real via compile_and_test.
        (
            probes.probe_multi_file_edit,
            dict(user_request="crie uma função absolute(n) e o teste dela"),
            [
                '<tool_call name="write_file">{"path": "calc.py", "content": '
                '"def absolute(n):\\n    return n if n >= 0 else -n\\n"}</tool_call>',
                '<tool_call name="write_file">{"path": "test_calc.py", "content": '
                '"from calc import absolute\\n\\ndef test_absolute():\\n    '
                'assert absolute(-5) == 5\\n"}</tool_call>',
                '<tool_call name="checker">{"language": "python", "operation": "compile_and_test", '
                '"files": [{"path": "calc.py", "content": "def absolute(n):\\n    '
                'return n if n >= 0 else -n\\n"}, {"path": "test_calc.py", "content": '
                '"from calc import absolute\\n\\ndef test_absolute():\\n    '
                'assert absolute(-5) == 5\\n"}]}</tool_call>',
                "<final>função e teste criados, suite passou</final>",
            ],
        ),
        # probe_frontend_checker_language_choice (D-eval-fase-g-probes): sintaxe SCSS real
        # (variável + mixin) só valida com language="scss" — html/css sozinho rejeitaria.
        (
            probes.probe_frontend_checker_language_choice,
            dict(user_request="crie um tema SCSS com uma variável de cor e um mixin de borda", expected_language="scss"),
            [
                '<tool_call name="write_file">{"path": "theme.scss", "content": '
                '"$primary: #3498db;\\n\\n@mixin rounded($radius) {\\n  border-radius: $radius;\\n}'
                '\\n\\n.card {\\n  color: $primary;\\n  @include rounded(8px);\\n}\\n"}</tool_call>',
                '<tool_call name="checker">{"language": "scss", "operation": "compile", "files": '
                '[{"path": "theme.scss", "content": "$primary: #3498db;\\n\\n@mixin rounded($radius) '
                '{\\n  border-radius: $radius;\\n}\\n\\n.card {\\n  color: $primary;\\n  '
                '@include rounded(8px);\\n}\\n"}]}</tool_call>',
                "<final>tema SCSS criado e compilado com sucesso</final>",
            ],
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

    return scenarios


def _run_scenarios(scenarios: list, make_runner) -> list:
    """`make_runner(script)` decide o `ModelRunner`: `--demo` ignora `script` e usa argumento
    posicional pra montar um `ScriptedModelRunner(script)`; `--adapter` ignora `script`
    inteiramente e devolve sempre a mesma instância de `TransformersModelRunner` já carregada."""
    registry = ToolExecutorRegistry()
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    results = []
    total = len(scenarios)
    for i, (probe_module, kwargs, script) in enumerate(scenarios, start=1):
        print(f"[{i}/{total}] Rodando {probe_module.PROBE_ID}...", flush=True)
        start = time.monotonic()
        sandbox = SandboxContext(policy=policy)
        try:
            runner = make_runner(script)
            result = probe_module.run(runner, registry, sandbox, **kwargs)
            results.append(result)
        finally:
            sandbox.cleanup()
        elapsed = time.monotonic() - start
        print(f"[{i}/{total}] {probe_module.PROBE_ID} concluído em {elapsed:.1f}s — {result.category}", flush=True)

    return results


def run_demo_probes() -> list:
    scenarios = _build_scenarios()
    return _run_scenarios(scenarios, make_runner=lambda script: ScriptedModelRunner(script))


def run_probes_against_adapter(base_model: str, adapter_path: str, device: str) -> list:
    from src.inference.transformers_runner import TransformersModelRunner

    print(f"Carregando {base_model} + adapter {adapter_path} (device={device})...", flush=True)
    runner = TransformersModelRunner(base_model, adapter_path=adapter_path, device=device)
    scenarios = _build_scenarios()
    # Mesma instância de runner reaproveitada em todos os cenários — generate() é sem estado
    # entre chamadas (cada probe monta seu próprio prompt via render_for_model()), recarregar o
    # modelo por cenário custaria minutos à toa numa GPU real.
    return _run_scenarios(scenarios, make_runner=lambda script: runner)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Roda os probes contra respostas roteirizadas")
    mode.add_argument("--adapter", type=str, help="Caminho do adapter LoRA treinado (ex.: outputs/adapter)")
    parser.add_argument(
        "--base-model", type=str, default="ibm-granite/granite-4.1-8b",
        help="Modelo base pro --adapter (default: ibm-granite/granite-4.1-8b, o de configs/train_l4.yaml)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device pro --adapter ('cuda' em GPU real; 'cpu' funciona como smoke-test de wiring, lento e sem refletir qualidade real)",
    )
    args = parser.parse_args()

    if args.demo:
        results = run_demo_probes()
    else:
        results = run_probes_against_adapter(args.base_model, args.adapter, args.device)

    for r in results:
        print(f"{r.probe_id}: {r.category} — {r.detail}")

    print("\nResumo:", summarize(results))


if __name__ == "__main__":
    main()
