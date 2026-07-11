#!/usr/bin/env python
"""Lote base do Gap 3 (docs/plan_dataset_expansion_error_recovery.md): tool-choice correto em
perguntas curtas/diretas com palavra-armadilha (soa a ferramenta, mas não exige uma).

Achado real: no probe #20 da suíte de 25 (testes pós-retreino), "responda em uma frase: para "
"que serve o comando import?" levou o modelo a chamar uma ferramenta sem necessidade —
provavelmente a palavra "comando" puxou associação com shell/tool.

Cada exemplo é puramente conversacional (task_type: direct_answer, num_steps: 0, sem
tool_call) — não há nada para executar de verdade aqui, só o padrão de raciocínio "isso é uma
pergunta conceitual, apesar da palavra X, não preciso de ferramenta nenhuma".

Uso:
    python scripts/gen_direct_answer_trapword_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"


def build_example(id_, domain, difficulty, user_request, think_text, final_text) -> dict:
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think_text}</think>")
    traj.append_raw(f"<final>{final_text}</final>")

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "python",
            "difficulty": difficulty,
            "tools_used": [],
            "num_steps": 0,
            "task_type": "direct_answer",
            "source": "curated_manual",
            "license": "synthetic-no-license-needed",
            "validation_status": "validated",
            "execution_performed": False,
            "split": "train",
            # D-checker-used-null-fix (docs/PLAN.md): omitido, não `null` — resposta direta,
            # nenhuma ferramenta usada, não existe checker pra reportar.
            "expected_result": {"passed": None},
            "execution_classification": "static_only",
        },
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


EXAMPLES = [
    dict(
        id_="gen-trapword-import",
        domain="clarificacao",
        difficulty="easy",
        user_request="Responda em uma frase: para que serve o comando import?",
        think_text=(
            "A palavra 'comando' aqui se refere à instrução import da linguagem, não a um "
            "comando de shell — é uma pergunta conceitual, não preciso de nenhuma ferramenta "
            "para responder."
        ),
        final_text="O comando import serve para trazer código de outro módulo ou biblioteca para o arquivo atual.",
    ),
    dict(
        id_="gen-trapword-debug-mode",
        domain="clarificacao",
        difficulty="easy",
        user_request="O que significa rodar um programa em modo debug, em poucas palavras?",
        think_text=(
            "'Rodar' aqui é parte da pergunta conceitual sobre o que é modo debug, não um "
            "pedido para eu executar algo de verdade — respondo direto, sem ferramenta."
        ),
        final_text="Rodar em modo debug significa executar o programa com ferramentas extras para inspecionar variáveis e pausar a execução, facilitando encontrar erros.",
    ),
    dict(
        id_="gen-trapword-unit-test",
        domain="clarificacao",
        difficulty="easy",
        user_request="Explique em uma frase o que é executar um teste unitário.",
        think_text=(
            "'Executar' aqui faz parte da pergunta conceitual sobre o que é um teste unitário "
            "— o usuário não está pedindo pra eu rodar nada de verdade, só uma explicação."
        ),
        final_text="Executar um teste unitário é rodar uma verificação automatizada que confirma se uma pequena parte do código (geralmente uma função) se comporta como esperado.",
    ),
    dict(
        id_="gen-trapword-config-file",
        domain="clarificacao",
        difficulty="easy",
        user_request="O que é um arquivo de configuração, de forma geral?",
        think_text=(
            "A palavra 'arquivo' aqui é só parte do conceito perguntado, não um pedido para "
            "criar ou ler um arquivo de verdade — respondo direto."
        ),
        final_text="Um arquivo de configuração guarda parâmetros que controlam o comportamento de um programa, sem precisar alterar o código-fonte.",
    ),
    dict(
        id_="gen-trapword-print-command",
        domain="clarificacao",
        difficulty="easy",
        user_request="Para que serve o comando print em Python, resumidamente?",
        think_text=(
            "De novo 'comando' se refere à instrução da linguagem, não a algo pra eu executar "
            "no terminal — é uma pergunta conceitual, respondo sem ferramenta."
        ),
        final_text="O comando print exibe um valor ou mensagem na saída padrão do terminal.",
    ),
    dict(
        id_="gen-trapword-compile-meaning",
        domain="clarificacao",
        difficulty="easy",
        user_request="O que significa compilar um programa, em uma frase?",
        think_text=(
            "O usuário está perguntando o CONCEITO de compilação, não pedindo pra eu compilar "
            "nada de verdade agora — respondo direto, sem ferramenta."
        ),
        final_text="Compilar um programa é traduzir o código-fonte escrito pelo desenvolvedor para uma forma que o computador consegue executar diretamente.",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for spec in EXAMPLES:
        example = build_example(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name}")
        written += 1
    print(f"\n{written} exemplos gerados em {OUT_DIR}")


if __name__ == "__main__":
    main()
