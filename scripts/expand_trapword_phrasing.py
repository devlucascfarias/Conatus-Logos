#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 3 (tool-choice em pergunta curta com
palavra-armadilha) (docs/plan_dataset_expansion_error_recovery.md). Mesmo método das
expansões anteriores: 4 variantes por tarefa-base. Como não há `<tool_call>`/`<tool_result>`
aqui (são exemplos puramente conversacionais), a variação é só na prosa do `<think>`/`<final>`
e no `user_request`.

Uso:
    python scripts/expand_trapword_phrasing.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "train"

_VARIANTS = [
    dict(
        think="{trap_note} — é uma pergunta conceitual, não preciso de ferramenta nenhuma pra responder.",
        final="{answer}",
    ),
    dict(
        think="{trap_note}. Não há nada pra criar/rodar/validar aqui, só uma explicação direta.",
        final="{answer}",
    ),
    dict(
        think="{trap_note}, então respondo direto, sem chamar nenhuma ferramenta.",
        final="{answer}",
    ),
    dict(
        think="{trap_note} — não é um pedido de ação, é só uma pergunta de conceito.",
        final="{answer}",
    ),
]

_REQUESTS = {
    "gen-trapword-import": [
        "Me explica rapidinho pra que serve o comando import?",
        "Numa frase: qual a função do comando import?",
        "O comando import faz o quê, exatamente?",
        "Qual o propósito do comando import em Python?",
    ],
    "gen-trapword-debug-mode": [
        "Explica rápido o que é rodar em modo debug.",
        "Numa frase: o que significa modo debug?",
        "O que quer dizer rodar um programa em debug?",
        "Qual a ideia por trás de rodar algo em modo debug?",
    ],
    "gen-trapword-unit-test": [
        "Me diz rapidinho o que é executar um teste unitário.",
        "Numa frase: o que significa rodar um teste unitário?",
        "O que quer dizer executar um teste unitário, no fundo?",
        "Qual a ideia de executar um teste unitário?",
    ],
    "gen-trapword-config-file": [
        "Me explica rapidinho o que é um arquivo de configuração.",
        "Numa frase: pra que serve um arquivo de configuração?",
        "O que é, no fundo, um arquivo de configuração?",
        "Qual a função de um arquivo de configuração?",
    ],
    "gen-trapword-print-command": [
        "Me diz rapidinho pra que serve o comando print.",
        "Numa frase: o que o comando print faz?",
        "O comando print serve pra quê, exatamente?",
        "Qual a função do comando print em Python?",
    ],
    "gen-trapword-compile-meaning": [
        "Me explica rapidinho o que é compilar um programa.",
        "Numa frase: o que significa compilar?",
        "O que quer dizer compilar um programa, no fundo?",
        "Qual a ideia por trás de compilar código?",
    ],
}

_TRAP_NOTES = {
    "gen-trapword-import": "A palavra 'comando' aqui é da instrução da linguagem, não de shell",
    "gen-trapword-debug-mode": "'Rodar' aqui é parte do conceito perguntado, não um pedido de execução real",
    "gen-trapword-unit-test": "'Executar' aqui é parte do conceito perguntado, não um pedido de execução real",
    "gen-trapword-config-file": "'Arquivo' aqui é parte do conceito perguntado, não um pedido pra criar/ler um arquivo",
    "gen-trapword-print-command": "A palavra 'comando' aqui é da instrução da linguagem, não de terminal",
    "gen-trapword-compile-meaning": "O usuário quer o conceito de compilação, não um pedido pra compilar algo agora",
}

_ANSWERS = {
    "gen-trapword-import": "O comando import serve para trazer código de outro módulo ou biblioteca para o arquivo atual.",
    "gen-trapword-debug-mode": "Rodar em modo debug significa executar o programa com ferramentas extras para inspecionar variáveis e pausar a execução, facilitando encontrar erros.",
    "gen-trapword-unit-test": "Executar um teste unitário é rodar uma verificação automatizada que confirma se uma pequena parte do código se comporta como esperado.",
    "gen-trapword-config-file": "Um arquivo de configuração guarda parâmetros que controlam o comportamento de um programa, sem precisar alterar o código-fonte.",
    "gen-trapword-print-command": "O comando print exibe um valor ou mensagem na saída padrão do terminal.",
    "gen-trapword-compile-meaning": "Compilar um programa é traduzir o código-fonte para uma forma que o computador consegue executar diretamente.",
}


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))

        slots = {"trap_note": _TRAP_NOTES[base_id], "answer": _ANSWERS[base_id]}

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
            think = variant["think"].format(**slots)
            final = variant["final"].format(**slots)
            raw_text = f"<think>{think}</think><final>{final}</final>"

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 3 (tool-choice em pergunta com palavra-armadilha).")


if __name__ == "__main__":
    main()
