#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 1 (recuperação de falha do checker por infra)
(docs/plan_dataset_expansion_error_recovery.md). Mesmo método das expansões anteriores: 4
variantes por tarefa-base, reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade.

Forma fixa de cada trajetória-base (5 segmentos, em ordem):
    think, tool_call, tool_result, think, tool_call, tool_result, think, final
(8 segmentos: write_file + checker, cada um com think/tool_call/tool_result, mais o think final
e o final)

Uso:
    python scripts/expand_checker_infra_recovery_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_EXPECTED_SHAPE = ("think", "tool_call", "tool_result", "think", "tool_call", "tool_result", "think", "final")

_VARIANTS = [
    dict(
        think_before_write="Vou implementar {summary}.",
        think_before_checker="Antes de responder, vou tentar validar com o checker.",
        think_after_failure="O checker não tem backend para {lang} neste ambiente — é limitação de infraestrutura, não bug no código. Repetir a mesma chamada não muda nada, então vou responder com o código já pronto, sendo honesto sobre a falta de validação.",
        final="Implementei {summary}:\n\n```{lang_tag}\n{code}\n```\n\nEste ambiente não tem suporte de validação (`checker`) para {lang} — só Python e Go são suportados aqui — então não consegui compilar/testar automaticamente. O código não foi validado.",
    ),
    dict(
        think_before_write="Vou escrever {summary} agora.",
        think_before_checker="Vou rodar o checker pra confirmar antes de responder ao usuário.",
        think_after_failure="Essa falha é do ambiente (sem backend de {lang} no checker), não do código que escrevi. Não faz sentido insistir na mesma chamada — vou responder com o código e avisar que não pôde ser validado aqui.",
        final="{summary} está pronto:\n\n```{lang_tag}\n{code}\n```\n\nO checker deste ambiente não suporta {lang} (só Python e Go têm backend registrado), então não consegui validar automaticamente. O código acima não foi testado.",
    ),
    dict(
        think_before_write="Vou implementar {summary} conforme pedido.",
        think_before_checker="Preciso confirmar com o checker antes de fechar a resposta.",
        think_after_failure="A falha do checker aqui é de infraestrutura (falta de backend para {lang}), não do meu código. Repetir a chamada não resolveria, então vou apresentar o resultado já com essa ressalva.",
        final="Aqui está {summary}:\n\n```{lang_tag}\n{code}\n```\n\nComo este ambiente não tem backend de checker para {lang} (só Python e Go), não consegui validar o código automaticamente — trate como não testado.",
    ),
    dict(
        think_before_write="Vou codificar {summary}.",
        think_before_checker="Vou tentar validar isso com o checker antes de responder.",
        think_after_failure="Não é um bug no código — o checker simplesmente não suporta {lang} neste ambiente. Ficar repetindo a mesma chamada não ajudaria, então vou responder com o código pronto e essa ressalva clara.",
        final="Segue {summary}:\n\n```{lang_tag}\n{code}\n```\n\nEste ambiente só valida Python e Go via `checker` — {lang} não tem backend registrado aqui, então o código acima não foi compilado/testado automaticamente.",
    ),
]

_REQUESTS = {
    "gen-checkerinfra-js-is-even": [
        "Preciso de uma função em JavaScript que diga se um número é par.",
        "Pode escrever uma função JS que verifique paridade de um número?",
        "Implementa isEven em JavaScript pra mim.",
        "Quero uma função JavaScript que retorne true se o número for par.",
    ],
    "gen-checkerinfra-ts-sum": [
        "Preciso de uma função TypeScript que some dois números.",
        "Pode escrever uma soma tipada em TypeScript?",
        "Implementa sum(a, b) em TypeScript.",
        "Quero uma função TypeScript simples de soma, com tipos.",
    ],
    "gen-checkerinfra-js-reverse-string": [
        "Preciso inverter uma string em JavaScript.",
        "Pode escrever uma função JS que inverta o texto?",
        "Implementa reverseString em JavaScript.",
        "Quero uma função JavaScript que devolva a string ao contrário.",
    ],
    "gen-checkerinfra-js-factorial": [
        "Preciso calcular fatorial em JavaScript.",
        "Pode escrever uma função JS de fatorial, tratando número negativo?",
        "Implementa factorial(n) em JavaScript.",
        "Quero uma função JavaScript que calcule o fatorial de um número.",
    ],
    "gen-checkerinfra-js-validate-email": [
        "Preciso validar e-mail em JavaScript.",
        "Pode escrever uma função JS que valide formato de e-mail?",
        "Implementa isValidEmail em JavaScript.",
        "Quero uma validação simples de e-mail em JavaScript.",
    ],
    "gen-checkerinfra-ts-filter-evens": [
        "Preciso filtrar números pares de uma lista em TypeScript.",
        "Pode escrever uma função TS que filtre só os pares?",
        "Implementa filterEvens em TypeScript, com tipos.",
        "Quero uma função TypeScript que retorne só os números pares de uma lista.",
    ],
}

_SUMMARIES = {
    "gen-checkerinfra-js-is-even": "isEven(n), verificando paridade",
    "gen-checkerinfra-ts-sum": "sum(a, b), somando dois números com tipos",
    "gen-checkerinfra-js-reverse-string": "reverseString(text), invertendo uma string",
    "gen-checkerinfra-js-factorial": "factorial(n), com tratamento para número negativo",
    "gen-checkerinfra-js-validate-email": "isValidEmail(email), validando formato de e-mail",
    "gen-checkerinfra-ts-filter-evens": "filterEvens(numbers), filtrando números pares com tipos",
}

_LANGS = {
    "gen-checkerinfra-js-is-even": ("JavaScript", "javascript"),
    "gen-checkerinfra-ts-sum": ("TypeScript", "typescript"),
    "gen-checkerinfra-js-reverse-string": ("JavaScript", "javascript"),
    "gen-checkerinfra-js-factorial": ("JavaScript", "javascript"),
    "gen-checkerinfra-js-validate-email": ("JavaScript", "javascript"),
    "gen-checkerinfra-ts-filter-evens": ("TypeScript", "typescript"),
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, think6: str, final7: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think0}</think>"
        + original_raw_text[segs[1].start : segs[2].end]
        + f"<think>{think3}</think>"
        + original_raw_text[segs[4].start : segs[5].end]
        + f"<think>{think6}</think>"
        + f"<final>{final7}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        # extrai o código já escrito (dentro do write_file original) para reaproveitar no <final>
        write_call_start = original_raw_text.index('<tool_call name="write_file">') + len('<tool_call name="write_file">')
        write_call_end = original_raw_text.index("</tool_call>", write_call_start)
        write_args = json.loads(original_raw_text[write_call_start:write_call_end])
        code = write_args["content"]

        lang, lang_tag = _LANGS[base_id]
        slots = {"summary": _SUMMARIES[base_id], "lang": lang, "lang_tag": lang_tag, "code": code}

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
            think0 = variant["think_before_write"].format(**slots)
            think3 = variant["think_before_checker"].format(**slots)
            think6 = variant["think_after_failure"].format(**slots)
            final7 = variant["final"].format(**slots)

            raw_text = _rebuild_raw_text(original_raw_text, think0, think3, think6, final7)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 1 (recuperação de falha de checker por infra).")


if __name__ == "__main__":
    main()
