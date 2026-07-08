#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 2b (diagnóstico de arquivo com bug real entre
vários) (docs/plan_dataset_expansion_error_recovery.md). Mesmo método das expansões
anteriores: 4 variantes por tarefa-base, reaproveitando os fragmentos `<tool_call>`/
`<tool_result>` já executados de verdade.

Uso:
    python scripts/expand_multifile_diagnosis_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_EXPECTED_SHAPE = (
    "think", "tool_call", "tool_result",
    "think", "tool_call", "tool_result",
    "think", "tool_call", "tool_result",
    "final",
)

_VARIANTS = [
    dict(
        think_before_write="Vou implementar {summary}.",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        think_after_diagnosis="{diagnosis}",
        final="Implementei {summary}. O primeiro teste falhou por um bug no arquivo de teste, não na implementação — corrigi só o teste. A validação confirma que compila e o teste passa.",
    ),
    dict(
        think_before_write="Vou escrever {summary} agora.",
        think_before_first_checker="Com a implementação pronta, vou escrever o teste e rodar a validação.",
        think_after_diagnosis="{diagnosis}",
        final="{summary} está pronta. O primeiro teste falhou por um erro no próprio arquivo de teste (a implementação estava correta) — corrigi só o teste, e agora passa.",
    ),
    dict(
        think_before_write="Vou implementar {summary} conforme pedido.",
        think_before_first_checker="Agora escrevo o teste correspondente e valido tudo de uma vez.",
        think_after_diagnosis="{diagnosis}",
        final="Depois de identificar que o bug estava no arquivo de teste (não na implementação de {summary}), corrigi só o teste. A validação confirma que está tudo certo.",
    ),
    dict(
        think_before_write="Vou codificar {summary}.",
        think_before_first_checker="Vou escrever o teste em seguida e rodar tudo junto pelo checker.",
        think_after_diagnosis="{diagnosis}",
        final="{summary} implementada corretamente desde a primeira tentativa — o bug encontrado estava no arquivo de teste, que corrigi. A validação confirma que compila e o teste passa.",
    ),
]

_REQUESTS = {
    "gen-multifile-missing-import": [
        "Preciso de uma função que calcule a média de uma lista de números, com teste.",
        "Pode implementar average(numbers) com um teste correspondente?",
        "Quero uma função de média com teste incluso.",
        "Implementa average(numbers) e o teste dela.",
    ],
    "gen-multifile-wrong-import-name": [
        "Preciso converter Celsius para Fahrenheit, com um teste.",
        "Pode implementar celsius_to_fahrenheit com teste correspondente?",
        "Quero uma conversão de temperatura com teste incluso.",
        "Implementa celsius_to_fahrenheit e o teste dela.",
    ],
    "gen-multifile-wrong-arg-count": [
        "Preciso calcular a área de um retângulo, com um teste.",
        "Pode implementar rectangle_area com teste correspondente?",
        "Quero uma função de área de retângulo com teste incluso.",
        "Implementa rectangle_area e o teste dela.",
    ],
    "gen-multifile-undefined-variable-in-test": [
        "Preciso verificar se uma lista está ordenada, com um teste.",
        "Pode implementar is_sorted com teste correspondente?",
        "Quero uma checagem de lista ordenada com teste incluso.",
        "Implementa is_sorted e o teste dela.",
    ],
}

_SUMMARIES = {
    "gen-multifile-missing-import": "average(numbers)",
    "gen-multifile-wrong-import-name": "celsius_to_fahrenheit(celsius)",
    "gen-multifile-wrong-arg-count": "rectangle_area(width, height)",
    "gen-multifile-undefined-variable-in-test": "is_sorted(items)",
}

_DIAGNOSES = {
    "gen-multifile-missing-import": "O erro aponta 'NameError: name average is not defined' no arquivo de teste — ele esqueceu de importar a função. A implementação está correta; vou corrigir só o teste.",
    "gen-multifile-wrong-import-name": "O erro é 'ImportError: cannot import name celsius_to_farenheit' — typo no nome da função dentro do teste (faltando um h). A implementação tem o nome certo; vou corrigir só o teste.",
    "gen-multifile-wrong-arg-count": "O erro é 'TypeError: rectangle_area() takes 2 positional arguments but 3 were given' — o teste está passando um argumento a mais. A implementação está correta com 2 parâmetros; vou corrigir só a chamada no teste.",
    "gen-multifile-undefined-variable-in-test": "O erro é 'NameError: name dat is not defined' — typo no nome de uma variável dentro do próprio teste. A implementação está correta; vou corrigir só o teste.",
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, think6: str, final9: str) -> str:
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
        + original_raw_text[segs[7].start : segs[8].end]
        + f"<final>{final9}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        slots = {
            "summary": _SUMMARIES[base_id],
            "diagnosis": _DIAGNOSES[base_id],
        }

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
            think0 = variant["think_before_write"].format(**slots)
            think3 = variant["think_before_first_checker"].format(**slots)
            think6 = variant["think_after_diagnosis"].format(**slots)
            final9 = variant["final"].format(**slots)

            raw_text = _rebuild_raw_text(original_raw_text, think0, think3, think6, final9)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 2b (diagnóstico de arquivo com bug real).")


if __name__ == "__main__":
    main()
