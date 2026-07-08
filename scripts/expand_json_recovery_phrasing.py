#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 2a (recuperação de TOOL_CALL_PARSE_ERROR em
código complexo) (docs/plan_dataset_expansion_error_recovery.md). Mesmo método das expansões
anteriores: 4 variantes por tarefa-base, reaproveitando os fragmentos malformed/tool_result e
tool_call/tool_result já processados/executados de verdade.

Forma fixa de cada trajetória-base (10 segmentos, em ordem):
    think, malformed, tool_result, think, tool_call, tool_result, think, tool_call, tool_result, final

Uso:
    python scripts/expand_json_recovery_phrasing.py
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
    "think", "malformed", "tool_result",
    "think", "tool_call", "tool_result",
    "think", "tool_call", "tool_result",
    "final",
)

_VARIANTS = [
    dict(
        think_before_write="Vou implementar {summary}.",
        think_after_parse_error="O JSON da minha última chamada estava malformado (uma barra invertida solta antes de um espaço). É um erro meu de formatação, não da ferramenta — vou corrigir e tentar de novo.",
        think_before_final="JSON corrigido e código pronto. Vou validar.",
        final="Implementei {summary}. A validação confirma que compila e o teste passa.",
    ),
    dict(
        think_before_write="Vou escrever {summary} agora.",
        think_after_parse_error="Minha chamada anterior tinha um JSON malformado por minha causa, não da ferramenta — vou reescrever com o escaping correto.",
        think_before_final="Com o JSON corrigido, falta só rodar a validação.",
        final="{summary} está pronto e validado — o teste passa.",
    ),
    dict(
        think_before_write="Vou implementar {summary} conforme pedido.",
        think_after_parse_error="O erro foi na formatação do meu próprio JSON, não um problema de ambiente — vou corrigir e enviar de novo.",
        think_before_final="Já com o JSON certo, vou confirmar que está tudo correto.",
        final="Depois de corrigir o JSON, implementei {summary} com sucesso — validado e funcionando.",
    ),
    dict(
        think_before_write="Vou codificar {summary}.",
        think_after_parse_error="Foi um erro de formatação na minha própria chamada (JSON malformado), não algo externo — vou reenviar corrigido.",
        think_before_final="Código pronto com o JSON correto. Vou validar antes de responder.",
        final="{summary} implementado — depois de corrigir o JSON, a validação confirma que está tudo certo.",
    ),
]

_REQUESTS = {
    "gen-jsonrecovery-word-frequency": [
        "Preciso contar a frequência de cada palavra num texto, ignorando maiúsculas.",
        "Pode implementar uma contagem de palavras que ignore caixa alta/baixa?",
        "Quero saber quantas vezes cada palavra aparece num texto.",
        "Implementa word_frequency(text) pra mim.",
    ],
    "gen-jsonrecovery-extract-emails": [
        "Preciso extrair todos os e-mails de um texto usando regex.",
        "Pode implementar uma extração de e-mails com regex?",
        "Quero uma função que ache todos os e-mails dentro de um texto.",
        "Implementa extract_emails(text) usando expressão regular.",
    ],
    "gen-jsonrecovery-matrix-is-diagonal": [
        "Preciso verificar se uma matriz quadrada é diagonal.",
        "Pode implementar uma checagem de matriz diagonal?",
        "Quero saber se só a diagonal principal de uma matriz tem valores não-zero.",
        "Implementa is_diagonal_matrix(matrix) pra mim.",
    ],
    "gen-jsonrecovery-parse-date-br": [
        "Preciso extrair dia, mês e ano de uma data no formato brasileiro usando regex.",
        "Pode implementar um parser de data DD/MM/AAAA com regex?",
        "Quero separar dia, mês e ano de uma data brasileira em texto.",
        "Implementa parse_date_br(text) usando expressão regular.",
    ],
}

_SUMMARIES = {
    "gen-jsonrecovery-word-frequency": "word_frequency(text), contando frequência de palavras ignorando maiúsculas",
    "gen-jsonrecovery-extract-emails": "extract_emails(text), extraindo e-mails com regex",
    "gen-jsonrecovery-matrix-is-diagonal": "is_diagonal_matrix(matrix), verificando se é diagonal",
    "gen-jsonrecovery-parse-date-br": "parse_date_br(text), extraindo dia/mês/ano com regex",
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, think6: str, final9: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think0}</think>"
        + original_raw_text[segs[1].start : segs[2].end]  # malformed + tool_result (verbatim)
        + f"<think>{think3}</think>"
        + original_raw_text[segs[4].start : segs[5].end]  # write_file corrigido + tool_result (verbatim)
        + f"<think>{think6}</think>"
        + original_raw_text[segs[7].start : segs[8].end]  # checker + tool_result (verbatim)
        + f"<final>{final9}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        slots = {"summary": _SUMMARIES[base_id]}

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
            think0 = variant["think_before_write"].format(**slots)
            think3 = variant["think_after_parse_error"].format(**slots)
            think6 = variant["think_before_final"].format(**slots)
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

    print(f"{count} variantes geradas para o Gap 2a (recuperação de JSON malformado).")


if __name__ == "__main__":
    main()
