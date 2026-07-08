#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 4 (revisão de hipótese após 2ª falha real)
(docs/plan_dataset_expansion_hypothesis_revision.md). Mesmo método das expansões anteriores:
variantes por tarefa-base reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade (as duas falhas reais e o sucesso real nunca são reexecutados).

Forma fixa de cada trajetória-base (16 segmentos, 3 rodadas de write_file+checker):
    think, tool_call, tool_result, tool_call, tool_result,   (rodada 1 — falha real)
    think, tool_call, tool_result, tool_call, tool_result,   (rodada 2 — falha real, teoria errada)
    think, tool_call, tool_result, tool_call, tool_result,   (rodada 3 — sucesso real)
    final

Uso:
    python scripts/expand_hypothesis_revision_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_ROUND = ("think", "tool_call", "tool_result", "tool_call", "tool_result")
_EXPECTED_SHAPE = _ROUND + _ROUND + _ROUND + ("final",)

# 4 aberturas de think1/final estruturalmente diferentes; think2/think3 usam listas próprias
# por tarefa-base (_WRONG_THEORIES_BY_TASK/_REVISIONS_BY_TASK), não um slot compartilhado —
# isso evitou o erro de repetir a MESMA frase de diagnóstico em todas as variantes (só
# think1/final variavam antes, o que já causou D-generalization-gap/D-tool-pilots-phrasing).
_OPENINGS = [
    dict(
        think1="Vou implementar {summary}.",
        final="Implementei {summary}. {final_note} A validação confirma que compila e o teste passa.",
    ),
    dict(
        think1="Vou escrever {summary} agora.",
        final="{summary} está pronta. {final_note} Validado: compila e o teste passa.",
    ),
    dict(
        think1="Vou implementar {summary} conforme pedido.",
        final="Depois de duas tentativas, cheguei em {summary} correta. {final_note} A validação confirma que está tudo certo.",
    ),
    dict(
        think1="Vou codificar {summary}.",
        final="{summary} implementada. {final_note} A validação confirma que compila e o teste passa.",
    ),
]

_REQUESTS = {
    "gen-hyprevision-sum-up-to": [
        "Preciso somar todos os inteiros de 1 até n, incluindo n.",
        "Pode implementar uma soma de 1 até n (inclusive) para mim?",
        "Quero uma função que some de 1 até n, incluindo o próprio n.",
        "Implementa sum_up_to(n) somando até n inclusive.",
    ],
    "gen-hyprevision-is-in-range": [
        "Preciso verificar se um valor está dentro de um intervalo, incluindo os limites.",
        "Pode implementar uma checagem de intervalo com os dois limites inclusivos?",
        "Quero uma função que diga se um valor está entre dois limites, incluindo eles.",
        "Implementa is_in_range com limites inclusivos dos dois lados.",
    ],
    "gen-hyprevision-last-n-items": [
        "Preciso pegar os últimos n itens de uma lista.",
        "Pode implementar uma função que retorne o final de uma lista, com n itens?",
        "Quero os últimos n elementos de uma lista.",
        "Implementa last_n_items pegando o final da lista.",
    ],
    "gen-hyprevision-dedup-keep-order": [
        "Preciso remover duplicatas de uma lista mantendo a ordem original.",
        "Pode implementar uma remoção de duplicatas que preserve a ordem de aparição?",
        "Quero deduplicar uma lista sem embaralhar a ordem original.",
        "Implementa dedup_keep_order preservando a ordem de primeira aparição.",
    ],
}

_SUMMARIES = {
    "gen-hyprevision-sum-up-to": "sum_up_to(n)",
    "gen-hyprevision-is-in-range": "is_in_range(value, low, high)",
    "gen-hyprevision-last-n-items": "last_n_items(items, n)",
    "gen-hyprevision-dedup-keep-order": "dedup_keep_order(items)",
}

# 4 formulações genuinamente diferentes por tarefa-base (não um slot único reaproveitado) —
# cada uma muda ordem de cláusula, grau de formalidade, direta vs. indireta.
_WRONG_THEORIES_BY_TASK = {
    "gen-hyprevision-sum-up-to": [
        "O teste falhou (deu 10 em vez de 15). Talvez o problema seja o início do range — vou tentar começar de 0 em vez de 1.",
        "Deu 10 em vez de 15 no teste. Pode ser que o range devesse começar em 0, não em 1 — vou ajustar isso.",
        "O resultado (10) não bate com o esperado (15). Uma hipótese: o início do range está errado, deveria ser 0. Vou testar essa mudança.",
        "10 em vez de 15 — algo está errado no intervalo somado. Suspeito do início do range; vou trocar de 1 para 0 e ver se resolve.",
    ],
    "gen-hyprevision-is-in-range": [
        "O teste falhou nos limites. Talvez o problema seja só o limite superior — vou trocar high para inclusivo.",
        "Os casos de limite (low e high) falharam. Uma possibilidade é que só o high precise virar inclusivo — vou ajustar esse lado.",
        "Falhou nos extremos do intervalo. Suspeito que o limite superior esteja exclusivo por engano; vou corrigir só esse lado primeiro.",
        "Os testes de fronteira não passaram. Pode ser que high precise de <= em vez de < — vou tentar essa correção pontual.",
    ],
    "gen-hyprevision-last-n-items": [
        "O teste falhou (retornou os primeiros itens, não os últimos). Talvez o problema seja o lado do slice — vou tentar items[n:] para pegar do índice n até o fim.",
        "Voltou os primeiros itens em vez dos últimos. Uma hipótese: o slice está do lado errado — vou tentar items[n:] em vez de items[:n].",
        "O resultado trouxe o começo da lista, não o final. Suspeito que items[n:] resolva, pegando do índice n em diante.",
        "Peguei os itens errados (do início, não do fim). Pode ser que inverter o slice para items[n:] já resolva — vou testar.",
    ],
    "gen-hyprevision-dedup-keep-order": [
        "O teste falhou — set() não preserva ordem. Talvez falte um critério determinístico — vou tentar ordenar o resultado.",
        "set() removeu as duplicatas, mas bagunçou a ordem. Uma hipótese: falta determinismo, então vou tentar sorted() no resultado.",
        "A ordem saiu diferente da esperada depois do set(). Suspeito que ordenar o resultado resolva o problema de consistência.",
        "O set() não preserva a ordem original. Pode ser que aplicar sorted() por cima já garanta um resultado previsível — vou tentar.",
    ],
}

_REVISIONS_BY_TASK = {
    "gen-hyprevision-sum-up-to": [
        "Minha correção anterior (começar do 0) não resolveu — o resultado ainda está errado (10), o que mostra que a teoria estava errada: o problema nunca foi o início do range, e sim o fim, que precisa incluir n.",
        "Mudar o início para 0 não resolveu nada — o valor continua 10. Isso prova que a hipótese estava errada: o range precisa é incluir n no FIM, não mudar onde começa.",
        "A correção do início não fez diferença (ainda 10). Ou seja, minha teoria estava equivocada — o que falta é estender o range até n+1, não mexer no começo.",
        "Mesmo trocando o início para 0, o resultado não mudou (continua 10) — isso descarta minha primeira teoria. O ajuste certo é no fim do range, incluindo n.",
    ],
    "gen-hyprevision-is-in-range": [
        "Minha correção anterior (só o limite superior) não resolveu tudo — o limite inferior ainda falha. A teoria estava incompleta: os dois limites precisam ser inclusivos, não só um.",
        "Corrigir só o high não bastou — o teste do low continua falhando. Ou seja, a hipótese estava incompleta: os dois lados do intervalo precisam ser inclusivos.",
        "Ajustar apenas o limite superior não resolveu o caso do limite inferior. Isso mostra que a teoria parcial estava errada — falta tornar ambos os lados inclusivos.",
        "Mesmo com o high corrigido, o teste do low ainda falha. Minha primeira correção só resolveu metade do problema — os dois limites precisam de <=.",
    ],
    "gen-hyprevision-last-n-items": [
        "Minha correção anterior (items[n:]) também falhou, porque conta a partir do início, não do fim. A teoria estava errada: eu precisava de um índice negativo contado a partir do fim.",
        "items[n:] também não resolveu — esse slice ainda conta a partir do começo da lista, não do final. A hipótese estava errada; o que falta é um índice negativo.",
        "A tentativa com items[n:] falhou de novo, porque continua contando do início. Isso invalida minha teoria — o correto é usar um índice negativo, contado do fim.",
        "Mesmo trocando para items[n:], o resultado ainda partiu do início da lista. Minha teoria estava errada: preciso de -n, não n, no índice do slice.",
    ],
    "gen-hyprevision-dedup-keep-order": [
        "Minha correção anterior (ordenar) também falhou — ordenar dá determinismo, mas não é a ordem original pedida. A teoria estava errada: o problema nunca foi falta de determinismo, era preservar a ordem de primeira aparição.",
        "Ordenar o resultado também não resolveu — dá uma ordem previsível, mas não é a ordem original que foi pedida. Minha teoria estava errada desde o início.",
        "sorted() também falhou o teste, porque produz ordem alfabética, não a ordem de aparição. Isso mostra que o problema nunca foi falta de determinismo.",
        "Mesmo ordenando, o teste continuou falhando — a ordem pedida é a de primeira aparição, não a alfabética. Preciso rastrear os itens vistos, não ordenar nada.",
    ],
}

_FINAL_NOTES_BY_TASK = {
    "gen-hyprevision-sum-up-to": [
        "Minha primeira correção (mudar o início do range) não resolveu — o problema real era o fim do range.",
        "A primeira tentativa (começar do 0) não bastou — o que faltava era incluir n no fim do range.",
        "Mudar o início do range não adiantou nada — a causa real estava no fim, não no começo.",
        "A primeira hipótese (início do range) estava errada — o ajuste certo era no fim.",
    ],
    "gen-hyprevision-is-in-range": [
        "Minha primeira correção tratou só o limite superior como inclusivo, mas o limite inferior também precisava ser.",
        "A primeira tentativa corrigiu só um lado do intervalo — faltava tornar o outro lado inclusivo também.",
        "Corrigir só o high não bastou — o low também precisava de <=.",
        "A primeira hipótese resolveu só metade do problema — os dois limites precisavam de ajuste.",
    ],
    "gen-hyprevision-last-n-items": [
        "Minha primeira correção (items[n:]) ainda contava a partir do início da lista, não do fim.",
        "A primeira tentativa (items[n:]) errou o lado — continuava contando do começo.",
        "Trocar para items[n:] não resolveu — faltava um índice negativo, contado do fim.",
        "A primeira hipótese sobre o slice estava errada — o índice precisava ser negativo.",
    ],
    "gen-hyprevision-dedup-keep-order": [
        "Minhas duas primeiras tentativas (set() e sorted(set())) removiam duplicatas mas não preservavam a ordem original.",
        "Nem set() nem sorted() preservaram a ordem original de aparição, só a implementação com rastreamento resolveu.",
        "As duas primeiras abordagens (set puro e ordenado) erravam a ordem — só rastrear os itens vistos manteve a ordem certa.",
        "Set() e sorted() falharam na ordem original — a solução certa precisava percorrer a lista rastreando o que já apareceu.",
    ],
}


def _rebuild_raw_text(original_raw_text: str, think1: str, think2: str, think3: str, final_text: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think1}</think>"
        + original_raw_text[segs[1].start : segs[4].end]  # rodada 1 (write_file + checker), verbatim
        + f"<think>{think2}</think>"
        + original_raw_text[segs[6].start : segs[9].end]  # rodada 2 (write_file + checker), verbatim
        + f"<think>{think3}</think>"
        + original_raw_text[segs[11].start : segs[14].end]  # rodada 3 (write_file + checker), verbatim
        + f"<final>{final_text}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        for k, (user_request, opening) in enumerate(zip(request_variants, _OPENINGS), start=1):
            idx = k - 1
            slots = {
                "summary": _SUMMARIES[base_id],
                "final_note": _FINAL_NOTES_BY_TASK[base_id][idx],
            }
            think1 = opening["think1"].format(**slots)
            think2 = _WRONG_THEORIES_BY_TASK[base_id][idx]
            think3 = _REVISIONS_BY_TASK[base_id][idx]
            final_text = opening["final"].format(**slots)

            raw_text = _rebuild_raw_text(original_raw_text, think1, think2, think3, final_text)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 4 (revisão de hipótese após 2ª falha real).")


if __name__ == "__main__":
    main()
