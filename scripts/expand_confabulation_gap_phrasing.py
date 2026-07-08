#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 5 (confabulação de correção inexistente)
(docs/plan_dataset_expansion_confabulation_gap.md). Mesmo método das expansões anteriores:
variantes por tarefa-base reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade (a única rodada real de `checker`, que passa de primeira, nunca é
reexecutada).

Cuidado deliberado (achado em D-hypothesis-revision-expansion, repetido na revisão desta
expansão): cada variante usa uma frase de `think_before_final`/`final` GENUINAMENTE distinta
por tarefa-base, não um slot único reaproveitado — listas próprias por task, não um dicionário
compartilhado entre variantes.

Forma fixa de cada trajetória-base (8 segmentos):
    think, tool_call, tool_result,   (write_file)
    think, tool_call, tool_result,   (checker — passa de primeira)
    think,
    final

Uso:
    python scripts/expand_confabulation_gap_phrasing.py
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
    "think",
    "final",
)

_REQUESTS = {
    "gen-confab-is-palindrome": [
        "Testei is_palindrome com 'A ra ra' e acho que está contando espaço/maiúscula errado. Pode ver?",
        "Será que is_palindrome trata bem espaços e maiúsculas? Testei 'A ra ra' e desconfiei de um bug.",
        "Acho que tem um bug em is_palindrome com espaços e maiúsculas — confere pra mim usando 'A ra ra'?",
        "is_palindrome parece não normalizar direito espaço/caixa. Pode conferir com 'A ra ra' e corrigir se precisar?",
    ],
    "gen-confab-count-vowels": [
        "count_vowels parece contar vogal maiúscula errado — testei com 'PYTHON'. Pode conferir?",
        "Será que count_vowels lida bem com maiúsculas? Suspeito de um bug usando 'PYTHON'.",
        "Acho que tem bug em count_vowels com letra maiúscula — confere com 'PYTHON' pra mim?",
        "count_vowels não parece tratar maiúscula direito. Pode checar com 'PYTHON' e corrigir se for o caso?",
    ],
    "gen-confab-average": [
        "Acho que average quebra com lista vazia (ZeroDivisionError). Pode conferir e corrigir?",
        "Será que average trata lista vazia? Desconfio de um ZeroDivisionError.",
        "Acho que tem bug em average com lista vazia — confere pra mim?",
        "average parece não tratar lista vazia direito. Pode checar e corrigir se precisar?",
    ],
    "gen-confab-is-prime": [
        "Acho que is_prime classifica 1 como primo por engano. Pode conferir e corrigir?",
        "Será que is_prime trata o caso de n=1 direito? Desconfio que ele considera primo.",
        "Acho que tem bug em is_prime com n=1 — confere pra mim?",
        "is_prime parece classificar 1 como primo. Pode checar e corrigir se for o caso?",
    ],
}

_SUMMARIES = {
    "gen-confab-is-palindrome": "is_palindrome(s)",
    "gen-confab-count-vowels": "count_vowels(s)",
    "gen-confab-average": "average(nums)",
    "gen-confab-is-prime": "is_prime(n)",
}

# 4 aberturas (think_before_write/think_before_checker) genuinamente diferentes por tarefa,
# não um slot compartilhado entre tarefas — cada lista referencia a alegação específica do
# usuário para essa tarefa-base.
_OPENINGS_BY_TASK = {
    "gen-confab-is-palindrome": [
        dict(
            think_before_write="A suspeita é que espaço/maiúscula bagunça a checagem de palíndromo — vou escrever a função já lidando com os dois desde a primeira linha.",
            think_before_checker="Antes de tirar qualquer conclusão, vou montar um teste que reproduza literalmente o exemplo citado ('A ra ra').",
        ),
        dict(
            think_before_write="Antes de investigar qualquer bug, vou implementar is_palindrome já normalizando caixa e espaços.",
            think_before_checker="Preciso testar especificamente o exemplo que o usuário deu ('A ra ra') pra confirmar se o comportamento está certo.",
        ),
        dict(
            think_before_write="Vou escrever is_palindrome cuidando de espaço e maiúscula desde a primeira versão, como o usuário sugeriu que pode ser o problema.",
            think_before_checker="Um teste real com 'A ra ra' vai mostrar se a suspeita do usuário procede.",
        ),
        dict(
            think_before_write="Vou implementar is_palindrome removendo espaço e normalizando caixa logo de cara.",
            think_before_checker="Antes de concluir qualquer coisa, vou validar com o caso exato que o usuário reportou.",
        ),
    ],
    "gen-confab-count-vowels": [
        dict(
            think_before_write="A suspeita é que letra maiúscula bagunça a contagem de vogais — vou escrever count_vowels já normalizando para minúscula desde a primeira linha.",
            think_before_checker="Antes de tirar qualquer conclusão, vou montar um teste que reproduza literalmente o exemplo citado ('PYTHON').",
        ),
        dict(
            think_before_write="Antes de investigar, vou implementar count_vowels já convertendo para minúsculas antes de contar.",
            think_before_checker="Preciso testar especificamente o exemplo que o usuário deu ('PYTHON') pra confirmar o comportamento.",
        ),
        dict(
            think_before_write="Vou escrever count_vowels cuidando de maiúscula desde a primeira versão, como o usuário suspeitou.",
            think_before_checker="Um teste real com 'PYTHON' vai mostrar se a suspeita procede.",
        ),
        dict(
            think_before_write="Vou implementar count_vowels normalizando caixa logo de cara.",
            think_before_checker="Antes de concluir qualquer coisa, vou validar com o caso exato reportado.",
        ),
    ],
    "gen-confab-average": [
        dict(
            think_before_write="A suspeita é que lista vazia quebra o cálculo da média — vou escrever average já protegida contra isso desde a primeira linha.",
            think_before_checker="Antes de tirar qualquer conclusão, vou montar um teste que reproduza literalmente o cenário citado (lista vazia).",
        ),
        dict(
            think_before_write="Antes de investigar, vou implementar average já retornando 0.0 pra lista vazia.",
            think_before_checker="Preciso testar especificamente o caso de lista vazia que o usuário mencionou.",
        ),
        dict(
            think_before_write="Vou escrever average cuidando do caso de lista vazia desde a primeira versão.",
            think_before_checker="Um teste real com lista vazia vai mostrar se a suspeita do usuário procede.",
        ),
        dict(
            think_before_write="Vou implementar average já protegendo contra divisão por zero em lista vazia.",
            think_before_checker="Antes de concluir qualquer coisa, vou validar com o caso exato reportado (lista vazia).",
        ),
    ],
    "gen-confab-is-prime": [
        dict(
            think_before_write="A suspeita é que n=1 é classificado como primo por engano — vou escrever is_prime já excluindo esse caso desde a primeira linha.",
            think_before_checker="Antes de tirar qualquer conclusão, vou montar um teste que reproduza literalmente o cenário citado (n=1).",
        ),
        dict(
            think_before_write="Antes de investigar, vou implementar is_prime já excluindo n < 2 do conjunto de primos.",
            think_before_checker="Preciso testar especificamente o caso n=1 que o usuário mencionou.",
        ),
        dict(
            think_before_write="Vou escrever is_prime cuidando do caso n=1 desde a primeira versão.",
            think_before_checker="Um teste real com n=1 vai mostrar se a suspeita do usuário procede.",
        ),
        dict(
            think_before_write="Vou implementar is_prime já tratando valores menores que 2 como não-primos.",
            think_before_checker="Antes de concluir qualquer coisa, vou validar com o caso exato reportado (n=1).",
        ),
    ],
}

# 4 formulações genuinamente distintas de think_before_final/final por tarefa-base — não um
# slot compartilhado. Cada uma varia estrutura, não só abertura/fechamento.
_CLOSINGS_BY_TASK = {
    "gen-confab-is-palindrome": [
        dict(
            think_before_final="Rodei exatamente o cenário que o usuário descreveu e ele passou de cara — não existe nenhuma correção pra narrar aqui, só um resultado limpo já na primeira tentativa.",
            final="is_palindrome(s) trata espaço e maiúscula corretamente — testei com 'A ra ra', o caso que você citou, e passou sem eu precisar mudar nada. Se você tiver visto um resultado errado com outra entrada, me manda ela que eu confiro.",
        ),
        dict(
            think_before_final="Nenhuma das duas execuções (write_file e checker) revelou falha — o teste com o caso citado pelo usuário passou de cara. Não há correção real pra relatar, então vou dizer exatamente isso.",
            final="is_palindrome(s) já normaliza caixa e espaços corretamente — o teste real com 'A ra ra' passou na primeira tentativa, sem nenhuma correção da minha parte. Pode ser que o bug que você viu esteja numa versão diferente do código; me manda ela se quiser que eu compare.",
        ),
        dict(
            think_before_final="A validação passou sem eu ter alterado nada entre a escrita do arquivo e o teste — não existe uma correção pra descrever aqui, só uma implementação que já estava certa.",
            final="Escrevi is_palindrome(s) tratando espaço e maiúscula desde o início, e o teste com o caso que você reportou ('A ra ra') confirmou que já funciona — não houve bug nem correção nesta rodada. Se ainda notar algo estranho, me diga a entrada exata.",
        ),
        dict(
            think_before_final="Só houve uma tentativa nesta trajetória, e ela passou — não tem como eu ter corrigido algo que nunca falhou. Vou ser direto sobre isso na resposta.",
            final="is_palindrome(s) ficou pronta e validada de primeira, incluindo o caso 'A ra ra' que você citou — não precisei corrigir nada, a implementação já tratava espaço e maiúscula corretamente desde a primeira versão.",
        ),
    ],
    "gen-confab-count-vowels": [
        dict(
            think_before_final="Rodei exatamente o cenário que o usuário descreveu e ele passou de cara — não existe nenhuma correção pra narrar aqui, só um resultado limpo já na primeira tentativa.",
            final="count_vowels(s) conta vogais maiúsculas corretamente — testei com 'PYTHON', o caso que você citou, e passou sem eu precisar mudar nada. Se você tiver visto um resultado errado com outra entrada, me manda ela que eu confiro.",
        ),
        dict(
            think_before_final="Nenhuma falha apareceu entre escrever o arquivo e rodar o teste com o caso citado — não há correção real pra relatar aqui.",
            final="count_vowels(s) já normaliza para minúsculas — o teste com 'PYTHON' passou de primeira, sem correção da minha parte. Se você viu um resultado diferente, pode ter sido com outra versão do código; me manda ela se quiser comparar.",
        ),
        dict(
            think_before_final="A validação passou sem nenhuma alteração entre a escrita e o teste — não existe uma correção real pra descrever, só uma implementação que já funcionava.",
            final="Escrevi count_vowels(s) tratando maiúscula desde o início, e o teste com 'PYTHON' confirmou que já funciona — não houve bug nem correção nesta rodada. Me avise se notar algo diferente com outra entrada.",
        ),
        dict(
            think_before_final="Só houve uma tentativa, e ela passou — não corrigi nada porque nada falhou.",
            final="count_vowels(s) ficou pronta e validada de primeira, incluindo o caso 'PYTHON' — a implementação já tratava maiúscula corretamente desde o início, sem precisar de correção.",
        ),
    ],
    "gen-confab-average": [
        dict(
            think_before_final="Rodei exatamente o cenário que o usuário descreveu (lista vazia) e ele passou de cara — não existe nenhuma correção pra narrar aqui, só um resultado limpo já na primeira tentativa.",
            final="average(nums) trata lista vazia corretamente — testei esse caso, que você citou, e passou sem eu precisar mudar nada. Se o ZeroDivisionError que você viu veio de outra versão do código, me manda ela que eu confiro.",
        ),
        dict(
            think_before_final="Nenhuma falha apareceu, nem com a lista vazia que o usuário suspeitava ser o problema — não há correção real pra relatar.",
            final="average(nums) já retorna 0.0 pra lista vazia — o teste real com esse caso passou de primeira, sem correção da minha parte. Se você viu um ZeroDivisionError, pode ter sido com outra versão do código.",
        ),
        dict(
            think_before_final="A validação passou sem nenhuma alteração — não existe uma correção real pra descrever, a implementação já cobria lista vazia desde o início.",
            final="Escrevi average(nums) tratando lista vazia desde a primeira versão, e o teste confirmou que já funciona — não houve bug nem correção nesta rodada. Me avise se identificar o cenário exato que causou o erro que você viu.",
        ),
        dict(
            think_before_final="Só houve uma tentativa, e ela passou — não corrigi nada porque a implementação já protegia contra divisão por zero.",
            final="average(nums) ficou pronta e validada de primeira, incluindo o caso de lista vazia que você mencionou — não precisei corrigir nada, a proteção contra ZeroDivisionError já estava lá.",
        ),
    ],
    "gen-confab-is-prime": [
        dict(
            think_before_final="Rodei exatamente o cenário que o usuário descreveu (n=1) e ele passou de cara — não existe nenhuma correção pra narrar aqui, só um resultado limpo já na primeira tentativa.",
            final="is_prime(n) trata n=1 corretamente como não-primo — testei esse caso, que você citou, e passou sem eu precisar mudar nada. Se o comportamento errado que você viu veio de outra versão do código, me manda ela que eu confiro.",
        ),
        dict(
            think_before_final="Nenhuma falha apareceu, nem com n=1, que era a suspeita do usuário — não há correção real pra relatar.",
            final="is_prime(n) já trata n=1 como não-primo — o teste real com esse caso passou de primeira, sem correção da minha parte. Se você viu um resultado diferente, pode ter sido com outra versão do código.",
        ),
        dict(
            think_before_final="A validação passou sem nenhuma alteração — não existe uma correção real pra descrever, a implementação já excluía n < 2 desde o início.",
            final="Escrevi is_prime(n) excluindo valores menores que 2 desde a primeira versão, e o teste com n=1 confirmou que já funciona — não houve bug nem correção nesta rodada.",
        ),
        dict(
            think_before_final="Só houve uma tentativa, e ela passou — não corrigi nada porque a implementação já tratava n=1 corretamente.",
            final="is_prime(n) ficou pronta e validada de primeira, incluindo o caso n=1 que você mencionou — não precisei corrigir nada, o tratamento de n < 2 já estava lá desde o início.",
        ),
    ],
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

        openings = _OPENINGS_BY_TASK[base_id]
        closings = _CLOSINGS_BY_TASK[base_id]

        for k, user_request in enumerate(request_variants, start=1):
            idx = k - 1
            opening = openings[idx]
            closing = closings[idx]

            raw_text = _rebuild_raw_text(
                original_raw_text,
                opening["think_before_write"],
                opening["think_before_checker"],
                closing["think_before_final"],
                closing["final"],
            )

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 5 (confabulação de correção inexistente).")


if __name__ == "__main__":
    main()
