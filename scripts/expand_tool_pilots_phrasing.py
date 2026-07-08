#!/usr/bin/env python
"""Expansão de diversidade de frase para os pilotos de recuperação/sucesso de ferramenta
(D-tool-recovery-pilot, D-tool-success-pilot) — mesmo problema que D-generalization-gap, dessa
vez no texto de RESOLUÇÃO (`<think>`/`<final>`), não só no pedido.

Ao revisar os 10 exemplos gerados por `gen_tool_recovery_pilot.py`/`gen_tool_success_pilot.py`,
o `<think>`/`<final>` de fechamento seguia o MESMO gabarito quase palavra por palavra em todos
os 5 de cada categoria (ex.: "A implementação está pronta. Vou validar com o checker." e "A
validação confirma que compila e o(s) teste(s) passa(m)." repetidos sem variação). Isso é
exatamente o padrão que causou D-generalization-gap — só que na prosa de resolução em vez do
pedido do usuário.

Este script gera 4 paráfrases adicionais por exemplo-base (pedido do usuário E o
`<think>`/`<final>` ao redor), reaproveitando os MESMOS fragmentos de `<tool_call>`/
`<tool_result>` já executados de verdade (extraídos via `parse_segments`, nunca reescritos) —
nenhuma execução nova é necessária.

Cada trajetória-base segue a forma fixa (10 segmentos, em ordem):
    think, tool_call, tool_result, think, tool_call, tool_result, think, tool_call, tool_result, final
Só os 4 segmentos de prosa (índices 0, 3, 6, 9) são substituídos por variante; os 6 de
tool_call/tool_result (índices 1, 2, 4, 5, 7, 8) são copiados verbatim do original.

Uso:
    python scripts/expand_tool_pilots_phrasing.py
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

# --- variantes de fala para a categoria "recuperação de erro" (busca falha) -------------------

_RECOVERY_VARIANTS = [
    dict(
        think_before_search="Vou pesquisar {topic} antes de implementar, só para confirmar a abordagem.",
        think_after="A busca falhou por um problema de configuração do ambiente, não por causa da tarefa. Vou seguir com o que já conheço.",
        think_before_final="Pronto, agora preciso confirmar que está tudo certo.",
        final="Não tive acesso à pesquisa nessa sessão (falha de ambiente), então segui com o conhecimento que já tinha para implementar {func_desc}. Testei e confirmei que funciona corretamente.",
    ),
    dict(
        think_before_search="Antes de escrever o código, vou ver se acho algo sobre {topic}.",
        think_after="Isso não é um problema no meu raciocínio, é a ferramenta de busca que não está disponível agora. Sigo em frente mesmo assim.",
        think_before_final="Terminei a implementação; hora de rodar a validação.",
        final="A busca não funcionou aqui (limitação do ambiente, não da tarefa), mas isso não impede de entregar o resultado: implementei e validei {func_desc} com sucesso.",
    ),
    dict(
        think_before_search="Deixa eu conferir {topic} rapidamente antes de codificar.",
        think_after="Sem pesquisa disponível, tudo bem — uso o que já sei sobre o assunto.",
        think_before_final="Código pronto, só falta confirmar que compila e funciona.",
        final="Mesmo sem conseguir pesquisar (a ferramenta não está configurada aqui), entreguei uma implementação de {func_desc} validada e funcionando.",
    ),
    dict(
        think_before_search="Vou tentar confirmar {topic} via busca antes de partir para o código.",
        think_after="A falha veio da configuração do ambiente de busca, então não faz sentido insistir nisso agora. Vou implementar direto.",
        think_before_final="Já implementei; agora vou conferir se está tudo correto.",
        final="Como a pesquisa não estava disponível, implementei {func_desc} usando o que já sabia sobre o problema, e a validação confirma que está funcionando.",
    ),
]

_RECOVERY_REQUESTS = {
    "gen-toolrecovery-dot_product": [
        "Preciso de uma função em Python que calcule o produto interno de dois vetores.",
        "Pode escrever uma função que calcula o dot product de dois vetores?",
        "Implementa aí uma função de produto escalar entre dois vetores.",
        "Quero uma função Python para produto interno (dot product) de vetores.",
    ],
    "gen-toolrecovery-leap_year": [
        "Preciso de uma função que diga se um ano é bissexto ou não.",
        "Pode escrever algo que verifique se um ano é bissexto?",
        "Implementa uma checagem de ano bissexto em Python.",
        "Quero uma função is_leap_year que funcione corretamente.",
    ],
    "gen-toolrecovery-transpose": [
        "Preciso de uma função que transponha uma matriz.",
        "Pode escrever código que transpõe uma matriz representada como lista de listas?",
        "Implementa a transposição de matriz em Python.",
        "Quero uma função que inverta linhas e colunas de uma matriz.",
    ],
    "gen-toolrecovery-count_words": [
        "Preciso contar quantas palavras tem uma string.",
        "Pode escrever uma função que conta palavras num texto?",
        "Implementa uma contagem de palavras em Python.",
        "Quero uma função que retorne o número de palavras de uma frase.",
    ],
    "gen-toolrecovery-palindrome_clean": [
        "Preciso verificar se uma frase é um palíndromo, ignorando espaços e maiúsculas.",
        "Pode escrever uma checagem de palíndromo que ignore pontuação e caixa?",
        "Implementa uma função que detecta palíndromos mesmo com espaços/maiúsculas.",
        "Quero saber se uma string é palíndromo, desconsiderando formatação.",
    ],
}

_RECOVERY_TOPICS = {
    "gen-toolrecovery-dot_product": "a convenção de assinatura mais comum para produto interno",
    "gen-toolrecovery-leap_year": "a regra exata de ano bissexto",
    "gen-toolrecovery-transpose": "a forma mais idiomática de transpor uma matriz",
    "gen-toolrecovery-count_words": "se split() já lida bem com espaços múltiplos",
    "gen-toolrecovery-palindrome_clean": "a forma mais robusta de normalizar a string",
}

_RECOVERY_FUNC_DESC = {
    "gen-toolrecovery-dot_product": "dot_product(a, b) (soma do produto elemento a elemento)",
    "gen-toolrecovery-leap_year": "is_leap_year(year) (divisível por 4, exceto séculos não divisíveis por 400)",
    "gen-toolrecovery-transpose": "transpose(matrix) (usando zip(*matrix))",
    "gen-toolrecovery-count_words": "count_words(text) (split() já ignora espaços extras)",
    "gen-toolrecovery-palindrome_clean": "is_palindrome_clean(text) (normalizando e comparando com o reverso)",
}

# --- variantes de fala para a categoria "sucesso de busca" (busca funciona, ainda assim código real) --

_SUCCESS_VARIANTS = [
    dict(
        think_before_search="Vou dar uma olhada em {topic} antes de codificar.",
        think_after="Beleza, achei o que precisava: {fact}. Bora implementar com isso.",
        think_before_final="Terminei o código; falta só validar.",
        final="Pesquisei sobre {topic} e confirmei que {fact}. Implementei {func_desc} com base nisso, e a validação mostra que está tudo certo.",
    ),
    dict(
        think_before_search="Melhor confirmar {topic} antes de escrever qualquer coisa.",
        think_after="A pesquisa confirmou: {fact}. Vou seguir com isso na implementação.",
        think_before_final="Código pronto — hora de rodar o checker.",
        final="Depois de pesquisar sobre {topic} e confirmar que {fact}, implementei {func_desc}. A validação passou sem problemas.",
    ),
    dict(
        think_before_search="Antes de partir pro código, vou checar {topic}.",
        think_after="Ótimo, os resultados confirmam que {fact}. Vou usar isso.",
        think_before_final="Pronto o suficiente para validar agora.",
        final="Busquei informação sobre {topic}: {fact}. Usei isso para implementar {func_desc}, validado com sucesso pelo checker.",
    ),
    dict(
        think_before_search="Quero ter certeza sobre {topic} antes de implementar.",
        think_after="Confirmado pela busca: {fact}. Vou aplicar isso agora.",
        think_before_final="Já escrevi o código, só falta testar.",
        final="A pesquisa sobre {topic} confirmou que {fact}, e com isso implementei {func_desc}. Testei e validei que funciona corretamente.",
    ),
]

_SUCCESS_REQUESTS = {
    "gen-toolsuccess-fourier_transform": [
        "Escreva uma função Python que calcule a transformada de Fourier de um sinal.",
        "Preciso de código que calcule a FFT de um sinal usando numpy.",
        "Pode implementar a transformada de Fourier discreta em Python?",
        "Quero uma função que aplique FFT a um sinal.",
    ],
    "gen-toolsuccess-merge_sort": [
        "Escreva uma função que ordene uma lista usando mergesort.",
        "Preciso de uma implementação do algoritmo mergesort em Python.",
        "Pode fazer uma ordenação por mergesort?",
        "Quero uma função merge_sort que ordene listas.",
    ],
    "gen-toolsuccess-gcd_euclid": [
        "Implemente o cálculo do MDC usando o algoritmo de Euclides.",
        "Preciso de uma função que calcule o máximo divisor comum via Euclides.",
        "Pode escrever o algoritmo de Euclides para MDC?",
        "Quero uma função gcd que use o método de Euclides.",
    ],
    "gen-toolsuccess-binary_search": [
        "Escreva uma implementação de busca binária em Python.",
        "Preciso de uma função que faça busca binária numa lista ordenada.",
        "Pode implementar binary search?",
        "Quero uma função que busque um elemento com busca binária.",
    ],
    "gen-toolsuccess-is_prime": [
        "Crie uma função eficiente para verificar se um número é primo.",
        "Preciso de um teste de primalidade eficiente em Python.",
        "Pode escrever uma função is_prime otimizada?",
        "Quero verificar rapidamente se um número é primo.",
    ],
}

_SUCCESS_TOPICS = {
    "gen-toolsuccess-fourier_transform": "a função certa do numpy para FFT",
    "gen-toolsuccess-merge_sort": "os passos clássicos do mergesort",
    "gen-toolsuccess-gcd_euclid": "a versão iterativa do algoritmo de Euclides",
    "gen-toolsuccess-binary_search": "a abordagem iterativa clássica de busca binária",
    "gen-toolsuccess-is_prime": "a forma mais eficiente de testar primalidade para números pequenos/moderados",
}

_SUCCESS_FACTS = {
    "gen-toolsuccess-fourier_transform": "numpy.fft.fft calcula a transformada discreta de Fourier em 1D",
    "gen-toolsuccess-merge_sort": "divide-se a lista ao meio recursivamente até sublistas de 1 elemento, depois mescla em ordem",
    "gen-toolsuccess-gcd_euclid": "a versão iterativa (a, b = b, a % b) é mais eficiente em memória que a recursiva",
    "gen-toolsuccess-binary_search": "usar ponteiros low/high, calculando mid a cada iteração, reduz o intervalo pela metade",
    "gen-toolsuccess-is_prime": "divisão por tentativa só até a raiz quadrada de n, pulando pares, já é suficiente",
}

_SUCCESS_FUNC_DESC = {
    "gen-toolsuccess-fourier_transform": "fourier_transform(signal)",
    "gen-toolsuccess-merge_sort": "merge_sort(arr)",
    "gen-toolsuccess-gcd_euclid": "gcd_euclid(a, b)",
    "gen-toolsuccess-binary_search": "binary_search(arr, target)",
    "gen-toolsuccess-is_prime": "is_prime(n)",
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, think6: str, final9: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think0}</think>"
        + original_raw_text[segs[1].start : segs[2].end]  # tool_call+tool_result da busca (verbatim)
        + f"<think>{think3}</think>"
        + original_raw_text[segs[4].start : segs[5].end]  # tool_call+tool_result do write_file (verbatim)
        + f"<think>{think6}</think>"
        + original_raw_text[segs[7].start : segs[8].end]  # tool_call+tool_result do checker (verbatim)
        + f"<final>{final9}</final>"
    )


def _expand_category(requests: dict, variants: list, slot_maps: dict) -> int:
    count = 0
    for base_id, request_variants in requests.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        for k, (user_request, variant) in enumerate(zip(request_variants, variants), start=1):
            slots = {name: mapping[base_id] for name, mapping in slot_maps.items()}
            think0 = variant["think_before_search"].format(**slots)
            think3 = variant["think_after"].format(**slots)
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
    return count


def main() -> None:
    n1 = _expand_category(
        _RECOVERY_REQUESTS,
        _RECOVERY_VARIANTS,
        {"topic": _RECOVERY_TOPICS, "func_desc": _RECOVERY_FUNC_DESC},
    )
    n2 = _expand_category(
        _SUCCESS_REQUESTS,
        _SUCCESS_VARIANTS,
        {"topic": _SUCCESS_TOPICS, "fact": _SUCCESS_FACTS, "func_desc": _SUCCESS_FUNC_DESC},
    )
    print(f"Recuperação: {n1} variantes geradas. Sucesso: {n2} variantes geradas.")


if __name__ == "__main__":
    main()
