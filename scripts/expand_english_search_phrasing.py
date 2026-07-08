#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Busca bem-sucedida em inglês"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4). Mesmo método de
`scripts/expand_tool_pilots_phrasing.py`: 8 variantes por tarefa-base, reaproveitando os
fragmentos `<tool_call>`/`<tool_result>` já executados de verdade. Cada variante mantém o
reconhecimento explícito de idioma (inglês na busca, português na resposta) com frases
estruturalmente diferentes entre si.

Uso:
    python scripts/expand_english_search_phrasing.py
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
        think_before_search="Vou pesquisar sobre {topic} antes de implementar {summary}.",
        think_after="O conteúdo encontrado está em inglês, mas a conversa é em português — vou responder no idioma do usuário mesmo assim. A busca confirma: {fact}.",
        think_before_final="A implementação está pronta com base no que a busca confirmou. Vou validar antes de responder.",
        final="Pesquisei sobre {topic} (o resultado veio em inglês, mas respondo em português) e confirmei que {fact}. Implementei {summary} com base nisso, e a validação passou.",
    ),
    dict(
        think_before_search="Antes de codificar, quero confirmar {topic} pesquisando.",
        think_after="Reparo que o conteúdo retornado está em inglês; como a conversa é em português, vou continuar respondendo em português. Confirmado: {fact}.",
        think_before_final="Código pronto; falta rodar a validação antes de fechar a resposta.",
        final="A busca sobre {topic} trouxe conteúdo em inglês, mas mantive a resposta em português: {fact}. Com isso implementei {summary}, validado com sucesso.",
    ),
    dict(
        think_before_search="Melhor pesquisar {topic} para confirmar a abordagem certa antes de escrever código.",
        think_after="O material que encontrei está em inglês — vou continuar em português na resposta, só usando isso como referência técnica. Confirma-se que {fact}.",
        think_before_final="Com o código pronto, o próximo passo é validar antes de responder.",
        final="Encontrei informação em inglês sobre {topic}, mas a resposta segue em português: {fact}. Implementei {summary} a partir disso, e a validação confirma que funciona.",
    ),
    dict(
        think_before_search="Vou confirmar {topic} com uma busca antes de implementar {summary}.",
        think_after="Idioma da busca (inglês) e da conversa (português) são diferentes, mas isso não muda como respondo — sigo em português. O achado central: {fact}.",
        think_before_final="Falta só confirmar que o código está correto antes de responder ao usuário.",
        final="A pesquisa sobre {topic} veio em inglês; respondo em português mesmo assim. Confirmado que {fact}, implementei {summary} e validei com sucesso.",
    ),
    dict(
        think_before_search="Antes de partir para o código, vou checar {topic}.",
        think_after="Só destacando: o conteúdo da busca está em inglês, mas continuo respondendo em português por ser o idioma da conversa. O ponto principal: {fact}.",
        think_before_final="Código pronto. Rodo a validação agora para confirmar antes de responder.",
        final="Busquei sobre {topic} (conteúdo em inglês) e mantive a resposta em português. Com {fact} confirmado, implementei {summary}, validado sem problemas.",
    ),
    dict(
        think_before_search="Quero ter certeza sobre {topic} antes de implementar — vou pesquisar.",
        think_after="O idioma do resultado (inglês) não é o da conversa (português), então vou traduzir mentalmente o essencial e responder em português. Confirmado: {fact}.",
        think_before_final="Antes de finalizar, preciso rodar a validação para confirmar que está tudo certo.",
        final="A pesquisa sobre {topic} retornou conteúdo em inglês, mas a resposta é em português: {fact}. Implementei {summary} com base nisso, validado corretamente.",
    ),
    dict(
        think_before_search="Vou pesquisar {topic} para não implementar algo baseado só em suposição.",
        think_after="Percebo que o resultado veio em inglês — como o usuário fala português, é assim que vou responder, usando o conteúdo só como base técnica. Acha-se que {fact}.",
        think_before_final="Terminei a implementação; agora preciso validar antes de responder ao usuário.",
        final="Pesquisei {topic}, com resultado em inglês, mas sigo respondendo em português: {fact}. Implementei {summary}, e a validação confirma que está correto.",
    ),
    dict(
        think_before_search="Antes de escrever qualquer código, vou pesquisar {topic}.",
        think_after="O material retornado está em inglês; a conversa é em português, então é nesse idioma que vou continuar. O que importa: {fact}.",
        think_before_final="Só falta validar o código antes de considerar a tarefa concluída.",
        final="A busca sobre {topic} trouxe material em inglês, mas a resposta ao usuário segue em português. Confirmado {fact}, implementei {summary} e validei com sucesso.",
    ),
]

_REQUESTS = {
    "gen-ensearch-ball-bounce": [
        "Pesquisa como fazer uma bolinha quicar na tela e implementa a lógica com pygame.",
        "Preciso da lógica de uma bolinha quicando, tipo pygame — pode pesquisar e implementar?",
        "Quero a física de quique de uma bolinha em pygame, pesquisada e implementada.",
        "Implementa a lógica de bolinha quicando na tela, pesquisando antes como fazer.",
        "Cria a lógica de quique de bola com pygame, depois de pesquisar a abordagem certa.",
        "Estou precisando da lógica de bolinha quicando (pygame), pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando a física de bolinha quicando em pygame?",
        "A lógica de bolinha quicando com pygame, por favor, pesquisada antes de implementar.",
    ],
    "gen-ensearch-quicksort": [
        "Pesquisa o algoritmo quicksort e implementa em Python.",
        "Preciso do quicksort — pode pesquisar a abordagem e implementar?",
        "Quero uma implementação de quicksort, pesquisada antes.",
        "Implementa quicksort em Python, pesquisando o particionamento certo antes.",
        "Cria uma função de quicksort, depois de pesquisar como funciona.",
        "Estou precisando de quicksort implementado, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando quicksort em Python?",
        "Quicksort em Python, por favor, pesquisado antes de implementar.",
    ],
    "gen-ensearch-dijkstra": [
        "Pesquisa o algoritmo de Dijkstra e implementa o caminho mais curto num grafo.",
        "Preciso de Dijkstra — pode pesquisar e implementar o cálculo de caminho mínimo?",
        "Quero uma implementação de Dijkstra, pesquisada antes.",
        "Implementa Dijkstra em Python, pesquisando a abordagem com fila de prioridade.",
        "Cria uma função de caminho mais curto (Dijkstra), depois de pesquisar como funciona.",
        "Estou precisando do algoritmo de Dijkstra implementado, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando Dijkstra em Python?",
        "Dijkstra em Python, por favor, pesquisado antes de implementar.",
    ],
    "gen-ensearch-hash-table": [
        "Pesquisa como implementar uma tabela hash com encadeamento separado e implementa.",
        "Preciso de uma hash table com chaining — pode pesquisar e implementar?",
        "Quero uma tabela hash com encadeamento separado, pesquisada antes.",
        "Implementa uma hash table em Python, pesquisando a técnica de encadeamento.",
        "Cria uma tabela hash com colisão tratada por encadeamento, depois de pesquisar.",
        "Estou precisando de uma hash table implementada, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma tabela hash com chaining?",
        "Uma tabela hash com encadeamento separado, por favor, pesquisada antes.",
    ],
    "gen-ensearch-binary-tree-inorder": [
        "Pesquisa travessia in-order de árvore binária e implementa de forma iterativa.",
        "Preciso de in-order traversal — pode pesquisar a versão iterativa e implementar?",
        "Quero uma travessia in-order de árvore binária, pesquisada antes.",
        "Implementa in-order traversal em Python, pesquisando a versão com pilha.",
        "Cria uma travessia in-order iterativa, depois de pesquisar como funciona.",
        "Estou precisando de travessia in-order implementada, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando in-order traversal iterativo?",
        "Travessia in-order de árvore binária, por favor, pesquisada antes de implementar.",
    ],
    "gen-ensearch-asyncio-basics": [
        "Pesquisa o básico de asyncio em Python e implementa uma função assíncrona.",
        "Preciso entender asyncio — pode pesquisar e implementar algo assíncrono simples?",
        "Quero uma função assíncrona básica, pesquisada antes.",
        "Implementa uma coroutine simples em Python, pesquisando o básico de asyncio.",
        "Cria uma função async, depois de pesquisar como o event loop funciona.",
        "Estou precisando de asyncio básico implementado, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma função assíncrona em Python?",
        "Uma função assíncrona básica, por favor, pesquisada antes de implementar.",
    ],
    "gen-ensearch-regex-lookahead": [
        "Pesquisa lookahead/lookbehind em regex e implementa um separador de camelCase.",
        "Preciso separar camelCase com regex — pode pesquisar lookahead e implementar?",
        "Quero uma função que separe camelCase usando regex, pesquisada antes.",
        "Implementa split de camelCase em Python, pesquisando lookahead/lookbehind.",
        "Cria uma função que quebre camelCase em palavras, depois de pesquisar a regex certa.",
        "Estou precisando separar camelCase com regex, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando um separador de camelCase com regex?",
        "Um separador de camelCase com regex, por favor, pesquisado antes.",
    ],
    "gen-ensearch-dataclasses": [
        "Pesquisa dataclasses em Python e implementa uma classe Point com esse recurso.",
        "Preciso entender dataclasses — pode pesquisar e implementar uma classe Point?",
        "Quero uma classe Point usando @dataclass, pesquisada antes.",
        "Implementa Point com dataclass em Python, pesquisando a sintaxe certa.",
        "Cria uma classe Point usando dataclasses, depois de pesquisar como funciona.",
        "Estou precisando de uma classe Point com dataclass, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma classe Point com @dataclass?",
        "Uma classe Point com dataclass, por favor, pesquisada antes de implementar.",
    ],
    "gen-ensearch-base64": [
        "Pesquisa como codificar e decodificar base64 em Python e implementa as duas funções.",
        "Preciso de encode/decode base64 — pode pesquisar e implementar?",
        "Quero funções de codificação e decodificação base64, pesquisadas antes.",
        "Implementa encode_text e decode_text em base64, pesquisando o módulo certo.",
        "Cria funções de base64 em Python, depois de pesquisar como usar o módulo.",
        "Estou precisando de codificação base64 implementada, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando encode/decode em base64?",
        "Funções de encode/decode base64, por favor, pesquisadas antes de implementar.",
    ],
    "gen-ensearch-linked-list-cycle": [
        "Pesquisa o algoritmo de Floyd para detectar ciclo em lista encadeada e implementa.",
        "Preciso detectar ciclo em linked list — pode pesquisar Floyd e implementar?",
        "Quero uma detecção de ciclo em lista encadeada, pesquisada antes.",
        "Implementa detecção de ciclo em Python, pesquisando o algoritmo de Floyd.",
        "Cria uma função has_cycle, depois de pesquisar a técnica dos dois ponteiros.",
        "Estou precisando detectar ciclo em lista encadeada, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando detecção de ciclo com Floyd?",
        "Detecção de ciclo em lista encadeada, por favor, pesquisada antes.",
    ],
    "gen-ensearch-rsa-basic": [
        "Pesquisa como funciona o RSA e implementa uma versão simplificada didática.",
        "Preciso entender RSA — pode pesquisar e implementar uma versão simples de chaves?",
        "Quero uma implementação didática de RSA, pesquisada antes.",
        "Implementa geração de chaves RSA simplificada, pesquisando os passos do algoritmo.",
        "Cria uma versão simples de RSA (encrypt/decrypt), depois de pesquisar como funciona.",
        "Estou precisando de uma versão didática de RSA, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma versão simplificada de RSA?",
        "Uma versão didática de RSA, por favor, pesquisada antes de implementar.",
    ],
    "gen-ensearch-pandas-groupby": [
        "Pesquisa groupby do pandas e implementa uma soma de vendas por categoria.",
        "Preciso agrupar vendas por categoria com pandas — pode pesquisar e implementar?",
        "Quero uma soma de vendas por categoria usando pandas, pesquisada antes.",
        "Implementa agregação de vendas com groupby, pesquisando o padrão split-apply-combine.",
        "Cria uma função que some vendas por categoria com pandas, depois de pesquisar.",
        "Estou precisando somar vendas por categoria com pandas, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma agregação de vendas com groupby?",
        "Uma soma de vendas por categoria com pandas, por favor, pesquisada antes.",
    ],
    "gen-ensearch-bubble-sort": [
        "Pesquisa bubble sort e implementa com a otimização de parar cedo.",
        "Preciso de bubble sort otimizado — pode pesquisar e implementar?",
        "Quero uma implementação de bubble sort com flag de troca, pesquisada antes.",
        "Implementa bubble sort em Python, pesquisando a otimização de parada antecipada.",
        "Cria uma função de bubble sort otimizada, depois de pesquisar como funciona.",
        "Estou precisando de bubble sort implementado, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando bubble sort com otimização?",
        "Bubble sort otimizado, por favor, pesquisado antes de implementar.",
    ],
    "gen-ensearch-multiprocessing-vs-threading": [
        "Pesquisa a diferença entre multiprocessing e threading e implementa uma função que recomende qual usar.",
        "Preciso saber quando usar threading ou multiprocessing — pode pesquisar e implementar uma recomendação?",
        "Quero uma função que diga qual abordagem de concorrência usar, pesquisada antes.",
        "Implementa uma recomendação de concorrência, pesquisando a diferença entre as duas abordagens.",
        "Cria uma função que escolha entre multiprocessing e threading, depois de pesquisar.",
        "Estou precisando de uma recomendação sobre threading vs multiprocessing, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma função que recomende a abordagem certa de concorrência?",
        "Uma recomendação de concorrência (threading vs multiprocessing), por favor, pesquisada antes.",
    ],
    "gen-ensearch-balanced-parentheses": [
        "Pesquisa verificação de parênteses balanceados com pilha e implementa para múltiplos tipos de colchetes.",
        "Preciso verificar parênteses balanceados — pode pesquisar a abordagem com pilha e implementar?",
        "Quero uma função que valide parênteses balanceados, pesquisada antes.",
        "Implementa verificação de parênteses balanceados, pesquisando a técnica com pilha.",
        "Cria uma função is_balanced para múltiplos tipos de colchetes, depois de pesquisar.",
        "Estou precisando validar parênteses balanceados, pode pesquisar primeiro?",
        "Me ajuda pesquisando e implementando uma verificação de parênteses balanceados?",
        "Uma verificação de parênteses balanceados, por favor, pesquisada antes de implementar.",
    ],
}

_TOPICS = {
    "gen-ensearch-ball-bounce": "a lógica clássica de bolinha quicando com pygame",
    "gen-ensearch-quicksort": "o esquema de particionamento do quicksort",
    "gen-ensearch-dijkstra": "a abordagem de Dijkstra com fila de prioridade",
    "gen-ensearch-hash-table": "a estrutura de tabela hash com encadeamento separado",
    "gen-ensearch-binary-tree-inorder": "a travessia in-order iterativa de árvore binária",
    "gen-ensearch-asyncio-basics": "o básico de asyncio e coroutines",
    "gen-ensearch-regex-lookahead": "lookahead/lookbehind em regex do Python",
    "gen-ensearch-dataclasses": "o decorator @dataclass do Python",
    "gen-ensearch-base64": "codificação e decodificação base64",
    "gen-ensearch-linked-list-cycle": "o algoritmo de Floyd para detecção de ciclo",
    "gen-ensearch-rsa-basic": "os passos de geração de chave do RSA",
    "gen-ensearch-pandas-groupby": "o padrão split-apply-combine do groupby",
    "gen-ensearch-bubble-sort": "a otimização de parada antecipada do bubble sort",
    "gen-ensearch-multiprocessing-vs-threading": "a diferença entre multiprocessing e threading",
    "gen-ensearch-balanced-parentheses": "a verificação de parênteses balanceados com pilha",
}

_FACTS = {
    "gen-ensearch-ball-bounce": "a velocidade deve ser guardada como vetor e o componente correspondente invertido ao bater na borda",
    "gen-ensearch-quicksort": "o esquema de Lomuto usa o último elemento como pivô e particiona os menores à esquerda",
    "gen-ensearch-dijkstra": "usar heapq como fila de prioridade, atualizando distâncias a partir do nó de menor custo",
    "gen-ensearch-hash-table": "cada posição do array guarda uma lista de pares chave-valor que colidiram no mesmo índice",
    "gen-ensearch-binary-tree-inorder": "empilhar os filhos à esquerda até None, desempilhar, registrar o valor e mover para a direita",
    "gen-ensearch-asyncio-basics": "usar 'async def' para criar a coroutine e asyncio.run() para executá-la",
    "gen-ensearch-regex-lookahead": "lookahead e lookbehind são asserções de largura zero que não consomem caracteres",
    "gen-ensearch-dataclasses": "@dataclass gera __init__, __repr__ e __eq__ automaticamente a partir dos atributos anotados",
    "gen-ensearch-base64": "codificar exige converter para bytes antes de aplicar base64.b64encode",
    "gen-ensearch-linked-list-cycle": "dois ponteiros (lento e rápido) se encontram se, e somente se, houver ciclo",
    "gen-ensearch-rsa-basic": "a chave pública/privada vem do totiente de Euler e do inverso modular do expoente escolhido",
    "gen-ensearch-pandas-groupby": "groupby seguido de uma agregação resume os dados por grupo",
    "gen-ensearch-bubble-sort": "uma flag de troca permite encerrar mais cedo quando nenhuma troca ocorre numa passada",
    "gen-ensearch-multiprocessing-vs-threading": "o GIL faz threading ideal para I/O-bound e multiprocessing ideal para CPU-bound",
    "gen-ensearch-balanced-parentheses": "só contar quantidades não detecta aninhamento errado, por isso a pilha com mapeamento de pares é necessária",
}

_SUMMARIES = {
    "gen-ensearch-ball-bounce": "bounce_update(rect, velocity, screen_width, screen_height)",
    "gen-ensearch-quicksort": "quicksort(arr)",
    "gen-ensearch-dijkstra": "dijkstra(graph, start)",
    "gen-ensearch-hash-table": "a classe HashTable com insert e get",
    "gen-ensearch-binary-tree-inorder": "inorder_traversal(root)",
    "gen-ensearch-asyncio-basics": "greet_all(names) como coroutine",
    "gen-ensearch-regex-lookahead": "split_camel_case(text)",
    "gen-ensearch-dataclasses": "a classe Point com distance_to",
    "gen-ensearch-base64": "encode_text e decode_text",
    "gen-ensearch-linked-list-cycle": "has_cycle(head)",
    "gen-ensearch-rsa-basic": "generate_keypair, encrypt e decrypt",
    "gen-ensearch-pandas-groupby": "total_sales_by_category(records)",
    "gen-ensearch-bubble-sort": "bubble_sort(arr)",
    "gen-ensearch-multiprocessing-vs-threading": "choose_concurrency_strategy(is_cpu_bound)",
    "gen-ensearch-balanced-parentheses": "is_balanced(text)",
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


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        slots = {
            "topic": _TOPICS[base_id],
            "fact": _FACTS[base_id],
            "summary": _SUMMARIES[base_id],
        }

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
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

    print(f"{count} variantes geradas para a categoria Busca-em-inglês.")


if __name__ == "__main__":
    main()
