#!/usr/bin/env python
"""Lote base da categoria "Busca bem-sucedida em inglês" (docs/plan_dataset_expansion_oop_shell.md,
seção 2). Complementa `scripts/gen_tool_success_pilot.py` (que já cobre 5 exemplos desse padrão)
com mais 15 tarefas-base, cobrindo tópicos técnicos variados cujo conteúdo de busca vem
naturalmente em inglês, mesmo com a conversa em português.

Achado real que motivou esta categoria: pedido de jogo em pygame e de FFT, com `web_search`
funcionando de verdade — o modelo pesquisou, leu a documentação em inglês, mas respondeu em
inglês e nunca produziu/validou código. Aqui, cada exemplo inclui um `<think>` EXPLÍCITO
reconhecendo o descompasso de idioma antes de prosseguir, conforme decidido em conversa (seção 3
do plano) — não deixar essa regra implícita só na diferença de idioma entre a busca e o `<final>`.

Cada exemplo: `<think>` -> `web_search` (via `MockSearchBackend` com fixtures reais, coletadas de
buscas de verdade em `data/fixtures/web_search/`) -> `<think>` com o reconhecimento explícito do
idioma -> `write_file` real -> `<think>` -> `checker` real (`compile_and_test`) -> `<final>` em
português. Nenhum `tool_result` é fabricado.

Uso:
    python scripts/gen_english_search_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402
from src.search.mock_backend import MockSearchBackend  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402
from src.tools.executors import web_search_tool  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures" / "web_search"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()


def _sanitize_checker_result(result):
    if result.passed and result.data.get("stderr"):
        result.data["stderr"] = ""
    return result


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    if name == "checker":
        result = _sanitize_checker_result(result)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


def build_example(
    id_, domain, difficulty, user_request, search_query,
    think_before_search, think_after_search,
    file_path, file_content, test_path, test_content,
    think_before_final, final_text,
) -> dict:
    web_search_tool._cached_backend = MockSearchBackend(FIXTURES_DIR)

    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_search}</think>")
        search_result = _tool_call(traj, sandbox, "web_search", {"query": search_query})
        assert search_result.passed, f"esperava web_search OK (fixture mock) em {id_}: {search_result.to_json()}"
        assert search_result.data.get("results"), f"fixture de busca vazia para {id_} ({search_query})"

        traj.append_raw(f"<think>{think_after_search}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": file_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": file_content},
                {"path": test_path, "content": test_content},
            ],
        }
        traj.append_raw(f"<think>{think_before_final}</think>")
        checker_result = _tool_call(traj, sandbox, "checker", checker_args)
        assert checker_result.passed, f"checker falhou inesperadamente em {id_}: {checker_result.to_json()}"

        traj.append_raw(f"<final>{final_text}</final>")
    finally:
        sandbox.cleanup()
        web_search_tool.reset_cached_backend()

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "python",
            "difficulty": difficulty,
            "tools_used": ["web_search", "write_file", "checker"],
            "num_steps": 3,
            "task_type": "multi_tool_call",
            "source": "curated_manual_real_execution",
            "license": "synthetic-no-license-needed",
            "validation_status": "validated",
            "execution_performed": True,
            "split": "train",
            "checker_used": "checker-python-1.0",
            "expected_result": {"passed": True},
            "execution_classification": "tested",
        },
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


_LANG_ACK = (
    "O conteúdo que a busca retornou está em inglês, mas a conversa com o usuário é em "
    "português — vou continuar respondendo em português mesmo assim, usando o que encontrei "
    "só como base técnica."
)

EXAMPLES = [
    dict(
        id_="gen-ensearch-ball-bounce",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise como fazer uma bolinha quicar na tela e implemente a lógica de quique usando pygame.",
        search_query="pygame bouncing ball game tutorial python",
        think_before_search="Vou pesquisar a abordagem clássica de bolinha quicando com pygame antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: guardar a velocidade como vetor e "
            "inverter o componente correspondente ao bater na borda da tela. Vou implementar a "
            "lógica de física (sem depender de janela gráfica, que não roda no ambiente de "
            "validação) usando pygame.Rect e pygame.math.Vector2, que não exigem inicializar "
            "vídeo."
        ),
        file_path="ball_bounce.py",
        file_content=(
            "import pygame\n\n\n"
            "def bounce_update(rect, velocity, screen_width, screen_height):\n"
            "    new_x = rect.x + velocity.x\n"
            "    new_y = rect.y + velocity.y\n"
            "    if new_x <= 0 or new_x + rect.width >= screen_width:\n"
            "        velocity.x *= -1\n"
            "    if new_y <= 0 or new_y + rect.height >= screen_height:\n"
            "        velocity.y *= -1\n"
            "    rect.x = max(0, min(new_x, screen_width - rect.width))\n"
            "    rect.y = max(0, min(new_y, screen_height - rect.height))\n"
            "    return rect, velocity\n"
        ),
        test_path="test_ball_bounce.py",
        test_content=(
            "import pygame\n\n"
            "from ball_bounce import bounce_update\n\n\n"
            "def test_bounce_update_reverses_at_edge():\n"
            "    rect = pygame.Rect(0, 50, 20, 20)\n"
            "    velocity = pygame.math.Vector2(-5, 0)\n"
            "    rect, velocity = bounce_update(rect, velocity, screen_width=200, screen_height=200)\n"
            "    assert velocity.x == 5\n\n\n"
            "def test_bounce_update_keeps_direction_away_from_edge():\n"
            "    rect = pygame.Rect(100, 100, 20, 20)\n"
            "    velocity = pygame.math.Vector2(3, 2)\n"
            "    rect, velocity = bounce_update(rect, velocity, screen_width=200, screen_height=200)\n"
            "    assert (velocity.x, velocity.y) == (3, 2)\n"
        ),
        think_before_final="A lógica de quique está implementada e testável sem depender de janela gráfica. Vou validar.",
        final_text=(
            "Pesquisei a abordagem clássica de bolinha quicando com pygame (a documentação está "
            "em inglês, mas respondo em português) e implementei bounce_update(rect, velocity, "
            "screen_width, screen_height), que inverte a velocidade ao bater nas bordas, usando "
            "pygame.Rect e Vector2 sem precisar abrir uma janela. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-quicksort",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise o algoritmo quicksort e implemente ele em Python.",
        search_query="quicksort algorithm implementation python",
        think_before_search="Vou confirmar o esquema de particionamento mais comum antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma o esquema de particionamento de Lomuto: "
            "escolher o último elemento como pivô, reorganizar os menores à esquerda, os "
            "maiores à direita, e aplicar recursivamente aos dois lados."
        ),
        file_path="quicksort.py",
        file_content=(
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[-1]\n"
            "    less = [x for x in arr[:-1] if x <= pivot]\n"
            "    greater = [x for x in arr[:-1] if x > pivot]\n"
            "    return quicksort(less) + [pivot] + quicksort(greater)\n"
        ),
        test_path="test_quicksort.py",
        test_content=(
            "from quicksort import quicksort\n\n\n"
            "def test_quicksort():\n"
            "    assert quicksort([5, 3, 8, 1, 9, 2]) == [1, 2, 3, 5, 8, 9]\n"
            "    assert quicksort([]) == []\n"
            "    assert quicksort([1]) == [1]\n"
        ),
        think_before_final="O particionamento estilo Lomuto está implementado corretamente. Vou validar.",
        final_text=(
            "Pesquisei o algoritmo quicksort (conteúdo em inglês, resposta em português) e "
            "implementei quicksort(arr) usando particionamento estilo Lomuto com o último "
            "elemento como pivô. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-dijkstra",
        domain="algorithms",
        difficulty="hard",
        user_request="Pesquise o algoritmo de Dijkstra e implemente o cálculo de caminho mais curto em um grafo.",
        search_query="dijkstra shortest path algorithm python implementation",
        think_before_search="Vou confirmar a abordagem com fila de prioridade antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma o uso de heapq como fila de prioridade: "
            "distâncias inicializadas em infinito exceto a origem (0), e a cada passo remove-se "
            "o nó de menor distância, atualizando os vizinhos."
        ),
        file_path="dijkstra.py",
        file_content=(
            "import heapq\n\n\n"
            "def dijkstra(graph, start):\n"
            "    distances = {node: float('inf') for node in graph}\n"
            "    distances[start] = 0\n"
            "    queue = [(0, start)]\n"
            "    while queue:\n"
            "        current_distance, current_node = heapq.heappop(queue)\n"
            "        if current_distance > distances[current_node]:\n"
            "            continue\n"
            "        for neighbor, weight in graph[current_node].items():\n"
            "            distance = current_distance + weight\n"
            "            if distance < distances[neighbor]:\n"
            "                distances[neighbor] = distance\n"
            "                heapq.heappush(queue, (distance, neighbor))\n"
            "    return distances\n"
        ),
        test_path="test_dijkstra.py",
        test_content=(
            "from dijkstra import dijkstra\n\n\n"
            "def test_dijkstra():\n"
            "    graph = {\n"
            "        'A': {'B': 1, 'C': 4},\n"
            "        'B': {'A': 1, 'C': 2, 'D': 5},\n"
            "        'C': {'A': 4, 'B': 2, 'D': 1},\n"
            "        'D': {'B': 5, 'C': 1},\n"
            "    }\n"
            "    result = dijkstra(graph, 'A')\n"
            "    assert result == {'A': 0, 'B': 1, 'C': 3, 'D': 4}\n"
        ),
        think_before_final="A fila de prioridade e a atualização de distâncias seguem o algoritmo padrão. Vou validar.",
        final_text=(
            "Pesquisei o algoritmo de Dijkstra (conteúdo em inglês, resposta em português) e "
            "implementei dijkstra(graph, start) usando heapq como fila de prioridade. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-hash-table",
        domain="algorithms",
        difficulty="hard",
        user_request="Pesquise como implementar uma tabela hash com encadeamento separado e implemente em Python.",
        search_query="hash table separate chaining implementation python",
        think_before_search="Vou confirmar a estrutura de encadeamento separado antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: cada posição do array guarda uma lista "
            "(corrente) de pares chave-valor que colidiram no mesmo índice, calculado com "
            "hash() e módulo da capacidade."
        ),
        file_path="hash_table.py",
        file_content=(
            "class HashTable:\n"
            "    def __init__(self, capacity=16):\n"
            "        self.capacity = capacity\n"
            "        self.buckets = [[] for _ in range(capacity)]\n\n"
            "    def _index(self, key):\n"
            "        return hash(key) % self.capacity\n\n"
            "    def insert(self, key, value):\n"
            "        bucket = self.buckets[self._index(key)]\n"
            "        for i, (existing_key, _) in enumerate(bucket):\n"
            "            if existing_key == key:\n"
            "                bucket[i] = (key, value)\n"
            "                return\n"
            "        bucket.append((key, value))\n\n"
            "    def get(self, key):\n"
            "        bucket = self.buckets[self._index(key)]\n"
            "        for existing_key, value in bucket:\n"
            "            if existing_key == key:\n"
            "                return value\n"
            "        raise KeyError(key)\n"
        ),
        test_path="test_hash_table.py",
        test_content=(
            "import pytest\n\n"
            "from hash_table import HashTable\n\n\n"
            "def test_hash_table():\n"
            "    table = HashTable(capacity=4)\n"
            "    table.insert('a', 1)\n"
            "    table.insert('b', 2)\n"
            "    table.insert('a', 3)\n"
            "    assert table.get('a') == 3\n"
            "    assert table.get('b') == 2\n"
            "    with pytest.raises(KeyError):\n"
            "        table.get('c')\n"
        ),
        think_before_final="O encadeamento separado trata colisões e atualização de chave existente. Vou validar.",
        final_text=(
            "Pesquisei tabela hash com encadeamento separado (conteúdo em inglês, resposta em "
            "português) e implementei HashTable com insert e get, tratando colisões com listas "
            "por posição. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-binary-tree-inorder",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise travessia in-order de árvore binária e implemente de forma iterativa.",
        search_query="binary tree inorder traversal python implementation",
        think_before_search="Vou confirmar a versão iterativa com pilha antes de implementar, para evitar limite de recursão em árvores grandes.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma a abordagem iterativa: empilhar os filhos "
            "à esquerda até None, desempilhar, registrar o valor, e mover para o filho à direita."
        ),
        file_path="binary_tree_inorder.py",
        file_content=(
            "class TreeNode:\n"
            "    def __init__(self, val, left=None, right=None):\n"
            "        self.val = val\n"
            "        self.left = left\n"
            "        self.right = right\n\n\n"
            "def inorder_traversal(root):\n"
            "    result, stack = [], []\n"
            "    current = root\n"
            "    while current or stack:\n"
            "        while current:\n"
            "            stack.append(current)\n"
            "            current = current.left\n"
            "        current = stack.pop()\n"
            "        result.append(current.val)\n"
            "        current = current.right\n"
            "    return result\n"
        ),
        test_path="test_binary_tree_inorder.py",
        test_content=(
            "from binary_tree_inorder import TreeNode, inorder_traversal\n\n\n"
            "def test_inorder_traversal():\n"
            "    root = TreeNode(2, TreeNode(1), TreeNode(3))\n"
            "    assert inorder_traversal(root) == [1, 2, 3]\n"
            "    assert inorder_traversal(None) == []\n"
        ),
        think_before_final="A travessia iterativa com pilha produz a ordem esquerda-raiz-direita esperada. Vou validar.",
        final_text=(
            "Pesquisei travessia in-order de árvore binária (conteúdo em inglês, resposta em "
            "português) e implementei a versão iterativa com pilha, evitando limite de "
            "recursão. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-asyncio-basics",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise o básico de asyncio em Python e implemente uma função assíncrona simples.",
        search_query="python asyncio event loop basics tutorial",
        think_before_search="Vou confirmar a forma recomendada de rodar uma coroutine antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: usar 'async def' para criar uma coroutine "
            "e asyncio.run() como forma recomendada de executá-la, sem manipular o event loop "
            "diretamente."
        ),
        file_path="async_greeting.py",
        file_content=(
            "import asyncio\n\n\n"
            "async def greet_all(names):\n"
            "    greetings = []\n"
            "    for name in names:\n"
            "        await asyncio.sleep(0)\n"
            "        greetings.append(f'ola, {name}')\n"
            "    return greetings\n"
        ),
        test_path="test_async_greeting.py",
        test_content=(
            "import asyncio\n\n"
            "from async_greeting import greet_all\n\n\n"
            "def test_greet_all():\n"
            "    result = asyncio.run(greet_all(['ana', 'bruno']))\n"
            "    assert result == ['ola, ana', 'ola, bruno']\n"
        ),
        think_before_final="A coroutine usa asyncio.run() como recomendado pela documentação, sem tocar no event loop diretamente. Vou validar.",
        final_text=(
            "Pesquisei o básico de asyncio (conteúdo em inglês, resposta em português) e "
            "implementei greet_all(names) como coroutine, executada com asyncio.run(). A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-regex-lookahead",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise sobre lookahead/lookbehind em regex do Python e implemente uma função que separe palavras em camelCase.",
        search_query="regular expression lookahead python re module",
        think_before_search="Vou confirmar a sintaxe de lookahead/lookbehind antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma que são asserções de largura zero: "
            "(?=padrao) para lookahead positivo e uma posição usando (?<=padrao) para "
            "lookbehind, sem consumir os caracteres da correspondência."
        ),
        file_path="camel_case_splitter.py",
        file_content=(
            "import re\n\n\n"
            "def split_camel_case(text):\n"
            "    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)\n"
            "    return spaced\n"
        ),
        test_path="test_camel_case_splitter.py",
        test_content=(
            "from camel_case_splitter import split_camel_case\n\n\n"
            "def test_split_camel_case():\n"
            "    assert split_camel_case('helloWorldExample') == 'hello World Example'\n"
            "    assert split_camel_case('single') == 'single'\n"
        ),
        think_before_final="O lookbehind/lookahead combinados inserem espaço só na transição minúscula->maiúscula. Vou validar.",
        final_text=(
            "Pesquisei lookahead/lookbehind em regex (conteúdo em inglês, resposta em "
            "português) e implementei split_camel_case(text), usando as duas asserções de "
            "largura zero combinadas. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-dataclasses",
        domain="algorithms",
        difficulty="easy",
        user_request="Pesquise sobre dataclasses em Python e implemente uma classe Point usando esse recurso.",
        search_query="python dataclasses basics tutorial",
        think_before_search="Vou confirmar a sintaxe do decorator @dataclass antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: @dataclass gera __init__, __repr__ e "
            "__eq__ automaticamente a partir dos atributos anotados com tipo, evitando "
            "boilerplate."
        ),
        file_path="point_dataclass.py",
        file_content=(
            "import math\n"
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: float\n"
            "    y: float\n\n"
            "    def distance_to(self, other):\n"
            "        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)\n"
        ),
        test_path="test_point_dataclass.py",
        test_content=(
            "from point_dataclass import Point\n\n\n"
            "def test_point_dataclass():\n"
            "    p1 = Point(0, 0)\n"
            "    p2 = Point(3, 4)\n"
            "    assert abs(p1.distance_to(p2) - 5.0) < 1e-9\n"
            "    assert Point(1, 2) == Point(1, 2)\n"
        ),
        think_before_final="@dataclass já fornece __eq__ automaticamente, e distance_to usa os atributos gerados. Vou validar.",
        final_text=(
            "Pesquisei dataclasses em Python (conteúdo em inglês, resposta em português) e "
            "implementei Point como @dataclass, com distance_to aproveitando o __init__/__eq__ "
            "gerados automaticamente. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-base64",
        domain="algorithms",
        difficulty="easy",
        user_request="Pesquise como codificar e decodificar texto em base64 em Python e implemente as duas funções.",
        search_query="base64 encoding python implementation",
        think_before_search="Vou confirmar o módulo padrão certo antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: converter a string para bytes com "
            ".encode(), aplicar base64.b64encode(), e decodificar de volta com .decode() "
            "quando necessário."
        ),
        file_path="base64_util.py",
        file_content=(
            "import base64\n\n\n"
            "def encode_text(text):\n"
            "    return base64.b64encode(text.encode('utf-8')).decode('ascii')\n\n\n"
            "def decode_text(encoded):\n"
            "    return base64.b64decode(encoded.encode('ascii')).decode('utf-8')\n"
        ),
        test_path="test_base64_util.py",
        test_content=(
            "from base64_util import encode_text, decode_text\n\n\n"
            "def test_base64_util():\n"
            "    encoded = encode_text('ola mundo')\n"
            "    assert decode_text(encoded) == 'ola mundo'\n"
            "    assert encoded != 'ola mundo'\n"
        ),
        think_before_final="Codificação e decodificação usam o módulo base64 padrão da biblioteca. Vou validar.",
        final_text=(
            "Pesquisei codificação base64 (conteúdo em inglês, resposta em português) e "
            "implementei encode_text e decode_text usando o módulo base64 padrão. A validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-linked-list-cycle",
        domain="algorithms",
        difficulty="hard",
        user_request="Pesquise o algoritmo de Floyd para detectar ciclo em lista encadeada e implemente.",
        search_query="linked list cycle detection floyd algorithm python",
        think_before_search="Vou confirmar a técnica dos dois ponteiros antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: dois ponteiros (lento e rápido) partem da "
            "cabeça, o lento anda um passo e o rápido dois; se houver ciclo eles se encontram, "
            "senão o rápido chega a None primeiro."
        ),
        file_path="linked_list_cycle.py",
        file_content=(
            "class Node:\n"
            "    def __init__(self, value, next=None):\n"
            "        self.value = value\n"
            "        self.next = next\n\n\n"
            "def has_cycle(head):\n"
            "    slow = head\n"
            "    fast = head\n"
            "    while fast is not None and fast.next is not None:\n"
            "        slow = slow.next\n"
            "        fast = fast.next.next\n"
            "        if slow is fast:\n"
            "            return True\n"
            "    return False\n"
        ),
        test_path="test_linked_list_cycle.py",
        test_content=(
            "from linked_list_cycle import Node, has_cycle\n\n\n"
            "def test_has_cycle_true():\n"
            "    a = Node(1)\n"
            "    b = Node(2)\n"
            "    c = Node(3)\n"
            "    a.next = b\n"
            "    b.next = c\n"
            "    c.next = a\n"
            "    assert has_cycle(a) is True\n\n\n"
            "def test_has_cycle_false():\n"
            "    a = Node(1, Node(2, Node(3)))\n"
            "    assert has_cycle(a) is False\n"
        ),
        think_before_final="O algoritmo de dois ponteiros detecta o ciclo em O(n) sem espaço extra. Vou validar.",
        final_text=(
            "Pesquisei o algoritmo de Floyd para detecção de ciclo (conteúdo em inglês, "
            "resposta em português) e implementei has_cycle(head) com os dois ponteiros lento "
            "e rápido. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-rsa-basic",
        domain="algorithms",
        difficulty="hard",
        user_request="Pesquise como funciona o algoritmo RSA e implemente uma versão simplificada de geração de chaves, criptografia e decriptografia (só para fins didáticos).",
        search_query="RSA encryption algorithm basics python",
        think_before_search="Vou confirmar os passos de geração de chave (p, q, n, totiente, e, d) antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma os passos: escolher primos p e q, calcular "
            "n = p*q e o totiente de Euler, escolher e coprimo com o totiente, e derivar d "
            "como o inverso modular de e. Encriptar é c = m^e mod n; decriptar é m = c^d mod n."
        ),
        file_path="rsa_basic.py",
        file_content=(
            "def _mod_inverse(e, phi):\n"
            "    for d in range(2, phi):\n"
            "        if (d * e) % phi == 1:\n"
            "            return d\n"
            "    raise ValueError('inverso modular nao encontrado')\n\n\n"
            "def generate_keypair(p, q, e=7):\n"
            "    n = p * q\n"
            "    phi = (p - 1) * (q - 1)\n"
            "    d = _mod_inverse(e, phi)\n"
            "    return (e, n), (d, n)\n\n\n"
            "def encrypt(message, public_key):\n"
            "    e, n = public_key\n"
            "    return pow(message, e, n)\n\n\n"
            "def decrypt(ciphertext, private_key):\n"
            "    d, n = private_key\n"
            "    return pow(ciphertext, d, n)\n"
        ),
        test_path="test_rsa_basic.py",
        test_content=(
            "from rsa_basic import generate_keypair, encrypt, decrypt\n\n\n"
            "def test_rsa_basic_roundtrip():\n"
            "    public_key, private_key = generate_keypair(p=3, q=11, e=7)\n"
            "    message = 4\n"
            "    ciphertext = encrypt(message, public_key)\n"
            "    assert ciphertext != message\n"
            "    assert decrypt(ciphertext, private_key) == message\n"
        ),
        think_before_final="O ciclo completo de geração de chave, encriptação e decriptação bate com o exemplo didático. Vou validar.",
        final_text=(
            "Pesquisei o funcionamento do RSA (conteúdo em inglês, resposta em português) e "
            "implementei uma versão simplificada e didática (não segura para uso real) de "
            "generate_keypair, encrypt e decrypt. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-ensearch-pandas-groupby",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise groupby do pandas e implemente uma função que some vendas por categoria.",
        search_query="pandas groupby aggregation tutorial",
        think_before_search="Vou confirmar o padrão split-apply-combine antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma o padrão split-apply-combine: agrupar por "
            "uma coluna, aplicar uma agregação (soma, média) e combinar os resultados num "
            "DataFrame resumido."
        ),
        file_path="sales_summary.py",
        file_content=(
            "import pandas as pd\n\n\n"
            "def total_sales_by_category(records):\n"
            "    df = pd.DataFrame(records)\n"
            "    return df.groupby('category')['amount'].sum().to_dict()\n"
        ),
        test_path="test_sales_summary.py",
        test_content=(
            "from sales_summary import total_sales_by_category\n\n\n"
            "def test_total_sales_by_category():\n"
            "    records = [\n"
            "        {'category': 'livros', 'amount': 50},\n"
            "        {'category': 'livros', 'amount': 30},\n"
            "        {'category': 'eletronicos', 'amount': 200},\n"
            "    ]\n"
            "    result = total_sales_by_category(records)\n"
            "    assert result == {'livros': 80, 'eletronicos': 200}\n"
        ),
        think_before_final="O agrupamento por categoria e a soma dos valores estão corretos. Vou validar.",
        final_text=(
            "Pesquisei groupby do pandas (conteúdo em inglês, resposta em português) e "
            "implementei total_sales_by_category(records), agrupando e somando por categoria. "
            "A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-bubble-sort",
        domain="algorithms",
        difficulty="easy",
        user_request="Pesquise o algoritmo bubble sort e implemente com a otimização de parar cedo se a lista já estiver ordenada.",
        search_query="bubble sort algorithm python implementation",
        think_before_search="Vou confirmar a otimização com flag de troca antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma a otimização: usar uma flag 'swapped' para "
            "encerrar mais cedo se nenhuma troca ocorreu numa passada, dando O(n) no melhor "
            "caso."
        ),
        file_path="bubble_sort.py",
        file_content=(
            "def bubble_sort(arr):\n"
            "    items = list(arr)\n"
            "    n = len(items)\n"
            "    for i in range(n):\n"
            "        swapped = False\n"
            "        for j in range(n - i - 1):\n"
            "            if items[j] > items[j + 1]:\n"
            "                items[j], items[j + 1] = items[j + 1], items[j]\n"
            "                swapped = True\n"
            "        if not swapped:\n"
            "            break\n"
            "    return items\n"
        ),
        test_path="test_bubble_sort.py",
        test_content=(
            "from bubble_sort import bubble_sort\n\n\n"
            "def test_bubble_sort():\n"
            "    assert bubble_sort([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]\n"
            "    assert bubble_sort([1, 2, 3]) == [1, 2, 3]\n"
            "    assert bubble_sort([]) == []\n"
        ),
        think_before_final="A flag de troca antecipa o término quando a lista já está ordenada. Vou validar.",
        final_text=(
            "Pesquisei bubble sort (conteúdo em inglês, resposta em português) e implementei "
            "bubble_sort(arr) com a otimização da flag de troca para encerrar mais cedo. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-multiprocessing-vs-threading",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise a diferença entre multiprocessing e threading em Python e implemente uma função que recomende qual usar conforme o tipo de tarefa.",
        search_query="python multiprocessing vs threading difference",
        think_before_search="Vou confirmar a diferença prática (GIL, I/O-bound vs CPU-bound) antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: por causa do GIL, threading serve bem "
            "para tarefas de I/O (rede, disco), enquanto multiprocessing é melhor para tarefas "
            "de CPU, já que cada processo tem seu próprio interpretador."
        ),
        file_path="concurrency_advisor.py",
        file_content=(
            "def choose_concurrency_strategy(is_cpu_bound):\n"
            "    return 'multiprocessing' if is_cpu_bound else 'threading'\n"
        ),
        test_path="test_concurrency_advisor.py",
        test_content=(
            "from concurrency_advisor import choose_concurrency_strategy\n\n\n"
            "def test_choose_concurrency_strategy():\n"
            "    assert choose_concurrency_strategy(is_cpu_bound=True) == 'multiprocessing'\n"
            "    assert choose_concurrency_strategy(is_cpu_bound=False) == 'threading'\n"
        ),
        think_before_final="A recomendação segue a regra prática confirmada pela documentação (GIL, I/O vs CPU). Vou validar.",
        final_text=(
            "Pesquisei a diferença entre multiprocessing e threading (conteúdo em inglês, "
            "resposta em português) e implementei choose_concurrency_strategy(is_cpu_bound), "
            "recomendando multiprocessing para CPU-bound e threading para I/O-bound. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ensearch-balanced-parentheses",
        domain="algorithms",
        difficulty="medium",
        user_request="Pesquise como verificar parênteses balanceados usando pilha e implemente cobrindo múltiplos tipos de colchetes.",
        search_query="balanced parentheses check using stack python",
        think_before_search="Vou confirmar a abordagem com pilha e mapeamento de pares antes de implementar.",
        think_after_search=(
            f"{_LANG_ACK} A documentação confirma: empilhar aberturas, e ao encontrar um "
            "fechamento, verificar se o topo da pilha corresponde ao par certo — só contar "
            "quantidades não detecta aninhamento errado como '([)]'."
        ),
        file_path="balanced_parentheses.py",
        file_content=(
            "def is_balanced(text):\n"
            "    stack = []\n"
            "    pairs = {')': '(', '}': '{', ']': '['}\n"
            "    openers = set(pairs.values())\n"
            "    for char in text:\n"
            "        if char in openers:\n"
            "            stack.append(char)\n"
            "        elif char in pairs:\n"
            "            if not stack or stack.pop() != pairs[char]:\n"
            "                return False\n"
            "    return len(stack) == 0\n"
        ),
        test_path="test_balanced_parentheses.py",
        test_content=(
            "from balanced_parentheses import is_balanced\n\n\n"
            "def test_is_balanced():\n"
            "    assert is_balanced('([]{})') is True\n"
            "    assert is_balanced('([)]') is False\n"
            "    assert is_balanced('(((') is False\n"
        ),
        think_before_final="O mapeamento de pares detecta corretamente o aninhamento errado, não só a contagem. Vou validar.",
        final_text=(
            "Pesquisei verificação de parênteses balanceados (conteúdo em inglês, resposta em "
            "português) e implementei is_balanced(text) com pilha e mapeamento de pares, "
            "cobrindo múltiplos tipos de colchetes. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for spec in EXAMPLES:
        example = build_example(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name} (busca simulada com fixture real + checker passou de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
