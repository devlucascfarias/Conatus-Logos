#!/usr/bin/env python
"""Lote multi-turno da wave 2 (docs/plan_dataset_expansion_wave2_identity_multiturn.md, seção
5) — as duas categorias que ficaram bloqueadas até a extensão de schema (D-dataset-history-schema)
e a máscara de loss (D-dataset-history-loss-mask) existirem:

1. Mudança abrupta de direção: um turno TREINADO (o "âncora") que chega depois de 1-2 turnos
   de contexto sobre um assunto completamente diferente, na mesma sessão. O `<think>` do
   âncora precisa tratar o pedido pelo próprio mérito, não herdar nada do turno anterior.
2. Pedido de ferramenta avulso/tardio: o âncora é um pedido de ferramenta (list_files/
   read_file/shell) que chega SOZINHO, sem estar embutido num pedido composto, depois de
   turnos de contexto irrelevantes — é o padrão exato que falhou ao vivo na CLI (ver
   D-cli-session-history, achado "Liste os arquivos nesse diretório").

Mecanismo: um banco pequeno de turnos de CONTEXTO reutilizáveis (`FILLERS`) — sem ferramenta,
diretos, cada um genuinamente distinto — vira `history` em várias combinações. O que treina de
verdade em cada exemplo é só o turno ÂNCORA (`user_request`/`raw_text` de topo), e cada âncora
é escrito com profundidade técnica de verdade (não é o turno de contexto que carrega peso
pedagógico, é o âncora). Reutilizar os mesmos FILLERS como contexto em sessões diferentes não é
o tipo de duplicação que compromete o dataset (a seção "AVISO CRÍTICO" dos gaps anteriores era
sobre duplicar o texto que TREINA, não o contexto que só aparece mascarado no prefixo).

Uso:
    python scripts/gen_multiturn_wave2_pilot.py
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import HistoryTurn, Trajectory  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()


# --- banco de turnos de contexto (FILLERS) — sem ferramenta, curtos, cada um distinto -------

FILLERS = {
    "identity": ("Quem é você?", "Pergunta direta de identidade.", "Sou o Logos-3, da família Conatus."),
    "who_trained": (
        "Quem te treinou?",
        "Pergunta sobre origem do projeto.",
        "Fui treinado como parte do projeto Conatus.",
    ),
    "mul": (
        "Quanto é 7 vezes 8?",
        "Multiplicação simples, respondo direto.",
        "7 vezes 8 é 56.",
    ),
    "sqrt": (
        "Qual a raiz quadrada de 144?",
        "Cálculo direto, sem necessidade de ferramenta.",
        "A raiz quadrada de 144 é 12.",
    ),
    "partial_derivative": (
        "O que é uma derivada parcial?",
        "Pergunta conceitual de cálculo.",
        "Uma derivada parcial mede a taxa de variação de uma função em relação a uma variável, mantendo as outras constantes.",
    ),
    "power": (
        "Quanto é 2 elevado a 10?",
        "Potenciação simples.",
        "2 elevado a 10 é 1024.",
    ),
    "decorator": (
        "O que é um decorator em Python, resumidamente?",
        "Pergunta conceitual sobre a linguagem.",
        "Um decorator envolve uma função com outra, adicionando comportamento extra sem alterar o código original.",
    ),
    "generator": (
        "O que é um generator em Python?",
        "Pergunta conceitual sobre a linguagem.",
        "Um generator produz valores um de cada vez sob demanda, usando yield, em vez de montar uma lista inteira na memória.",
    ),
    "big_o": (
        "O que significa complexidade O(n log n)?",
        "Pergunta conceitual de complexidade.",
        "Significa que o tempo de execução cresce proporcionalmente a n multiplicado pelo logaritmo de n, típico de algoritmos de ordenação eficientes como merge sort.",
    ),
    "dependency_injection": (
        "O que é injeção de dependência?",
        "Pergunta conceitual de design de software.",
        "É passar as dependências de um componente de fora, em vez de criá-las internamente, facilitando testes e substituição.",
    ),
    "race_condition": (
        "O que é uma race condition?",
        "Pergunta conceitual de concorrência.",
        "É quando o resultado de um programa depende da ordem imprevisível em que threads ou processos acessam um recurso compartilhado.",
    ),
    "tail_recursion": (
        "O que é recursão de cauda?",
        "Pergunta conceitual sobre recursão.",
        "É quando a chamada recursiva é a última operação da função, permitindo que alguns compiladores otimizem isso sem crescer a pilha.",
    ),
}


def _filler_trajectory(key: str) -> HistoryTurn:
    user_request, think, final = FILLERS[key]
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think}</think><final>{final}</final>")
    return HistoryTurn(user_request=traj.user_request, raw_text=traj.raw_text)


def _history(*filler_keys: str) -> list[HistoryTurn]:
    return [_filler_trajectory(k) for k in filler_keys]


def _base_metadata(id_, domain, language, difficulty, tools_used, task_type, num_steps, execution_performed, execution_classification):
    return {
        "id": id_,
        "domain": domain,
        "language": language,
        "difficulty": difficulty,
        "tools_used": tools_used,
        "num_steps": num_steps,
        "task_type": task_type,
        "source": "curated_manual_real_execution",
        "license": "synthetic-no-license-needed",
        "validation_status": "validated",
        "execution_performed": execution_performed,
        "split": "train",
        "execution_classification": execution_classification,
    }


# Mesmo achado/correção de D-style-upgrade-wave2-pilot: pytest_asyncio emite um warning de
# depreciação que inclui o caminho absoluto da instalação local do Python, tanto em stdout
# quanto (mais sutilmente) dentro de error_message — um campo separado de `data` que
# to_json() copia pro campo "message" do corpo de erro. Sanitiza os dois.
_LOCAL_PATH_WARNING = re.compile(
    r"[A-Za-z]:\\{1,2}Users\\{1,2}[^\r\n]*?PytestDeprecationWarning.*?warnings\.warn\([^)]*\)\)?",
    re.DOTALL,
)


def _sanitize_value(value):
    if isinstance(value, str):
        return _LOCAL_PATH_WARNING.sub("", value)
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return value


def _sanitize_checker_result(result):
    sanitized_data = _sanitize_value(result.data)
    sanitized_data = sanitized_data if isinstance(sanitized_data, dict) else result.data
    sanitized_message = (
        _LOCAL_PATH_WARNING.sub("", result.error_message) if result.error_message else result.error_message
    )
    return dataclasses.replace(result, data=sanitized_data, error_message=sanitized_message)


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    if name == "checker":
        result = _sanitize_checker_result(result)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


# --- construtor: âncora sem ferramenta, com histórico ---------------------------------------


def build_no_tool_anchor(id_, domain, difficulty, task_type, user_request, think_text, final_text, history_keys):
    traj = Trajectory(
        system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request, history=_history(*history_keys)
    )
    traj.append_raw(f"<think>{think_text}</think><final>{final_text}</final>")
    return {
        "metadata": _base_metadata(id_, domain, "python", difficulty, [], task_type, 0, False, "static_only"),
        "trajectory": traj.to_example_dict(),
    }


# --- construtor: âncora write_file + checker, com histórico ---------------------------------


def build_write_checker_anchor(
    id_, domain, difficulty, task_type, user_request, think1, file_path, content, think2, checker_args, final_text,
    history_keys, language="python",
):
    sandbox = SandboxContext(policy=_POLICY)
    traj = Trajectory(
        system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request, history=_history(*history_keys)
    )
    traj.append_raw(f"<think>{think1}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": content})
    assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    checker_result = _tool_call(traj, sandbox, "checker", checker_args)
    assert checker_result.passed, f"esperava checker passando em {id_}: {checker_result.to_json()}"
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, language, difficulty, ["write_file", "checker"], task_type, 2, True, "tested"
        ),
        "trajectory": traj.to_example_dict(),
    }


# --- construtor: âncora de pedido de ferramenta avulso/tardio -------------------------------


def build_late_tool_anchor(
    id_, domain, difficulty, user_request, think1, tool_name, tool_args, think2, final_text,
    history_keys, preexisting_files=None, expect_pass=True,
):
    # `expect_pass=False` cobre o caso real de `grep` sem nenhuma ocorrência: retorna código 1
    # (convenção do próprio grep pra "nenhum match", não um erro de execução de verdade), o
    # `shell_tool` trata isso como `status="error"`/RUNTIME_ERROR — o `<think>` do âncora
    # precisa interpretar isso corretamente como "não achou nada", não como falha real.
    sandbox = SandboxContext(policy=_POLICY)
    for rel_path, content in (preexisting_files or {}).items():
        target = sandbox.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    traj = Trajectory(
        system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request, history=_history(*history_keys)
    )
    traj.append_raw(f"<think>{think1}</think>")
    result = _tool_call(traj, sandbox, tool_name, tool_args)
    if expect_pass:
        assert result.passed, f"esperava sucesso real em {id_}: {result.to_json()}"
    else:
        assert not result.passed, f"esperava retorno sem match real em {id_}: {result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, "python", difficulty, [tool_name], "single_tool_call", 1, True, "tested"
        ),
        "trajectory": traj.to_example_dict(),
    }


# =============================================================================================
# Categoria 1: mudança abrupta de direção (~45 âncoras)
# =============================================================================================

ABRUPT_ANCHORS = []

# --- algoritmos / complexidade ---------------------------------------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-quicksort-worst-case",
    domain="complexidade", difficulty="medium", task_type="complexity_analysis",
    user_request="Por que um quicksort pode degradar pra O(n²)?",
    think_text=(
        "Pergunta nova sobre algoritmos, sem relação com o assunto anterior, respondo direto "
        "com o que sei sobre o pior caso do quicksort. O pior caso acontece quando o pivô "
        "escolhido é sempre o menor ou o maior elemento da partição, o que faz cada chamada "
        "recursiva reduzir o tamanho do problema em só um elemento, gerando n níveis de "
        "recursão em vez de log n."
    ),
    final_text="O quicksort degrada pra O(n²) quando o pivô escolhido é sempre o extremo (menor ou maior valor) da partição, tipicamente em entradas já ordenadas com pivô fixo no início ou fim. Isso faz o algoritmo virar n níveis de recursão em vez de log n, cada um processando quase todos os elementos restantes.",
    history_keys=["identity"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-stack-overflow",
    domain="debugging", difficulty="easy", task_type="direct_answer",
    user_request="O que causa stack overflow em recursão infinita?",
    think_text=(
        "Pergunta técnica nova, sem ligação com o assunto anterior. Cada chamada recursiva "
        "empilha um novo quadro de pilha com suas variáveis locais; sem um caso base que "
        "interrompa as chamadas, esses quadros se acumulam até esgotar o espaço de pilha "
        "reservado pelo processo."
    ),
    final_text="Stack overflow acontece porque cada chamada recursiva reserva espaço na pilha de execução para suas variáveis locais. Sem um caso base que pare as chamadas, esses espaços se acumulam infinitamente até estourar o limite de pilha do processo.",
    history_keys=["mul"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-float-money",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="Por que usar float pra representar dinheiro é perigoso?",
    think_text=(
        "Assunto completamente diferente do anterior, trato como pergunta nova. Float usa "
        "representação binária de ponto flutuante, que não consegue representar exatamente "
        "certas frações decimais, como 0.1, gerando pequenos erros de arredondamento que se "
        "acumulam em operações sucessivas."
    ),
    final_text="Float é perigoso pra dinheiro porque a representação binária de ponto flutuante não consegue guardar exatamente certos valores decimais, como 0.1. Esses pequenos erros de arredondamento se acumulam em somas e multiplicações sucessivas, causando divergências reais em cálculos financeiros. O ideal é usar tipos decimais exatos, como Decimal em Python, ou trabalhar em centavos como inteiros.",
    history_keys=["decorator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-mutable-aliasing",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="O que é aliasing em listas mutáveis do Python?",
    think_text=(
        "Pergunta técnica nova, não relacionada ao turno anterior. Aliasing acontece quando "
        "duas variáveis apontam pro mesmo objeto na memória, então mudar uma através de um "
        "método mutável, como append, afeta a outra também, porque não existe cópia "
        "nenhuma envolvida."
    ),
    final_text="Aliasing é quando duas variáveis referenciam o mesmo objeto de lista na memória, não cópias independentes. Se você faz b = a e depois b.append(1), a também muda, porque a e b apontam pro mesmo objeto. Pra evitar isso, use a.copy() ou list(a) quando precisar de uma cópia independente.",
    history_keys=["sqrt"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-float-precision",
    domain="conceitos", difficulty="easy", task_type="direct_answer",
    user_request="Por que 0.1 mais 0.2 não dá exatamente 0.3 em código?",
    think_text=(
        "Pergunta nova, sem relação com o turno anterior. Números binários de ponto flutuante "
        "não representam exatamente frações decimais como 0.1 e 0.2, cada um já carrega um "
        "pequeno erro de arredondamento antes mesmo da soma acontecer."
    ),
    final_text="0.1 e 0.2 não têm representação binária exata em ponto flutuante, cada um já vem com um pequeno erro de arredondamento embutido. Ao somar os dois, esses erros se combinam e o resultado fica ligeiramente diferente de 0.3, mesmo que pareça óbvio que deveria bater exatamente.",
    history_keys=["identity", "mul"],
)))

# --- teoria/CS ---------------------------------------------------------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-np-complete",
    domain="teoria", difficulty="hard", task_type="direct_answer",
    user_request="O que é NP-completo, de forma simples?",
    think_text=(
        "Mudança completa de assunto, trato como pergunta independente. NP-completo é a "
        "classe de problemas pra que uma solução proposta pode ser VERIFICADA rapidamente, "
        "mas nenhum algoritmo conhecido consegue ENCONTRAR essa solução rapidamente no pior "
        "caso, e resolver qualquer um deles rápido resolveria todos os outros da classe."
    ),
    final_text="NP-completo é a classe de problemas em que verificar uma solução proposta é rápido, mas encontrar essa solução do zero não tem algoritmo eficiente conhecido no pior caso. Resolver qualquer problema NP-completo em tempo eficiente implicaria resolver todos os outros da mesma classe também, por isso são considerados igualmente difíceis entre si.",
    history_keys=["partial_derivative"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-bfs-vs-dfs",
    domain="algoritmos", difficulty="medium", task_type="direct_answer",
    user_request="Qual a diferença entre BFS e DFS?",
    think_text=(
        "Pergunta nova de algoritmos de grafo, sem relação com o assunto anterior. BFS "
        "explora nível por nível usando uma fila, garantindo o caminho mais curto em grafos "
        "não ponderados; DFS explora um ramo inteiro antes de voltar, usando uma pilha "
        "(ou recursão), o que pode achar QUALQUER caminho primeiro, não necessariamente o "
        "mais curto."
    ),
    final_text="BFS (busca em largura) explora o grafo nível por nível usando uma fila, e garante achar o caminho mais curto em grafos não ponderados. DFS (busca em profundidade) mergulha num ramo inteiro antes de voltar, usando uma pilha ou recursão, e pode achar um caminho bem mais longo antes do correto.",
    history_keys=["power"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-greedy",
    domain="algoritmos", difficulty="medium", task_type="direct_answer",
    user_request="O que é um algoritmo guloso (greedy)?",
    think_text=(
        "Assunto novo, sem conexão com o turno anterior. Um algoritmo guloso escolhe, a cada "
        "passo, a opção que parece melhor NAQUELE momento, sem reconsiderar escolhas "
        "anteriores — funciona bem quando o problema tem a propriedade de que ótimos locais "
        "levam ao ótimo global, mas nem sempre é o caso."
    ),
    final_text="Um algoritmo guloso toma, a cada passo, a decisão que parece melhor naquele momento, sem voltar atrás pra reconsiderar. Funciona perfeitamente quando o problema garante que escolhas ótimas locais somam um resultado ótimo global (como no troco de moedas em sistemas canônicos), mas pode dar resultado subótimo em problemas sem essa propriedade.",
    history_keys=["generator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-hashmap-worst-case",
    domain="algoritmos", difficulty="medium", task_type="complexity_analysis",
    user_request="Por que hash maps são O(1) em média mas O(n) no pior caso?",
    think_text=(
        "Pergunta técnica nova. Em média, o hash distribui as chaves de forma uniforme pelos "
        "buckets, então acessar uma chave é praticamente direto. No pior caso, se muitas "
        "chaves colidirem no mesmo bucket (por má distribuição do hash ou ataque "
        "deliberado), o acesso vira uma busca linear dentro daquele bucket."
    ),
    final_text="Em média, a função hash distribui as chaves uniformemente entre os buckets, então acessar qualquer chave é praticamente O(1). No pior caso, se muitas chaves colidem no mesmo bucket (má distribuição do hash, ou um ataque deliberado explorando isso), o acesso dentro daquele bucket vira uma busca linear entre as chaves colididas, O(n).",
    history_keys=["big_o"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-collatz",
    domain="matematica", difficulty="easy", task_type="direct_answer",
    user_request="O que é a conjectura de Collatz?",
    think_text=(
        "Pergunta nova, sem relação com o assunto anterior. A conjectura diz que, partindo de "
        "qualquer número positivo e aplicando repetidamente a regra (dividir por 2 se par, "
        "multiplicar por 3 e somar 1 se ímpar), sempre se chega em 1, embora ninguém tenha "
        "provado isso pra todos os números."
    ),
    final_text="A conjectura de Collatz afirma que, começando de qualquer número inteiro positivo e aplicando repetidamente a regra (se par, divide por 2; se ímpar, multiplica por 3 e soma 1), a sequência sempre termina em 1. É verificada computacionalmente pra números enormes, mas continua sem prova matemática geral.",
    history_keys=["tail_recursion"],
)))

# --- escrita de código, com execução real -----------------------------------------------------
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-is-prime",
    domain="algoritmos", difficulty="medium", task_type="multi_tool_call",
    user_request="Escreva uma função que verifica se um número é primo.",
    think1=(
        "Pedido novo, totalmente diferente do assunto anterior. Vou checar divisibilidade só "
        "até a raiz quadrada do número, já que qualquer fator maior que isso teria um par "
        "correspondente menor que já teria sido encontrado antes."
    ),
    file_path="primes.py",
    content=(
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    i = 2\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 1\n"
        "    return True\n"
    ),
    think2="Vou testar com um primo, um composto e um número menor que 2, pra cobrir os casos de borda.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "primes.py", "content": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n"},
            {"path": "test_primes.py", "content": "from primes import is_prime\n\ndef test_seven_is_prime():\n    assert is_prime(7) is True\n\ndef test_eight_is_not_prime():\n    assert is_prime(8) is False\n\ndef test_one_is_not_prime():\n    assert is_prime(1) is False\n"},
        ],
    },
    final_text="Implementei is_prime checando divisibilidade só até a raiz quadrada de n. Testei com um primo, um composto e um caso de borda menor que 2, todos passaram.",
    history_keys=["who_trained"],
)))
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-fib-memo",
    domain="algoritmos", difficulty="medium", task_type="multi_tool_call",
    user_request="Escreva uma função fibonacci com memoização.",
    think1=(
        "Assunto novo, sem relação com o anterior. Fibonacci sem memoização recalcula os "
        "mesmos subproblemas repetidamente, é O(2^n). Vou usar um dicionário como cache pra "
        "guardar resultados já calculados e reduzir isso pra O(n)."
    ),
    file_path="fib.py",
    content=(
        "def fib(n, cache=None):\n"
        "    if cache is None:\n"
        "        cache = {}\n"
        "    if n in cache:\n"
        "        return cache[n]\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    cache[n] = fib(n - 1, cache) + fib(n - 2, cache)\n"
        "    return cache[n]\n"
    ),
    think2="Vou testar com um valor pequeno conhecido e um valor maior, pra confirmar que a memoização não quebra o resultado.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "fib.py", "content": "def fib(n, cache=None):\n    if cache is None:\n        cache = {}\n    if n in cache:\n        return cache[n]\n    if n <= 1:\n        return n\n    cache[n] = fib(n - 1, cache) + fib(n - 2, cache)\n    return cache[n]\n"},
            {"path": "test_fib.py", "content": "from fib import fib\n\ndef test_fib_small():\n    assert fib(10) == 55\n\ndef test_fib_larger():\n    assert fib(20) == 6765\n"},
        ],
    },
    final_text="Implementei fib com memoização via dicionário cache, reduzindo de O(2^n) pra O(n). Testei com fib(10) e fib(20), os dois bateram com os valores esperados.",
    history_keys=["sqrt", "decorator"],
)))
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-balanced-parens",
    domain="algoritmos", difficulty="medium", task_type="multi_tool_call",
    user_request="Escreva um validador de parênteses balanceados.",
    think1=(
        "Pedido novo, sem ligação com o turno anterior. Vou usar uma pilha: empilho cada "
        "abertura, e a cada fechamento verifico se ela corresponde ao topo da pilha; no "
        "final, a pilha precisa estar vazia pra estar tudo balanceado."
    ),
    file_path="balanced.py",
    content=(
        "def is_balanced(s):\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for ch in s:\n"
        "        if ch in '([{':\n"
        "            stack.append(ch)\n"
        "        elif ch in pairs:\n"
        "            if not stack or stack.pop() != pairs[ch]:\n"
        "                return False\n"
        "    return not stack\n"
    ),
    think2="Vou testar com um caso balanceado, um desbalanceado por tipo errado e um com fechamento sobrando.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "balanced.py", "content": "def is_balanced(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n"},
            {"path": "test_balanced.py", "content": "from balanced import is_balanced\n\ndef test_balanced_mixed():\n    assert is_balanced('([{}])') is True\n\ndef test_wrong_type_order():\n    assert is_balanced('([)]') is False\n\ndef test_extra_closing():\n    assert is_balanced('())') is False\n"},
        ],
    },
    final_text="Implementei is_balanced usando uma pilha pra rastrear aberturas e conferir cada fechamento contra o topo. Testei balanceado, tipo errado e fechamento sobrando, todos corretos.",
    history_keys=["race_condition"],
)))

# --- conceitos de linguagem/design ------------------------------------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-immutability",
    domain="conceitos", difficulty="easy", task_type="direct_answer",
    user_request="O que é imutabilidade e por que importa?",
    think_text=(
        "Pergunta nova, sem relação com o assunto anterior. Um objeto imutável não pode ter "
        "seu estado interno alterado depois de criado — qualquer 'mudança' na verdade cria um "
        "objeto novo. Isso importa porque elimina uma classe inteira de bugs de mutação "
        "compartilhada inesperada."
    ),
    final_text="Imutabilidade significa que um objeto não pode ter seu estado alterado depois de criado, qualquer mudança gera um objeto novo. Importa porque elimina bugs onde uma parte do código muda um valor que outra parte não esperava que mudasse, especialmente em código concorrente.",
    history_keys=["mul", "sqrt"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-eq-vs-is",
    domain="conceitos", difficulty="easy", task_type="direct_answer",
    user_request="Qual a diferença entre == e is em Python?",
    think_text=(
        "Pergunta nova sobre a linguagem, sem relação com o turno anterior. == compara "
        "IGUALDADE de valor (chama __eq__); is compara IDENTIDADE, se são literalmente o "
        "mesmo objeto na memória."
    ),
    final_text="== compara se dois valores são iguais (chama o método __eq__). is compara identidade: se as duas variáveis apontam pro exato mesmo objeto na memória. Dois objetos podem ter valores iguais (== True) sem serem o mesmo objeto (is False).",
    history_keys=["identity"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-context-manager",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="O que é um context manager (o with) em Python?",
    think_text=(
        "Assunto novo, sem conexão com o anterior. Um context manager garante que um recurso "
        "(arquivo, conexão, lock) seja liberado corretamente mesmo se uma exceção acontecer "
        "no meio do bloco, via os métodos __enter__/__exit__."
    ),
    final_text="Um context manager, usado com with, garante que um recurso (como um arquivo aberto) seja liberado corretamente ao final do bloco, mesmo que uma exceção aconteça no meio. Isso é feito pelos métodos __enter__ (setup) e __exit__ (cleanup garantido).",
    history_keys=["power"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-late-binding-closure",
    domain="conceitos", difficulty="hard", task_type="direct_answer",
    user_request="O que é late binding em closures e por que causa bug em loops?",
    think_text=(
        "Pergunta técnica nova. Closures em Python capturam a VARIÁVEL, não o valor no "
        "momento da criação — então se várias funções são criadas dentro de um loop "
        "referenciando a variável do loop, todas acabam vendo o valor FINAL dela, não o "
        "valor de quando foram criadas."
    ),
    final_text="Late binding significa que uma closure captura a variável em si, não o valor dela no momento em que a closure foi criada. Isso causa bug clássico em loops: se você cria várias funções dentro de um for referenciando a variável do loop, todas elas veem o valor final da variável quando forem chamadas depois, não o valor de cada iteração. A correção comum é capturar o valor via um argumento padrão.",
    history_keys=["generator", "decorator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-globals-bad-practice",
    domain="conceitos", difficulty="easy", task_type="direct_answer",
    user_request="Por que variáveis globais são consideradas uma prática ruim?",
    think_text=(
        "Pergunta nova, sem relação com o assunto anterior. Globais criam acoplamento "
        "invisível: qualquer parte do código pode ler ou mudar o valor, tornando difícil "
        "rastrear de onde veio um bug ou prever o comportamento sem ler o programa inteiro."
    ),
    final_text="Globais são arriscadas porque criam acoplamento invisível: qualquer função pode ler ou modificar o valor sem isso aparecer na assinatura da função, dificultando rastrear a origem de um bug ou testar partes do código isoladamente.",
    history_keys=["big_o"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-sql-injection",
    domain="seguranca", difficulty="medium", task_type="direct_answer",
    user_request="O que é injeção de SQL e como evitar?",
    think_text=(
        "Assunto novo, sem relação com o turno anterior. Injeção de SQL acontece quando "
        "entrada do usuário é concatenada diretamente numa query SQL sem escapar, permitindo "
        "que o atacante injete comandos SQL arbitrários. A defesa real é usar consultas "
        "parametrizadas, nunca concatenação de string."
    ),
    final_text="Injeção de SQL acontece quando entrada não confiável do usuário é concatenada direto numa query SQL, permitindo que um atacante injete comandos SQL próprios. A defesa correta é sempre usar consultas parametrizadas (placeholders), nunca montar a query por concatenação de string.",
    history_keys=["big_o"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-serialization",
    domain="conceitos", difficulty="easy", task_type="direct_answer",
    user_request="O que é serialização, em termos simples?",
    think_text=(
        "Pergunta nova, sem conexão com o assunto anterior. Serialização é transformar uma "
        "estrutura de dados em memória (objeto, dicionário) num formato que pode ser "
        "guardado em arquivo ou enviado pela rede, como JSON ou bytes."
    ),
    final_text="Serialização é converter uma estrutura de dados que existe na memória do programa (um objeto, um dicionário) num formato que pode ser salvo em arquivo ou transmitido pela rede, como JSON, e depois reconstruído (desserializado) do outro lado.",
    history_keys=["tail_recursion"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-process-vs-thread",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="Qual a diferença entre processo e thread?",
    think_text=(
        "Pergunta nova, sem relação com o turno anterior. Um processo tem seu próprio espaço "
        "de memória isolado; threads dentro do mesmo processo compartilham memória entre si, "
        "o que é mais rápido pra comunicação, mas exige cuidado com acesso concorrente."
    ),
    final_text="Um processo tem seu próprio espaço de memória isolado do sistema operacional. Threads existem dentro de um processo e compartilham a mesma memória entre si, o que torna comunicação entre elas mais rápida, mas exige sincronização cuidadosa pra evitar condições de corrida.",
    history_keys=["mul"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-deadlock",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="O que é deadlock?",
    think_text=(
        "Assunto novo, sem relação com o anterior. Deadlock acontece quando duas ou mais "
        "threads/processos esperam eternamente por um recurso que a outra parte segura, e "
        "nenhuma libera o que tem até conseguir o que falta."
    ),
    final_text="Deadlock é quando duas ou mais threads ficam presas esperando eternamente por um recurso que a outra está segurando, e nenhuma libera o que já tem até conseguir o que falta, travando o sistema por completo.",
    history_keys=["power", "identity"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-idempotency",
    domain="conceitos", difficulty="medium", task_type="direct_answer",
    user_request="O que é idempotência numa API?",
    think_text=(
        "Pergunta nova de API, sem relação com o assunto anterior. Uma operação idempotente "
        "produz o mesmo resultado final não importa quantas vezes seja chamada com os "
        "mesmos parâmetros — importante pra retentativas seguras em rede instável."
    ),
    final_text="Idempotência significa que chamar a mesma operação várias vezes com os mesmos parâmetros produz o mesmo resultado final da primeira chamada, sem efeitos colaterais extras. É importante porque permite retentativas seguras quando uma resposta se perde na rede, sem risco de duplicar a ação.",
    history_keys=["sqrt"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-singleton",
    domain="design_patterns", difficulty="medium", task_type="direct_answer",
    user_request="O que é o design pattern singleton?",
    think_text=(
        "Pergunta nova, sem conexão com o anterior. Singleton garante que uma classe tenha "
        "só UMA instância durante toda a execução do programa, e fornece um ponto de acesso "
        "global a ela — útil, mas também criticado por dificultar testes."
    ),
    final_text="Singleton é um padrão que garante que uma classe tenha apenas uma instância durante toda a execução, com um ponto de acesso global a ela. É útil pra recursos compartilhados como configuração ou conexão de log, mas é criticado por criar estado global implícito, dificultando testes isolados.",
    history_keys=["decorator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-composition-vs-inheritance",
    domain="design_patterns", difficulty="medium", task_type="direct_answer",
    user_request="Qual a diferença entre composição e herança?",
    think_text=(
        "Assunto novo, sem relação com o turno anterior. Herança modela uma relação 'é um' "
        "(Cachorro é um Animal), fixada em tempo de compilação/definição; composição modela "
        "'tem um' (Carro tem um Motor), mais flexível porque pode trocar o componente em "
        "tempo de execução."
    ),
    final_text="Herança modela uma relação 'é um' (Cachorro é um Animal), definida na estrutura de classes. Composição modela 'tem um' (Carro tem um Motor), montando o objeto a partir de componentes menores, geralmente mais flexível porque o componente pode ser trocado sem alterar a hierarquia de classes.",
    history_keys=["big_o", "power"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-modulo",
    domain="matematica", difficulty="easy", task_type="direct_answer",
    user_request="O que é o operador módulo (%) e pra que serve em código?",
    think_text=(
        "Pergunta nova de matemática/código, sem relação com o assunto anterior. Módulo "
        "devolve o RESTO de uma divisão inteira — muito usado pra checar paridade (n % 2), "
        "fazer wraparound em índices circulares, ou limitar um valor a um intervalo."
    ),
    final_text="O operador módulo (%) devolve o resto de uma divisão inteira. É usado pra checar se um número é par ou ímpar (n % 2), fazer índices circulares (i % tamanho_da_lista), ou distribuir itens em grupos de tamanho fixo.",
    history_keys=["identity"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-binary-to-decimal",
    domain="matematica", difficulty="easy", task_type="direct_answer",
    user_request="Como converter um número binário pra decimal, manualmente?",
    think_text=(
        "Pergunta nova, sem conexão com o turno anterior. Cada dígito binário representa uma "
        "potência de 2, da direita pra esquerda começando em 2 elevado a 0. Basta somar as "
        "potências correspondentes aos dígitos que são 1."
    ),
    final_text="Cada posição do número binário, da direita pra esquerda, representa uma potência de 2 crescente (2⁰, 2¹, 2², ...). Some as potências correspondentes aos dígitos que são 1. Por exemplo, 1011 em binário é 2³+2¹+2⁰ = 8+2+1 = 11 em decimal.",
    history_keys=["mul", "identity"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-scientific-notation",
    domain="matematica", difficulty="easy", task_type="direct_answer",
    user_request="O que é notação científica e quando usar em código?",
    think_text=(
        "Pergunta nova, sem relação com o assunto anterior. Notação científica representa um "
        "número como mantissa vezes potência de 10, útil pra números muito grandes ou muito "
        "pequenos onde a forma decimal comum ficaria difícil de ler ou processar com precisão."
    ),
    final_text="Notação científica representa um número como uma mantissa multiplicada por uma potência de 10, como 1.5e10 pra 15 bilhões. Em código, é útil pra escrever ou exibir números muito grandes ou muito pequenos de forma compacta e legível, sem contar zeros manualmente.",
    history_keys=["power"],
)))

# --- ferramentas de dev (git/venv/ci) ----------------------------------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-git-rebase",
    domain="ferramentas", difficulty="medium", task_type="documentation_usage",
    user_request="O que faz o comando git rebase?",
    think_text=(
        "Pergunta nova sobre ferramentas de versionamento, sem relação com o assunto "
        "anterior. Rebase reaplica os commits de um branch em cima de outro ponto base, "
        "reescrevendo o histórico pra parecer linear, em vez de criar um commit de merge."
    ),
    final_text="git rebase reaplica os commits do seu branch atual em cima de outro ponto (geralmente a versão mais recente de main), reescrevendo o histórico como se você tivesse começado a trabalhar a partir dali, sem criar um commit de merge extra.",
    history_keys=["sqrt"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-merge-vs-rebase",
    domain="ferramentas", difficulty="medium", task_type="documentation_usage",
    user_request="Qual a diferença entre git merge e git rebase?",
    think_text=(
        "Assunto novo, sem conexão com o turno anterior. merge combina dois históricos "
        "criando um commit de merge que preserva a ramificação real; rebase reescreve o "
        "histórico do seu branch como se ele tivesse partido de um ponto mais recente, "
        "resultando num histórico linear sem esse commit extra."
    ),
    final_text="git merge combina os dois históricos criando um commit de merge, preservando a ramificação real que aconteceu. git rebase reescreve o histórico do seu branch como se ele tivesse partido de um ponto mais recente, resultando num histórico linear, sem esse commit extra, mas alterando os commits originais.",
    history_keys=["decorator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-venv",
    domain="ferramentas", difficulty="easy", task_type="documentation_usage",
    user_request="O que é um virtual environment em Python e por que usar?",
    think_text=(
        "Pergunta nova sobre ferramentas, sem relação com o assunto anterior. Um virtual "
        "environment isola as dependências de um projeto das dependências globais do "
        "sistema, evitando conflito de versões entre projetos diferentes."
    ),
    final_text="Um virtual environment cria um ambiente Python isolado, com suas próprias dependências separadas das instaladas globalmente no sistema. Isso evita que projetos diferentes, que precisam de versões diferentes da mesma biblioteca, entrem em conflito.",
    history_keys=["generator"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-cicd",
    domain="ferramentas", difficulty="easy", task_type="documentation_usage",
    user_request="O que é CI/CD, resumidamente?",
    think_text=(
        "Assunto novo, sem relação com o turno anterior. CI (integração contínua) roda "
        "testes automaticamente a cada mudança de código; CD (entrega/implantação contínua) "
        "automatiza o processo de levar essa mudança validada pra produção."
    ),
    final_text="CI (integração contínua) roda testes automaticamente toda vez que código novo é enviado, pegando problemas cedo. CD (entrega ou implantação contínua) automatiza o processo de levar esse código já validado até produção, sem passos manuais repetitivos.",
    history_keys=["mul", "big_o"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-linter",
    domain="ferramentas", difficulty="easy", task_type="documentation_usage",
    user_request="O que é um linter e por que rodar um?",
    think_text=(
        "Pergunta nova, sem conexão com o assunto anterior. Um linter analisa o código sem "
        "executar ele, apontando problemas de estilo, padrões arriscados e possíveis bugs "
        "antes mesmo do código rodar."
    ),
    final_text="Um linter analisa o código estaticamente, sem executar ele, apontando problemas de estilo, padrões arriscados e possíveis bugs antes mesmo de rodar. Rodar um regularmente pega erros bobos cedo e mantém o código consistente entre diferentes pessoas do time.",
    history_keys=["power"],
)))

# --- mais algoritmos -----------------------------------------------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-dijkstra",
    domain="algoritmos", difficulty="hard", task_type="direct_answer",
    user_request="Como funciona o algoritmo de Dijkstra?",
    think_text=(
        "Pergunta nova de grafos, sem relação com o turno anterior. Dijkstra mantém a menor "
        "distância conhecida até cada nó, sempre expandindo o nó não visitado com menor "
        "distância acumulada, atualizando os vizinhos se encontrar um caminho mais curto — "
        "não funciona com pesos negativos."
    ),
    final_text="Dijkstra encontra o caminho mais curto de um nó origem até todos os outros num grafo com pesos não negativos. Ele mantém a menor distância conhecida até cada nó, sempre expande o nó não visitado com menor distância acumulada, e atualiza os vizinhos se achar um caminho mais curto passando por ali. Não funciona corretamente com pesos negativos.",
    history_keys=["identity", "sqrt"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-dynamic-programming",
    domain="algoritmos", difficulty="hard", task_type="direct_answer",
    user_request="O que é programação dinâmica?",
    think_text=(
        "Assunto novo, sem ligação com o anterior. Programação dinâmica resolve um problema "
        "quebrando em subproblemas menores sobrepostos, guardando o resultado de cada "
        "subproblema (memoização ou tabela) pra nunca recalcular o mesmo trabalho duas vezes."
    ),
    final_text="Programação dinâmica resolve um problema quebrando ele em subproblemas menores que se repetem, guardando o resultado de cada subproblema já resolvido (via memoização ou uma tabela) pra nunca recalcular o mesmo trabalho duas vezes. Fibonacci com cache é o exemplo clássico mais simples.",
    history_keys=["tail_recursion"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-stack-vs-queue",
    domain="algoritmos", difficulty="easy", task_type="direct_answer",
    user_request="Qual a diferença entre pilha e fila?",
    think_text=(
        "Pergunta nova, sem conexão com o turno anterior. Pilha é LIFO (o último que entra é "
        "o primeiro que sai); fila é FIFO (o primeiro que entra é o primeiro que sai)."
    ),
    final_text="Pilha segue a ordem LIFO (último a entrar, primeiro a sair), como uma pilha de pratos. Fila segue FIFO (primeiro a entrar, primeiro a sair), como uma fila de pessoas.",
    history_keys=["power"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-heap",
    domain="algoritmos", difficulty="medium", task_type="direct_answer",
    user_request="O que é um heap (fila de prioridade)?",
    think_text=(
        "Pergunta nova de estrutura de dados, sem relação com o assunto anterior. Um heap "
        "mantém, em tempo O(log n) por operação, acesso rápido ao elemento de maior (ou "
        "menor) prioridade, sem precisar manter a coleção inteira ordenada."
    ),
    final_text="Um heap é uma estrutura de dados que mantém acesso rápido ao elemento de maior (ou menor) prioridade, em O(log n) por inserção ou remoção, sem precisar manter a coleção inteira ordenada o tempo todo. É a base de filas de prioridade e do heapsort.",
    history_keys=["decorator", "mul"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-binary-search",
    domain="algoritmos", difficulty="easy", task_type="direct_answer",
    user_request="Como funciona a busca binária?",
    think_text=(
        "Pergunta nova, sem relação com o turno anterior. Busca binária exige entrada "
        "ordenada: compara o elemento do meio com o alvo, e descarta metade da busca a cada "
        "passo, dependendo se o alvo é maior ou menor que o meio."
    ),
    final_text="Busca binária exige que a entrada esteja ordenada. A cada passo, compara o elemento do meio com o valor procurado: se o meio for maior, descarta a metade direita; se for menor, descarta a metade esquerda. Isso reduz o espaço de busca pela metade a cada comparação, dando complexidade O(log n).",
    history_keys=["identity"],
)))

# --- debugging real, com execução -------------------------------------------------------------
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-off-by-one",
    domain="debugging", difficulty="medium", task_type="debugging",
    user_request="Essa função devia somar todos os elementos exceto o último, mas está dando erro. def sum_except_last(items): return sum(items[0:len(items)])",
    think1=(
        "Assunto novo, sem relação com o turno anterior. O slice items[0:len(items)] pega a "
        "lista INTEIRA, não exclui o último elemento — o slice correto pra excluir o último "
        "item é items[0:len(items) - 1] ou simplesmente items[:-1]."
    ),
    file_path="sumexcept.py",
    content="def sum_except_last(items):\n    return sum(items[:-1])\n",
    think2="Vou testar com uma lista de vários elementos e confirmar que o último realmente fica de fora da soma.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "sumexcept.py", "content": "def sum_except_last(items):\n    return sum(items[:-1])\n"},
            {"path": "test_sumexcept.py", "content": "from sumexcept import sum_except_last\n\ndef test_excludes_last_element():\n    assert sum_except_last([1, 2, 3, 10]) == 6\n"},
        ],
    },
    final_text="O bug era o slice items[0:len(items)], que pega a lista inteira em vez de excluir o último elemento. Corrigi pra items[:-1], que exclui o último item corretamente. Testado com [1, 2, 3, 10], a soma ficou 6 como esperado.",
    history_keys=["who_trained"],
)))
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-mutable-default",
    domain="debugging", difficulty="hard", task_type="debugging",
    user_request="Por que essa função acumula itens entre chamadas diferentes? def add_item(item, bucket=[]): bucket.append(item); return bucket",
    think1=(
        "Mudança completa de assunto. O problema é o argumento padrão mutável bucket=[] — "
        "listas padrão em Python são criadas UMA vez, na definição da função, não a cada "
        "chamada, então todas as chamadas sem argumento explícito compartilham a MESMA lista."
    ),
    file_path="additem.py",
    content="def add_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket\n",
    think2="Vou testar chamando a função duas vezes sem passar bucket, confirmando que cada chamada agora começa vazia.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "additem.py", "content": "def add_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket\n"},
            {"path": "test_additem.py", "content": "from additem import add_item\n\ndef test_calls_do_not_share_state():\n    first = add_item('a')\n    second = add_item('b')\n    assert first == ['a']\n    assert second == ['b']\n"},
        ],
    },
    final_text="O bug é o argumento padrão mutável bucket=[], que é criado uma única vez na definição da função e compartilhado entre TODAS as chamadas sem argumento explícito. Corrigi usando None como padrão e criando a lista nova dentro da função. Testei duas chamadas seguidas e confirmei que não compartilham mais estado.",
    history_keys=["big_o", "identity"],
)))
ABRUPT_ANCHORS.append(("write_checker", dict(
    id_="gen-mt-abrupt-wrong-comparison",
    domain="debugging", difficulty="easy", task_type="debugging",
    user_request="Essa função devia checar se a lista está vazia, mas sempre retorna False. def is_empty(items): return items == None",
    think1=(
        "Assunto novo, sem relação com o turno anterior. Comparar com None não detecta lista "
        "vazia, uma lista vazia é [] e [] == None é False. O jeito certo de checar vazio é "
        "len(items) == 0 ou simplesmente not items."
    ),
    file_path="isempty.py",
    content="def is_empty(items):\n    return not items\n",
    think2="Vou testar com uma lista vazia e uma lista com elementos, pra confirmar os dois lados.",
    checker_args={
        "language": "python", "operation": "compile_and_test",
        "files": [
            {"path": "isempty.py", "content": "def is_empty(items):\n    return not items\n"},
            {"path": "test_isempty.py", "content": "from isempty import is_empty\n\ndef test_empty_list_is_true():\n    assert is_empty([]) is True\n\ndef test_nonempty_list_is_false():\n    assert is_empty([1]) is False\n"},
        ],
    },
    final_text="O bug era comparar items == None, que nunca é verdadeiro pra uma lista vazia de verdade ([] é diferente de None). Corrigi pra not items, que é True tanto pra lista vazia quanto pra None. Testei lista vazia e lista com elemento, os dois corretos agora.",
    history_keys=["sqrt", "generator"],
)))

# --- reação de confusão (mirror do bug real observado ao vivo) --------------------------------
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-confused-1",
    domain="clarificacao", difficulty="easy", task_type="insufficient_information",
    user_request="O quê?",
    think_text=(
        "O usuário está reagindo com confusão a algo que eu disse antes, não fazendo um "
        "pedido novo com conteúdo próprio. Não sei exatamente qual parte confundiu, preciso "
        "perguntar em vez de assumir ou repetir uma resposta genérica."
    ),
    final_text="Desculpa, não ficou claro o que confundiu. Pode me dizer qual parte da resposta anterior não fez sentido?",
    history_keys=["partial_derivative"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-confused-2",
    domain="clarificacao", difficulty="easy", task_type="insufficient_information",
    user_request="Isso não faz sentido, o que você quis dizer?",
    think_text=(
        "O usuário está pedindo esclarecimento sobre algo que eu já falei, não fazendo um "
        "pedido totalmente novo. Não tenho certeza de qual parte especificamente causou "
        "confusão, preciso perguntar em vez de tentar adivinhar e explicar a coisa errada."
    ),
    final_text="Sem problema, vamos esclarecer. Qual trecho especificamente não fez sentido pra você?",
    history_keys=["dependency_injection"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-confused-3",
    domain="clarificacao", difficulty="easy", task_type="insufficient_information",
    user_request="Pera, do que você estava falando mesmo?",
    think_text=(
        "O usuário perdeu o fio da conversa e está pedindo pra eu recapitular. Posso "
        "resumir o assunto do turno anterior com segurança, já que isso está registrado na "
        "própria sessão."
    ),
    final_text="Sem problema. No turno anterior eu estava respondendo sobre injeção de dependência. Quer que eu retome esse assunto, ou prefere seguir com outra coisa?",
    history_keys=["dependency_injection"],
)))
ABRUPT_ANCHORS.append(("no_tool", dict(
    id_="gen-mt-abrupt-confused-4",
    domain="clarificacao", difficulty="easy", task_type="insufficient_information",
    user_request="Não entendi nada, explica de novo mas mais simples?",
    think_text=(
        "Pedido de reformulação mais simples do que já expliquei, não é um assunto novo. "
        "Sei exatamente qual foi o conteúdo do turno anterior, posso reformular sem "
        "inventar nada novo."
    ),
    final_text="Claro, de forma mais simples: uma race condition é quando duas partes do programa tentam usar a mesma coisa ao mesmo tempo, numa ordem que ninguém controla, e isso bagunça o resultado.",
    history_keys=["race_condition"],
)))

# =============================================================================================
# Categoria 2: pedido de ferramenta avulso/tardio numa sessão poluída (~40 âncoras)
# =============================================================================================

LATE_TOOL_ANCHORS = []

# --- list_files (10) ---------------------------------------------------------------------------
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-empty-1",
    domain="ferramentas", difficulty="easy",
    user_request="Liste os arquivos nesse diretório.",
    think1="Pedido avulso de listagem, sem relação com os turnos anteriores. Preciso listar o workspace de verdade pra responder, não posso adivinhar o que tem lá.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem veio vazia, não há nenhum arquivo ainda nesse workspace.",
    final_text="No momento não há nenhum arquivo no diretório de trabalho.",
    history_keys=["identity", "partial_derivative"],
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-with-files-1",
    domain="ferramentas", difficulty="easy",
    user_request="Quais arquivos existem nesse projeto?",
    think1="Pedido de listagem chegando sozinho, depois de assuntos sem relação. Vou listar o workspace real em vez de tentar lembrar de algo que não foi dito nesta sessão.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem mostrou dois arquivos reais no workspace.",
    final_text="O diretório contém dois arquivos: config.py e main.py.",
    history_keys=["mul", "decorator"],
    preexisting_files={"config.py": "TIMEOUT = 30\n", "main.py": "print('ola')\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-nested-1",
    domain="ferramentas", difficulty="medium",
    user_request="Tem alguma pasta dentro desse projeto?",
    think1="Pedido novo de listagem, não relacionado ao que discutimos antes. Preciso listar o workspace de verdade pra ver a estrutura real.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem mostrou um arquivo dentro de uma subpasta chamada utils.",
    final_text="Sim, há uma pasta utils contendo um arquivo helpers.py.",
    history_keys=["big_o"],
    preexisting_files={"utils/helpers.py": "def helper():\n    pass\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-after-tool-1",
    domain="ferramentas", difficulty="easy",
    user_request="E agora, o que tem no diretório?",
    think1="Pedido avulso de listagem, chegando depois de um assunto sem relação nenhuma com arquivos. Vou listar de verdade em vez de supor.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem confirmou um único arquivo README.md no workspace.",
    final_text="O diretório contém só o README.md por enquanto.",
    history_keys=["race_condition", "sqrt"],
    preexisting_files={"README.md": "# Projeto\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-python-files-1",
    domain="ferramentas", difficulty="easy",
    user_request="Quantos arquivos Python tem nesse diretório?",
    think1="Pedido de listagem novo, sem relação com o assunto anterior. Vou listar o workspace real pra contar de verdade, não estimar.",
    tool_name="list_files", tool_args={"path": ".", "glob": "*.py"},
    think2="A listagem mostrou dois arquivos .py no diretório.",
    final_text="Há dois arquivos Python: app.py e settings.py.",
    history_keys=["dependency_injection"],
    preexisting_files={"app.py": "print('app')\n", "settings.py": "DEBUG = True\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-empty-2",
    domain="ferramentas", difficulty="easy",
    user_request="Dá uma olhada no que já existe nesse workspace?",
    think1="Pedido avulso pra inspecionar o workspace, sem relação com os turnos anteriores. Preciso listar de verdade antes de responder qualquer coisa sobre o conteúdo.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem veio vazia, esse workspace ainda não tem nada.",
    final_text="Esse workspace está vazio no momento, nenhum arquivo criado ainda.",
    history_keys=["tail_recursion", "power"],
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-config-1",
    domain="ferramentas", difficulty="easy",
    user_request="Existe algum arquivo de configuração já criado?",
    think1="Pedido novo de inspeção, sem relação com o assunto anterior. Vou listar o workspace real em vez de assumir que existe ou não.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem mostrou um arquivo settings.json já presente.",
    final_text="Sim, já existe settings.json no workspace.",
    history_keys=["identity"],
    preexisting_files={"settings.json": "{\"debug\": true}\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-tests-1",
    domain="ferramentas", difficulty="easy",
    user_request="Já existe algum teste escrito nesse projeto?",
    think1="Pedido avulso, sem relação com os turnos de contexto anteriores. Vou listar o workspace de verdade em vez de supor que não há testes.",
    tool_name="list_files", tool_args={"path": ".", "glob": "test_*.py"},
    think2="A listagem confirmou um arquivo de teste já presente.",
    final_text="Sim, já existe test_utils.py no projeto.",
    history_keys=["mul", "generator"],
    preexisting_files={"test_utils.py": "def test_something():\n    assert True\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-after-two-1",
    domain="ferramentas", difficulty="medium",
    user_request="Quais arquivos estão no diretório atual, exatamente?",
    think1="Pedido de listagem chegando sozinho, depois de dois assuntos sem relação nenhuma entre si. Vou listar de verdade pra dar uma resposta precisa.",
    tool_name="list_files", tool_args={"path": "."},
    think2="A listagem mostrou três arquivos reais no workspace.",
    final_text="O diretório tem três arquivos: models.py, views.py e urls.py.",
    history_keys=["big_o", "identity", "power"],
    preexisting_files={"models.py": "class Model:\n    pass\n", "views.py": "def view():\n    pass\n", "urls.py": "urls = []\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-list-yaml-1",
    domain="ferramentas", difficulty="easy",
    user_request="Tem arquivo de configuração YAML já criado?",
    think1="Pedido novo e independente do que veio antes. Vou listar o workspace de verdade filtrando por extensão yaml.",
    tool_name="list_files", tool_args={"path": ".", "glob": "*.yaml"},
    think2="A listagem não retornou nenhum arquivo .yaml.",
    final_text="Não, não há nenhum arquivo YAML nesse workspace ainda.",
    history_keys=["decorator", "sqrt"],
    preexisting_files={"config.json": "{}\n"},
))

# --- read_file (10) ------------------------------------------------------------------------
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-config-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="O que tem dentro do config.py?",
    think1="Pedido avulso de leitura, sem relação com o assunto anterior. Preciso ler o arquivo de verdade, não posso inventar o conteúdo.",
    tool_name="read_file", tool_args={"path": "config.py"},
    think2="O arquivo define uma única constante de timeout.",
    final_text="O arquivo config.py contém apenas TIMEOUT = 30.",
    history_keys=["identity"],
    preexisting_files={"config.py": "TIMEOUT = 30\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-main-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="O que a função em main.py faz?",
    think1="Pedido de leitura chegando sozinho, sem relação com o turno anterior. Vou ler o arquivo real antes de responder.",
    tool_name="read_file", tool_args={"path": "main.py"},
    think2="A função soma dois números e retorna o resultado.",
    final_text="A função add(a, b) em main.py retorna a soma de a e b.",
    history_keys=["mul"],
    preexisting_files={"main.py": "def add(a, b):\n    return a + b\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-utils-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="Pode ler o utils.py pra mim?",
    think1="Pedido avulso de leitura, sem conexão com o assunto anterior. Vou ler o conteúdo real do arquivo.",
    tool_name="read_file", tool_args={"path": "utils.py"},
    think2="O arquivo tem uma função que formata datas.",
    final_text="utils.py contém a função format_date(d), que formata uma data no padrão dd/mm/aaaa.",
    history_keys=["big_o", "power"],
    preexisting_files={"utils.py": "def format_date(d):\n    return d.strftime('%d/%m/%Y')\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-readme-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="O README já tem alguma coisa escrita?",
    think1="Pedido novo de leitura, sem relação com os turnos anteriores. Vou ler o arquivo de verdade em vez de assumir.",
    tool_name="read_file", tool_args={"path": "README.md"},
    think2="O README só tem um título, sem descrição ainda.",
    final_text="O README.md só tem o título '# Projeto', sem descrição ainda.",
    history_keys=["dependency_injection"],
    preexisting_files={"README.md": "# Projeto\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-requirements-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="Quais dependências estão no requirements.txt?",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou ler o arquivo real pra listar as dependências corretas.",
    tool_name="read_file", tool_args={"path": "requirements.txt"},
    think2="O arquivo lista duas dependências com versões fixadas.",
    final_text="requirements.txt lista requests==2.31.0 e pytest==7.4.0.",
    history_keys=["race_condition"],
    preexisting_files={"requirements.txt": "requests==2.31.0\npytest==7.4.0\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-model-1",
    domain="leitura_de_repositorio", difficulty="medium",
    user_request="O que a classe em models.py representa?",
    think1="Pedido de leitura chegando avulso, sem relação com o turno anterior. Vou ler o arquivo real antes de descrever a classe.",
    tool_name="read_file", tool_args={"path": "models.py"},
    think2="A classe representa um usuário com nome e email.",
    final_text="models.py define a classe User, com os campos name e email.",
    history_keys=["identity", "mul"],
    preexisting_files={"models.py": "class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-gitignore-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="O que está no .gitignore desse projeto?",
    think1="Pedido novo e independente. Vou ler o arquivo de verdade em vez de supor um conteúdo padrão.",
    tool_name="read_file", tool_args={"path": ".gitignore"},
    think2="O arquivo ignora cache de bytecode e ambientes virtuais.",
    final_text="O .gitignore ignora __pycache__/ e .venv/.",
    history_keys=["decorator"],
    preexisting_files={".gitignore": "__pycache__/\n.venv/\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-env-example-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="Quais variáveis de ambiente esse projeto espera?",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou ler o arquivo de exemplo real pra saber quais variáveis são esperadas.",
    tool_name="read_file", tool_args={"path": ".env.example"},
    think2="O arquivo lista duas variáveis esperadas.",
    final_text="O projeto espera DATABASE_URL e SECRET_KEY, conforme .env.example.",
    history_keys=["tail_recursion"],
    preexisting_files={".env.example": "DATABASE_URL=\nSECRET_KEY=\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-changelog-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="Tem alguma coisa registrada no CHANGELOG?",
    think1="Pedido novo de leitura, sem conexão com o turno anterior. Vou ler o arquivo real em vez de assumir que está vazio ou não.",
    tool_name="read_file", tool_args={"path": "CHANGELOG.md"},
    think2="O changelog tem uma única entrada registrada.",
    final_text="O CHANGELOG.md tem uma entrada: '0.1.0: versão inicial'.",
    history_keys=["big_o"],
    preexisting_files={"CHANGELOG.md": "0.1.0: versao inicial\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-read-constants-1",
    domain="leitura_de_repositorio", difficulty="easy",
    user_request="O que tem em constants.py?",
    think1="Pedido avulso de leitura, sem relação com o assunto anterior. Vou ler o arquivo real de verdade.",
    tool_name="read_file", tool_args={"path": "constants.py"},
    think2="O arquivo define duas constantes de limite.",
    final_text="constants.py define MAX_RETRIES = 3 e TIMEOUT_SECONDS = 10.",
    history_keys=["power", "identity"],
    preexisting_files={"constants.py": "MAX_RETRIES = 3\nTIMEOUT_SECONDS = 10\n"},
))

# --- shell ls (8) ----------------------------------------------------------------------------
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-1",
    domain="shell_real", difficulty="easy",
    user_request="Roda um ls aí pra ver o que tem.",
    think1="Pedido avulso de shell, sem relação com o assunto anterior. Vou rodar ls de verdade em vez de inventar uma listagem.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída do ls confirmou um único arquivo no diretório.",
    final_text="O ls mostrou só um arquivo: app.py.",
    history_keys=["mul"],
    preexisting_files={"app.py": "print('oi')\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-2",
    domain="shell_real", difficulty="easy",
    user_request="Confirma pra mim o que tem na pasta atual usando o terminal.",
    think1="Pedido novo, sem relação com o turno anterior. Vou usar shell/ls de verdade pra confirmar o conteúdo real.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída do ls veio vazia, sem arquivos.",
    final_text="O terminal confirmou: a pasta atual está vazia.",
    history_keys=["decorator", "sqrt"],
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-3",
    domain="shell_real", difficulty="easy",
    user_request="Dá um ls -la nesse diretório?",
    think1="Pedido avulso, sem conexão com o assunto anterior. Vou rodar o comando de verdade.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída mostrou dois arquivos no diretório.",
    final_text="ls -la mostrou dois arquivos: script.py e data.csv.",
    history_keys=["race_condition"],
    preexisting_files={"script.py": "pass\n", "data.csv": "a,b\n1,2\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-4",
    domain="shell_real", difficulty="medium",
    user_request="Verifica no terminal se já tem algum arquivo de log.",
    think1="Pedido novo de shell, sem relação com o turno anterior. Vou rodar ls de verdade em vez de supor.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A listagem confirmou um arquivo de log já presente.",
    final_text="Sim, já existe app.log no diretório.",
    history_keys=["identity", "big_o"],
    preexisting_files={"app.log": "iniciado\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-5",
    domain="shell_real", difficulty="easy",
    user_request="Usa o terminal e me diz o que tem aqui.",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou rodar ls de verdade pra dar uma resposta precisa.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída confirmou um arquivo de configuração presente.",
    final_text="O terminal mostrou um arquivo: settings.ini.",
    history_keys=["tail_recursion"],
    preexisting_files={"settings.ini": "[core]\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-6",
    domain="shell_real", difficulty="easy",
    user_request="Roda ls de novo, quero confirmar o estado atual.",
    think1="Pedido novo e independente do assunto anterior. Vou rodar o comando de verdade, não vou supor que nada mudou.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída mostrou um arquivo de teste presente.",
    final_text="O estado atual: só test_app.py está presente no diretório.",
    history_keys=["power", "decorator"],
    preexisting_files={"test_app.py": "def test_x():\n    assert True\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-7",
    domain="shell_real", difficulty="easy",
    user_request="Me confirma via terminal quantos arquivos tem aqui.",
    think1="Pedido avulso de shell, sem relação com o turno anterior. Vou rodar ls de verdade pra contar certo.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída mostrou três arquivos no total.",
    final_text="São três arquivos: a.py, b.py e c.py.",
    history_keys=["mul", "identity"],
    preexisting_files={"a.py": "x = 1\n", "b.py": "y = 2\n", "c.py": "z = 3\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-ls-8",
    domain="shell_real", difficulty="easy",
    user_request="Roda um ls pra eu ver se o build já gerou algo.",
    think1="Pedido novo, sem conexão com o assunto anterior. Vou checar de verdade via ls em vez de supor que o build rodou.",
    tool_name="shell", tool_args={"binary": "ls", "args": ["-la"]},
    think2="A saída veio vazia, nada foi gerado ainda.",
    final_text="Não, o diretório está vazio, o build ainda não gerou nada.",
    history_keys=["big_o"],
))

# --- shell cat (6) ----------------------------------------------------------------------------
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-1",
    domain="shell_real", difficulty="easy",
    user_request="Dá um cat no config.py.",
    think1="Pedido avulso de shell, sem relação com o assunto anterior. Vou rodar cat de verdade pra mostrar o conteúdo real.",
    tool_name="shell", tool_args={"binary": "cat", "args": ["config.py"]},
    think2="O cat confirmou o conteúdo real do arquivo.",
    final_text="config.py contém apenas: DEBUG = False.",
    history_keys=["identity"],
    preexisting_files={"config.py": "DEBUG = False\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-2",
    domain="shell_real", difficulty="easy",
    user_request="Usa cat pra ver o que tem no main.py.",
    think1="Pedido novo, sem relação com o turno anterior. Vou rodar cat de verdade.",
    tool_name="shell", tool_args={"binary": "cat", "args": ["main.py"]},
    think2="O cat mostrou uma função de saudação simples.",
    final_text="main.py contém uma função greet() que imprime 'ola mundo'.",
    history_keys=["decorator", "mul"],
    preexisting_files={"main.py": "def greet():\n    print('ola mundo')\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-3",
    domain="shell_real", difficulty="easy",
    user_request="Cat no requirements.txt, por favor.",
    think1="Pedido avulso, sem conexão com o assunto anterior. Vou rodar cat de verdade pra listar as dependências reais.",
    tool_name="shell", tool_args={"binary": "cat", "args": ["requirements.txt"]},
    think2="O cat mostrou uma única dependência.",
    final_text="requirements.txt contém só: flask==3.0.0.",
    history_keys=["race_condition"],
    preexisting_files={"requirements.txt": "flask==3.0.0\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-4",
    domain="shell_real", difficulty="easy",
    user_request="Roda cat no arquivo de log agora.",
    think1="Pedido novo de shell, sem relação com o turno anterior. Vou rodar cat de verdade em vez de supor o conteúdo.",
    tool_name="shell", tool_args={"binary": "cat", "args": ["app.log"]},
    think2="O log mostrou uma única linha registrada.",
    final_text="O app.log tem uma linha: 'servidor iniciado na porta 8000'.",
    history_keys=["tail_recursion", "identity"],
    preexisting_files={"app.log": "servidor iniciado na porta 8000\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-5",
    domain="shell_real", difficulty="easy",
    user_request="Dá uma olhada com cat no arquivo de versão.",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou rodar cat de verdade pra confirmar a versão real.",
    tool_name="shell", tool_args={"binary": "cat", "args": ["VERSION"]},
    think2="O cat confirmou a versão atual registrada.",
    final_text="O arquivo VERSION contém: 1.2.0.",
    history_keys=["power"],
    preexisting_files={"VERSION": "1.2.0\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-cat-6",
    domain="shell_real", difficulty="easy",
    user_request="Confirma com cat o que tem no .env.example.",
    think1="Pedido novo, sem relação com o turno anterior. Vou rodar cat de verdade pra listar as variáveis reais esperadas.",
    tool_name="shell", tool_args={"binary": "cat", "args": [".env.example"]},
    think2="O cat mostrou uma variável de ambiente esperada.",
    final_text="O .env.example espera só a variável API_KEY.",
    history_keys=["big_o", "mul"],
    preexisting_files={".env.example": "API_KEY=\n"},
))

# --- shell grep (6) ---------------------------------------------------------------------------
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-1",
    domain="shell_real", difficulty="medium",
    user_request="Procura por TODO no código com grep.",
    think1="Pedido avulso de busca, sem relação com o assunto anterior. Vou rodar grep de verdade pra achar ocorrências reais.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "TODO", "."]},
    think2="O grep encontrou uma ocorrência real de TODO no código.",
    final_text="Encontrei um TODO em app.py, linha 2: '# TODO: adicionar validacao'.",
    history_keys=["identity"],
    preexisting_files={"app.py": "def run():\n    # TODO: adicionar validacao\n    pass\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-2",
    domain="shell_real", difficulty="medium",
    user_request="Usa grep pra achar onde 'import os' aparece.",
    think1="Pedido novo de busca, sem relação com o turno anterior. Vou rodar grep de verdade em vez de adivinhar onde está.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "import os", "."]},
    think2="O grep achou a ocorrência real do import.",
    final_text="'import os' aparece em utils.py, linha 1.",
    history_keys=["decorator"],
    preexisting_files={"utils.py": "import os\n\ndef path():\n    return os.getcwd()\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-3",
    domain="shell_real", difficulty="medium",
    user_request="Tem algum print esquecido no código? Usa grep pra checar.",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou rodar grep de verdade em vez de assumir que não tem.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "print(", "."]},
    think2="O grep encontrou um print esquecido no código.",
    final_text="Sim, tem um print esquecido em debug.py, linha 3.",
    history_keys=["race_condition", "power"],
    preexisting_files={"debug.py": "def run():\n    x = 1\n    print(x)\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-4",
    domain="shell_real", difficulty="medium",
    user_request="Procura por 'FIXME' em todo o projeto.",
    think1="Pedido novo de busca, sem conexão com o turno anterior. Vou rodar grep de verdade pra confirmar se existe.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "FIXME", "."]},
    think2="O grep voltou com código 1, que é a convenção padrão pra 'nenhuma ocorrência encontrada', não um erro de execução real.",
    final_text="Não encontrei nenhuma ocorrência de FIXME no projeto.",
    history_keys=["big_o", "identity"],
    preexisting_files={"app.py": "def run():\n    pass\n"},
    expect_pass=False,
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-5",
    domain="shell_real", difficulty="medium",
    user_request="Onde a função connect é definida? Usa grep.",
    think1="Pedido avulso, sem relação com o assunto anterior. Vou rodar grep de verdade pra localizar a definição real.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "def connect", "."]},
    think2="O grep achou a definição real da função.",
    final_text="A função connect é definida em db.py, linha 1.",
    history_keys=["tail_recursion"],
    preexisting_files={"db.py": "def connect():\n    pass\n"},
))
LATE_TOOL_ANCHORS.append(dict(
    id_="gen-mt-late-shell-grep-6",
    domain="shell_real", difficulty="medium",
    user_request="Confirma com grep se a variável SECRET_KEY está hardcoded em algum lugar.",
    think1="Pedido novo e sensível, sem relação com o turno anterior. Vou rodar grep de verdade pra confirmar antes de afirmar qualquer coisa.",
    tool_name="shell", tool_args={"binary": "grep", "args": ["-rn", "SECRET_KEY", "."]},
    think2="O grep encontrou a variável hardcoded de verdade no código.",
    final_text="Sim, SECRET_KEY está hardcoded em settings.py, linha 1. Isso deveria vir de uma variável de ambiente, não do código.",
    history_keys=["mul", "decorator"],
    preexisting_files={"settings.py": "SECRET_KEY = 'abc123'\n"},
))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_seen = set()
    written = 0

    for kind, spec in ABRUPT_ANCHORS:
        if spec["id_"] in ids_seen:
            raise SystemExit(f"id duplicado: {spec['id_']}")
        ids_seen.add(spec["id_"])
        example = build_no_tool_anchor(**spec) if kind == "no_tool" else build_write_checker_anchor(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name}")
        written += 1

    for spec in LATE_TOOL_ANCHORS:
        if spec["id_"] in ids_seen:
            raise SystemExit(f"id duplicado: {spec['id_']}")
        ids_seen.add(spec["id_"])
        example = build_late_tool_anchor(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name}")
        written += 1

    print(f"\n{written} exemplos gerados em {OUT_DIR}")


if __name__ == "__main__":
    main()

