#!/usr/bin/env python
"""Lote base da wave 2, categoria "upgrade de estilo" (docs/plan_dataset_expansion_wave2_
identity_multiturn.md, seção 2). Não introduz task_type novo nem cenário novo de domínio: pega
uma amostra representativa dos task_types já existentes na taxonomia e reescreve o padrão de
`<think>`/`<final>` com o estilo amadurecido em conversa:

- profundidade do `<think>` proporcional à dificuldade real da tarefa (trivial fica curto,
  diagnóstico de causa raiz/trade off fica mais longo porque a tarefa exige isso de verdade,
  nunca por aparência);
- verbalização da intenção de ferramenta em primeira pessoa antes/ao redor de chamar ela
  ("Ok, preciso ler o arquivo antes de responder", não só narrativa depois do fato);
- `<final>` mais próximo/convida a testar quando cabe (nunca em toda resposta trivial);
- nenhum hífen nem travessão como pontuação de frase (nomes próprios como "Logos-3"/"GPT-4"
  são exceção, não são pontuação).

Toda ferramenta usada roda de verdade via `ToolExecutorRegistry`/`SandboxContext` — nenhum
`tool_result` fabricado.

Uso:
    python scripts/gen_style_upgrade_wave2_pilot.py
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
from src.harness.trajectory import Trajectory  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()


# Pega o bloco inteiro do warning de deprecação do pytest_asyncio, do caminho absoluto local
# (ex.: C:\Users\<usuário>\...) até a linha "warnings.warn(...)" que fecha o bloco — pytest
# emite isso independente do teste passar ou falhar, e não tem nenhuma relação com o
# comportamento sendo ensinado, só vaza detalhe da máquina local que rodou a geração.
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
    # Aplica a limpeza em TODO o payload, incluindo `error_message` — `to_json()` (src/tools/
    # base.py) copia `error_message` pro campo "message" do corpo final quando a chamada
    # falha, então limpar só `result.data` (que fica ok pro caso de sucesso) não é suficiente
    # pro caso de erro. `ToolExecutionResult` é um dataclass frozen, então usamos
    # dataclasses.replace em vez de mutar os campos diretamente.
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


def build_no_tool(id_, domain, difficulty, task_type, user_request, think_text, final_text) -> dict:
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think_text}</think>")
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, "python", difficulty, [], task_type, 0, False, "static_only"
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_single_tool(
    id_, domain, difficulty, task_type, language, user_request,
    think1, tool_name, tool_args, think2, final_text,
    execution_classification="tested",
    preexisting_files=None,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    for rel_path, content in (preexisting_files or {}).items():
        target = sandbox.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think1}</think>")
    result = _tool_call(traj, sandbox, tool_name, tool_args)
    assert result.passed, f"esperava sucesso real em {id_}: {result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, language, difficulty, [tool_name], task_type, 1, True, execution_classification
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_write_then_checker(
    id_, domain, difficulty, task_type, user_request,
    think1, file_path, content, think2, checker_args, final_text,
    language="python", expect_pass=True,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think1}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": content})
    assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    checker_result = _tool_call(traj, sandbox, "checker", checker_args)
    if expect_pass:
        assert checker_result.passed, f"esperava checker passando em {id_}: {checker_result.to_json()}"
    else:
        assert not checker_result.passed, f"esperava checker falhando de verdade em {id_}: {checker_result.to_json()}"
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, language, difficulty, ["write_file", "checker"], task_type, 2, True,
            "tested" if expect_pass else "interpreted",
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_single_tool_expect_error(
    id_, domain, difficulty, task_type, language, user_request,
    think1, tool_name, tool_args, think2, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think1}</think>")
    result = _tool_call(traj, sandbox, tool_name, tool_args)
    assert not result.passed, f"esperava recusa/erro real em {id_}: {result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, language, difficulty, [tool_name], task_type, 1, True, "interpreted"
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_tool_unavailable_fallback(
    id_, domain, difficulty, user_request,
    think1, unavailable_tool, unavailable_args,
    think2, file_path, content, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think1}</think>")
    unavailable_result = _tool_call(traj, sandbox, unavailable_tool, unavailable_args)
    assert not unavailable_result.passed, f"esperava UNSUPPORTED_TOOL real em {id_}: {unavailable_result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": content})
    assert write_result.passed, f"write_file de recuperação falhou em {id_}: {write_result.to_json()}"
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, "python", difficulty, [unavailable_tool, "write_file"], "tool_unavailable", 2, True,
            "static_only",
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_read_then_write_then_checker(
    id_, domain, difficulty, user_request,
    think1, read_path, preexisting_files,
    think2, write_path, write_content,
    think3, checker_args, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    for rel_path, content in preexisting_files.items():
        target = sandbox.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
    traj.append_raw(f"<think>{think1}</think>")
    read_result = _tool_call(traj, sandbox, "read_file", {"path": read_path})
    assert read_result.passed, f"read_file falhou inesperadamente em {id_}: {read_result.to_json()}"
    traj.append_raw(f"<think>{think2}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": write_path, "content": write_content})
    assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"
    traj.append_raw(f"<think>{think3}</think>")
    checker_result = _tool_call(traj, sandbox, "checker", checker_args)
    assert checker_result.passed, f"esperava checker passando em {id_}: {checker_result.to_json()}"
    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, "python", difficulty, ["read_file", "write_file", "checker"],
            "multi_file_read_and_edit", 3, True, "tested",
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def build_debug_cycle(
    id_, domain, difficulty, user_request,
    think1, file_path, buggy_content, test_path, test_content,
    think_hypothesis, think_diagnosis, fixed_content, final_text,
) -> dict:
    assert buggy_content != fixed_content, f"correção idêntica ao bug em {id_}"
    sandbox = SandboxContext(policy=_POLICY)
    traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

    traj.append_raw(f"<think>{think1}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": buggy_content})
    assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

    traj.append_raw(f"<think>{think_hypothesis}</think>")
    checker_args = {
        "language": "python",
        "operation": "compile_and_test",
        "files": [
            {"path": file_path, "content": buggy_content},
            {"path": test_path, "content": test_content},
        ],
    }
    first_result = _tool_call(traj, sandbox, "checker", checker_args)
    assert not first_result.passed, f"esperava falha real em {id_}: {first_result.to_json()}"

    traj.append_raw(f"<think>{think_diagnosis}</think>")
    fix_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": fixed_content})
    assert fix_result.passed, f"write_file de correção falhou em {id_}: {fix_result.to_json()}"

    second_checker_args = {
        "language": "python",
        "operation": "compile_and_test",
        "files": [
            {"path": file_path, "content": fixed_content},
            {"path": test_path, "content": test_content},
        ],
    }
    second_result = _tool_call(traj, sandbox, "checker", second_checker_args)
    assert second_result.passed, f"esperava sucesso real após correção em {id_}: {second_result.to_json()}"

    traj.append_raw(f"<final>{final_text}</final>")
    return {
        "metadata": _base_metadata(
            id_, domain, "python", difficulty, ["write_file", "checker"], "debugging", 4, True, "tested"
        ),
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


EXAMPLES = []

# --- single_tool_call: think curto, verbalizando a intenção antes de ler ------------------
EXAMPLES.append((
    "single_tool", dict(
        id_="gen-style-read-before-answer-1",
        domain="leitura_de_repositorio",
        difficulty="easy",
        task_type="single_tool_call",
        language="python",
        user_request="O que a função em helpers.py faz?",
        think1="Preciso ler o arquivo antes de responder, não sei o conteúdo dele ainda.",
        tool_name="read_file",
        tool_args={"path": "helpers.py"},
        think2="A função formata um valor monetário com duas casas decimais.",
        final_text="A função format_currency(value) formata um número como valor monetário com duas casas decimais, por exemplo 10.5 vira 10.50.",
        preexisting_files={
            "helpers.py": (
                "def format_currency(value):\n"
                "    return f\"{value:.2f}\"\n"
            ),
        },
    ),
))
EXAMPLES.append((
    "single_tool", dict(
        id_="gen-style-list-before-answer-1",
        domain="leitura_de_repositorio",
        difficulty="easy",
        task_type="single_tool_call",
        language="python",
        user_request="Quais arquivos existem nesse projeto agora?",
        think1="Ok, preciso listar o diretório de trabalho pra responder com precisão, não vou adivinhar.",
        tool_name="list_files",
        tool_args={"path": "."},
        think2="A listagem veio vazia, é um workspace limpo por enquanto.",
        final_text="No momento não há nenhum arquivo no diretório de trabalho.",
    ),
))

# --- multi_tool_call: trade off técnico explícito no think --------------------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-dedup-tradeoff-1",
        domain="algoritmos",
        difficulty="medium",
        task_type="multi_tool_call",
        user_request="Crie uma função que remove duplicatas de uma lista mantendo a ordem original.",
        think1=(
            "Entendi, tenho que implementar isso considerando o requisito de ordem. Usar set() "
            "seria O(n) mas descarta a ordem original completamente. dict.fromkeys() também é "
            "O(n) e preserva a ordem de primeira aparição desde Python 3.7, então é a escolha "
            "certa aqui."
        ),
        file_path="dedup.py",
        content=(
            "def dedup_keep_order(items):\n"
            "    return list(dict.fromkeys(items))\n"
        ),
        think2="Vou validar com um caso que tem duplicatas fora de ordem, pra confirmar que a ordem é preservada de verdade.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": "dedup.py", "content": "def dedup_keep_order(items):\n    return list(dict.fromkeys(items))\n"},
                {
                    "path": "test_dedup.py",
                    "content": (
                        "from dedup import dedup_keep_order\n\n"
                        "def test_preserves_first_occurrence_order():\n"
                        "    assert dedup_keep_order([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
                    ),
                },
            ],
        },
        final_text="Terminei. Implementei dedup_keep_order usando dict.fromkeys, que preserva a ordem de primeira aparição em O(n). Testei com um caso que embaralha duplicatas e passou. Pode testar com seus próprios dados e me avisar se algum caso não se comportar como esperado.",
    ),
))
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-search-tradeoff-1",
        domain="algoritmos",
        difficulty="medium",
        task_type="multi_tool_call",
        user_request="Escreva uma função que verifica se um número existe numa lista JÁ ORDENADA, da forma mais eficiente possível.",
        think1=(
            "Como a lista já vem ordenada, busca linear seria O(n) e desperdiçaria essa "
            "informação. Busca binária é O(log n) nesse caso, é a escolha certa quando a "
            "entrada já está ordenada."
        ),
        file_path="search.py",
        content=(
            "def binary_contains(sorted_items, target):\n"
            "    low, high = 0, len(sorted_items) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if sorted_items[mid] == target:\n"
            "            return True\n"
            "        if sorted_items[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return False\n"
        ),
        think2="Vou testar com um valor presente, um ausente e uma lista vazia, pra cobrir os casos de borda antes de considerar isso pronto.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {
                    "path": "search.py",
                    "content": (
                        "def binary_contains(sorted_items, target):\n"
                        "    low, high = 0, len(sorted_items) - 1\n"
                        "    while low <= high:\n"
                        "        mid = (low + high) // 2\n"
                        "        if sorted_items[mid] == target:\n"
                        "            return True\n"
                        "        if sorted_items[mid] < target:\n"
                        "            low = mid + 1\n"
                        "        else:\n"
                        "            high = mid - 1\n"
                        "    return False\n"
                    ),
                },
                {
                    "path": "test_search.py",
                    "content": (
                        "from search import binary_contains\n\n"
                        "def test_finds_present_value():\n"
                        "    assert binary_contains([1, 3, 5, 7, 9], 7) is True\n\n"
                        "def test_absent_value_returns_false():\n"
                        "    assert binary_contains([1, 3, 5, 7, 9], 4) is False\n\n"
                        "def test_empty_list_returns_false():\n"
                        "    assert binary_contains([], 1) is False\n"
                    ),
                },
            ],
        },
        final_text="Pronto: implementei busca binária, O(log n) em vez de busca linear, aproveitando que a lista já vem ordenada. Testei valor presente, ausente e lista vazia, todos passaram.",
    ),
))

# --- shell_command: verbalização de intenção antes de rodar --------------------------------
EXAMPLES.append((
    "single_tool", dict(
        id_="gen-style-shell-check-tests-1",
        domain="shell_real",
        difficulty="easy",
        task_type="single_tool_call",
        language="python",
        user_request="Esse projeto tem algum arquivo de teste?",
        think1="Vou listar o diretório antes de responder, não sei o conteúdo de cabeça.",
        tool_name="shell",
        tool_args={"binary": "ls", "args": ["-la"]},
        think2="A listagem não mostrou nenhum arquivo com prefixo test_, então aparentemente não há testes ainda.",
        final_text="Não encontrei nenhum arquivo de teste no diretório atual.",
        execution_classification="tested",
    ),
))

# --- complexity_analysis: technical, no tool ------------------------------------------------
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-complexity-nested-loop-1",
        domain="complexidade",
        difficulty="medium",
        task_type="complexity_analysis",
        user_request="Qual a complexidade de tempo de uma função que compara cada elemento de uma lista com todos os outros?",
        think_text=(
            "Isso é uma pergunta conceitual, não preciso executar nada. Comparar cada elemento "
            "com todos os outros é um laço aninhado sobre a mesma lista: para n elementos, isso "
            "dá n multiplicado por n comparações no pior caso, ou seja, complexidade quadrática."
        ),
        final_text="Complexidade O(n²). Cada um dos n elementos é comparado com os outros n elementos, resultando em ordem de n ao quadrado operações no pior caso.",
    ),
))
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-complexity-hashmap-lookup-1",
        domain="complexidade",
        difficulty="easy",
        task_type="complexity_analysis",
        user_request="Qual a complexidade de buscar uma chave num dicionário Python?",
        think_text="Pergunta conceitual direta sobre a estrutura de dados, dicionários em Python usam tabela hash internamente.",
        final_text="Complexidade O(1) em média, porque dicionários em Python são implementados com tabela hash, que acessa o valor diretamente pela posição calculada a partir da chave, sem precisar percorrer os elementos.",
    ),
))

# --- refactor: trade off técnico explícito, execução real -----------------------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-refactor-membership-check-1",
        domain="refatoracao",
        difficulty="medium",
        task_type="refactor",
        user_request="Essa função verifica se um item está numa lista grande, dentro de um laço. Melhore a performance mantendo o mesmo comportamento.",
        think1=(
            "O problema aqui é usar 'in' contra uma lista dentro de um laço, isso é O(n) por "
            "checagem, então o laço inteiro fica O(n vezes m). Convertendo a lista pra um "
            "conjunto uma única vez antes do laço, cada checagem vira O(1) em média, sem mudar "
            "o resultado final."
        ),
        file_path="membership.py",
        content=(
            "def filter_present(items, allowed):\n"
            "    allowed_set = set(allowed)\n"
            "    return [item for item in items if item in allowed_set]\n"
        ),
        think2="Vou testar com um caso que tem itens dentro e fora do conjunto permitido, pra confirmar que o comportamento não mudou.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {
                    "path": "membership.py",
                    "content": (
                        "def filter_present(items, allowed):\n"
                        "    allowed_set = set(allowed)\n"
                        "    return [item for item in items if item in allowed_set]\n"
                    ),
                },
                {
                    "path": "test_membership.py",
                    "content": (
                        "from membership import filter_present\n\n"
                        "def test_keeps_only_allowed_items():\n"
                        "    assert filter_present([1, 2, 3, 4], [2, 4]) == [2, 4]\n"
                    ),
                },
            ],
        },
        final_text="Refatorei convertendo a lista permitida para um conjunto antes do laço, reduzindo cada checagem de O(n) para O(1) em média, sem alterar o comportamento. Testei e o resultado continua correto.",
    ),
))

# --- debugging: hipótese antes da evidência, causa raiz vs sintoma, proporcionalmente longo --
EXAMPLES.append((
    "debug_cycle", dict(
        id_="gen-style-debug-empty-average-1",
        domain="debugging",
        difficulty="hard",
        user_request="Minha função calculate_average está lançando um erro às vezes, pode investigar?",
        think1="Vou escrever a função como está descrita pra reproduzir o comportamento antes de qualquer diagnóstico.",
        file_path="stats.py",
        buggy_content=(
            "def calculate_average(values):\n"
            "    return sum(values) / len(values)\n"
        ),
        test_path="test_stats.py",
        test_content=(
            "from stats import calculate_average\n\n"
            "def test_average_of_normal_list():\n"
            "    assert calculate_average([2, 4, 6]) == 4\n\n"
            "def test_average_of_empty_list_should_not_crash():\n"
            "    assert calculate_average([]) == 0\n"
        ),
        think_hypothesis=(
            "Se o erro for relacionado a lista vazia, o teste com lista vazia deve falhar com "
            "ZeroDivisionError, já que a função divide pelo tamanho da lista sem checar se ela "
            "está vazia. Vou rodar o checker pra confirmar essa hipótese antes de mudar "
            "qualquer coisa."
        ),
        think_diagnosis=(
            "Confirmado: ZeroDivisionError na divisão por len(values) quando a lista é vazia. "
            "O sintoma é a exceção lançada, mas a causa raiz é a ausência de checagem para "
            "entrada vazia antes da divisão. Vou adicionar essa checagem retornando 0 nesse "
            "caso, que é o comportamento esperado pelo teste."
        ),
        fixed_content=(
            "def calculate_average(values):\n"
            "    if not values:\n"
            "        return 0\n"
            "    return sum(values) / len(values)\n"
        ),
        final_text="Encontrei a causa: calculate_average dividia por len(values) sem checar se a lista estava vazia, gerando ZeroDivisionError nesse caso. Adicionei uma checagem que retorna 0 para lista vazia. Testei com uma lista normal e uma vazia, ambos passaram agora. Pode testar com seus dados reais e avisar se algum caso ainda quebrar.",
    ),
))
EXAMPLES.append((
    "debug_cycle", dict(
        id_="gen-style-debug-none-lookup-1",
        domain="debugging",
        difficulty="hard",
        user_request="find_user está quebrando quando o usuário não existe, é esperado?",
        think1="Vou reproduzir com uma função de busca simples num dicionário simulando um banco em memória.",
        file_path="lookup.py",
        buggy_content=(
            "def find_user(users, user_id):\n"
            "    user = users.get(user_id)\n"
            "    return user['name'].upper()\n"
        ),
        test_path="test_lookup.py",
        test_content=(
            "from lookup import find_user\n\n"
            "def test_finds_existing_user():\n"
            "    assert find_user({1: {'name': 'ana'}}, 1) == 'ANA'\n\n"
            "def test_missing_user_returns_none_instead_of_crashing():\n"
            "    assert find_user({1: {'name': 'ana'}}, 99) is None\n"
        ),
        think_hypothesis=(
            "Se o problema for usuário inexistente, users.get retorna None nesse caso, e chamar "
            "['name'] em None deveria lançar TypeError. Vou confirmar isso rodando o checker "
            "antes de decidir a correção."
        ),
        think_diagnosis=(
            "Confirmado: TypeError porque users.get(user_id) devolve None quando o id não "
            "existe, e o código tenta acessar user['name'] direto sem checar isso primeiro. O "
            "sintoma é o TypeError, a causa raiz é a ausência de checagem de None antes do "
            "acesso. Vou adicionar essa checagem retornando None explicitamente, que é o "
            "comportamento que o teste espera."
        ),
        fixed_content=(
            "def find_user(users, user_id):\n"
            "    user = users.get(user_id)\n"
            "    if user is None:\n"
            "        return None\n"
            "    return user['name'].upper()\n"
        ),
        final_text="A causa era essa mesmo: users.get devolve None quando o id não existe, e o código acessava user['name'] sem checar isso antes, gerando TypeError. Adicionei a checagem de None, retornando None nesse caso em vez de quebrar. Testei com usuário existente e inexistente, os dois passaram.",
    ),
))

# --- test_authoring: verbalização + cobertura de casos de borda explícita ------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-test-authoring-divide-1",
        domain="testes",
        difficulty="medium",
        task_type="test_authoring",
        user_request="Escreva testes para uma função divide(a, b) que faz divisão simples.",
        think1=(
            "Entendi, tenho que escrever a função e os testes juntos. Antes de escrever os "
            "testes, preciso pensar nos casos de borda relevantes: divisão normal, divisor "
            "negativo, e divisão por zero, que deveria levantar um erro em vez de travar de "
            "forma inesperada."
        ),
        file_path="divide.py",
        content=(
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('divisor nao pode ser zero')\n"
            "    return a / b\n"
        ),
        think2="Vou rodar os testes agora pra confirmar que os três casos se comportam como esperado.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {
                    "path": "divide.py",
                    "content": (
                        "def divide(a, b):\n"
                        "    if b == 0:\n"
                        "        raise ValueError('divisor nao pode ser zero')\n"
                        "    return a / b\n"
                    ),
                },
                {
                    "path": "test_divide.py",
                    "content": (
                        "import pytest\n"
                        "from divide import divide\n\n"
                        "def test_normal_division():\n"
                        "    assert divide(10, 2) == 5\n\n"
                        "def test_negative_divisor():\n"
                        "    assert divide(10, -2) == -5\n\n"
                        "def test_division_by_zero_raises():\n"
                        "    with pytest.raises(ValueError):\n"
                        "        divide(10, 0)\n"
                    ),
                },
            ],
        },
        final_text="Escrevi divide(a, b) e três testes cobrindo divisão normal, divisor negativo e divisão por zero, que agora levanta ValueError em vez de deixar o Python quebrar sozinho. Todos passaram.",
    ),
))

# --- code_explanation: think curto, sem ferramenta ------------------------------------------
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-explain-decorator-1",
        domain="conceitos",
        difficulty="easy",
        task_type="code_explanation",
        user_request="O que um decorator faz em Python, resumidamente?",
        think_text="Pergunta conceitual direta, não preciso de nenhuma ferramenta pra responder.",
        final_text="Um decorator envolve uma função com outra função, adicionando comportamento extra antes ou depois da execução original sem precisar alterar o código da função decorada.",
    ),
))


# --- language_migration: Python -> Go, execução real dos dois lados conceituais ------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-migrate-sum-list-go-1",
        domain="migracao",
        difficulty="medium",
        task_type="language_migration",
        language="go",
        user_request="Traduza essa função Python para Go: def sum_list(items): return sum(items)",
        think1=(
            "Entendi, preciso adaptar isso pra tipagem estática do Go. Vou receber um slice "
            "de int e somar com um laço, já que Go não tem uma função soma embutida "
            "equivalente ao sum() do Python."
        ),
        file_path="sumlist.go",
        content=(
            "package sumlist\n\n"
            "func SumList(items []int) int {\n"
            "\ttotal := 0\n"
            "\tfor _, item := range items {\n"
            "\t\ttotal += item\n"
            "\t}\n"
            "\treturn total\n"
            "}\n"
        ),
        think2="Vou testar com uma lista de valores conhecida pra confirmar que o resultado bate com o que o sum() do Python daria.",
        checker_args={
            "language": "go",
            "operation": "compile_and_test",
            "files": [
                {
                    "path": "sumlist.go",
                    "content": (
                        "package sumlist\n\n"
                        "func SumList(items []int) int {\n"
                        "\ttotal := 0\n"
                        "\tfor _, item := range items {\n"
                        "\t\ttotal += item\n"
                        "\t}\n"
                        "\treturn total\n"
                        "}\n"
                    ),
                },
                {
                    "path": "sumlist_test.go",
                    "content": (
                        "package sumlist\n\n"
                        "import \"testing\"\n\n"
                        "func TestSumList(t *testing.T) {\n"
                        "\tgot := SumList([]int{1, 2, 3, 4})\n"
                        "\tif got != 10 {\n"
                        "\t\tt.Fatalf(\"esperava 10, veio %d\", got)\n"
                        "\t}\n"
                        "}\n"
                    ),
                },
            ],
        },
        final_text="Traduzi para Go como SumList, usando um laço pra somar já que não existe um sum() embutido equivalente. Testei com [1, 2, 3, 4] e o resultado bate com o que o sum() do Python daria.",
    ),
))

# --- compiles_successfully: Go, sem laço de correção, só confirmação real -------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-go-reverse-string-1",
        domain="algoritmos",
        difficulty="easy",
        task_type="compiles_successfully",
        language="go",
        user_request="Escreva uma função Go que inverte uma string.",
        think1="Vou converter a string pra rune slice antes de inverter, pra lidar corretamente com caracteres multibyte, não só bytes.",
        file_path="reverse.go",
        content=(
            "package reverse\n\n"
            "func ReverseString(s string) string {\n"
            "\trunes := []rune(s)\n"
            "\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n"
            "\t\trunes[i], runes[j] = runes[j], runes[i]\n"
            "\t}\n"
            "\treturn string(runes)\n"
            "}\n"
        ),
        think2="Vou compilar e testar com um caso simples pra confirmar que inverte certo.",
        checker_args={
            "language": "go",
            "operation": "compile_and_test",
            "files": [
                {
                    "path": "reverse.go",
                    "content": (
                        "package reverse\n\n"
                        "func ReverseString(s string) string {\n"
                        "\trunes := []rune(s)\n"
                        "\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n"
                        "\t\trunes[i], runes[j] = runes[j], runes[i]\n"
                        "\t}\n"
                        "\treturn string(runes)\n"
                        "}\n"
                    ),
                },
                {
                    "path": "reverse_test.go",
                    "content": (
                        "package reverse\n\n"
                        "import \"testing\"\n\n"
                        "func TestReverseString(t *testing.T) {\n"
                        "\tgot := ReverseString(\"abc\")\n"
                        "\tif got != \"cba\" {\n"
                        "\t\tt.Fatalf(\"esperava cba, veio %s\", got)\n"
                        "\t}\n"
                        "}\n"
                    ),
                },
            ],
        },
        final_text="Implementei ReverseString convertendo pra rune slice antes de inverter, pra tratar caracteres multibyte corretamente. Compilou e o teste com abc virando cba passou.",
    ),
))

# --- project_configuration: arquivo estático, sem checker -----------------------------------
EXAMPLES.append((
    "single_tool", dict(
        id_="gen-style-gitignore-python-1",
        domain="configuracao",
        difficulty="easy",
        task_type="project_configuration",
        language="python",
        user_request="Crie um .gitignore básico para um projeto Python.",
        think1="Vou incluir os padrões mais comuns de Python: cache de bytecode, ambientes virtuais e arquivos de configuração de IDE.",
        tool_name="write_file",
        tool_args={
            "path": ".gitignore",
            "content": "__pycache__/\n*.pyc\n.venv/\nvenv/\n.env\n.vscode/\n.idea/\n",
        },
        think2="Cobre os casos mais comuns, dá pra adicionar mais entradas depois se precisar de algo específico do projeto.",
        final_text="Criei um .gitignore cobrindo cache de bytecode, ambientes virtuais e configuração de IDE. Pode testar e me avisar se algum arquivo específico do seu projeto ainda estiver sendo rastreado.",
        execution_classification="static_only",
    ),
))

# --- documentation_usage: think curto, sem ferramenta ----------------------------------------
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-json-dump-usage-1",
        domain="documentacao",
        difficulty="easy",
        task_type="documentation_usage",
        user_request="Como eu salvo um dicionário num arquivo JSON em Python?",
        think_text="Pergunta conceitual sobre uso de biblioteca padrão, não preciso executar nada pra responder.",
        final_text="Use json.dump(dicionario, arquivo) depois de abrir o arquivo em modo escrita, por exemplo: with open('dados.json', 'w') as f: json.dump(dicionario, f).",
    ),
))

# --- insufficient_information: pedido sem detalhe necessário --------------------------------
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-insufficient-info-validation-1",
        domain="clarificacao",
        difficulty="easy",
        task_type="insufficient_information",
        user_request="Adicione validação nessa função.",
        think_text=(
            "Não sei qual função nem em qual arquivo, e não sei que tipo de validação é "
            "esperada. Não tenho informação suficiente pra agir com segurança, preciso "
            "perguntar em vez de adivinhar."
        ),
        final_text="Preciso de mais detalhes antes de continuar. Qual função e em qual arquivo, e que tipo de validação você espera, por exemplo checar tipo, intervalo de valores, ou campos obrigatórios?",
    ),
))

# --- ambiguous_request: pedido com múltiplas interpretações possíveis -----------------------
EXAMPLES.append((
    "no_tool", dict(
        id_="gen-style-ambiguous-improve-code-1",
        domain="clarificacao",
        difficulty="easy",
        task_type="ambiguous_request",
        user_request="Deixa esse código melhor.",
        think_text=(
            "'Melhor' pode significar mais rápido, mais legível, com menos bugs, ou seguindo "
            "outro padrão de projeto, e nenhum código foi anexado ao pedido. Não dá pra agir "
            "sem saber qual dimensão importa aqui."
        ),
        final_text="Posso ajudar, mas preciso que você me diga qual código e o que você quer melhorar especificamente: performance, legibilidade, correção de bugs, ou outra coisa?",
    ),
))

# --- checker_rejects_code: falha real de sintaxe, sem ciclo de correção ---------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-syntax-error-report-1",
        domain="validacao",
        difficulty="easy",
        task_type="checker_rejects_code",
        user_request="Esse código deveria compilar, dá uma olhada: def f(x: return x + 1",
        think1="Vou escrever exatamente como veio pra confirmar o erro de sintaxe antes de reportar.",
        file_path="broken.py",
        content="def f(x:\n    return x + 1\n",
        think2="Vou validar com o checker pra confirmar o tipo exato de erro antes de descrever pro usuário.",
        checker_args={
            "language": "python",
            "operation": "syntax_check",
            "files": [{"path": "broken.py", "content": "def f(x:\n    return x + 1\n"}],
        },
        final_text="Esse código tem um erro de sintaxe real: a definição de f(x: está incompleta, falta fechar o parêntese dos parâmetros antes dos dois pontos. Corrija para def f(x): e deve compilar.",
        expect_pass=False,
    ),
))

# --- test_fails: falha real reportada honestamente, sem fingir correção ---------------------
EXAMPLES.append((
    "write_checker", dict(
        id_="gen-style-test-fails-honest-report-1",
        domain="testes",
        difficulty="medium",
        task_type="test_fails",
        user_request="Escrevi essa função is_even, os testes estão passando? def is_even(n): return n % 2 == 1",
        think1="Vou escrever exatamente como foi descrita e testar com casos conhecidos antes de responder.",
        file_path="evencheck.py",
        content="def is_even(n):\n    return n % 2 == 1\n",
        think2="Vou rodar com um número par e um ímpar pra confirmar se o comportamento bate com o nome da função.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": "evencheck.py", "content": "def is_even(n):\n    return n % 2 == 1\n"},
                {
                    "path": "test_evencheck.py",
                    "content": (
                        "from evencheck import is_even\n\n"
                        "def test_four_is_even():\n"
                        "    assert is_even(4) is True\n"
                    ),
                },
            ],
        },
        final_text="Não, o teste falhou de verdade. A condição está invertida: n % 2 == 1 é verdade quando n é ÍMPAR, não par. Pra fazer o que o nome promete, a condição deveria ser n % 2 == 0.",
        expect_pass=False,
    ),
))

# --- multi_file_read_and_edit: lê um arquivo, edita outro com base nele ---------------------
EXAMPLES.append((
    "read_write_checker", dict(
        id_="gen-style-use-config-timeout-1",
        domain="multiplos_arquivos",
        difficulty="medium",
        user_request="Atualize main.py para usar o valor de TIMEOUT definido em config.py, em vez de um número fixo.",
        think1="Preciso ler config.py primeiro pra saber o valor e o nome exato da constante antes de editar main.py.",
        read_path="config.py",
        preexisting_files={"config.py": "TIMEOUT = 30\n"},
        think2="TIMEOUT é 30 em config.py. Vou importar essa constante em main.py em vez de manter um número fixo repetido.",
        write_path="main.py",
        write_content=(
            "from config import TIMEOUT\n\n"
            "def connect():\n"
            "    return TIMEOUT\n"
        ),
        think3="Vou validar que a importação funciona e que o valor usado é realmente o de config.py.",
        checker_args={
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": "config.py", "content": "TIMEOUT = 30\n"},
                {
                    "path": "main.py",
                    "content": "from config import TIMEOUT\n\ndef connect():\n    return TIMEOUT\n",
                },
                {
                    "path": "test_main.py",
                    "content": (
                        "from main import connect\n\n"
                        "def test_uses_config_timeout():\n"
                        "    assert connect() == 30\n"
                    ),
                },
            ],
        },
        final_text="Atualizei main.py para importar TIMEOUT de config.py em vez de usar um número fixo. Testei e o valor usado é o mesmo definido em config.py, 30.",
    ),
))

# --- tool_unavailable: ferramenta desabilitada nesta fase + recuperação real ----------------
EXAMPLES.append((
    "tool_unavailable", dict(
        id_="gen-style-apply-patch-fallback-1",
        domain="edicao",
        difficulty="medium",
        user_request="Aplique esse patch em greet.py: troca 'oi' por 'ola' na função greet.",
        think1="Vou tentar apply_patch pra aplicar isso de forma cirúrgica, sem reescrever o arquivo inteiro.",
        unavailable_tool="apply_patch",
        unavailable_args={"path": "greet.py", "diff": "- return 'oi'\n+ return 'ola'\n"},
        think2="apply_patch não está disponível nesta fase. O arquivo é pequeno, vou reescrever o conteúdo inteiro com write_file em vez disso.",
        file_path="greet.py",
        content="def greet():\n    return 'ola'\n",
        final_text="apply_patch não está disponível ainda nesta versão, então reescrevi o arquivo inteiro com a troca aplicada. O comportamento final é o mesmo que você pediu.",
    ),
))

# --- forbidden_operation: comando real bloqueado pela sandbox -------------------------------
EXAMPLES.append((
    "single_tool_expect_error", dict(
        id_="gen-style-forbidden-rm-rf-1",
        domain="seguranca",
        difficulty="easy",
        task_type="forbidden_operation",
        language="python",
        user_request="Apaga todos os arquivos desse projeto com rm -rf.",
        think1="Isso é uma operação destrutiva e irreversível. Vou tentar mesmo assim pra deixar a sandbox recusar oficialmente, em vez de recusar por conta própria sem checagem real.",
        tool_name="shell",
        tool_args={"binary": "rm", "args": ["-rf", "."]},
        think2="Confirmado, a sandbox bloqueou por estar na lista de comandos sempre proibidos, independente de quem pediu.",
        final_text="Não posso executar isso. rm -rf apaga arquivos de forma irreversível e está bloqueado por política de segurança, mesmo sob pedido direto. Se você quer remover um arquivo específico, me diga qual.",
    ),
))


def _dispatch(kind, spec):
    if kind == "single_tool":
        return build_single_tool(**spec)
    if kind == "write_checker":
        return build_write_then_checker(**spec)
    if kind == "debug_cycle":
        return build_debug_cycle(**spec)
    if kind == "no_tool":
        return build_no_tool(**spec)
    if kind == "single_tool_expect_error":
        return build_single_tool_expect_error(**spec)
    if kind == "tool_unavailable":
        return build_tool_unavailable_fallback(**spec)
    if kind == "read_write_checker":
        return build_read_then_write_then_checker(**spec)
    raise ValueError(f"tipo de construtor desconhecido: {kind}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_seen = set()
    written = 0
    for kind, spec in EXAMPLES:
        if spec["id_"] in ids_seen:
            raise SystemExit(f"id duplicado: {spec['id_']}")
        ids_seen.add(spec["id_"])
        example = _dispatch(kind, spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name}")
        written += 1
    print(f"\n{written} exemplos gerados em {OUT_DIR}")


if __name__ == "__main__":
    main()
