#!/usr/bin/env python
"""Lote base do Gap 6 (docs/plan_dataset_expansion_own_bug_diagnosis.md): quando o código
recém-escrito pelo PRÓPRIO modelo tem um `NameError` real (referência a um nome nunca
definido ou nunca importado), o diagnóstico precisa acertar a causa exata (definição faltando
vs. import faltando) e a retentativa precisa aplicar uma correção real e verificável — nunca
reenviar o mesmo payload dizendo que corrigiu algo.

Achado real: `TodoList.add_task` chamava `Task(name)` sem `Task` nunca ter sido definida.
`NameError` real. O `<think>` seguinte concluiu "esqueceu de importar a classe Task" — sem
sentido, não existe módulo `Task` para importar, faltava DEFINIR a classe. Pior: a
retentativa reenviou o mesmo payload, sem nenhuma correção real, bloqueada pela deduplicação
de chamadas do harness (`seen_calls` -> `MAX_STEPS_EXCEEDED`).

Cada exemplo: `<think>` -> `write_file` real (código com `NameError` genuíno) -> `<think>` ->
`checker` real -> falha real (`NameError`, apontado pelo próprio traceback) -> `<think>`
diagnosticando CORRETAMENTE o nome exato que falta e se é definição ou import faltando ->
`write_file`/`checker` real de novo, com uma correção VERIFICAVELMENTE diferente do payload
anterior (asserção no próprio script) -> passa -> `<final>`.

Uso:
    python scripts/gen_own_bug_diagnosis_pilot.py
"""

from __future__ import annotations

import json
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
    id_, domain, difficulty, user_request,
    think1, file_path, buggy_content, think_before_first_checker, test_path, test_content,
    think2, fixed_content,
    final_text,
) -> dict:
    assert buggy_content != fixed_content, (
        f"payload de correção idêntico ao da falha em {id_} — a retentativa precisa aplicar "
        "uma mudança real, não reenviar o mesmo conteúdo"
    )

    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think1}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": buggy_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        traj.append_raw(f"<think>{think_before_first_checker}</think>")
        first_checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": buggy_content},
                {"path": test_path, "content": test_content},
            ],
        }
        first_result = _tool_call(traj, sandbox, "checker", first_checker_args)
        assert not first_result.passed, f"esperava NameError real em {id_}: {first_result.to_json()}"
        assert "NameError" in json.dumps(first_result.to_json()), (
            f"esperava NameError especificamente no traceback em {id_}: {first_result.to_json()}"
        )

        traj.append_raw(f"<think>{think2}</think>")
        second_checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": fixed_content},
                {"path": test_path, "content": test_content},
            ],
        }
        second_result = _tool_call(traj, sandbox, "checker", second_checker_args)
        assert second_result.passed, f"checker (corrigido) falhou inesperadamente em {id_}: {second_result.to_json()}"

        traj.append_raw(f"<final>{final_text}</final>")
    finally:
        sandbox.cleanup()

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "python",
            "difficulty": difficulty,
            "tools_used": ["write_file", "checker"],
            "num_steps": 4,
            "task_type": "own_bug_diagnosis",
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


EXAMPLES = [
    dict(
        id_="gen-ownbug-todo-list",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Crie uma classe TodoList com add_task(name) e complete_task(name). "
            "complete_task deve levantar TaskNotFoundError se a tarefa não existir na lista. "
            "Valide com um teste real cobrindo sucesso e o caso de erro."
        ),
        think1=(
            "Vou implementar TodoList com add_task e complete_task, usando um objeto Task "
            "pra guardar o nome e o status de conclusão de cada tarefa."
        ),
        file_path="todo_list.py",
        buggy_content=(
            "class TaskNotFoundError(Exception):\n"
            "    pass\n\n\n"
            "class TodoList:\n"
            "    def __init__(self):\n"
            "        self._tasks = []\n\n"
            "    def add_task(self, name):\n"
            "        self._tasks.append(Task(name))\n\n"
            "    def complete_task(self, name):\n"
            "        for task in self._tasks:\n"
            "            if task.name == name:\n"
            "                task.completed = True\n"
            "                return task\n"
            "        raise TaskNotFoundError(name)\n"
        ),
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_todo_list.py",
        test_content=(
            "import pytest\n\n"
            "from todo_list import TodoList, TaskNotFoundError\n\n\n"
            "def test_todo_list():\n"
            "    todo = TodoList()\n"
            "    todo.add_task('comprar pao')\n"
            "    completed = todo.complete_task('comprar pao')\n"
            "    assert completed.completed is True\n"
            "    with pytest.raises(TaskNotFoundError):\n"
            "        todo.complete_task('inexistente')\n"
        ),
        think2=(
            "O erro é 'NameError: name Task is not defined' — a classe Task nunca foi "
            "definida em lugar nenhum deste arquivo. Isso não é um problema de import (não "
            "existe nenhum módulo 'Task' para importar); é falta de DEFINIÇÃO. Vou definir a "
            "classe Task antes de TodoList."
        ),
        fixed_content=(
            "class TaskNotFoundError(Exception):\n"
            "    pass\n\n\n"
            "class Task:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "        self.completed = False\n\n\n"
            "class TodoList:\n"
            "    def __init__(self):\n"
            "        self._tasks = []\n\n"
            "    def add_task(self, name):\n"
            "        self._tasks.append(Task(name))\n\n"
            "    def complete_task(self, name):\n"
            "        for task in self._tasks:\n"
            "            if task.name == name:\n"
            "                task.completed = True\n"
            "                return task\n"
            "        raise TaskNotFoundError(name)\n"
        ),
        final_text=(
            "Implementei TodoList com add_task e complete_task. O primeiro teste falhou com "
            "'NameError: name Task is not defined' — eu tinha usado Task(name) sem nunca "
            "definir essa classe. Defini Task com name e completed, e agora a validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ownbug-event-bus",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Implemente uma classe EventBus com subscribe(callback) e publish(event). "
            "publish deve chamar todos os callbacks inscritos com o evento. Valide com um "
            "teste real."
        ),
        think1=(
            "Vou implementar EventBus com subscribe e publish, guardando cada callback num "
            "objeto Subscriber pra facilitar extensões futuras (como filtro por tipo de "
            "evento)."
        ),
        file_path="event_bus.py",
        buggy_content=(
            "class EventBus:\n"
            "    def __init__(self):\n"
            "        self._subs = []\n\n"
            "    def subscribe(self, callback):\n"
            "        self._subs.append(Subscriber(callback))\n\n"
            "    def publish(self, event):\n"
            "        for sub in self._subs:\n"
            "            sub.callback(event)\n"
        ),
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_event_bus.py",
        test_content=(
            "from event_bus import EventBus\n\n\n"
            "def test_event_bus():\n"
            "    received = []\n"
            "    bus = EventBus()\n"
            "    bus.subscribe(lambda e: received.append(e))\n"
            "    bus.publish('hello')\n"
            "    assert received == ['hello']\n"
        ),
        think2=(
            "O erro é 'NameError: name Subscriber is not defined' — a classe Subscriber "
            "nunca foi definida em lugar nenhum deste arquivo. Não é um problema de import; "
            "é falta de DEFINIÇÃO. Vou definir Subscriber guardando o callback."
        ),
        fixed_content=(
            "class Subscriber:\n"
            "    def __init__(self, callback):\n"
            "        self.callback = callback\n\n\n"
            "class EventBus:\n"
            "    def __init__(self):\n"
            "        self._subs = []\n\n"
            "    def subscribe(self, callback):\n"
            "        self._subs.append(Subscriber(callback))\n\n"
            "    def publish(self, event):\n"
            "        for sub in self._subs:\n"
            "            sub.callback(event)\n"
        ),
        final_text=(
            "Implementei EventBus com subscribe e publish. O primeiro teste falhou com "
            "'NameError: name Subscriber is not defined' — eu tinha usado Subscriber(callback) "
            "sem nunca definir essa classe. Defini Subscriber guardando o callback, e agora a "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ownbug-parse-duration",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Escreva parse_duration(text) que converte um texto tipo '5s', '2m' ou '1h' para "
            "segundos. Valide com um teste real cobrindo as três unidades."
        ),
        think1=(
            "Vou implementar parse_duration extraindo o número e a unidade do texto, e "
            "convertendo para segundos com uma função auxiliar de mapeamento de unidade."
        ),
        file_path="duration.py",
        buggy_content=(
            "def parse_duration(text):\n"
            "    number = int(text[:-1])\n"
            "    unit = text[-1]\n"
            "    return number * _parse_unit(unit)\n"
        ),
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_duration.py",
        test_content=(
            "from duration import parse_duration\n\n\n"
            "def test_parse_duration():\n"
            "    assert parse_duration('5s') == 5\n"
            "    assert parse_duration('2m') == 120\n"
            "    assert parse_duration('1h') == 3600\n"
        ),
        think2=(
            "O erro é 'NameError: name _parse_unit is not defined' — eu referenciei essa "
            "função auxiliar mas nunca cheguei a escrevê-la. Não é um import faltando (não é "
            "de nenhuma biblioteca); é uma função que esqueci de definir. Vou escrever "
            "_parse_unit com o mapeamento de unidades."
        ),
        fixed_content=(
            "def _parse_unit(unit):\n"
            "    return {'s': 1, 'm': 60, 'h': 3600}[unit]\n\n\n"
            "def parse_duration(text):\n"
            "    number = int(text[:-1])\n"
            "    unit = text[-1]\n"
            "    return number * _parse_unit(unit)\n"
        ),
        final_text=(
            "Implementei parse_duration(text). O primeiro teste falhou com 'NameError: name "
            "_parse_unit is not defined' — eu tinha referenciado essa função auxiliar sem "
            "nunca escrevê-la. Defini _parse_unit com o mapeamento de unidades (s/m/h), e "
            "agora a validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-ownbug-to-cents",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Escreva to_cents(value) que converte um valor monetário em texto (ex.: '1.50') "
            "para centavos como inteiro, usando Decimal para evitar erro de ponto flutuante. "
            "Valide com um teste real."
        ),
        think1=(
            "Vou implementar to_cents usando Decimal para converter o texto com precisão, "
            "evitando os erros de arredondamento típicos de float com valores monetários."
        ),
        file_path="money.py",
        buggy_content="def to_cents(value):\n    return int(Decimal(value) * 100)\n",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_money.py",
        test_content=(
            "from money import to_cents\n\n\n"
            "def test_to_cents():\n"
            "    assert to_cents('1.50') == 150\n"
            "    assert to_cents('0.99') == 99\n"
        ),
        think2=(
            "O erro é 'NameError: name Decimal is not defined' — desta vez é o oposto do "
            "padrão comum: Decimal já existe na biblioteca padrão, não precisa ser definida, "
            "só faltou o import de decimal. Vou adicionar 'from decimal import Decimal'."
        ),
        fixed_content=(
            "from decimal import Decimal\n\n\n"
            "def to_cents(value):\n"
            "    return int(Decimal(value) * 100)\n"
        ),
        final_text=(
            "Implementei to_cents(value) usando Decimal. O primeiro teste falhou com "
            "'NameError: name Decimal is not defined' — eu tinha esquecido o import de "
            "decimal. Adicionei 'from decimal import Decimal', e agora a validação confirma "
            "que compila e o teste passa."
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
        print(f"OK: {out_path.name} (NameError real, diagnóstico correto, correção real diferente)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
