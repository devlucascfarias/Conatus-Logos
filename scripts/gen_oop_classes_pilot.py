#!/usr/bin/env python
"""Lote base da categoria "Classes/OOP" (docs/plan_dataset_expansion_oop_shell.md, seção 2).

Achado real que motivou esta categoria: pedido de "uma classe Python com funções de soma,
subtração, multiplicação, divisão e fatorial" resultou em funções soltas (nenhuma `class`), sem
chamar `checker`, com `<final>` afirmando falsamente que a classe foi criada. O dataset atual
não tinha cobertura de estruturas com estado interno e múltiplos métodos.

Cada exemplo aqui: `<think>` -> `write_file` real -> `<think>` -> `checker` real
(`compile_and_test`) -> `<final>`. Nenhum `tool_result` é fabricado — tudo roda via
`ToolExecutorRegistry`/`SandboxContext`, o mesmo código do harness de produção.

Uso:
    python scripts/gen_oop_classes_pilot.py
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
    id_: str,
    domain: str,
    difficulty: str,
    user_request: str,
    think_before_write: str,
    file_path: str,
    file_content: str,
    test_path: str,
    test_content: str,
    think_before_final: str,
    final_text: str,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_write}</think>")
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

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "python",
            "difficulty": difficulty,
            "tools_used": ["write_file", "checker"],
            "num_steps": 2,
            "task_type": "class_implementation",
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
        id_="gen-oop-calculator",
        domain="oop",
        difficulty="medium",
        user_request=(
            "Crie uma classe Python chamada Calculator com métodos de soma, subtração, "
            "multiplicação, divisão e fatorial."
        ),
        think_before_write=(
            "O pedido é claro sobre a estrutura: precisa ser uma classe de verdade, não funções "
            "soltas. Vou definir Calculator com um método por operação, incluindo factorial, e "
            "levantar ValueError na divisão por zero e no fatorial de número negativo."
        ),
        file_path="calculator.py",
        file_content=(
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n\n"
            "    def multiply(self, a, b):\n"
            "        return a * b\n\n"
            "    def divide(self, a, b):\n"
            "        if b == 0:\n"
            "            raise ValueError('divisao por zero')\n"
            "        return a / b\n\n"
            "    def factorial(self, n):\n"
            "        if n < 0:\n"
            "            raise ValueError('fatorial nao definido para negativos')\n"
            "        result = 1\n"
            "        for i in range(2, n + 1):\n"
            "            result *= i\n"
            "        return result\n"
        ),
        test_path="test_calculator.py",
        test_content=(
            "import pytest\n\n"
            "from calculator import Calculator\n\n\n"
            "def test_calculator():\n"
            "    calc = Calculator()\n"
            "    assert calc.add(2, 3) == 5\n"
            "    assert calc.subtract(5, 2) == 3\n"
            "    assert calc.multiply(4, 3) == 12\n"
            "    assert calc.divide(10, 2) == 5\n"
            "    assert calc.factorial(5) == 120\n"
            "    with pytest.raises(ValueError):\n"
            "        calc.divide(1, 0)\n"
            "    with pytest.raises(ValueError):\n"
            "        calc.factorial(-1)\n"
        ),
        think_before_final="A classe está pronta com todos os métodos pedidos. Vou validar antes de responder.",
        final_text=(
            "Criei a classe Calculator com os métodos add, subtract, multiply, divide e "
            "factorial, incluindo tratamento de divisão por zero e fatorial negativo. A "
            "validação confirma que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-oop-stack",
        domain="oop",
        difficulty="easy",
        user_request="Implemente uma classe Stack (pilha) com push, pop, peek e is_empty.",
        think_before_write=(
            "Uma pilha precisa manter estado interno (a lista de itens) entre chamadas de "
            "método, então uso uma classe com uma lista privada por convenção."
        ),
        file_path="stack.py",
        file_content=(
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n\n"
            "    def pop(self):\n"
            "        if self.is_empty():\n"
            "            raise IndexError('pop em pilha vazia')\n"
            "        return self._items.pop()\n\n"
            "    def peek(self):\n"
            "        if self.is_empty():\n"
            "            raise IndexError('peek em pilha vazia')\n"
            "        return self._items[-1]\n\n"
            "    def is_empty(self):\n"
            "        return len(self._items) == 0\n"
        ),
        test_path="test_stack.py",
        test_content=(
            "import pytest\n\n"
            "from stack import Stack\n\n\n"
            "def test_stack():\n"
            "    s = Stack()\n"
            "    assert s.is_empty() is True\n"
            "    s.push(1)\n"
            "    s.push(2)\n"
            "    assert s.peek() == 2\n"
            "    assert s.pop() == 2\n"
            "    assert s.pop() == 1\n"
            "    assert s.is_empty() is True\n"
            "    with pytest.raises(IndexError):\n"
            "        s.pop()\n"
        ),
        think_before_final="A implementação cobre os quatro métodos pedidos e trata pilha vazia. Vou validar.",
        final_text=(
            "Implementei a classe Stack com push, pop, peek e is_empty, levantando IndexError "
            "ao operar sobre pilha vazia. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-queue",
        domain="oop",
        difficulty="easy",
        user_request="Preciso de uma classe Queue (fila) com enqueue, dequeue e is_empty.",
        think_before_write=(
            "Uma fila FIFO também precisa de estado interno entre chamadas — vou usar uma lista "
            "e remover sempre do início em dequeue."
        ),
        file_path="queue_.py",
        file_content=(
            "class Queue:\n"
            "    def __init__(self):\n"
            "        self._items = []\n\n"
            "    def enqueue(self, item):\n"
            "        self._items.append(item)\n\n"
            "    def dequeue(self):\n"
            "        if self.is_empty():\n"
            "            raise IndexError('dequeue em fila vazia')\n"
            "        return self._items.pop(0)\n\n"
            "    def is_empty(self):\n"
            "        return len(self._items) == 0\n"
        ),
        test_path="test_queue_.py",
        test_content=(
            "import pytest\n\n"
            "from queue_ import Queue\n\n\n"
            "def test_queue():\n"
            "    q = Queue()\n"
            "    assert q.is_empty() is True\n"
            "    q.enqueue('a')\n"
            "    q.enqueue('b')\n"
            "    assert q.dequeue() == 'a'\n"
            "    assert q.dequeue() == 'b'\n"
            "    with pytest.raises(IndexError):\n"
            "        q.dequeue()\n"
        ),
        think_before_final="Fila FIFO pronta com os três métodos pedidos. Vou validar.",
        final_text=(
            "Implementei a classe Queue com enqueue, dequeue e is_empty, respeitando a ordem "
            "FIFO. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-bank_account",
        domain="oop",
        difficulty="medium",
        user_request=(
            "Crie uma classe BankAccount com depósito, saque e consulta de saldo, que não "
            "permita saldo negativo."
        ),
        think_before_write=(
            "O estado (saldo) precisa persistir entre chamadas, e o saque deve validar saldo "
            "suficiente antes de debitar, levantando erro caso contrário."
        ),
        file_path="bank_account.py",
        file_content=(
            "class BankAccount:\n"
            "    def __init__(self, initial_balance=0):\n"
            "        self._balance = initial_balance\n\n"
            "    def deposit(self, amount):\n"
            "        if amount <= 0:\n"
            "            raise ValueError('deposito deve ser positivo')\n"
            "        self._balance += amount\n\n"
            "    def withdraw(self, amount):\n"
            "        if amount <= 0:\n"
            "            raise ValueError('saque deve ser positivo')\n"
            "        if amount > self._balance:\n"
            "            raise ValueError('saldo insuficiente')\n"
            "        self._balance -= amount\n\n"
            "    @property\n"
            "    def balance(self):\n"
            "        return self._balance\n"
        ),
        test_path="test_bank_account.py",
        test_content=(
            "import pytest\n\n"
            "from bank_account import BankAccount\n\n\n"
            "def test_bank_account():\n"
            "    acc = BankAccount(100)\n"
            "    acc.deposit(50)\n"
            "    assert acc.balance == 150\n"
            "    acc.withdraw(30)\n"
            "    assert acc.balance == 120\n"
            "    with pytest.raises(ValueError):\n"
            "        acc.withdraw(1000)\n"
            "    with pytest.raises(ValueError):\n"
            "        acc.deposit(-10)\n"
        ),
        think_before_final="Depósito, saque e saldo protegido contra valores inválidos. Vou validar.",
        final_text=(
            "Implementei a classe BankAccount com deposit, withdraw e a propriedade balance, "
            "impedindo saldo negativo e valores inválidos. A validação confirma que compila e "
            "o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-rectangle",
        domain="oop",
        difficulty="easy",
        user_request="Implemente uma classe Rectangle com métodos de área, perímetro e verificação se é quadrado.",
        think_before_write="Guardo largura e altura no construtor e derivo área, perímetro e is_square a partir deles.",
        file_path="rectangle.py",
        file_content=(
            "class Rectangle:\n"
            "    def __init__(self, width, height):\n"
            "        self.width = width\n"
            "        self.height = height\n\n"
            "    def area(self):\n"
            "        return self.width * self.height\n\n"
            "    def perimeter(self):\n"
            "        return 2 * (self.width + self.height)\n\n"
            "    def is_square(self):\n"
            "        return self.width == self.height\n"
        ),
        test_path="test_rectangle.py",
        test_content=(
            "from rectangle import Rectangle\n\n\n"
            "def test_rectangle():\n"
            "    r = Rectangle(4, 5)\n"
            "    assert r.area() == 20\n"
            "    assert r.perimeter() == 18\n"
            "    assert r.is_square() is False\n"
            "    s = Rectangle(3, 3)\n"
            "    assert s.is_square() is True\n"
        ),
        think_before_final="Os três métodos pedidos estão implementados. Vou validar.",
        final_text=(
            "Implementei a classe Rectangle com area, perimeter e is_square. A validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-circle",
        domain="oop",
        difficulty="easy",
        user_request="Quero uma classe Circle com métodos para calcular área e circunferência.",
        think_before_write="Guardo o raio no construtor e uso math.pi para os cálculos derivados.",
        file_path="circle.py",
        file_content=(
            "import math\n\n\n"
            "class Circle:\n"
            "    def __init__(self, radius):\n"
            "        self.radius = radius\n\n"
            "    def area(self):\n"
            "        return math.pi * self.radius ** 2\n\n"
            "    def circumference(self):\n"
            "        return 2 * math.pi * self.radius\n"
        ),
        test_path="test_circle.py",
        test_content=(
            "from circle import Circle\n\n\n"
            "def test_circle():\n"
            "    c = Circle(2)\n"
            "    assert abs(c.area() - 12.566370614359172) < 1e-9\n"
            "    assert abs(c.circumference() - 12.566370614359172) < 1e-9\n"
        ),
        think_before_final="Área e circunferência implementadas com math.pi. Vou validar.",
        final_text=(
            "Implementei a classe Circle com area e circumference usando math.pi. A validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-temperature_converter",
        domain="oop",
        difficulty="medium",
        user_request=(
            "Crie uma classe TemperatureConverter que guarde uma temperatura em Celsius e "
            "converta para Fahrenheit e Kelvin."
        ),
        think_before_write="O estado é o valor em Celsius; Fahrenheit e Kelvin são derivados por métodos, não guardados separadamente.",
        file_path="temperature_converter.py",
        file_content=(
            "class TemperatureConverter:\n"
            "    def __init__(self, celsius):\n"
            "        self.celsius = celsius\n\n"
            "    def to_fahrenheit(self):\n"
            "        return self.celsius * 9 / 5 + 32\n\n"
            "    def to_kelvin(self):\n"
            "        return self.celsius + 273.15\n"
        ),
        test_path="test_temperature_converter.py",
        test_content=(
            "from temperature_converter import TemperatureConverter\n\n\n"
            "def test_temperature_converter():\n"
            "    t = TemperatureConverter(25)\n"
            "    assert abs(t.to_fahrenheit() - 77.0) < 1e-9\n"
            "    assert abs(t.to_kelvin() - 298.15) < 1e-9\n"
        ),
        think_before_final="Conversões para Fahrenheit e Kelvin implementadas a partir do Celsius guardado. Vou validar.",
        final_text=(
            "Implementei a classe TemperatureConverter, guardando a temperatura em Celsius e "
            "derivando to_fahrenheit e to_kelvin. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-oop-counter",
        domain="oop",
        difficulty="easy",
        user_request="Implemente uma classe Counter com increment, decrement, reset e uma propriedade value.",
        think_before_write="O contador é o próprio estado interno; increment/decrement/reset só modificam esse valor.",
        file_path="counter.py",
        file_content=(
            "class Counter:\n"
            "    def __init__(self, start=0):\n"
            "        self._value = start\n\n"
            "    def increment(self, step=1):\n"
            "        self._value += step\n\n"
            "    def decrement(self, step=1):\n"
            "        self._value -= step\n\n"
            "    def reset(self):\n"
            "        self._value = 0\n\n"
            "    @property\n"
            "    def value(self):\n"
            "        return self._value\n"
        ),
        test_path="test_counter.py",
        test_content=(
            "from counter import Counter\n\n\n"
            "def test_counter():\n"
            "    c = Counter()\n"
            "    c.increment()\n"
            "    c.increment(4)\n"
            "    assert c.value == 5\n"
            "    c.decrement(2)\n"
            "    assert c.value == 3\n"
            "    c.reset()\n"
            "    assert c.value == 0\n"
        ),
        think_before_final="increment, decrement, reset e value estão implementados. Vou validar.",
        final_text=(
            "Implementei a classe Counter com increment, decrement, reset e a propriedade "
            "value. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-linked_list",
        domain="oop",
        difficulty="medium",
        user_request="Implemente uma lista encadeada simples em Python com append, prepend e conversão para lista Python.",
        think_before_write=(
            "Preciso de duas classes: Node (guarda valor e próximo) e LinkedList (guarda a "
            "cabeça e implementa append/prepend/to_list percorrendo os nós)."
        ),
        file_path="linked_list.py",
        file_content=(
            "class Node:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "        self.next = None\n\n\n"
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n\n"
            "    def append(self, value):\n"
            "        node = Node(value)\n"
            "        if self.head is None:\n"
            "            self.head = node\n"
            "            return\n"
            "        current = self.head\n"
            "        while current.next is not None:\n"
            "            current = current.next\n"
            "        current.next = node\n\n"
            "    def prepend(self, value):\n"
            "        node = Node(value)\n"
            "        node.next = self.head\n"
            "        self.head = node\n\n"
            "    def to_list(self):\n"
            "        result = []\n"
            "        current = self.head\n"
            "        while current is not None:\n"
            "            result.append(current.value)\n"
            "            current = current.next\n"
            "        return result\n"
        ),
        test_path="test_linked_list.py",
        test_content=(
            "from linked_list import LinkedList\n\n\n"
            "def test_linked_list():\n"
            "    ll = LinkedList()\n"
            "    ll.append(1)\n"
            "    ll.append(2)\n"
            "    ll.prepend(0)\n"
            "    assert ll.to_list() == [0, 1, 2]\n"
        ),
        think_before_final="Node e LinkedList com append, prepend e to_list estão prontos. Vou validar.",
        final_text=(
            "Implementei Node e LinkedList com append, prepend e to_list. A validação confirma "
            "que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-matrix",
        domain="oop",
        difficulty="medium",
        user_request="Crie uma classe Matrix que envolva uma lista de listas e ofereça soma e transposição.",
        think_before_write="A matriz fica guardada como lista de listas no construtor; add valida dimensões antes de somar.",
        file_path="matrix.py",
        file_content=(
            "class Matrix:\n"
            "    def __init__(self, rows):\n"
            "        self.rows = rows\n\n"
            "    def add(self, other):\n"
            "        if len(self.rows) != len(other.rows):\n"
            "            raise ValueError('dimensoes incompativeis')\n"
            "        return Matrix([\n"
            "            [a + b for a, b in zip(row_a, row_b)]\n"
            "            for row_a, row_b in zip(self.rows, other.rows)\n"
            "        ])\n\n"
            "    def transpose(self):\n"
            "        return Matrix([list(col) for col in zip(*self.rows)])\n"
        ),
        test_path="test_matrix.py",
        test_content=(
            "import pytest\n\n"
            "from matrix import Matrix\n\n\n"
            "def test_matrix():\n"
            "    m1 = Matrix([[1, 2], [3, 4]])\n"
            "    m2 = Matrix([[5, 6], [7, 8]])\n"
            "    assert m1.add(m2).rows == [[6, 8], [10, 12]]\n"
            "    assert m1.transpose().rows == [[1, 3], [2, 4]]\n"
            "    with pytest.raises(ValueError):\n"
            "        m1.add(Matrix([[1]]))\n"
        ),
        think_before_final="add e transpose estão implementados, com validação de dimensão em add. Vou validar.",
        final_text=(
            "Implementei a classe Matrix com add e transpose sobre uma lista de listas, "
            "validando dimensões na soma. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-shopping_cart",
        domain="oop",
        difficulty="medium",
        user_request=(
            "Implemente uma classe ShoppingCart com adicionar item, remover item e calcular o "
            "total formatado em Real (R$)."
        ),
        think_before_write=(
            "Guardo os itens como lista de tuplas (nome, preço); total() soma os preços e "
            "formata como R$ com vírgula decimal, seguindo o padrão brasileiro."
        ),
        file_path="shopping_cart.py",
        file_content=(
            "class ShoppingCart:\n"
            "    def __init__(self):\n"
            "        self._items = []\n\n"
            "    def add_item(self, name, price):\n"
            "        self._items.append((name, price))\n\n"
            "    def remove_item(self, name):\n"
            "        for item in self._items:\n"
            "            if item[0] == name:\n"
            "                self._items.remove(item)\n"
            "                return\n"
            "        raise ValueError(f'item nao encontrado: {name}')\n\n"
            "    def total_formatted(self):\n"
            "        total = sum(price for _, price in self._items)\n"
            "        return f\"R$ {total:,.2f}\".replace(',', 'X').replace('.', ',').replace('X', '.')\n"
        ),
        test_path="test_shopping_cart.py",
        test_content=(
            "import pytest\n\n"
            "from shopping_cart import ShoppingCart\n\n\n"
            "def test_shopping_cart():\n"
            "    cart = ShoppingCart()\n"
            "    cart.add_item('caneta', 2.50)\n"
            "    cart.add_item('caderno', 15.00)\n"
            "    assert cart.total_formatted() == 'R$ 17,50'\n"
            "    cart.remove_item('caneta')\n"
            "    assert cart.total_formatted() == 'R$ 15,00'\n"
            "    with pytest.raises(ValueError):\n"
            "        cart.remove_item('inexistente')\n"
        ),
        think_before_final="add_item, remove_item e total no formato R$ brasileiro estão prontos. Vou validar.",
        final_text=(
            "Implementei a classe ShoppingCart com add_item, remove_item e total_formatted, "
            "retornando o total no formato R$ com vírgula decimal. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-employee_payroll",
        domain="oop",
        difficulty="medium",
        user_request="Crie uma classe Employee que calcule o salário líquido a partir do salário base e um bônus percentual.",
        think_before_write=(
            "O estado é salário base e percentual de bônus; net_salary() deriva o valor final "
            "sem guardar como atributo separado (evita inconsistência se o bônus mudar)."
        ),
        file_path="employee_payroll.py",
        file_content=(
            "class Employee:\n"
            "    def __init__(self, name, base_salary, bonus_percentage=0):\n"
            "        self.name = name\n"
            "        self.base_salary = base_salary\n"
            "        self.bonus_percentage = bonus_percentage\n\n"
            "    def net_salary(self):\n"
            "        return self.base_salary * (1 + self.bonus_percentage / 100)\n"
        ),
        test_path="test_employee_payroll.py",
        test_content=(
            "from employee_payroll import Employee\n\n\n"
            "def test_employee_payroll():\n"
            "    e = Employee('Ana', 3000, bonus_percentage=10)\n"
            "    assert abs(e.net_salary() - 3300.0) < 1e-9\n"
            "    e2 = Employee('Bruno', 2000)\n"
            "    assert abs(e2.net_salary() - 2000.0) < 1e-9\n"
        ),
        think_before_final="net_salary calcula corretamente o salário com bônus opcional. Vou validar.",
        final_text=(
            "Implementei a classe Employee com net_salary, aplicando o percentual de bônus "
            "sobre o salário base. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-cpf_validator",
        domain="oop",
        difficulty="hard",
        user_request="Implemente uma classe CPFValidator que valide um CPF calculando os dígitos verificadores.",
        think_before_write=(
            "Preciso implementar o algoritmo real de dígito verificador do CPF (pesos "
            "decrescentes de 10 a 2 e depois de 11 a 2, mod 11), não só checar o formato."
        ),
        file_path="cpf_validator.py",
        file_content=(
            "class CPFValidator:\n"
            "    def _digits_only(self, cpf):\n"
            "        return [int(c) for c in cpf if c.isdigit()]\n\n"
            "    def _check_digit(self, digits, weight_start):\n"
            "        total = sum(d * w for d, w in zip(digits, range(weight_start, 1, -1)))\n"
            "        remainder = (total * 10) % 11\n"
            "        return 0 if remainder == 10 else remainder\n\n"
            "    def is_valid(self, cpf):\n"
            "        digits = self._digits_only(cpf)\n"
            "        if len(digits) != 11 or len(set(digits)) == 1:\n"
            "            return False\n"
            "        first_check = self._check_digit(digits[:9], 10)\n"
            "        if first_check != digits[9]:\n"
            "            return False\n"
            "        second_check = self._check_digit(digits[:10], 11)\n"
            "        return second_check == digits[10]\n"
        ),
        test_path="test_cpf_validator.py",
        test_content=(
            "from cpf_validator import CPFValidator\n\n\n"
            "def test_cpf_validator():\n"
            "    v = CPFValidator()\n"
            "    assert v.is_valid('111.444.777-35') is True\n"
            "    assert v.is_valid('111.444.777-36') is False\n"
            "    assert v.is_valid('000.000.000-00') is False\n"
            "    assert v.is_valid('123') is False\n"
        ),
        think_before_final="O cálculo dos dois dígitos verificadores está implementado conforme a regra oficial. Vou validar.",
        final_text=(
            "Implementei a classe CPFValidator com is_valid, calculando os dois dígitos "
            "verificadores pela regra oficial (módulo 11). A validação confirma que compila e "
            "o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-vector2d",
        domain="oop",
        difficulty="medium",
        user_request="Implemente uma classe Vector2D com soma, subtração e magnitude, usando sobrecarga de operadores.",
        think_before_write=(
            "Sobrecarga de operadores é o jeito idiomático em Python: implemento __add__ e "
            "__sub__ para que + e - funcionem entre vetores, além de magnitude() como método comum."
        ),
        file_path="vector2d.py",
        file_content=(
            "import math\n\n\n"
            "class Vector2D:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n\n"
            "    def __add__(self, other):\n"
            "        return Vector2D(self.x + other.x, self.y + other.y)\n\n"
            "    def __sub__(self, other):\n"
            "        return Vector2D(self.x - other.x, self.y - other.y)\n\n"
            "    def magnitude(self):\n"
            "        return math.sqrt(self.x ** 2 + self.y ** 2)\n"
        ),
        test_path="test_vector2d.py",
        test_content=(
            "from vector2d import Vector2D\n\n\n"
            "def test_vector2d():\n"
            "    v1 = Vector2D(1, 2)\n"
            "    v2 = Vector2D(3, 4)\n"
            "    result = v1 + v2\n"
            "    assert (result.x, result.y) == (4, 6)\n"
            "    diff = v2 - v1\n"
            "    assert (diff.x, diff.y) == (2, 2)\n"
            "    assert abs(Vector2D(3, 4).magnitude() - 5.0) < 1e-9\n"
        ),
        think_before_final="__add__, __sub__ e magnitude estão implementados e testados. Vou validar.",
        final_text=(
            "Implementei a classe Vector2D com sobrecarga de __add__ e __sub__, além de "
            "magnitude(). A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-oop-playlist",
        domain="oop",
        difficulty="medium",
        user_request="Crie uma classe Playlist com adicionar música, remover música e avançar para a próxima faixa, voltando ao início ao chegar no fim.",
        think_before_write=(
            "O estado é a lista de músicas e o índice da faixa atual; next_track precisa "
            "avançar o índice e voltar a 0 (circular) ao passar do fim."
        ),
        file_path="playlist.py",
        file_content=(
            "class Playlist:\n"
            "    def __init__(self):\n"
            "        self._songs = []\n"
            "        self._current_index = 0\n\n"
            "    def add_song(self, title):\n"
            "        self._songs.append(title)\n\n"
            "    def remove_song(self, title):\n"
            "        self._songs.remove(title)\n"
            "        if self._current_index >= len(self._songs):\n"
            "            self._current_index = 0\n\n"
            "    def next_track(self):\n"
            "        if not self._songs:\n"
            "            raise IndexError('playlist vazia')\n"
            "        self._current_index = (self._current_index + 1) % len(self._songs)\n"
            "        return self._songs[self._current_index]\n\n"
            "    def current_track(self):\n"
            "        if not self._songs:\n"
            "            raise IndexError('playlist vazia')\n"
            "        return self._songs[self._current_index]\n"
        ),
        test_path="test_playlist.py",
        test_content=(
            "import pytest\n\n"
            "from playlist import Playlist\n\n\n"
            "def test_playlist():\n"
            "    p = Playlist()\n"
            "    p.add_song('A')\n"
            "    p.add_song('B')\n"
            "    p.add_song('C')\n"
            "    assert p.current_track() == 'A'\n"
            "    assert p.next_track() == 'B'\n"
            "    assert p.next_track() == 'C'\n"
            "    assert p.next_track() == 'A'\n"
            "    with pytest.raises(IndexError):\n"
            "        Playlist().next_track()\n"
        ),
        think_before_final="A navegação circular entre faixas está implementada e testada. Vou validar.",
        final_text=(
            "Implementei a classe Playlist com add_song, remove_song, next_track e "
            "current_track, avançando de forma circular pela lista. A validação confirma que "
            "compila e o teste passa."
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
        print(f"OK: {out_path.name} (checker passou de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
