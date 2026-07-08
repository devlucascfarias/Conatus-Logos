#!/usr/bin/env python
"""Lote base da categoria "Módulos realistas" (docs/plan_dataset_expansion_oop_shell.md, seção 2):
tarefas com formato mais próximo de engenharia de verdade (múltiplas funções relacionadas por
módulo), em vez de uma única função pura isolada. Parte usa domínio brasileiro sutil (CNPJ, IMC,
formatação de R$, feriados nacionais, telefone), conforme decidido em conversa.

Cada exemplo: `<think>` -> `write_file` real -> `<think>` -> `checker` real
(`compile_and_test`) -> `<final>`. Execução real via `ToolExecutorRegistry`/`SandboxContext`.

Uso:
    python scripts/gen_realistic_modules_pilot.py
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
    id_, domain, difficulty, user_request, think_before_write,
    file_path, file_content, test_path, test_content,
    think_before_final, final_text,
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
            "task_type": "realistic_module",
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
        id_="gen-module-inventory",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo de controle de estoque com adicionar, remover e listar itens com estoque baixo.",
        think_before_write="São três funções relacionadas que operam sobre o mesmo dicionário de estoque — module-level, sem precisar de classe aqui.",
        file_path="inventory_management.py",
        file_content=(
            "def add_stock(inventory, item, quantity):\n"
            "    inventory[item] = inventory.get(item, 0) + quantity\n"
            "    return inventory[item]\n\n\n"
            "def remove_stock(inventory, item, quantity):\n"
            "    if inventory.get(item, 0) < quantity:\n"
            "        raise ValueError(f'estoque insuficiente para remover {quantity} de {item}')\n"
            "    inventory[item] -= quantity\n"
            "    return inventory[item]\n\n\n"
            "def low_stock_items(inventory, threshold):\n"
            "    return [item for item, qty in inventory.items() if qty < threshold]\n"
        ),
        test_path="test_inventory_management.py",
        test_content=(
            "import pytest\n\n"
            "from inventory_management import add_stock, remove_stock, low_stock_items\n\n\n"
            "def test_inventory_management():\n"
            "    inventory = {}\n"
            "    add_stock(inventory, 'caneta', 20)\n"
            "    add_stock(inventory, 'caderno', 3)\n"
            "    remove_stock(inventory, 'caneta', 5)\n"
            "    assert inventory['caneta'] == 15\n"
            "    assert low_stock_items(inventory, threshold=10) == ['caderno']\n"
            "    with pytest.raises(ValueError):\n"
            "        remove_stock(inventory, 'caderno', 100)\n"
        ),
        think_before_final="As três funções cobrem o fluxo básico de controle de estoque, com validação no remove. Vou validar.",
        final_text=(
            "Implementei o módulo inventory_management com add_stock, remove_stock e "
            "low_stock_items. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-cnpj-validator",
        domain="modulos_realistas",
        difficulty="hard",
        user_request="Implemente uma função que valide um CNPJ calculando os dois dígitos verificadores pela regra oficial.",
        think_before_write="O CNPJ usa pesos diferentes do CPF para cada dígito verificador — preciso implementar os dois cálculos com os pesos corretos, não reaproveitar a lógica do CPF.",
        file_path="cnpj_validator.py",
        file_content=(
            "def _digits_only(cnpj):\n"
            "    return [int(c) for c in cnpj if c.isdigit()]\n\n\n"
            "def _check_digit(digits, weights):\n"
            "    total = sum(d * w for d, w in zip(digits, weights))\n"
            "    remainder = total % 11\n"
            "    return 0 if remainder < 2 else 11 - remainder\n\n\n"
            "def is_valid_cnpj(cnpj):\n"
            "    digits = _digits_only(cnpj)\n"
            "    if len(digits) != 14 or len(set(digits)) == 1:\n"
            "        return False\n"
            "    weights_first = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]\n"
            "    first_check = _check_digit(digits[:12], weights_first)\n"
            "    if first_check != digits[12]:\n"
            "        return False\n"
            "    weights_second = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]\n"
            "    second_check = _check_digit(digits[:13], weights_second)\n"
            "    return second_check == digits[13]\n"
        ),
        test_path="test_cnpj_validator.py",
        test_content=(
            "from cnpj_validator import is_valid_cnpj\n\n\n"
            "def test_is_valid_cnpj():\n"
            "    assert is_valid_cnpj('11.222.333/0001-81') is True\n"
            "    assert is_valid_cnpj('11.222.333/0001-82') is False\n"
            "    assert is_valid_cnpj('00.000.000/0000-00') is False\n"
            "    assert is_valid_cnpj('123') is False\n"
        ),
        think_before_final="Os pesos e o cálculo dos dois dígitos verificadores seguem a regra oficial do CNPJ. Vou validar.",
        final_text=(
            "Implementei is_valid_cnpj(cnpj), calculando os dois dígitos verificadores com os "
            "pesos corretos do CNPJ (diferentes do CPF). A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-module-imc-calculator",
        domain="modulos_realistas",
        difficulty="easy",
        user_request="Implemente um módulo que calcule o IMC de uma pessoa e classifique o resultado.",
        think_before_write="Duas funções: uma calcula o valor numérico, outra classifica a faixa — separadas para poderem ser testadas e reutilizadas de forma independente.",
        file_path="imc_calculator.py",
        file_content=(
            "def calculate_imc(weight_kg, height_m):\n"
            "    return weight_kg / (height_m ** 2)\n\n\n"
            "def classify_imc(imc):\n"
            "    if imc < 18.5:\n"
            "        return 'abaixo do peso'\n"
            "    if imc < 25:\n"
            "        return 'peso normal'\n"
            "    if imc < 30:\n"
            "        return 'sobrepeso'\n"
            "    return 'obesidade'\n"
        ),
        test_path="test_imc_calculator.py",
        test_content=(
            "from imc_calculator import calculate_imc, classify_imc\n\n\n"
            "def test_imc_calculator():\n"
            "    imc = calculate_imc(70, 1.75)\n"
            "    assert abs(imc - 22.857142857142858) < 1e-9\n"
            "    assert classify_imc(imc) == 'peso normal'\n"
            "    assert classify_imc(17) == 'abaixo do peso'\n"
            "    assert classify_imc(32) == 'obesidade'\n"
        ),
        think_before_final="O cálculo e a classificação em faixas estão corretos. Vou validar.",
        final_text=(
            "Implementei o módulo imc_calculator com calculate_imc e classify_imc, cobrindo as "
            "faixas padrão de classificação. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-national-holidays",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo que verifique se uma data é feriado nacional brasileiro de data fixa.",
        think_before_write="Vou cobrir só os feriados de data fixa (não os móveis como Carnaval e Páscoa, que dependem de cálculo litúrgico à parte) e guardar como dicionário (mês, dia) -> nome.",
        file_path="national_holidays.py",
        file_content=(
            "_FIXED_HOLIDAYS = {\n"
            "    (1, 1): 'Confraternizacao Universal',\n"
            "    (4, 21): 'Tiradentes',\n"
            "    (5, 1): 'Dia do Trabalho',\n"
            "    (9, 7): 'Independencia do Brasil',\n"
            "    (10, 12): 'Nossa Senhora Aparecida',\n"
            "    (11, 2): 'Finados',\n"
            "    (11, 15): 'Proclamacao da Republica',\n"
            "    (12, 25): 'Natal',\n"
            "}\n\n\n"
            "def is_national_holiday(day, month):\n"
            "    return (month, day) in _FIXED_HOLIDAYS\n\n\n"
            "def holiday_name(day, month):\n"
            "    return _FIXED_HOLIDAYS.get((month, day))\n"
        ),
        test_path="test_national_holidays.py",
        test_content=(
            "from national_holidays import is_national_holiday, holiday_name\n\n\n"
            "def test_national_holidays():\n"
            "    assert is_national_holiday(7, 9) is True\n"
            "    assert holiday_name(7, 9) == 'Independencia do Brasil'\n"
            "    assert is_national_holiday(3, 6) is False\n"
            "    assert holiday_name(3, 6) is None\n"
        ),
        think_before_final="Os feriados de data fixa estão cobertos e a busca por nome funciona. Vou validar.",
        final_text=(
            "Implementei o módulo national_holidays com is_national_holiday e holiday_name, "
            "cobrindo os feriados nacionais de data fixa. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-module-currency-format",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo de formatação com valor em Real (R$) e percentual, seguindo o padrão brasileiro de vírgula decimal.",
        think_before_write="O padrão brasileiro usa vírgula como separador decimal e ponto como separador de milhar — o inverso do padrão que f-string produz por padrão, então preciso trocar os dois.",
        file_path="currency_format.py",
        file_content=(
            "def format_brl(value):\n"
            "    formatted = f'{value:,.2f}'\n"
            "    return 'R$ ' + formatted.replace(',', 'X').replace('.', ',').replace('X', '.')\n\n\n"
            "def format_percentage(value):\n"
            "    return f'{value:.1f}%'.replace('.', ',')\n"
        ),
        test_path="test_currency_format.py",
        test_content=(
            "from currency_format import format_brl, format_percentage\n\n\n"
            "def test_currency_format():\n"
            "    assert format_brl(1234.5) == 'R$ 1.234,50'\n"
            "    assert format_brl(10) == 'R$ 10,00'\n"
            "    assert format_percentage(12.34) == '12,3%'\n"
        ),
        think_before_final="Os separadores de milhar e decimal seguem o padrão brasileiro. Vou validar.",
        final_text=(
            "Implementei o módulo currency_format com format_brl e format_percentage, seguindo "
            "o padrão brasileiro de separadores. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-module-password-generator",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente uma função que gere senhas aleatórias, permitindo fixar uma seed para reprodutibilidade em testes.",
        think_before_write="Preciso usar random.Random(seed) em vez do módulo random global, para que a mesma seed sempre produza a mesma senha, sem afetar outro código que use random.",
        file_path="password_generator.py",
        file_content=(
            "import random\n"
            "import string\n\n\n"
            "def generate_password(length=12, seed=None):\n"
            "    rng = random.Random(seed)\n"
            "    charset = string.ascii_letters + string.digits\n"
            "    return ''.join(rng.choice(charset) for _ in range(length))\n"
        ),
        test_path="test_password_generator.py",
        test_content=(
            "from password_generator import generate_password\n\n\n"
            "def test_generate_password():\n"
            "    pw1 = generate_password(10, seed=42)\n"
            "    pw2 = generate_password(10, seed=42)\n"
            "    assert pw1 == pw2\n"
            "    assert len(pw1) == 10\n"
        ),
        think_before_final="A mesma seed produz a mesma senha, e o comprimento é o esperado. Vou validar.",
        final_text=(
            "Implementei generate_password(length, seed), usando random.Random para permitir "
            "reprodutibilidade via seed. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-slug-generator",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente uma função slugify que converta texto em português (com acentos) em um slug seguro para URL.",
        think_before_write="Preciso normalizar caracteres acentuados para sua versão ASCII (unicodedata) antes de trocar espaços e pontuação por hífen.",
        file_path="slug_generator.py",
        file_content=(
            "import re\n"
            "import unicodedata\n\n\n"
            "def slugify(text):\n"
            "    normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')\n"
            "    normalized = normalized.lower()\n"
            "    normalized = re.sub(r'[^a-z0-9]+', '-', normalized).strip('-')\n"
            "    return normalized\n"
        ),
        test_path="test_slug_generator.py",
        test_content=(
            "from slug_generator import slugify\n\n\n"
            "def test_slugify():\n"
            "    assert slugify('Sao Paulo, Cafe!') == 'sao-paulo-cafe'\n"
            "    assert slugify('  Espacos   Extras  ') == 'espacos-extras'\n"
        ),
        think_before_final="A normalização de acentos e a limpeza de pontuação estão corretas. Vou validar.",
        final_text=(
            "Implementei slugify(text), normalizando acentos e convertendo o texto em um slug "
            "seguro para URL. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-csv-line-parser",
        domain="modulos_realistas",
        difficulty="hard",
        user_request="Implemente uma função que faça parse de uma linha CSV simples, tratando campos entre aspas que contenham o delimitador.",
        think_before_write="Preciso percorrer caractere a caractere, alternando um estado 'dentro de aspas' para não quebrar o campo quando o delimitador aparece dentro de aspas.",
        file_path="csv_line_parser.py",
        file_content=(
            "def parse_csv_line(line, delimiter=','):\n"
            "    fields = []\n"
            "    current = ''\n"
            "    in_quotes = False\n"
            "    for char in line:\n"
            "        if char == '\"':\n"
            "            in_quotes = not in_quotes\n"
            "        elif char == delimiter and not in_quotes:\n"
            "            fields.append(current)\n"
            "            current = ''\n"
            "        else:\n"
            "            current += char\n"
            "    fields.append(current)\n"
            "    return fields\n"
        ),
        test_path="test_csv_line_parser.py",
        test_content=(
            "from csv_line_parser import parse_csv_line\n\n\n"
            "def test_parse_csv_line():\n"
            "    assert parse_csv_line('a,b,c') == ['a', 'b', 'c']\n"
            "    assert parse_csv_line('\"a,b\",c') == ['a,b', 'c']\n"
            "    assert parse_csv_line('x;y', delimiter=';') == ['x', 'y']\n"
        ),
        think_before_final="Campos simples e campos entre aspas com delimitador embutido são tratados corretamente. Vou validar.",
        final_text=(
            "Implementei parse_csv_line(line, delimiter=','), tratando campos entre aspas que "
            "contêm o delimitador. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-user-registry",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo de cadastro de usuários com registro e busca por e-mail, impedindo e-mail duplicado.",
        think_before_write="Duas funções operando sobre a mesma lista de dicionários: register_user valida duplicidade antes de inserir, find_user_by_email só consulta.",
        file_path="user_registry.py",
        file_content=(
            "def register_user(registry, name, email):\n"
            "    if any(user['email'] == email for user in registry):\n"
            "        raise ValueError(f'email ja cadastrado: {email}')\n"
            "    user = {'name': name, 'email': email}\n"
            "    registry.append(user)\n"
            "    return user\n\n\n"
            "def find_user_by_email(registry, email):\n"
            "    for user in registry:\n"
            "        if user['email'] == email:\n"
            "            return user\n"
            "    return None\n"
        ),
        test_path="test_user_registry.py",
        test_content=(
            "import pytest\n\n"
            "from user_registry import register_user, find_user_by_email\n\n\n"
            "def test_user_registry():\n"
            "    registry = []\n"
            "    register_user(registry, 'Ana', 'ana@exemplo.com')\n"
            "    assert find_user_by_email(registry, 'ana@exemplo.com')['name'] == 'Ana'\n"
            "    assert find_user_by_email(registry, 'inexistente@exemplo.com') is None\n"
            "    with pytest.raises(ValueError):\n"
            "        register_user(registry, 'Outra Ana', 'ana@exemplo.com')\n"
        ),
        think_before_final="O registro impede duplicidade de e-mail e a busca funciona para casos existentes e inexistentes. Vou validar.",
        final_text=(
            "Implementei o módulo user_registry com register_user e find_user_by_email, "
            "impedindo cadastro de e-mail duplicado. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-module-discount-calculator",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo de cálculo de desconto simples e desconto progressivo por quantidade.",
        think_before_write="apply_bulk_discount reaproveita apply_discount internamente, escolhendo o percentual conforme a faixa de quantidade.",
        file_path="discount_calculator.py",
        file_content=(
            "def apply_discount(price, percentage):\n"
            "    if not 0 <= percentage <= 100:\n"
            "        raise ValueError('percentual de desconto deve estar entre 0 e 100')\n"
            "    return price * (1 - percentage / 100)\n\n\n"
            "def apply_bulk_discount(price, quantity):\n"
            "    if quantity >= 10:\n"
            "        return apply_discount(price, 20)\n"
            "    if quantity >= 5:\n"
            "        return apply_discount(price, 10)\n"
            "    return price\n"
        ),
        test_path="test_discount_calculator.py",
        test_content=(
            "import pytest\n\n"
            "from discount_calculator import apply_discount, apply_bulk_discount\n\n\n"
            "def test_discount_calculator():\n"
            "    assert apply_discount(100, 10) == 90\n"
            "    assert apply_bulk_discount(100, 12) == 80\n"
            "    assert apply_bulk_discount(100, 6) == 90\n"
            "    assert apply_bulk_discount(100, 2) == 100\n"
            "    with pytest.raises(ValueError):\n"
            "        apply_discount(100, 150)\n"
        ),
        think_before_final="As faixas de desconto progressivo e a validação do percentual estão corretas. Vou validar.",
        final_text=(
            "Implementei o módulo discount_calculator com apply_discount e "
            "apply_bulk_discount, com faixas progressivas por quantidade. A validação confirma "
            "que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-unit-converter",
        domain="modulos_realistas",
        difficulty="easy",
        user_request="Implemente um módulo de conversão de unidades entre quilômetros/milhas e quilogramas/libras.",
        think_before_write="Três funções de conversão direta, cada uma usando o fator de conversão padrão correspondente.",
        file_path="unit_converter.py",
        file_content=(
            "def km_to_miles(km):\n"
            "    return km * 0.621371\n\n\n"
            "def miles_to_km(miles):\n"
            "    return miles / 0.621371\n\n\n"
            "def kg_to_lb(kg):\n"
            "    return kg * 2.20462\n"
        ),
        test_path="test_unit_converter.py",
        test_content=(
            "from unit_converter import km_to_miles, miles_to_km, kg_to_lb\n\n\n"
            "def test_unit_converter():\n"
            "    assert abs(km_to_miles(10) - 6.21371) < 1e-5\n"
            "    assert abs(miles_to_km(6.21371) - 10) < 1e-4\n"
            "    assert abs(kg_to_lb(1) - 2.20462) < 1e-5\n"
        ),
        think_before_final="As três conversões usam os fatores padrão corretos. Vou validar.",
        final_text=(
            "Implementei o módulo unit_converter com km_to_miles, miles_to_km e kg_to_lb. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-text-statistics",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente um módulo que calcule a frequência de palavras em um texto e identifique a mais comum.",
        think_before_write="most_common_word reaproveita word_frequency para não duplicar a lógica de contagem.",
        file_path="text_statistics.py",
        file_content=(
            "def word_frequency(text):\n"
            "    words = text.lower().split()\n"
            "    frequency = {}\n"
            "    for word in words:\n"
            "        frequency[word] = frequency.get(word, 0) + 1\n"
            "    return frequency\n\n\n"
            "def most_common_word(text):\n"
            "    frequency = word_frequency(text)\n"
            "    return max(frequency, key=frequency.get)\n"
        ),
        test_path="test_text_statistics.py",
        test_content=(
            "from text_statistics import word_frequency, most_common_word\n\n\n"
            "def test_text_statistics():\n"
            "    text = 'o gato e o cachorro e o gato'\n"
            "    freq = word_frequency(text)\n"
            "    assert freq['o'] == 3\n"
            "    assert freq['gato'] == 2\n"
            "    assert most_common_word(text) == 'o'\n"
        ),
        think_before_final="A contagem de frequência e a identificação da palavra mais comum estão corretas. Vou validar.",
        final_text=(
            "Implementei o módulo text_statistics com word_frequency e most_common_word. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-todo-list",
        domain="modulos_realistas",
        difficulty="easy",
        user_request="Implemente um módulo de lista de tarefas com adicionar, concluir e listar pendentes.",
        think_before_write="Cada tarefa é um dicionário com title e done, guardados numa lista compartilhada entre as três funções.",
        file_path="todo_list.py",
        file_content=(
            "def add_task(tasks, title):\n"
            "    task = {'title': title, 'done': False}\n"
            "    tasks.append(task)\n"
            "    return task\n\n\n"
            "def complete_task(tasks, index):\n"
            "    tasks[index]['done'] = True\n\n\n"
            "def pending_tasks(tasks):\n"
            "    return [task for task in tasks if not task['done']]\n"
        ),
        test_path="test_todo_list.py",
        test_content=(
            "from todo_list import add_task, complete_task, pending_tasks\n\n\n"
            "def test_todo_list():\n"
            "    tasks = []\n"
            "    add_task(tasks, 'comprar pao')\n"
            "    add_task(tasks, 'estudar python')\n"
            "    complete_task(tasks, 0)\n"
            "    pending = pending_tasks(tasks)\n"
            "    assert len(pending) == 1\n"
            "    assert pending[0]['title'] == 'estudar python'\n"
        ),
        think_before_final="Adicionar, concluir e listar pendentes funcionam sobre a mesma lista compartilhada. Vou validar.",
        final_text=(
            "Implementei o módulo todo_list com add_task, complete_task e pending_tasks. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-phone-formatter",
        domain="modulos_realistas",
        difficulty="medium",
        user_request="Implemente uma função que formate um número de telefone brasileiro celular no padrão (DD) 9XXXX-XXXX, validando a quantidade de dígitos.",
        think_before_write="Preciso ignorar caracteres não numéricos antes de validar o total de 11 dígitos (DDD + 9 + 8 dígitos do número).",
        file_path="phone_formatter.py",
        file_content=(
            "def format_phone(digits):\n"
            "    only_digits = ''.join(c for c in digits if c.isdigit())\n"
            "    if len(only_digits) != 11:\n"
            "        raise ValueError('numero de telefone deve ter 11 digitos (DDD + 9 + numero)')\n"
            "    ddd = only_digits[:2]\n"
            "    first_part = only_digits[2:7]\n"
            "    second_part = only_digits[7:]\n"
            "    return f'({ddd}) {first_part}-{second_part}'\n"
        ),
        test_path="test_phone_formatter.py",
        test_content=(
            "import pytest\n\n"
            "from phone_formatter import format_phone\n\n\n"
            "def test_format_phone():\n"
            "    assert format_phone('11987654321') == '(11) 98765-4321'\n"
            "    assert format_phone('(11) 98765-4321') == '(11) 98765-4321'\n"
            "    with pytest.raises(ValueError):\n"
            "        format_phone('123')\n"
        ),
        think_before_final="A limpeza de caracteres não numéricos e a validação de 11 dígitos estão corretas. Vou validar.",
        final_text=(
            "Implementei format_phone(digits), formatando o número no padrão brasileiro (DD) "
            "9XXXX-XXXX e validando a quantidade de dígitos. A validação confirma que compila "
            "e o teste passa."
        ),
    ),
    dict(
        id_="gen-module-grade-calculator",
        domain="modulos_realistas",
        difficulty="easy",
        user_request="Implemente um módulo que calcule a média de notas de um aluno e classifique o resultado (aprovado, recuperação ou reprovado).",
        think_before_write="classify_result usa os limites comuns do sistema brasileiro (média >= 6 aprovado, >= 4 recuperação, abaixo reprovado).",
        file_path="grade_calculator.py",
        file_content=(
            "def calculate_average(grades):\n"
            "    return sum(grades) / len(grades)\n\n\n"
            "def classify_result(average):\n"
            "    if average >= 6:\n"
            "        return 'aprovado'\n"
            "    if average >= 4:\n"
            "        return 'recuperacao'\n"
            "    return 'reprovado'\n"
        ),
        test_path="test_grade_calculator.py",
        test_content=(
            "from grade_calculator import calculate_average, classify_result\n\n\n"
            "def test_grade_calculator():\n"
            "    average = calculate_average([8, 7, 6])\n"
            "    assert abs(average - 7.0) < 1e-9\n"
            "    assert classify_result(average) == 'aprovado'\n"
            "    assert classify_result(5) == 'recuperacao'\n"
            "    assert classify_result(2) == 'reprovado'\n"
        ),
        think_before_final="A média e as três classificações estão corretas. Vou validar.",
        final_text=(
            "Implementei o módulo grade_calculator com calculate_average e classify_result, "
            "seguindo as faixas usuais de aprovação. A validação confirma que compila e o "
            "teste passa."
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
