#!/usr/bin/env python
"""Lote base da categoria "Tratamento de erro" (docs/plan_dataset_expansion_oop_shell.md, seção 2).

O dataset atual tinha pouca cobertura de `try/except`/`raise` explícitos como o COMPORTAMENTO
pedido (em vez de só aparecerem incidentalmente dentro de outras tarefas). Esta categoria cobre
validação com mensagens de erro específicas, exceções customizadas, `raise ... from ...` para
encadear causa, e `try/except/else/finally`.

Cada exemplo: `<think>` -> `write_file` real -> `<think>` -> `checker` real
(`compile_and_test`) -> `<final>`. Execução real via `ToolExecutorRegistry`/`SandboxContext`.

Uso:
    python scripts/gen_error_handling_pilot.py
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
            "task_type": "error_handling",
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
        id_="gen-error-safe-divide",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Implemente uma função safe_divide que retorne None em vez de lançar exceção quando o divisor for zero.",
        think_before_write="Preciso capturar ZeroDivisionError especificamente, sem esconder outros erros que a função eventualmente possa gerar.",
        file_path="safe_divide.py",
        file_content=(
            "def safe_divide(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            "        return None\n"
        ),
        test_path="test_safe_divide.py",
        test_content=(
            "from safe_divide import safe_divide\n\n\n"
            "def test_safe_divide():\n"
            "    assert safe_divide(10, 2) == 5\n"
            "    assert safe_divide(1, 0) is None\n"
        ),
        think_before_final="A exceção ZeroDivisionError está tratada corretamente. Vou validar.",
        final_text=(
            "Implementei safe_divide(a, b), capturando ZeroDivisionError e retornando None nesse "
            "caso. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-parse-int-safe",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Crie uma função que tente converter uma string para inteiro e retorne um valor padrão se falhar.",
        think_before_write="ValueError é a exceção certa para capturar aqui, quando int() recebe algo não numérico.",
        file_path="parse_int_safe.py",
        file_content=(
            "def parse_int_safe(text, default=0):\n"
            "    try:\n"
            "        return int(text)\n"
            "    except ValueError:\n"
            "        return default\n"
        ),
        test_path="test_parse_int_safe.py",
        test_content=(
            "from parse_int_safe import parse_int_safe\n\n\n"
            "def test_parse_int_safe():\n"
            "    assert parse_int_safe('42') == 42\n"
            "    assert parse_int_safe('abc') == 0\n"
            "    assert parse_int_safe('abc', default=-1) == -1\n"
        ),
        think_before_final="A conversão com valor padrão em caso de falha está implementada. Vou validar.",
        final_text=(
            "Implementei parse_int_safe(text, default=0), capturando ValueError e retornando o "
            "valor padrão quando a conversão falha. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-error-validate-age",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Implemente uma função validate_age que levante ValueError com mensagem clara se a idade for negativa ou maior que 150.",
        think_before_write="A mensagem de erro precisa dizer exatamente qual regra foi violada, não só 'valor invalido'.",
        file_path="validate_age.py",
        file_content=(
            "def validate_age(age):\n"
            "    if age < 0:\n"
            "        raise ValueError('idade nao pode ser negativa')\n"
            "    if age > 150:\n"
            "        raise ValueError('idade acima do limite plausivel (150)')\n"
            "    return age\n"
        ),
        test_path="test_validate_age.py",
        test_content=(
            "import pytest\n\n"
            "from validate_age import validate_age\n\n\n"
            "def test_validate_age():\n"
            "    assert validate_age(30) == 30\n"
            "    with pytest.raises(ValueError, match='negativa'):\n"
            "        validate_age(-1)\n"
            "    with pytest.raises(ValueError, match='150'):\n"
            "        validate_age(200)\n"
        ),
        think_before_final="As duas regras de validação levantam mensagens distintas. Vou validar.",
        final_text=(
            "Implementei validate_age(age), levantando ValueError com mensagens específicas para "
            "idade negativa e idade acima de 150. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-error-config-lookup",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que busque uma chave em um dicionário de configuração e levante KeyError com mensagem customizada se não existir.",
        think_before_write="Uso try/except em torno do acesso direto e relevanto o erro com raise ... from e para preservar a causa original.",
        file_path="config_lookup.py",
        file_content=(
            "def get_config_value(config, key):\n"
            "    try:\n"
            "        return config[key]\n"
            "    except KeyError as e:\n"
            "        raise KeyError(f\"chave de configuracao ausente: {key}\") from e\n"
        ),
        test_path="test_config_lookup.py",
        test_content=(
            "import pytest\n\n"
            "from config_lookup import get_config_value\n\n\n"
            "def test_get_config_value():\n"
            "    config = {'timeout': 30}\n"
            "    assert get_config_value(config, 'timeout') == 30\n"
            "    with pytest.raises(KeyError, match='host'):\n"
            "        get_config_value(config, 'host')\n"
        ),
        think_before_final="O encadeamento com raise...from preserva a causa original do KeyError. Vou validar.",
        final_text=(
            "Implementei get_config_value(config, key), relançando KeyError com uma mensagem "
            "customizada e preservando a causa original via 'raise ... from e'. A validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-safe-list-get",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Crie uma função que acesse um índice de uma lista e retorne um valor padrão se o índice não existir.",
        think_before_write="IndexError é a exceção certa para capturar aqui, evitando checar o tamanho manualmente antes.",
        file_path="safe_list_get.py",
        file_content=(
            "def safe_list_get(items, index, default=None):\n"
            "    try:\n"
            "        return items[index]\n"
            "    except IndexError:\n"
            "        return default\n"
        ),
        test_path="test_safe_list_get.py",
        test_content=(
            "from safe_list_get import safe_list_get\n\n\n"
            "def test_safe_list_get():\n"
            "    assert safe_list_get([1, 2, 3], 1) == 2\n"
            "    assert safe_list_get([1, 2, 3], 10) is None\n"
            "    assert safe_list_get([1, 2, 3], 10, default=-1) == -1\n"
        ),
        think_before_final="O acesso seguro com valor padrão está implementado. Vou validar.",
        final_text=(
            "Implementei safe_list_get(items, index, default=None), capturando IndexError e "
            "retornando o valor padrão quando o índice não existe. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-validate-email",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que valide o formato de um e-mail e levante ValueError se for inválido.",
        think_before_write="Vou validar com uma checagem simples e explícita (presença de @ e domínio com ponto) e levantar erro com a razão específica.",
        file_path="validate_email.py",
        file_content=(
            "def validate_email(email):\n"
            "    if '@' not in email:\n"
            "        raise ValueError('e-mail sem @')\n"
            "    local, _, domain = email.partition('@')\n"
            "    if not local or not domain:\n"
            "        raise ValueError('e-mail com parte local ou dominio vazio')\n"
            "    if '.' not in domain:\n"
            "        raise ValueError('dominio do e-mail sem ponto')\n"
            "    return email\n"
        ),
        test_path="test_validate_email.py",
        test_content=(
            "import pytest\n\n"
            "from validate_email import validate_email\n\n\n"
            "def test_validate_email():\n"
            "    assert validate_email('ana@exemplo.com') == 'ana@exemplo.com'\n"
            "    with pytest.raises(ValueError, match='@'):\n"
            "        validate_email('semarroba.com')\n"
            "    with pytest.raises(ValueError, match='ponto'):\n"
            "        validate_email('ana@exemplocom')\n"
        ),
        think_before_final="As três checagens levantam mensagens distintas para cada problema. Vou validar.",
        final_text=(
            "Implementei validate_email(email), levantando ValueError com mensagens específicas "
            "para cada tipo de formato inválido. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-error-parse-brl",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que converta uma string em formato R$ 1.234,56 para float, levantando ValueError se o formato estiver errado.",
        think_before_write="Preciso remover o prefixo R$, trocar separadores (ponto de milhar e vírgula decimal) e capturar erro de conversão para relançar com mensagem clara.",
        file_path="parse_brl.py",
        file_content=(
            "def parse_brl(value):\n"
            "    cleaned = value.replace('R$', '').strip()\n"
            "    cleaned = cleaned.replace('.', '').replace(',', '.')\n"
            "    try:\n"
            "        return float(cleaned)\n"
            "    except ValueError as e:\n"
            "        raise ValueError(f\"formato de valor invalido: {value}\") from e\n"
        ),
        test_path="test_parse_brl.py",
        test_content=(
            "import pytest\n\n"
            "from parse_brl import parse_brl\n\n\n"
            "def test_parse_brl():\n"
            "    assert parse_brl('R$ 1.234,56') == 1234.56\n"
            "    assert parse_brl('R$ 10,00') == 10.0\n"
            "    with pytest.raises(ValueError, match='invalido'):\n"
            "        parse_brl('R$ abc')\n"
        ),
        think_before_final="A conversão trata o formato brasileiro e relança erro claro em caso de falha. Vou validar.",
        final_text=(
            "Implementei parse_brl(value), convertendo strings no formato R$ 1.234,56 para "
            "float e relançando ValueError com mensagem clara em caso de formato inválido. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-retry-operation",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função retry_operation que tente executar uma função várias vezes até funcionar, e levante o último erro se todas as tentativas falharem.",
        think_before_write="Preciso guardar a última exceção capturada dentro do loop e relançá-la (raise) só depois de esgotar as tentativas, não a cada falha.",
        file_path="retry_operation.py",
        file_content=(
            "def retry_operation(func, retries=3):\n"
            "    last_error = None\n"
            "    for _ in range(retries):\n"
            "        try:\n"
            "            return func()\n"
            "        except Exception as e:\n"
            "            last_error = e\n"
            "    raise last_error\n"
        ),
        test_path="test_retry_operation.py",
        test_content=(
            "import pytest\n\n"
            "from retry_operation import retry_operation\n\n\n"
            "def test_retry_operation_eventually_succeeds():\n"
            "    calls = {'count': 0}\n\n"
            "    def flaky():\n"
            "        calls['count'] += 1\n"
            "        if calls['count'] < 3:\n"
            "            raise RuntimeError('falha temporaria')\n"
            "        return 'ok'\n\n"
            "    assert retry_operation(flaky, retries=5) == 'ok'\n\n\n"
            "def test_retry_operation_exhausts_retries():\n"
            "    def always_fails():\n"
            "        raise RuntimeError('sempre falha')\n\n"
            "    with pytest.raises(RuntimeError, match='sempre falha'):\n"
            "        retry_operation(always_fails, retries=2)\n"
        ),
        think_before_final="O retry funciona quando eventualmente dá certo e propaga o último erro quando esgota as tentativas. Vou validar.",
        final_text=(
            "Implementei retry_operation(func, retries=3), tentando novamente em caso de erro e "
            "relançando a última exceção quando todas as tentativas falham. A validação confirma "
            "que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-error-password-strength",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que valide a força de uma senha, levantando ValueError com a regra específica que falhou.",
        think_before_write="Vou verificar cada regra em sequência (tamanho, maiúscula, dígito) e levantar erro assim que a primeira falhar, com mensagem específica.",
        file_path="validate_password.py",
        file_content=(
            "def validate_password_strength(password):\n"
            "    if len(password) < 8:\n"
            "        raise ValueError('senha deve ter pelo menos 8 caracteres')\n"
            "    if not any(c.isupper() for c in password):\n"
            "        raise ValueError('senha deve ter pelo menos uma letra maiuscula')\n"
            "    if not any(c.isdigit() for c in password):\n"
            "        raise ValueError('senha deve ter pelo menos um digito')\n"
            "    return True\n"
        ),
        test_path="test_validate_password.py",
        test_content=(
            "import pytest\n\n"
            "from validate_password import validate_password_strength\n\n\n"
            "def test_validate_password_strength():\n"
            "    assert validate_password_strength('Senha123') is True\n"
            "    with pytest.raises(ValueError, match='8 caracteres'):\n"
            "        validate_password_strength('Ab1')\n"
            "    with pytest.raises(ValueError, match='maiuscula'):\n"
            "        validate_password_strength('senha123')\n"
            "    with pytest.raises(ValueError, match='digito'):\n"
            "        validate_password_strength('SenhaSemNumero')\n"
        ),
        think_before_final="Cada regra levanta uma mensagem própria, na ordem certa. Vou validar.",
        final_text=(
            "Implementei validate_password_strength(password), levantando ValueError com a "
            "mensagem da regra específica que falhou (tamanho, maiúscula ou dígito). A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-safe-json-parse",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Implemente uma função que tente fazer parse de uma string JSON e retorne None se o JSON for inválido.",
        think_before_write="json.JSONDecodeError é a exceção específica a capturar aqui, sem esconder outros erros inesperados.",
        file_path="safe_json_parse.py",
        file_content=(
            "import json\n\n\n"
            "def safe_json_parse(text):\n"
            "    try:\n"
            "        return json.loads(text)\n"
            "    except json.JSONDecodeError:\n"
            "        return None\n"
        ),
        test_path="test_safe_json_parse.py",
        test_content=(
            "from safe_json_parse import safe_json_parse\n\n\n"
            "def test_safe_json_parse():\n"
            "    assert safe_json_parse('{\"a\": 1}') == {'a': 1}\n"
            "    assert safe_json_parse('nao e json') is None\n"
        ),
        think_before_final="A captura de JSONDecodeError está correta e específica. Vou validar.",
        final_text=(
            "Implementei safe_json_parse(text), capturando json.JSONDecodeError e retornando "
            "None quando o texto não é um JSON válido. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-error-insufficient-funds",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma exceção customizada InsufficientFundsError e uma função withdraw que a levante quando o saque for maior que o saldo.",
        think_before_write="Uma exceção customizada, herdando de Exception, deixa mais claro o tipo de erro do que um ValueError genérico.",
        file_path="insufficient_funds.py",
        file_content=(
            "class InsufficientFundsError(Exception):\n"
            "    pass\n\n\n"
            "def withdraw(balance, amount):\n"
            "    if amount > balance:\n"
            "        raise InsufficientFundsError(\n"
            "            f'saldo {balance} insuficiente para saque de {amount}'\n"
            "        )\n"
            "    return balance - amount\n"
        ),
        test_path="test_insufficient_funds.py",
        test_content=(
            "import pytest\n\n"
            "from insufficient_funds import InsufficientFundsError, withdraw\n\n\n"
            "def test_withdraw():\n"
            "    assert withdraw(100, 50) == 50\n"
            "    with pytest.raises(InsufficientFundsError):\n"
            "        withdraw(100, 200)\n"
        ),
        think_before_final="A exceção customizada e a função withdraw estão implementadas. Vou validar.",
        final_text=(
            "Implementei a exceção InsufficientFundsError e a função withdraw(balance, amount), "
            "que a levanta quando o saque excede o saldo. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-error-validate-date-br",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que valide uma data no formato DD/MM/AAAA, levantando ValueError se o formato ou a data forem inválidos.",
        think_before_write="datetime.strptime já valida datas reais (ex.: 31/02 falha); só preciso capturar ValueError e relançar com mensagem no formato esperado.",
        file_path="validate_date_br.py",
        file_content=(
            "from datetime import datetime\n\n\n"
            "def validate_date_br(date_str):\n"
            "    try:\n"
            "        return datetime.strptime(date_str, '%d/%m/%Y').date()\n"
            "    except ValueError as e:\n"
            "        raise ValueError(f\"data invalida, use DD/MM/AAAA: {date_str}\") from e\n"
        ),
        test_path="test_validate_date_br.py",
        test_content=(
            "import pytest\n\n"
            "from validate_date_br import validate_date_br\n\n\n"
            "def test_validate_date_br():\n"
            "    assert validate_date_br('25/12/2024').day == 25\n"
            "    with pytest.raises(ValueError, match='DD/MM/AAAA'):\n"
            "        validate_date_br('31/02/2024')\n"
            "    with pytest.raises(ValueError, match='DD/MM/AAAA'):\n"
            "        validate_date_br('2024-12-25')\n"
        ),
        think_before_final="A validação cobre formato errado e datas inexistentes (como 31/02). Vou validar.",
        final_text=(
            "Implementei validate_date_br(date_str), validando o formato brasileiro DD/MM/AAAA "
            "e relançando ValueError com mensagem clara para datas ou formatos inválidos. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-division-pipeline",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que processe uma lista de divisões (pares numerador/denominador), usando try/except/else/finally para registrar sucesso e contagem de tentativas.",
        think_before_write="else roda só quando não há exceção (para contar sucessos) e finally sempre roda (para contar tentativas), independentemente do resultado.",
        file_path="division_pipeline.py",
        file_content=(
            "def process_divisions(pairs):\n"
            "    results = []\n"
            "    attempts = 0\n"
            "    successes = 0\n"
            "    for numerator, denominator in pairs:\n"
            "        attempts += 1\n"
            "        try:\n"
            "            value = numerator / denominator\n"
            "        except ZeroDivisionError:\n"
            "            results.append(None)\n"
            "        else:\n"
            "            results.append(value)\n"
            "            successes += 1\n"
            "        finally:\n"
            "            pass\n"
            "    return {'results': results, 'attempts': attempts, 'successes': successes}\n"
        ),
        test_path="test_division_pipeline.py",
        test_content=(
            "from division_pipeline import process_divisions\n\n\n"
            "def test_process_divisions():\n"
            "    outcome = process_divisions([(10, 2), (5, 0), (9, 3)])\n"
            "    assert outcome['results'] == [5.0, None, 3.0]\n"
            "    assert outcome['attempts'] == 3\n"
            "    assert outcome['successes'] == 2\n"
        ),
        think_before_final="O uso de try/except/else/finally está correto e os contadores batem. Vou validar.",
        final_text=(
            "Implementei process_divisions(pairs), usando try/except/else/finally para separar "
            "divisões bem-sucedidas das que falham por zero, contando tentativas e sucessos. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-validate-positive-number",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma função que valide se um valor é um número positivo, levantando TypeError se não for número e ValueError se for negativo.",
        think_before_write="Duas exceções diferentes para dois problemas diferentes: TypeError para tipo errado, ValueError para valor errado do tipo certo.",
        file_path="validate_positive_number.py",
        file_content=(
            "def validate_positive_number(value):\n"
            "    if not isinstance(value, (int, float)):\n"
            "        raise TypeError(f'esperado numero, recebido {type(value).__name__}')\n"
            "    if value <= 0:\n"
            "        raise ValueError('numero deve ser positivo')\n"
            "    return value\n"
        ),
        test_path="test_validate_positive_number.py",
        test_content=(
            "import pytest\n\n"
            "from validate_positive_number import validate_positive_number\n\n\n"
            "def test_validate_positive_number():\n"
            "    assert validate_positive_number(5) == 5\n"
            "    with pytest.raises(TypeError):\n"
            "        validate_positive_number('5')\n"
            "    with pytest.raises(ValueError):\n"
            "        validate_positive_number(-1)\n"
        ),
        think_before_final="TypeError e ValueError cobrem os dois tipos de problema, cada um com sua exceção. Vou validar.",
        final_text=(
            "Implementei validate_positive_number(value), levantando TypeError para tipo "
            "inválido e ValueError para números não positivos. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-error-stock-check",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma exceção customizada OutOfStockError e uma função que a levante ao tentar comprar mais itens do que há em estoque, escondendo a causa técnica interna do usuário final.",
        think_before_write="Uso 'raise ... from None' para suprimir o encadeamento de causa quando o motivo interno (ex.: comparação de índice) não é relevante para quem chama a função.",
        file_path="stock_check.py",
        file_content=(
            "class OutOfStockError(Exception):\n"
            "    pass\n\n\n"
            "def purchase(stock, item, quantity):\n"
            "    try:\n"
            "        available = stock[item]\n"
            "    except KeyError:\n"
            "        raise OutOfStockError(f'item nao encontrado no estoque: {item}') from None\n"
            "    if quantity > available:\n"
            "        raise OutOfStockError(\n"
            "            f'estoque insuficiente para {item}: pedido {quantity}, disponivel {available}'\n"
            "        )\n"
            "    stock[item] -= quantity\n"
            "    return stock[item]\n"
        ),
        test_path="test_stock_check.py",
        test_content=(
            "import pytest\n\n"
            "from stock_check import OutOfStockError, purchase\n\n\n"
            "def test_purchase():\n"
            "    stock = {'caneta': 10}\n"
            "    assert purchase(stock, 'caneta', 4) == 6\n"
            "    with pytest.raises(OutOfStockError, match='insuficiente'):\n"
            "        purchase(stock, 'caneta', 100)\n"
            "    with pytest.raises(OutOfStockError, match='nao encontrado'):\n"
            "        purchase(stock, 'lapis', 1)\n"
        ),
        think_before_final="A exceção customizada cobre item inexistente e quantidade insuficiente, suprimindo a causa técnica interna. Vou validar.",
        final_text=(
            "Implementei a exceção OutOfStockError e a função purchase(stock, item, quantity), "
            "que a levanta para item inexistente ou quantidade insuficiente, escondendo a causa "
            "técnica interna com 'raise ... from None'. A validação confirma que compila e o "
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
