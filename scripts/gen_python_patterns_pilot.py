#!/usr/bin/env python
"""Lote base da categoria "Padrões Python" (docs/plan_dataset_expansion_oop_shell.md, seção 2):
decorator, generator e context manager — 5 tarefas-base de cada, escolhidas para serem
determinísticas e testáveis (sem dependência de relógio real ou I/O externo).

Cada exemplo: `<think>` -> `write_file` real -> `<think>` -> `checker` real
(`compile_and_test`) -> `<final>`. Execução real via `ToolExecutorRegistry`/`SandboxContext`.

Uso:
    python scripts/gen_python_patterns_pilot.py
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
            "task_type": "python_pattern",
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
    # --- decorators -------------------------------------------------------------------------
    dict(
        id_="gen-pattern-count-calls",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um decorator count_calls que conte quantas vezes a função decorada foi chamada.",
        think_before_write="O contador precisa ficar acessível de fora, então guardo em um atributo da própria função wrapper, não numa variável local perdida no closure.",
        file_path="count_calls.py",
        file_content=(
            "def count_calls(func):\n"
            "    def wrapper(*args, **kwargs):\n"
            "        wrapper.calls += 1\n"
            "        return func(*args, **kwargs)\n"
            "    wrapper.calls = 0\n"
            "    return wrapper\n"
        ),
        test_path="test_count_calls.py",
        test_content=(
            "from count_calls import count_calls\n\n\n"
            "@count_calls\n"
            "def greet(name):\n"
            "    return f'ola {name}'\n\n\n"
            "def test_count_calls():\n"
            "    assert greet('ana') == 'ola ana'\n"
            "    greet('bruno')\n"
            "    greet('carla')\n"
            "    assert greet.calls == 3\n"
        ),
        think_before_final="O decorator preserva o comportamento original e conta as chamadas corretamente. Vou validar.",
        final_text=(
            "Implementei o decorator count_calls, que expõe o número de chamadas no atributo "
            "calls da função decorada. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-memoize",
        domain="python_patterns",
        difficulty="medium",
        user_request="Crie um decorator memoize que armazene em cache o resultado de uma função pura por argumentos.",
        think_before_write="O cache precisa viver no closure do decorator (não reiniciar a cada chamada) e usar os args como chave do dicionário.",
        file_path="memoize.py",
        file_content=(
            "def memoize(func):\n"
            "    cache = {}\n\n"
            "    def wrapper(*args):\n"
            "        if args not in cache:\n"
            "            cache[args] = func(*args)\n"
            "        return cache[args]\n\n"
            "    return wrapper\n"
        ),
        test_path="test_memoize.py",
        test_content=(
            "from memoize import memoize\n\n\n"
            "def test_memoize():\n"
            "    calls = []\n\n"
            "    @memoize\n"
            "    def square(x):\n"
            "        calls.append(x)\n"
            "        return x * x\n\n"
            "    assert square(4) == 16\n"
            "    assert square(4) == 16\n"
            "    assert calls == [4]\n"
            "    assert square(5) == 25\n"
            "    assert calls == [4, 5]\n"
        ),
        think_before_final="O cache evita recomputação para os mesmos argumentos, confirmado pela lista de chamadas reais. Vou validar.",
        final_text=(
            "Implementei o decorator memoize, guardando resultados em cache por argumentos. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-retry-decorator",
        domain="python_patterns",
        difficulty="hard",
        user_request="Implemente um decorator retry(max_attempts) que tente novamente a função decorada em caso de exceção.",
        think_before_write="Isso é um decorator com parâmetro, então preciso de uma fábrica de decorators: uma função externa que recebe max_attempts e retorna o decorator de verdade.",
        file_path="retry_decorator.py",
        file_content=(
            "def retry(max_attempts=3):\n"
            "    def decorator(func):\n"
            "        def wrapper(*args, **kwargs):\n"
            "            last_error = None\n"
            "            for _ in range(max_attempts):\n"
            "                try:\n"
            "                    return func(*args, **kwargs)\n"
            "                except Exception as e:\n"
            "                    last_error = e\n"
            "            raise last_error\n"
            "        return wrapper\n"
            "    return decorator\n"
        ),
        test_path="test_retry_decorator.py",
        test_content=(
            "import pytest\n\n"
            "from retry_decorator import retry\n\n\n"
            "def test_retry_decorator_eventually_succeeds():\n"
            "    state = {'count': 0}\n\n"
            "    @retry(max_attempts=5)\n"
            "    def flaky():\n"
            "        state['count'] += 1\n"
            "        if state['count'] < 3:\n"
            "            raise RuntimeError('falha temporaria')\n"
            "        return 'ok'\n\n"
            "    assert flaky() == 'ok'\n\n\n"
            "def test_retry_decorator_exhausts():\n"
            "    @retry(max_attempts=2)\n"
            "    def always_fails():\n"
            "        raise RuntimeError('sempre falha')\n\n"
            "    with pytest.raises(RuntimeError, match='sempre falha'):\n"
            "        always_fails()\n"
        ),
        think_before_final="A fábrica de decorator com parâmetro está correta e o comportamento de retry funciona nos dois casos. Vou validar.",
        final_text=(
            "Implementei o decorator retry(max_attempts), como uma fábrica de decorators que "
            "tenta novamente em caso de exceção e relança o último erro ao esgotar as "
            "tentativas. A validação confirma que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-pattern-require-positive",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um decorator require_positive que valide que todos os argumentos numéricos posicionais são positivos antes de executar a função.",
        think_before_write="O decorator precisa inspecionar os args antes de chamar a função original, levantando ValueError se algum número não-positivo aparecer.",
        file_path="require_positive.py",
        file_content=(
            "def require_positive(func):\n"
            "    def wrapper(*args, **kwargs):\n"
            "        for value in args:\n"
            "            if isinstance(value, (int, float)) and value <= 0:\n"
            "                raise ValueError(f'argumento deve ser positivo: {value}')\n"
            "        return func(*args, **kwargs)\n"
            "    return wrapper\n"
        ),
        test_path="test_require_positive.py",
        test_content=(
            "import pytest\n\n"
            "from require_positive import require_positive\n\n\n"
            "@require_positive\n"
            "def sqrt_like(x):\n"
            "    return x ** 0.5\n\n\n"
            "def test_require_positive():\n"
            "    assert sqrt_like(9) == 3.0\n"
            "    with pytest.raises(ValueError):\n"
            "        sqrt_like(-4)\n"
        ),
        think_before_final="A validação ocorre antes da execução da função original, como esperado. Vou validar.",
        final_text=(
            "Implementei o decorator require_positive, que valida os argumentos numéricos "
            "antes de executar a função decorada. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-pattern-log-calls",
        domain="python_patterns",
        difficulty="medium",
        user_request="Crie um decorator log_calls que registre argumentos e resultado de cada chamada da função decorada.",
        think_before_write="O registro precisa ficar acessível de fora, então uso um atributo log na própria função wrapper, similar ao count_calls.",
        file_path="log_calls.py",
        file_content=(
            "def log_calls(func):\n"
            "    def wrapper(*args, **kwargs):\n"
            "        result = func(*args, **kwargs)\n"
            "        wrapper.log.append((args, kwargs, result))\n"
            "        return result\n"
            "    wrapper.log = []\n"
            "    return wrapper\n"
        ),
        test_path="test_log_calls.py",
        test_content=(
            "from log_calls import log_calls\n\n\n"
            "@log_calls\n"
            "def add(a, b):\n"
            "    return a + b\n\n\n"
            "def test_log_calls():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(4, 5) == 9\n"
            "    assert log_calls_log_length() == 2\n\n\n"
            "def log_calls_log_length():\n"
            "    return len(add.log)\n"
        ),
        think_before_final="O log acumula args, kwargs e resultado de cada chamada. Vou validar.",
        final_text=(
            "Implementei o decorator log_calls, que registra args, kwargs e resultado de cada "
            "chamada no atributo log da função decorada. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    # --- generators ---------------------------------------------------------------------------
    dict(
        id_="gen-pattern-fibonacci-generator",
        domain="python_patterns",
        difficulty="easy",
        user_request="Implemente um generator que produza os primeiros n números da sequência de Fibonacci.",
        think_before_write="Um generator com yield evita guardar toda a sequência em memória antes de precisar dela — ideal para esse tipo de sequência.",
        file_path="fibonacci_generator.py",
        file_content=(
            "def fibonacci_sequence(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        yield a\n"
            "        a, b = b, a + b\n"
        ),
        test_path="test_fibonacci_generator.py",
        test_content=(
            "from fibonacci_generator import fibonacci_sequence\n\n\n"
            "def test_fibonacci_sequence():\n"
            "    assert list(fibonacci_sequence(6)) == [0, 1, 1, 2, 3, 5]\n"
            "    assert list(fibonacci_sequence(0)) == []\n"
        ),
        think_before_final="A sequência gerada bate com os valores esperados. Vou validar.",
        final_text=(
            "Implementei fibonacci_sequence(n) como generator, produzindo a sequência sob "
            "demanda com yield. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-chunked-generator",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um generator chunked que divida um iterável em blocos de tamanho fixo.",
        think_before_write="Preciso acumular itens até atingir o tamanho do bloco, dar yield, e no final ainda produzir o bloco parcial restante, se houver.",
        file_path="chunked_generator.py",
        file_content=(
            "def chunked(iterable, size):\n"
            "    chunk = []\n"
            "    for item in iterable:\n"
            "        chunk.append(item)\n"
            "        if len(chunk) == size:\n"
            "            yield chunk\n"
            "            chunk = []\n"
            "    if chunk:\n"
            "        yield chunk\n"
        ),
        test_path="test_chunked_generator.py",
        test_content=(
            "from chunked_generator import chunked\n\n\n"
            "def test_chunked():\n"
            "    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]\n"
            "    assert list(chunked([], 3)) == []\n"
        ),
        think_before_final="Os blocos completos e o bloco parcial final estão corretos. Vou validar.",
        final_text=(
            "Implementei chunked(iterable, size) como generator, incluindo o bloco parcial "
            "final quando o total não é múltiplo do tamanho. A validação confirma que compila "
            "e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-countdown-generator",
        domain="python_patterns",
        difficulty="easy",
        user_request="Crie um generator countdown que conte de n até 1.",
        think_before_write="Um loop simples com yield decrescente resolve, sem precisar montar a lista inteira antes.",
        file_path="countdown_generator.py",
        file_content=(
            "def countdown(n):\n"
            "    while n > 0:\n"
            "        yield n\n"
            "        n -= 1\n"
        ),
        test_path="test_countdown_generator.py",
        test_content=(
            "from countdown_generator import countdown\n\n\n"
            "def test_countdown():\n"
            "    assert list(countdown(5)) == [5, 4, 3, 2, 1]\n"
            "    assert list(countdown(0)) == []\n"
        ),
        think_before_final="A contagem regressiva está correta, inclusive para n=0. Vou validar.",
        final_text=(
            "Implementei countdown(n) como generator, contando de n até 1. A validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-flatten-generator",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um generator flatten que achate uma lista de listas aninhada em qualquer profundidade.",
        think_before_write="Preciso de recursão combinada com yield from para repassar os valores de generators internos sem montar listas intermediárias.",
        file_path="flatten_generator.py",
        file_content=(
            "def flatten(nested):\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            yield from flatten(item)\n"
            "        else:\n"
            "            yield item\n"
        ),
        test_path="test_flatten_generator.py",
        test_content=(
            "from flatten_generator import flatten\n\n\n"
            "def test_flatten():\n"
            "    assert list(flatten([1, [2, 3, [4]], 5])) == [1, 2, 3, 4, 5]\n"
            "    assert list(flatten([])) == []\n"
        ),
        think_before_final="A recursão com yield from achata qualquer nível de aninhamento. Vou validar.",
        final_text=(
            "Implementei flatten(nested) como generator recursivo com yield from, achatando "
            "listas aninhadas em qualquer profundidade. A validação confirma que compila e o "
            "teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-take-generator",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um generator infinito de números naturais e uma função take que extraia os primeiros n valores dele.",
        think_before_write="O generator infinito nunca deve terminar sozinho; quem controla quantos valores extrair é a função take, chamando next() repetidamente.",
        file_path="take_generator.py",
        file_content=(
            "def natural_numbers():\n"
            "    n = 1\n"
            "    while True:\n"
            "        yield n\n"
            "        n += 1\n\n\n"
            "def take(generator, n):\n"
            "    result = []\n"
            "    for _ in range(n):\n"
            "        result.append(next(generator))\n"
            "    return result\n"
        ),
        test_path="test_take_generator.py",
        test_content=(
            "from take_generator import natural_numbers, take\n\n\n"
            "def test_take():\n"
            "    assert take(natural_numbers(), 5) == [1, 2, 3, 4, 5]\n"
            "    gen = natural_numbers()\n"
            "    assert take(gen, 2) == [1, 2]\n"
            "    assert take(gen, 2) == [3, 4]\n"
        ),
        think_before_final="O generator infinito e a extração parcial de valores funcionam corretamente, mantendo o estado entre chamadas de take. Vou validar.",
        final_text=(
            "Implementei natural_numbers() como generator infinito e take(generator, n) para "
            "extrair os primeiros n valores. A validação confirma que compila e o teste passa."
        ),
    ),
    # --- context managers -----------------------------------------------------------------------
    dict(
        id_="gen-pattern-suppress-error",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um context manager suppress_error que suprima exceções de tipos específicos, similar a contextlib.suppress.",
        think_before_write="Preciso implementar __enter__/__exit__: __exit__ retorna True (suprime) só quando o tipo da exceção corresponde a um dos tipos configurados.",
        file_path="suppress_error.py",
        file_content=(
            "class suppress_error:\n"
            "    def __init__(self, *exception_types):\n"
            "        self.exception_types = exception_types\n\n"
            "    def __enter__(self):\n"
            "        return self\n\n"
            "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
            "        return exc_type is not None and issubclass(exc_type, self.exception_types)\n"
        ),
        test_path="test_suppress_error.py",
        test_content=(
            "import pytest\n\n"
            "from suppress_error import suppress_error\n\n\n"
            "def test_suppress_error_suppresses_matching():\n"
            "    with suppress_error(ZeroDivisionError):\n"
            "        1 / 0\n\n\n"
            "def test_suppress_error_lets_other_propagate():\n"
            "    with pytest.raises(ValueError):\n"
            "        with suppress_error(ZeroDivisionError):\n"
            "            raise ValueError('outro erro')\n"
        ),
        think_before_final="A supressão funciona só para o tipo configurado, deixando outras exceções propagarem. Vou validar.",
        final_text=(
            "Implementei o context manager suppress_error, que suprime apenas os tipos de "
            "exceção configurados, deixando os demais propagarem. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-temporary-override",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um context manager que sobrescreva temporariamente uma chave de um dicionário, restaurando o valor original ao sair.",
        think_before_write="Preciso guardar se a chave já existia e qual era o valor antigo em __enter__, para decidir em __exit__ entre restaurar o valor antigo ou remover a chave.",
        file_path="temporary_override.py",
        file_content=(
            "class temporary_override:\n"
            "    def __init__(self, config, key, value):\n"
            "        self.config = config\n"
            "        self.key = key\n"
            "        self.value = value\n\n"
            "    def __enter__(self):\n"
            "        self._had_key = self.key in self.config\n"
            "        self._old_value = self.config.get(self.key)\n"
            "        self.config[self.key] = self.value\n"
            "        return self.config\n\n"
            "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
            "        if self._had_key:\n"
            "            self.config[self.key] = self._old_value\n"
            "        else:\n"
            "            del self.config[self.key]\n"
            "        return False\n"
        ),
        test_path="test_temporary_override.py",
        test_content=(
            "from temporary_override import temporary_override\n\n\n"
            "def test_temporary_override_existing_key():\n"
            "    config = {'timeout': 30}\n"
            "    with temporary_override(config, 'timeout', 5):\n"
            "        assert config['timeout'] == 5\n"
            "    assert config['timeout'] == 30\n\n\n"
            "def test_temporary_override_new_key():\n"
            "    config = {}\n"
            "    with temporary_override(config, 'debug', True):\n"
            "        assert config['debug'] is True\n"
            "    assert 'debug' not in config\n"
        ),
        think_before_final="O valor é restaurado corretamente tanto para chave preexistente quanto para chave nova. Vou validar.",
        final_text=(
            "Implementei o context manager temporary_override, que sobrescreve uma chave "
            "temporariamente e restaura (ou remove) o valor original ao sair do bloco. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-transaction-contextmanager",
        domain="python_patterns",
        difficulty="hard",
        user_request="Implemente um context manager transaction (usando @contextmanager) que registre início, commit em sucesso e rollback em caso de exceção.",
        think_before_write="Com @contextmanager, o código antes do yield roda no __enter__ e o try/except ao redor do yield decide entre commit (sucesso) e rollback (exceção, relançada depois).",
        file_path="transaction_contextmanager.py",
        file_content=(
            "from contextlib import contextmanager\n\n\n"
            "@contextmanager\n"
            "def transaction(log):\n"
            "    log.append('begin')\n"
            "    try:\n"
            "        yield log\n"
            "        log.append('commit')\n"
            "    except Exception:\n"
            "        log.append('rollback')\n"
            "        raise\n"
        ),
        test_path="test_transaction_contextmanager.py",
        test_content=(
            "import pytest\n\n"
            "from transaction_contextmanager import transaction\n\n\n"
            "def test_transaction_commits_on_success():\n"
            "    log = []\n"
            "    with transaction(log):\n"
            "        pass\n"
            "    assert log == ['begin', 'commit']\n\n\n"
            "def test_transaction_rolls_back_on_error():\n"
            "    log = []\n"
            "    with pytest.raises(RuntimeError):\n"
            "        with transaction(log):\n"
            "            raise RuntimeError('falha no meio da transacao')\n"
            "    assert log == ['begin', 'rollback']\n"
        ),
        think_before_final="O log confirma begin/commit no sucesso e begin/rollback com a exceção relançada. Vou validar.",
        final_text=(
            "Implementei o context manager transaction usando @contextmanager, registrando "
            "begin/commit no sucesso e begin/rollback (relançando a exceção) em caso de "
            "erro. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-timer-contextmanager",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um context manager Timer que meça o tempo decorrido dentro do bloco, permitindo injetar uma fonte de tempo customizada para facilitar testes.",
        think_before_write="Para o teste ser determinístico, o relógio precisa ser injetável (parâmetro clock), em vez de depender de time.perf_counter fixo.",
        file_path="timer_contextmanager.py",
        file_content=(
            "import time as _time\n\n\n"
            "class Timer:\n"
            "    def __init__(self, clock=None):\n"
            "        self._clock = clock or _time.perf_counter\n"
            "        self.elapsed = None\n\n"
            "    def __enter__(self):\n"
            "        self._start = self._clock()\n"
            "        return self\n\n"
            "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
            "        self.elapsed = self._clock() - self._start\n"
            "        return False\n"
        ),
        test_path="test_timer_contextmanager.py",
        test_content=(
            "from timer_contextmanager import Timer\n\n\n"
            "def test_timer_with_fake_clock():\n"
            "    values = iter([10.0, 15.0])\n"
            "    with Timer(clock=lambda: next(values)) as t:\n"
            "        pass\n"
            "    assert t.elapsed == 5.0\n"
        ),
        think_before_final="Com o relógio falso injetado, o tempo decorrido é determinístico e testável. Vou validar.",
        final_text=(
            "Implementei o context manager Timer, com uma fonte de tempo injetável para "
            "permitir testes determinísticos do tempo decorrido. A validação confirma que "
            "compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-pattern-resource-pool-contextmanager",
        domain="python_patterns",
        difficulty="medium",
        user_request="Implemente um context manager (usando @contextmanager) que registre a aquisição e a liberação de um recurso, garantindo a liberação mesmo se ocorrer erro dentro do bloco.",
        think_before_write="O finally dentro do @contextmanager garante que o registro de liberação aconteça mesmo se uma exceção for levantada dentro do bloco with.",
        file_path="resource_pool_contextmanager.py",
        file_content=(
            "from contextlib import contextmanager\n\n\n"
            "@contextmanager\n"
            "def acquire_resource(pool, name):\n"
            "    pool.append(f'acquired:{name}')\n"
            "    try:\n"
            "        yield name\n"
            "    finally:\n"
            "        pool.append(f'released:{name}')\n"
        ),
        test_path="test_resource_pool_contextmanager.py",
        test_content=(
            "import pytest\n\n"
            "from resource_pool_contextmanager import acquire_resource\n\n\n"
            "def test_acquire_resource_success():\n"
            "    pool = []\n"
            "    with acquire_resource(pool, 'db') as name:\n"
            "        assert name == 'db'\n"
            "        assert pool == ['acquired:db']\n"
            "    assert pool == ['acquired:db', 'released:db']\n\n\n"
            "def test_acquire_resource_releases_on_error():\n"
            "    pool = []\n"
            "    with pytest.raises(RuntimeError):\n"
            "        with acquire_resource(pool, 'cache'):\n"
            "            raise RuntimeError('erro dentro do bloco')\n"
            "    assert pool == ['acquired:cache', 'released:cache']\n"
        ),
        think_before_final="A liberação acontece tanto no caminho de sucesso quanto no de erro, confirmado pelo registro no pool. Vou validar.",
        final_text=(
            "Implementei o context manager acquire_resource usando @contextmanager, "
            "garantindo a liberação do recurso mesmo em caso de erro dentro do bloco. A "
            "validação confirma que compila e o teste passa."
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
