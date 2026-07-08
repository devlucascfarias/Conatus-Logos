#!/usr/bin/env python
"""Lote base do Gap 2b (docs/plan_dataset_expansion_error_recovery.md): diagnosticar QUAL
arquivo (implementação vs. teste) tem o bug real, quando o `checker` recebe vários.

Achado real: pedido de jogo em pygame, o teste gerado esquecia de importar `main` de
`pong_game.py` — bug real no arquivo de TESTE, não na implementação. O modelo resubmeteu
repetidamente `pong_game.py` (que nunca teve o problema), culpando "o ambiente"/"a
ferramenta", sem examinar o traceback para ver ONDE o erro realmente estava.

Cada exemplo: `<think>` -> `write_file` real (implementação correta) -> `<think>` -> `checker`
real com [implementação correta, teste com bug real] -> falha real (NameError/ImportError/
TypeError, apontado pelo próprio traceback do pytest) -> `<think>` examinando o traceback e
identificando CORRETAMENTE que o bug está no arquivo de TESTE, não na implementação -> `checker`
real de novo com [implementação inalterada, teste corrigido] -> passa -> `<final>`.

Uso:
    python scripts/gen_multifile_diagnosis_pilot.py
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
    think_before_write, file_path, file_content,
    think_before_first_checker, test_path, buggy_test_content,
    think_after_diagnosis, fixed_test_content,
    final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_write}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": file_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        traj.append_raw(f"<think>{think_before_first_checker}</think>")
        first_checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": file_content},
                {"path": test_path, "content": buggy_test_content},
            ],
        }
        first_checker_result = _tool_call(traj, sandbox, "checker", first_checker_args)
        assert not first_checker_result.passed, f"esperava falha real do teste com bug em {id_}"

        traj.append_raw(f"<think>{think_after_diagnosis}</think>")
        second_checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": file_content},
                {"path": test_path, "content": fixed_test_content},
            ],
        }
        second_checker_result = _tool_call(traj, sandbox, "checker", second_checker_args)
        assert second_checker_result.passed, f"checker (corrigido) falhou inesperadamente em {id_}: {second_checker_result.to_json()}"

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
            "num_steps": 3,
            "task_type": "multi_file_root_cause_diagnosis",
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
        id_="gen-multifile-missing-import",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que calcule a média de uma lista de números, com um teste correspondente.",
        think_before_write="Vou implementar average(numbers) primeiro.",
        file_path="average.py",
        file_content="def average(numbers):\n    return sum(numbers) / len(numbers)\n",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_average.py",
        buggy_test_content=(
            "def test_average():\n"
            "    assert average([2, 4, 6]) == 4\n"
        ),
        think_after_diagnosis=(
            "O erro do checker aponta 'NameError: name average is not defined' na linha do "
            "test_average.py — o problema está no arquivo de TESTE, que esqueceu de importar "
            "average de average.py. A implementação em average.py está correta e não precisa "
            "mudar. Vou corrigir só o teste."
        ),
        fixed_test_content=(
            "from average import average\n\n\n"
            "def test_average():\n"
            "    assert average([2, 4, 6]) == 4\n"
        ),
        final_text=(
            "Implementei average(numbers). O primeiro teste falhou porque o arquivo de teste "
            "esquecia de importar a função (bug no teste, não na implementação) — corrigi só o "
            "teste. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-multifile-wrong-import-name",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que converta Celsius para Fahrenheit, com um teste correspondente.",
        think_before_write="Vou implementar celsius_to_fahrenheit(celsius) primeiro.",
        file_path="temperature.py",
        file_content="def celsius_to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32\n",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_temperature.py",
        buggy_test_content=(
            "from temperature import celsius_to_farenheit\n\n\n"
            "def test_celsius_to_fahrenheit():\n"
            "    assert celsius_to_farenheit(0) == 32\n"
        ),
        think_after_diagnosis=(
            "O erro do checker é 'ImportError: cannot import name celsius_to_farenheit' no "
            "test_temperature.py — o nome da função no teste tem um typo (farenheit, faltando "
            "um h). A implementação em temperature.py está correta com o nome certo "
            "(celsius_to_fahrenheit). O bug é só no teste — vou corrigir a importação e a "
            "chamada."
        ),
        fixed_test_content=(
            "from temperature import celsius_to_fahrenheit\n\n\n"
            "def test_celsius_to_fahrenheit():\n"
            "    assert celsius_to_fahrenheit(0) == 32\n"
        ),
        final_text=(
            "Implementei celsius_to_fahrenheit(celsius). O primeiro teste falhou por um typo "
            "no nome da função importada no arquivo de teste (bug no teste, não na "
            "implementação) — corrigi só o teste. A validação confirma que compila e o teste "
            "passa."
        ),
    ),
    dict(
        id_="gen-multifile-wrong-arg-count",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que calcule a área de um retângulo, com um teste correspondente.",
        think_before_write="Vou implementar rectangle_area(width, height) primeiro.",
        file_path="rectangle.py",
        file_content="def rectangle_area(width, height):\n    return width * height\n",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_rectangle.py",
        buggy_test_content=(
            "from rectangle import rectangle_area\n\n\n"
            "def test_rectangle_area():\n"
            "    assert rectangle_area(4, 5, 6) == 20\n"
        ),
        think_after_diagnosis=(
            "O erro do checker é 'TypeError: rectangle_area() takes 2 positional arguments "
            "but 3 were given' no test_rectangle.py — o teste está chamando a função com um "
            "argumento a mais. A implementação em rectangle.py está correta (2 parâmetros, "
            "como pedido). O bug é só no teste — vou corrigir a chamada."
        ),
        fixed_test_content=(
            "from rectangle import rectangle_area\n\n\n"
            "def test_rectangle_area():\n"
            "    assert rectangle_area(4, 5) == 20\n"
        ),
        final_text=(
            "Implementei rectangle_area(width, height). O primeiro teste falhou porque "
            "chamava a função com um argumento a mais (bug no teste, não na implementação) — "
            "corrigi só o teste. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-multifile-undefined-variable-in-test",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que verifique se uma lista está ordenada, com um teste correspondente.",
        think_before_write="Vou implementar is_sorted(items) primeiro.",
        file_path="is_sorted.py",
        file_content="def is_sorted(items):\n    return all(items[i] <= items[i + 1] for i in range(len(items) - 1))\n",
        think_before_first_checker="Vou escrever o teste e validar tudo junto.",
        test_path="test_is_sorted.py",
        buggy_test_content=(
            "from is_sorted import is_sorted\n\n\n"
            "def test_is_sorted():\n"
            "    data = [1, 2, 3]\n"
            "    assert is_sorted(dat) is True\n"
        ),
        think_after_diagnosis=(
            "O erro do checker é 'NameError: name dat is not defined' no test_is_sorted.py — "
            "o teste tem um typo, referenciando 'dat' em vez de 'data'. A implementação em "
            "is_sorted.py está correta. O bug é só no teste — vou corrigir o nome da "
            "variável."
        ),
        fixed_test_content=(
            "from is_sorted import is_sorted\n\n\n"
            "def test_is_sorted():\n"
            "    data = [1, 2, 3]\n"
            "    assert is_sorted(data) is True\n"
        ),
        final_text=(
            "Implementei is_sorted(items). O primeiro teste falhou por um typo no nome de uma "
            "variável dentro do próprio teste (bug no teste, não na implementação) — corrigi "
            "só o teste. A validação confirma que compila e o teste passa."
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
        print(f"OK: {out_path.name} (bug real no teste, diagnosticado e corrigido de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
