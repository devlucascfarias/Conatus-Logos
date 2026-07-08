#!/usr/bin/env python
"""Lote base do Gap 5 (docs/plan_dataset_expansion_confabulation_gap.md): quando o usuário
alega um bug que na verdade NÃO existe, o `<final>` não pode inventar uma narrativa de
diagnóstico/correção que nunca aconteceu na trajetória real.

Achado real: pedido de "descobrir a causa raiz e corrigir" um teste supostamente falhando —
o código que o próprio modelo escreveu passou de primeira (nunca existiu bug real), mas o
`<final>` alegou "a soma estava sendo subtraída... corrigi só o teste", uma história sem
nenhuma base nas chamadas de ferramenta reais da trajetória.

Cada exemplo: `<think>` -> `write_file` real (implementação já correta) -> `<think>` ->
`checker` real com [implementação, teste cobrindo exatamente o caso que o usuário alegou] ->
PASSA de primeira (nunca há uma segunda rodada) -> `<think>` reconhecendo explicitamente que
a validação passou sem nenhuma correção real ter sido necessária -> `<final>` honesto, que
nunca alega ter corrigido algo que não foi alterado.

Uso:
    python scripts/gen_confabulation_gap_pilot.py
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
    think_before_checker, test_path, test_content,
    think_before_final, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_write}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": file_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        traj.append_raw(f"<think>{think_before_checker}</think>")
        checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": file_content},
                {"path": test_path, "content": test_content},
            ],
        }
        checker_result = _tool_call(traj, sandbox, "checker", checker_args)
        assert checker_result.passed, (
            f"esperava sucesso real de primeira (sem bug nenhum) em {id_}: {checker_result.to_json()}"
        )

        traj.append_raw(f"<think>{think_before_final}</think>")
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
            "task_type": "no_bug_found_report",
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
        id_="gen-confab-is-palindrome",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Acho que minha função is_palindrome está contando espaços e maiúsculas errado — "
            "testei com 'A ra ra' e não sei se está tratando certo. Pode conferir e corrigir "
            "se precisar?"
        ),
        think_before_write=(
            "O usuário alega que espaços/maiúsculas quebram a checagem de palíndromo. Vou "
            "implementar tratando isso desde o início — normalizando caixa e removendo "
            "espaços antes de comparar com o reverso."
        ),
        file_path="is_palindrome.py",
        file_content=(
            "def is_palindrome(s):\n"
            "    cleaned = s.lower().replace(' ', '')\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        think_before_checker=(
            "Vou escrever um teste cobrindo exatamente o caso que o usuário citou ('A ra ra') "
            "antes de assumir que existe algum bug."
        ),
        test_path="test_is_palindrome.py",
        test_content=(
            "from is_palindrome import is_palindrome\n\n\n"
            "def test_is_palindrome():\n"
            "    assert is_palindrome('A ra ra') is True\n"
            "    assert is_palindrome('arara') is True\n"
            "    assert is_palindrome('teste') is False\n"
        ),
        think_before_final=(
            "O teste, incluindo o caso exato mencionado pelo usuário, passou já na primeira "
            "execução real — não fiz nenhuma correção porque não havia nada quebrado nesta "
            "implementação. Vou reportar isso com precisão, sem inventar um diagnóstico que "
            "não aconteceu."
        ),
        final_text=(
            "Implementei is_palindrome(s) normalizando caixa e espaços antes de comparar. Os "
            "testes reais, incluindo o caso com espaços e maiúsculas que você mencionou ('A ra "
            "ra'), passaram já na primeira execução — não precisei corrigir nada. Se o "
            "comportamento que você viu de errado usava uma entrada diferente da que testei "
            "aqui, me diga o valor exato para eu conferir esse caso específico."
        ),
    ),
    dict(
        id_="gen-confab-count-vowels",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "O count_vowels parece estar contando vogais maiúsculas errado, tipo em 'PYTHON'. "
            "Confere e corrige se for o caso?"
        ),
        think_before_write=(
            "O usuário alega que maiúsculas quebram a contagem de vogais. Vou implementar "
            "normalizando para minúsculas antes de contar, desde o início."
        ),
        file_path="count_vowels.py",
        file_content="def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        think_before_checker=(
            "Vou escrever um teste cobrindo exatamente o caso citado ('PYTHON') antes de "
            "assumir que existe algum bug."
        ),
        test_path="test_count_vowels.py",
        test_content=(
            "from count_vowels import count_vowels\n\n\n"
            "def test_count_vowels():\n"
            "    assert count_vowels('PYTHON') == 1\n"
            "    assert count_vowels('banana') == 3\n"
            "    assert count_vowels('') == 0\n"
        ),
        think_before_final=(
            "O teste, incluindo o caso das maiúsculas que o usuário mencionou, passou já na "
            "primeira execução real — não houve correção nenhuma, porque não havia bug nesta "
            "implementação. Vou reportar isso com precisão."
        ),
        final_text=(
            "Implementei count_vowels(s) normalizando para minúsculas antes de contar. O "
            "teste real, incluindo o caso com 'PYTHON' que você mencionou, passou já na "
            "primeira execução — não precisei corrigir nada. Se o comportamento errado que "
            "você viu veio de uma entrada diferente, me diga qual para eu conferir."
        ),
    ),
    dict(
        id_="gen-confab-average",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Acho que minha função average quebra com lista vazia (ZeroDivisionError). Pode "
            "conferir e corrigir?"
        ),
        think_before_write=(
            "O usuário alega que lista vazia quebra o cálculo da média. Vou implementar já "
            "tratando esse caso desde o início, retornando 0.0 para lista vazia."
        ),
        file_path="average.py",
        file_content="def average(nums):\n    return sum(nums) / len(nums) if nums else 0.0\n",
        think_before_checker=(
            "Vou escrever um teste cobrindo exatamente o caso de lista vazia que o usuário "
            "citou antes de assumir que existe algum bug."
        ),
        test_path="test_average.py",
        test_content=(
            "from average import average\n\n\n"
            "def test_average():\n"
            "    assert average([]) == 0.0\n"
            "    assert average([2, 4, 6]) == 4.0\n"
            "    assert average([5]) == 5.0\n"
        ),
        think_before_final=(
            "O teste, incluindo o caso de lista vazia que o usuário mencionou, passou já na "
            "primeira execução real — não houve correção nenhuma, porque essa implementação "
            "já trata lista vazia. Vou reportar isso com precisão."
        ),
        final_text=(
            "Implementei average(nums) já tratando lista vazia (retorna 0.0). O teste real, "
            "incluindo o caso de lista vazia que você mencionou, passou já na primeira "
            "execução — não precisei corrigir nada. Se o `ZeroDivisionError` que você viu veio "
            "de outra versão do código, me manda ela que eu confiro."
        ),
    ),
    dict(
        id_="gen-confab-is-prime",
        domain="tratamento_erro",
        difficulty="hard",
        user_request=(
            "Acho que is_prime está classificando 1 como primo por engano. Pode conferir e "
            "corrigir?"
        ),
        think_before_write=(
            "O usuário alega que 1 é classificado como primo. Vou implementar já tratando "
            "n < 2 como não-primo desde o início."
        ),
        file_path="is_prime.py",
        file_content=(
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        think_before_checker=(
            "Vou escrever um teste cobrindo exatamente o caso de n=1 que o usuário citou antes "
            "de assumir que existe algum bug."
        ),
        test_path="test_is_prime.py",
        test_content=(
            "from is_prime import is_prime\n\n\n"
            "def test_is_prime():\n"
            "    assert is_prime(1) is False\n"
            "    assert is_prime(2) is True\n"
            "    assert is_prime(17) is True\n"
            "    assert is_prime(15) is False\n"
        ),
        think_before_final=(
            "O teste, incluindo o caso de n=1 que o usuário mencionou, passou já na primeira "
            "execução real — não houve correção nenhuma, porque essa implementação já trata "
            "n=1 como não-primo. Vou reportar isso com precisão."
        ),
        final_text=(
            "Implementei is_prime(n) já tratando n < 2 como não-primo. O teste real, incluindo "
            "o caso de n=1 que você mencionou, passou já na primeira execução — não precisei "
            "corrigir nada. Se o comportamento errado que você viu veio de outra versão do "
            "código, me manda ela que eu confiro."
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
        print(f"OK: {out_path.name} (passou de primeira, sem bug real, <final> honesto)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
