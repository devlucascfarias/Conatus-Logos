#!/usr/bin/env python
"""Lote base do Gap 4 (docs/plan_dataset_expansion_hypothesis_revision.md): revisão de
hipótese depois de uma SEGUNDA falha real — diferente dos Gaps 2a/2b (que sempre acertam o
diagnóstico na 2ª tentativa), aqui a 1ª correção também falha de verdade, baseada numa teoria
plausível mas errada.

Achado real: nos casos Go e Sudoku, o modelo repetiu a MESMA chamada até `MAX_STEPS_EXCEEDED`
sem nunca revisar a própria hipótese. Aqui ensinamos: falha real -> correção com teoria
plausível -> falha real de NOVO (prova que a teoria estava errada) -> reconhecimento
explícito de que a correção anterior não resolveu -> segunda teoria, agora certa -> sucesso
real.

Cada exemplo: 3 rodadas de `<think>` -> `write_file` real -> `checker` real, sendo as duas
primeiras rodadas falhas GENUÍNAS (nunca fabricadas) e só a terceira passa.

Uso:
    python scripts/gen_hypothesis_revision_pilot.py
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


def _round(traj, sandbox, think_text, file_path, content, test_path, test_content):
    traj.append_raw(f"<think>{think_text}</think>")
    write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": content})
    assert write_result.passed, f"write_file falhou inesperadamente: {write_result.to_json()}"
    checker_args = {
        "language": "python",
        "operation": "compile_and_test",
        "files": [
            {"path": file_path, "content": content},
            {"path": test_path, "content": test_content},
        ],
    }
    return _tool_call(traj, sandbox, "checker", checker_args)


def build_example(
    id_, domain, difficulty, user_request, file_path, test_path, test_content,
    think1, content1,
    think2, content2,
    think3, content3,
    final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        r1 = _round(traj, sandbox, think1, file_path, content1, test_path, test_content)
        assert not r1.passed, f"esperava falha real na 1ª rodada em {id_}: {r1.to_json()}"

        r2 = _round(traj, sandbox, think2, file_path, content2, test_path, test_content)
        assert not r2.passed, f"esperava falha real na 2ª rodada (teoria errada) em {id_}: {r2.to_json()}"

        r3 = _round(traj, sandbox, think3, file_path, content3, test_path, test_content)
        assert r3.passed, f"esperava sucesso real na 3ª rodada em {id_}: {r3.to_json()}"

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
            "num_steps": 6,
            "task_type": "hypothesis_revision_after_failure",
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
        id_="gen-hyprevision-sum-up-to",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que some todos os inteiros de 1 até n, incluindo n.",
        file_path="sum_up_to.py",
        test_path="test_sum_up_to.py",
        test_content=(
            "from sum_up_to import sum_up_to\n\n\n"
            "def test_sum_up_to():\n"
            "    assert sum_up_to(5) == 15\n"
            "    assert sum_up_to(1) == 1\n"
        ),
        think1="Vou implementar sum_up_to(n) somando um range até n.",
        content1="def sum_up_to(n):\n    return sum(range(1, n))\n",
        think2=(
            "O teste falhou (sum_up_to(5) deu 10, esperava 15). Talvez o problema seja o "
            "início do range — vou tentar começar de 0 em vez de 1."
        ),
        content2="def sum_up_to(n):\n    return sum(range(0, n))\n",
        think3=(
            "Minha correção anterior (começar do 0) não resolveu — o teste ainda falha com o "
            "mesmo valor (10 em vez de 15), o que mostra que a teoria estava errada: o "
            "problema nunca foi o início do range. range(1, n) e range(0, n) excluem o próprio "
            "n de qualquer forma — preciso incluir n somando até n+1."
        ),
        content3="def sum_up_to(n):\n    return sum(range(1, n + 1))\n",
        final_text=(
            "Implementei sum_up_to(n). Minha primeira correção (mudar o início do range) não "
            "resolveu — o problema real era o fim do range, que precisava incluir n. Com "
            "range(1, n + 1), a validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-hyprevision-is-in-range",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que verifique se um valor está dentro de um intervalo, incluindo os dois limites.",
        file_path="is_in_range.py",
        test_path="test_is_in_range.py",
        test_content=(
            "from is_in_range import is_in_range\n\n\n"
            "def test_is_in_range():\n"
            "    assert is_in_range(1, 1, 5) is True\n"
            "    assert is_in_range(5, 1, 5) is True\n"
            "    assert is_in_range(3, 1, 5) is True\n"
            "    assert is_in_range(0, 1, 5) is False\n"
        ),
        think1="Vou implementar is_in_range(value, low, high) comparando os limites.",
        content1="def is_in_range(value, low, high):\n    return low < value < high\n",
        think2=(
            "O teste falhou nos limites (1 e 5 deram False, deveriam ser True). Talvez o "
            "problema seja só o limite superior — vou trocar high para inclusivo."
        ),
        content2="def is_in_range(value, low, high):\n    return low < value <= high\n",
        think3=(
            "Minha correção anterior (só o limite superior) não resolveu tudo — is_in_range(1, "
            "1, 5) ainda falha, porque o limite inferior continua exclusivo. A teoria estava "
            "incompleta: os DOIS limites precisam ser inclusivos, não só um."
        ),
        content3="def is_in_range(value, low, high):\n    return low <= value <= high\n",
        final_text=(
            "Implementei is_in_range(value, low, high). Minha primeira correção tratou só o "
            "limite superior como inclusivo, mas o limite inferior também precisava ser — com "
            "os dois corrigidos, a validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-hyprevision-last-n-items",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que retorne os últimos n itens de uma lista.",
        file_path="last_n_items.py",
        test_path="test_last_n_items.py",
        test_content=(
            "from last_n_items import last_n_items\n\n\n"
            "def test_last_n_items():\n"
            "    assert last_n_items([1, 2, 3, 4, 5], 2) == [4, 5]\n"
            "    assert last_n_items([1, 2, 3], 1) == [3]\n"
        ),
        think1="Vou implementar last_n_items(items, n) com slicing.",
        content1="def last_n_items(items, n):\n    return items[:n]\n",
        think2=(
            "O teste falhou (retornou os PRIMEIROS n itens, não os últimos). Talvez o "
            "problema seja o lado do slice — vou tentar items[n:] para pegar do índice n até "
            "o fim."
        ),
        content2="def last_n_items(items, n):\n    return items[n:]\n",
        think3=(
            "Minha correção anterior (items[n:]) também falhou — para [1,2,3,4,5] com n=2, "
            "isso retorna [3,4,5] (3 itens, não 2), porque items[n:] conta a partir do início, "
            "não do fim. A teoria estava errada: eu precisava de um índice NEGATIVO contado a "
            "partir do fim, não um índice positivo a partir do começo."
        ),
        content3="def last_n_items(items, n):\n    return items[-n:]\n",
        final_text=(
            "Implementei last_n_items(items, n). Minha primeira correção (items[n:]) ainda "
            "contava a partir do início da lista, não do fim — com items[-n:], a validação "
            "confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-hyprevision-dedup-keep-order",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que remova itens duplicados de uma lista, preservando a ordem original de primeira aparição.",
        file_path="dedup_keep_order.py",
        test_path="test_dedup_keep_order.py",
        test_content=(
            "from dedup_keep_order import dedup_keep_order\n\n\n"
            "def test_dedup_keep_order():\n"
            "    assert dedup_keep_order(['banana', 'apple', 'cherry', 'apple', 'banana']) == "
            "['banana', 'apple', 'cherry']\n"
        ),
        think1="Vou implementar dedup_keep_order(items) usando set() para remover duplicatas.",
        content1="def dedup_keep_order(items):\n    return list(set(items))\n",
        think2=(
            "O teste falhou — set() não preserva a ordem original. Talvez o problema seja a "
            "falta de um critério determinístico — vou tentar ordenar o resultado."
        ),
        content2="def dedup_keep_order(items):\n    return sorted(set(items))\n",
        think3=(
            "Minha correção anterior (ordenar) também falhou — ordenar dá um resultado "
            "determinístico, mas não é a ORDEM ORIGINAL pedida, é ordem alfabética. A teoria "
            "estava errada: o problema nunca foi falta de determinismo, era preservar a ordem "
            "de primeira aparição — preciso percorrer a lista original e rastrear o que já vi, "
            "não usar set() nem sorted() diretamente no resultado."
        ),
        content3=(
            "def dedup_keep_order(items):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        final_text=(
            "Implementei dedup_keep_order(items). Minhas duas primeiras tentativas (set() e "
            "sorted(set())) removiam duplicatas mas não preservavam a ordem original de "
            "aparição — rastreando os itens já vistos enquanto percorro a lista, a validação "
            "confirma que compila e o teste passa."
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
        print(f"OK: {out_path.name} (2 falhas reais + 1 sucesso real, teoria revisada de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
