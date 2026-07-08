#!/usr/bin/env python
"""Lote piloto ensinando o caminho de SUCESSO da busca (PLAN.md seção 17/8.6) — complementa
`scripts/gen_tool_recovery_pilot.py`, que só cobre `web_search` FALHANDO.

Achado real: pedido de código de FFT, com `web_search` funcionando de verdade (chave da Ollama
configurada) → o modelo pesquisou, achou a documentação certa, mas NUNCA escreveu nem validou
código nenhum — respondeu só com texto (em inglês, terminando com "Example code follows." sem
nenhum código depois). O dataset não tinha nenhum exemplo do padrão "busca funciona → ainda
assim preciso produzir e validar um artefato real".

Cada exemplo aqui: `<think>` → `web_search` bem-sucedido (via `MockSearchBackend` com fixtures
em `data/fixtures/web_search/`, coletadas de buscas reais, NUNCA fabricadas por texto solto) →
`<think>` sintetizando o que foi encontrado → `write_file` real → `checker` real
(`compile_and_test`) → `<final>` limpo, em português, descrevendo o que foi feito.

Uso:
    python scripts/gen_tool_success_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402
from src.search.mock_backend import MockSearchBackend  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402
from src.tools.executors import web_search_tool  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures" / "web_search"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()


def _sanitize_checker_result(result):
    """Mesma sanitização de `gen_tool_recovery_pilot.py`: um aviso de depreciação do
    pytest-asyncio (só instalado nesta máquina local, não é dependência do projeto) polui o
    stderr com um caminho de arquivo do Windows — não deve entrar no dataset."""
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
    search_query: str,
    think_before_search: str,
    think_after_search: str,
    file_path: str,
    file_content: str,
    test_path: str,
    test_content: str,
    think_before_final: str,
    final_text: str,
) -> dict:
    # Força o MockSearchBackend com as fixtures reais coletadas — nunca a config de produção
    # (que continua em "ollama"); isolado por chamada para não vazar estado entre exemplos.
    web_search_tool._cached_backend = MockSearchBackend(FIXTURES_DIR)

    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_search}</think>")
        search_result = _tool_call(traj, sandbox, "web_search", {"query": search_query})
        assert search_result.passed, f"esperava web_search OK (fixture mock) em {id_}: {search_result.to_json()}"
        assert search_result.data.get("results"), f"fixture de busca vazia para {id_} ({search_query})"

        traj.append_raw(f"<think>{think_after_search}</think>")
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
        web_search_tool.reset_cached_backend()

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "python",
            "difficulty": difficulty,
            "tools_used": ["web_search", "write_file", "checker"],
            "num_steps": 3,
            "task_type": "multi_tool_call",
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
        id_="gen-toolsuccess-fourier_transform",
        domain="algorithms",
        difficulty="medium",
        user_request="Faça um código que calcule a transformada de Fourier de um sinal usando numpy.",
        search_query="numpy fft",
        think_before_search="Vou confirmar a função certa do numpy antes de implementar.",
        think_after_search="A documentação confirma: numpy.fft.fft calcula a transformada discreta de Fourier em 1D. Vou implementar com base nisso.",
        file_path="fourier_transform.py",
        file_content="import numpy as np\n\n\ndef fourier_transform(signal):\n    return np.fft.fft(signal)\n",
        test_path="test_fourier_transform.py",
        test_content=(
            "from fourier_transform import fourier_transform\n\n\n"
            "def test_fourier_transform():\n"
            "    result = fourier_transform([1, 2, 3, 4])\n"
            "    assert len(result) == 4\n"
            "    assert abs(result[0].real - 10) < 1e-9\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "Confirmei pela documentação do numpy que numpy.fft.fft é a função certa para a "
            "transformada discreta de Fourier em 1D e implementei fourier_transform(signal) com "
            "base nisso. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-toolsuccess-merge_sort",
        domain="algorithms",
        difficulty="medium",
        user_request="Escreva uma função que ordene uma lista usando o algoritmo mergesort.",
        search_query="python mergesort algorithm implementation",
        think_before_search="Vou confirmar os passos clássicos do mergesort antes de implementar.",
        think_after_search="A documentação confirma: dividir a lista ao meio recursivamente até sublistas de 1 elemento, depois mesclar em ordem. Vou implementar com base nisso.",
        file_path="merge_sort.py",
        file_content=(
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return _merge(left, right)\n\n\n"
            "def _merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    result.extend(left[i:])\n"
            "    result.extend(right[j:])\n"
            "    return result\n"
        ),
        test_path="test_merge_sort.py",
        test_content=(
            "from merge_sort import merge_sort\n\n\n"
            "def test_merge_sort():\n"
            "    assert merge_sort([5, 3, 4, 1, 2]) == [1, 2, 3, 4, 5]\n"
            "    assert merge_sort([]) == []\n"
            "    assert merge_sort([1]) == [1]\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "Confirmei pela documentação a estrutura clássica do mergesort (dividir "
            "recursivamente e mesclar em ordem) e implementei merge_sort(arr) com base nisso. "
            "A validação confirma que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-toolsuccess-gcd_euclid",
        domain="algorithms",
        difficulty="easy",
        user_request="Implemente uma função que calcula o máximo divisor comum usando o algoritmo de Euclides.",
        search_query="python euclidean algorithm gcd implementation",
        think_before_search="Vou confirmar a versão iterativa do algoritmo de Euclides antes de implementar.",
        think_after_search="A documentação confirma: a versão iterativa (a, b = b, a % b até b chegar a 0) é mais eficiente em memória que a recursiva. Vou implementar com base nisso.",
        file_path="gcd_euclid.py",
        file_content="def gcd_euclid(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        test_path="test_gcd_euclid.py",
        test_content=(
            "from gcd_euclid import gcd_euclid\n\n\n"
            "def test_gcd_euclid():\n"
            "    assert gcd_euclid(48, 18) == 6\n"
            "    assert gcd_euclid(17, 5) == 1\n"
            "    assert gcd_euclid(0, 5) == 5\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "Confirmei pela documentação que a versão iterativa do algoritmo de Euclides é a "
            "mais eficiente e implementei gcd_euclid(a, b) com base nisso. A validação confirma "
            "que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-toolsuccess-binary_search",
        domain="algorithms",
        difficulty="easy",
        user_request="Escreva um código que implementa busca binária em uma lista ordenada.",
        search_query="python binary search implementation",
        think_before_search="Vou confirmar a abordagem iterativa clássica antes de implementar.",
        think_after_search="A documentação confirma: usar ponteiros low/high, calcular mid a cada iteração e reduzir o intervalo pela metade. Vou implementar com base nisso.",
        file_path="binary_search.py",
        file_content=(
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
        test_path="test_binary_search.py",
        test_content=(
            "from binary_search import binary_search\n\n\n"
            "def test_binary_search():\n"
            "    assert binary_search([1, 3, 5, 7, 9], 5) == 2\n"
            "    assert binary_search([1, 3, 5, 7, 9], 4) == -1\n"
            "    assert binary_search([], 1) == -1\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "Confirmei pela documentação a abordagem iterativa clássica de busca binária e "
            "implementei binary_search(arr, target) com base nisso. A validação confirma que "
            "compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-toolsuccess-is_prime",
        domain="algorithms",
        difficulty="easy",
        user_request="Crie uma função que verifica se um número é primo de forma eficiente.",
        search_query="python efficient primality test algorithm",
        think_before_search="Vou confirmar a abordagem mais eficiente para números pequenos/moderados antes de implementar.",
        think_after_search="A documentação confirma: divisão por tentativa só até a raiz quadrada de n, pulando pares depois do 2, já é eficiente para esse caso. Vou implementar com base nisso.",
        file_path="is_prime.py",
        file_content=(
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n == 2:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n"
        ),
        test_path="test_is_prime.py",
        test_content=(
            "from is_prime import is_prime\n\n\n"
            "def test_is_prime():\n"
            "    assert is_prime(17) is True\n"
            "    assert is_prime(1) is False\n"
            "    assert is_prime(2) is True\n"
            "    assert is_prime(15) is False\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "Confirmei pela documentação que divisão por tentativa até a raiz quadrada de n, "
            "pulando pares, é eficiente para esse caso, e implementei is_prime(n) com base "
            "nisso. A validação confirma que compila e os testes passam."
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
        print(f"OK: {out_path.name} (busca simulada com fixture real + checker passou de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
