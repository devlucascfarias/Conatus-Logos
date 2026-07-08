#!/usr/bin/env python
"""Lote piloto de exemplos ensinando recuperação de erro de ferramenta (PLAN.md seção 17,
risco "final_segment_leaks_internal_tags só detecta tags, não conteúdo copiado").

Achado real: pedido de código de FFT no Colab → `web_search` falhou por
`PRAXIS_OLLAMA_SEARCH_API_KEY` não configurada (falha de AMBIENTE, não do modelo) → o modelo
copiou a mensagem de erro interna verbatim no `<final>` (vazando nome de variável de ambiente e
referência a seção do PLAN.md) e desistiu da tarefa, em vez de implementar direto sem depender
da busca.

Diferente da expansão por paráfrase (scripts/expand_dataset_paraphrases.py), aqui a trajetória
é NOVA, não reaproveitada — por isso, seguindo a discussão sobre "destilação de raciocínio",
TODO `<tool_result>` vem de execução real via `ToolExecutorRegistry`/`SandboxContext` (o mesmo
código do harness de produção), nunca fabricado por texto. O `web_search` falha de propósito
(sem `PRAXIS_OLLAMA_SEARCH_API_KEY` neste ambiente de dev — mesma condição do bug real), e o
`write_file`/`checker` rodam de verdade contra código correto que eu escrevi e validei.

Uso:
    python scripts/gen_tool_recovery_pilot.py
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
    """`compile_and_test` roda pytest de verdade no ambiente de quem gera o exemplo — plugins
    de dev instalados só nesta máquina (ex.: pytest-asyncio, não declarado em
    requirements.txt/requirements-train.txt) podem imprimir avisos de depreciação no stderr,
    incluindo caminho de arquivo local. Isso não reflete o ambiente real do Colab e não deve
    entrar no dataset — em execução passante, stderr vira "" (mesmo padrão dos exemplos
    existentes que passam sem aviso nenhum)."""
    if result.passed and result.data.get("stderr"):
        result.data["stderr"] = ""
    return result


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict) -> "object":
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
    think_after_search_fails: str,
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

        traj.append_raw(f"<think>{think_before_search}</think>")
        search_result = _tool_call(traj, sandbox, "web_search", {"query": search_query})
        assert not search_result.passed, (
            f"esperava que web_search falhasse neste ambiente (sem PRAXIS_OLLAMA_SEARCH_API_KEY) "
            f"para {id_}, mas passou — PRAXIS_OLLAMA_SEARCH_API_KEY está definida?"
        )

        traj.append_raw(f"<think>{think_after_search_fails}</think>")
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
            "tools_used": ["web_search", "write_file", "checker"],
            "num_steps": 3,
            "task_type": "tool_unavailable",
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


_RECOVERY_TEXT = (
    "Não consegui confirmar isso pela busca — não está disponível neste ambiente — mas "
    "implementei a função com base no que já sei; vou validar antes de responder."
)


EXAMPLES = [
    dict(
        id_="gen-toolrecovery-dot_product",
        domain="algorithms",
        difficulty="easy",
        user_request="Escreva uma função em Python que calcula o produto interno (dot product) de dois vetores.",
        search_query="python dot product two vectors implementation",
        think_before_search="Vou confirmar a convenção de assinatura mais comum antes de implementar.",
        think_after_search_fails=_RECOVERY_TEXT,
        file_path="dot_product.py",
        file_content="def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))\n",
        test_path="test_dot_product.py",
        test_content=(
            "from dot_product import dot_product\n\n"
            "def test_dot_product():\n"
            "    assert dot_product([1, 2, 3], [4, 5, 6]) == 32\n"
            "    assert dot_product([0, 0], [1, 1]) == 0\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "A pesquisa não estava disponível neste ambiente, mas implementei dot_product(a, b) "
            "com base no que já sei — soma do produto elemento a elemento. A validação confirma "
            "que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-toolrecovery-leap_year",
        domain="algorithms",
        difficulty="easy",
        user_request="Crie uma função que verifica se um ano é bissexto.",
        search_query="regra ano bissexto divisível por 4 100 400",
        think_before_search="Vou confirmar a regra exata (múltiplos de 100 não são bissextos, exceto múltiplos de 400) antes de codificar.",
        think_after_search_fails=_RECOVERY_TEXT,
        file_path="leap_year.py",
        file_content="def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n",
        test_path="test_leap_year.py",
        test_content=(
            "from leap_year import is_leap_year\n\n"
            "def test_is_leap_year():\n"
            "    assert is_leap_year(2000) is True\n"
            "    assert is_leap_year(1900) is False\n"
            "    assert is_leap_year(2024) is True\n"
            "    assert is_leap_year(2023) is False\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "A pesquisa não estava disponível neste ambiente, mas implementei is_leap_year(year) "
            "com base na regra que já conhecia (divisível por 4, exceto séculos não divisíveis "
            "por 400). A validação confirma que compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-toolrecovery-transpose",
        domain="algorithms",
        difficulty="easy",
        user_request="Escreva uma função que transpõe uma matriz representada como lista de listas.",
        search_query="python transpose matrix list of lists zip",
        think_before_search="Vou confirmar a forma mais idiomática antes de implementar.",
        think_after_search_fails=_RECOVERY_TEXT,
        file_path="transpose.py",
        file_content="def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
        test_path="test_transpose.py",
        test_content=(
            "from transpose import transpose\n\n"
            "def test_transpose():\n"
            "    assert transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]\n"
            "    assert transpose([[1]]) == [[1]]\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "A pesquisa não estava disponível neste ambiente, mas implementei transpose(matrix) "
            "com base no que já sei (zip(*matrix)). A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-toolrecovery-count_words",
        domain="algorithms",
        difficulty="easy",
        user_request="Implemente uma função que conta quantas palavras tem uma string.",
        search_query="python count words in string split whitespace",
        think_before_search="Vou confirmar se split() sem argumento já lida bem com espaços múltiplos antes de implementar.",
        think_after_search_fails=_RECOVERY_TEXT,
        file_path="count_words.py",
        file_content="def count_words(text):\n    return len(text.split())\n",
        test_path="test_count_words.py",
        test_content=(
            "from count_words import count_words\n\n"
            "def test_count_words():\n"
            "    assert count_words('um dois tres') == 3\n"
            "    assert count_words('  espacos   extras  ') == 2\n"
            "    assert count_words('') == 0\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "A pesquisa não estava disponível neste ambiente, mas implementei count_words(text) "
            "com base no que já sei (split() já ignora espaços extras). A validação confirma que "
            "compila e os testes passam."
        ),
    ),
    dict(
        id_="gen-toolrecovery-palindrome_clean",
        domain="algorithms",
        difficulty="medium",
        user_request="Crie uma função que verifica se uma string é um palíndromo, ignorando espaços, pontuação e maiúsculas.",
        search_query="python check palindrome ignore spaces punctuation case",
        think_before_search="Vou confirmar a forma mais robusta de normalizar a string antes de implementar.",
        think_after_search_fails=_RECOVERY_TEXT,
        file_path="palindrome_clean.py",
        file_content=(
            "def is_palindrome_clean(text):\n"
            "    normalized = ''.join(c.lower() for c in text if c.isalnum())\n"
            "    return normalized == normalized[::-1]\n"
        ),
        test_path="test_palindrome_clean.py",
        test_content=(
            "from palindrome_clean import is_palindrome_clean\n\n"
            "def test_is_palindrome_clean():\n"
            "    assert is_palindrome_clean('A man a plan a canal Panama') is True\n"
            "    assert is_palindrome_clean('hello') is False\n"
            "    assert is_palindrome_clean('') is True\n"
        ),
        think_before_final="A implementação está pronta. Vou validar com o checker.",
        final_text=(
            "A pesquisa não estava disponível neste ambiente, mas implementei is_palindrome_clean(text) "
            "com base no que já sei (normalizar removendo não-alfanuméricos e comparar com o "
            "reverso). A validação confirma que compila e os testes passam."
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
