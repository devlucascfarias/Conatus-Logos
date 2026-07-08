#!/usr/bin/env python
"""Lote base do Gap 2a (docs/plan_dataset_expansion_error_recovery.md): recuperação de
`TOOL_CALL_PARSE_ERROR` em código mais longo/complexo, com execução real de ponta a ponta.

Achado real: pedido de validador de Sudoku, o código gerado (loop aninhado + regex) teve uma
sequência de escape inválida (`\\` solto antes de espaço) quebrando o JSON do `tool_call`. O
modelo repetiu a MESMA chamada malformada 6 vezes, culpando "a ferramenta"/"o ambiente".

Diferente dos 10 exemplos antigos de `invalid_call_then_correction` (typos triviais de uma
linha, nunca executados de verdade), aqui: a primeira chamada malformada é processada pela
MESMA função que o harness usa em produção (`src.parsers.parse_last_segment`), então o
`TOOL_CALL_PARSE_ERROR` é real, não fabricado; a correção é uma chamada `write_file` de verdade
via `ToolExecutorRegistry`, seguida de `checker` real.

Cada exemplo: `<think>` -> `<tool_call name="write_file">` malformado (JSON inválido de
propósito, capturado por `parse_last_segment` de verdade) -> `<tool_result ... status="error">`
(gerado pelo mesmo mecanismo do harness) -> `<think>` reconhecendo, em tom neutro, que o
PRÓPRIO JSON estava malformado (não "a ferramenta"/"o ambiente") -> `write_file` real (JSON
válido) -> `checker` real -> `<final>`.

Uso:
    python scripts/gen_json_recovery_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402
from src.parsers import parse_last_segment  # noqa: E402
from src.parsers.segments import MalformedSegment  # noqa: E402
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


def _malformed_write_file_call(traj: Trajectory, path: str, malformed_body: str) -> None:
    """Injeta um <tool_call name="write_file"> com JSON deliberadamente inválido e processa
    através do MESMO parser que o harness usa em produção (`parse_last_segment`), obtendo um
    `TOOL_CALL_PARSE_ERROR` real — nunca fabricado."""
    tool_call_text = f'<tool_call name="write_file">{malformed_body}</tool_call>'
    traj.append_raw(tool_call_text)
    segment = parse_last_segment(traj.raw_text)
    assert isinstance(segment, MalformedSegment), f"esperava MalformedSegment, veio {type(segment).__name__}"
    assert segment.reason == "TOOL_CALL_PARSE_ERROR", f"esperava TOOL_CALL_PARSE_ERROR, veio {segment.reason}"
    traj.append_tool_result(
        segment.attempted_name or "write_file", "error", {"code": segment.reason, "message": segment.message}
    )


def build_example(
    id_, domain, difficulty, user_request,
    think_before_write, file_path, malformed_body,
    think_after_parse_error, corrected_content,
    test_path, test_content, think_before_final, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_write}</think>")
        _malformed_write_file_call(traj, file_path, malformed_body)

        traj.append_raw(f"<think>{think_after_parse_error}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": corrected_content})
        assert write_result.passed, f"write_file (corrigido) falhou inesperadamente em {id_}: {write_result.to_json()}"

        checker_args = {
            "language": "python",
            "operation": "compile_and_test",
            "files": [
                {"path": file_path, "content": corrected_content},
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
            "num_steps": 3,
            "task_type": "tool_call_json_recovery",
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
        id_="gen-jsonrecovery-word-frequency",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que conte a frequência de cada palavra em um texto, ignorando maiúsculas.",
        think_before_write="Vou implementar word_frequency(text) com um loop aninhado simples sobre as palavras.",
        file_path="word_frequency.py",
        malformed_body=(
            '{"path": "word_frequency.py", "content": '
            '"def word_frequency(text):\\n    freq = {}\\n    words = text.lower().split()\\n'
            '    for word in words:\\        if word in freq:\\n            freq[word] += 1\\n'
            '        else:\\n            freq[word] = 1\\n    return freq\\n"}'
        ),
        think_after_parse_error=(
            "O JSON da minha última chamada estava malformado (uma barra invertida solta antes "
            "de um espaço, quebrando a string). O problema é na minha formatação, não na "
            "ferramenta nem no ambiente — vou reescrever o conteúdo corretamente e tentar de novo."
        ),
        corrected_content=(
            "def word_frequency(text):\n"
            "    freq = {}\n"
            "    words = text.lower().split()\n"
            "    for word in words:\n"
            "        if word in freq:\n"
            "            freq[word] += 1\n"
            "        else:\n"
            "            freq[word] = 1\n"
            "    return freq\n"
        ),
        test_path="test_word_frequency.py",
        test_content=(
            "from word_frequency import word_frequency\n\n\n"
            "def test_word_frequency():\n"
            "    result = word_frequency('O gato e o cachorro e o gato')\n"
            "    assert result['o'] == 3\n"
            "    assert result['gato'] == 2\n"
        ),
        think_before_final="Agora o JSON está correto e o código está pronto. Vou validar.",
        final_text=(
            "Implementei word_frequency(text), contando a frequência de cada palavra "
            "ignorando maiúsculas. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-jsonrecovery-extract-emails",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que extraia todos os e-mails de um texto usando regex.",
        think_before_write="Vou implementar extract_emails(text) usando re.findall com um padrão de e-mail.",
        file_path="extract_emails.py",
        malformed_body=(
            '{"path": "extract_emails.py", "content": '
            '"import re\\n\\n\\ndef extract_emails(text):\\n    pattern = r\'[\\\\w.+-]+@[\\\\w-]+\\.[\\\\w.-]+\'\\n'
            '    matches = []\\    for m in re.findall(pattern, text):\\n        matches.append(m)\\n    return matches\\n"}'
        ),
        think_after_parse_error=(
            "O JSON da minha última chamada estava malformado — uma barra invertida solta "
            "antes de um espaço quebrou a string. É um problema na minha formatação, não na "
            "ferramenta — vou reescrever com o escaping correto e tentar de novo."
        ),
        corrected_content=(
            "import re\n\n\n"
            "def extract_emails(text):\n"
            "    pattern = r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+'\n"
            "    return re.findall(pattern, text)\n"
        ),
        test_path="test_extract_emails.py",
        test_content=(
            "from extract_emails import extract_emails\n\n\n"
            "def test_extract_emails():\n"
            "    text = 'contato: ana@exemplo.com ou bruno@teste.com.br'\n"
            "    result = extract_emails(text)\n"
            "    assert result == ['ana@exemplo.com', 'bruno@teste.com.br']\n"
        ),
        think_before_final="O JSON está correto agora e a regex está certa. Vou validar.",
        final_text=(
            "Implementei extract_emails(text) usando re.findall com um padrão de e-mail. A "
            "validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-jsonrecovery-matrix-is-diagonal",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que verifique se uma matriz quadrada é diagonal (só a diagonal principal tem valores não-zero).",
        think_before_write="Vou implementar is_diagonal_matrix(matrix) com um loop aninhado sobre linhas e colunas.",
        file_path="matrix_diagonal.py",
        malformed_body=(
            '{"path": "matrix_diagonal.py", "content": '
            '"def is_diagonal_matrix(matrix):\\n    n = len(matrix)\\n    for i in range(n):\\        for j in range(n):\\n'
            '            if i != j and matrix[i][j] != 0:\\n                return False\\n    return True\\n"}'
        ),
        think_after_parse_error=(
            "O JSON estava malformado — uma barra invertida solta antes de um espaço na "
            "string do conteúdo. É um erro meu na formatação da chamada, não da ferramenta — "
            "vou corrigir e tentar de novo."
        ),
        corrected_content=(
            "def is_diagonal_matrix(matrix):\n"
            "    n = len(matrix)\n"
            "    for i in range(n):\n"
            "        for j in range(n):\n"
            "            if i != j and matrix[i][j] != 0:\n"
            "                return False\n"
            "    return True\n"
        ),
        test_path="test_matrix_diagonal.py",
        test_content=(
            "from matrix_diagonal import is_diagonal_matrix\n\n\n"
            "def test_is_diagonal_matrix():\n"
            "    assert is_diagonal_matrix([[1, 0], [0, 2]]) is True\n"
            "    assert is_diagonal_matrix([[1, 1], [0, 2]]) is False\n"
        ),
        think_before_final="JSON corrigido, código pronto. Vou validar.",
        final_text=(
            "Implementei is_diagonal_matrix(matrix), verificando se só a diagonal principal "
            "tem valores não-zero. A validação confirma que compila e o teste passa."
        ),
    ),
    dict(
        id_="gen-jsonrecovery-parse-date-br",
        domain="tratamento_erro",
        difficulty="hard",
        user_request="Implemente uma função que extraia dia, mês e ano de uma data no formato brasileiro (DD/MM/AAAA) usando regex.",
        think_before_write="Vou implementar parse_date_br(text) usando re.match com grupos nomeados.",
        file_path="parse_date_br.py",
        malformed_body=(
            '{"path": "parse_date_br.py", "content": '
            '"import re\\n\\n\\ndef parse_date_br(text):\\n    match = re.match(r\'(\\\\d{2})/(\\\\d{2})/(\\\\d{4})\', text)\\n'
            '    if not match:\\        return None\\n    day, month, year = match.groups()\\n    return int(day), int(month), int(year)\\n"}'
        ),
        think_after_parse_error=(
            "O JSON estava malformado — barra invertida solta antes de um espaço quebrando a "
            "string. É um erro meu de formatação na chamada, não da ferramenta — vou corrigir "
            "e tentar de novo."
        ),
        corrected_content=(
            "import re\n\n\n"
            "def parse_date_br(text):\n"
            "    match = re.match(r'(\\d{2})/(\\d{2})/(\\d{4})', text)\n"
            "    if not match:\n"
            "        return None\n"
            "    day, month, year = match.groups()\n"
            "    return int(day), int(month), int(year)\n"
        ),
        test_path="test_parse_date_br.py",
        test_content=(
            "from parse_date_br import parse_date_br\n\n\n"
            "def test_parse_date_br():\n"
            "    assert parse_date_br('25/12/2024') == (25, 12, 2024)\n"
            "    assert parse_date_br('data invalida') is None\n"
        ),
        think_before_final="JSON corrigido e a regex está certa. Vou validar.",
        final_text=(
            "Implementei parse_date_br(text), extraindo dia, mês e ano com regex. A validação "
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
        print(f"OK: {out_path.name} (TOOL_CALL_PARSE_ERROR real + correção real + checker passou)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
