#!/usr/bin/env python
"""Lote piloto reforçando o padrão "criar arquivo + validar sintaxe" (D-segment-smuggling,
docs/PLAN.md) — achado real, adapter L4 pós-fix do harness: `probe_paraphrase_generalization`
falhou 5/5 porque o modelo, nesse tipo específico de pedido ("crie um script X.py que faça Y,
depois confira a sintaxe"), inventa nomes de ferramenta inexistentes (`create_file`,
`check_syntax`) e sintaxe self-closing (`<tool_call name="x" args="{...}"/>`) nunca vista no
treino, em vez da gramática canônica (`<tool_call name="X">{json}</tool_call>`) com os nomes
reais do registro (`write_file`, `checker`). O dataset não tinha exemplos reforçando ESSE padrão
específico com nomes/sintaxe corretos, nem demonstrando autocorreção depois de uma tentativa com
nome de ferramenta errado.

Cada exemplo: `<think>` → tentativa MALFORMADA de propósito (mesmos padrões reais observados:
tag self-closing OU nome de ferramenta inexistente com gramática válida) → o `<tool_result>` de
rejeição não é fabricado por texto solto — vem do parser/registro REAIS
(`src.parsers.parse_segments` / `ToolExecutorRegistry.get_spec`), garantindo que o texto do erro
bate exatamente com o que o harness de produção geraria hoje → `<think>` de autocorreção →
`write_file` real → `checker` real (`syntax_check`, o mesmo pedido do probe: "confira a
sintaxe") → `<final>` limpo.

Uso:
    python scripts/gen_create_validate_recovery_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.checker import errors as error_codes  # noqa: E402
from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402
from src.parsers import parse_segments  # noqa: E402
from src.parsers.segments import MalformedSegment, ToolCallSegment  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()


def _sanitize_checker_result(result):
    """Mesma sanitização de `gen_tool_success_pilot.py`: aviso de depreciação do pytest-asyncio
    (só instalado nesta máquina local) não deve entrar no dataset."""
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


def _attempt_malformed_tool_call(traj: Trajectory, raw_snippet: str, id_: str) -> None:
    """Anexa uma tentativa de tool_call MAL-FORMADA de propósito (mesma forma dos achados reais)
    e o `<tool_result>` de rejeição correspondente — derivado do parser/registro REAIS, nunca
    fabricado por texto solto, pra garantir que bate exatamente com o que
    `src/harness/loop.py` geraria hoje para o mesmo texto."""
    traj.append_raw(raw_snippet)
    segments = parse_segments(raw_snippet)
    assert len(segments) == 1, f"esperava exatamente 1 segmento no snippet malformado ({id_}): {segments}"
    segment = segments[0]

    if isinstance(segment, MalformedSegment):
        # Sintaxe self-closing (`.../>`) não bate com `_ANY_OPEN` (src/parsers/grammar.py) —
        # vira texto solto, exatamente como no achado real do "saudacao.py".
        traj.append_tool_result(
            segment.attempted_name or "unknown",
            "error",
            {"code": segment.reason, "message": segment.message},
        )
        return

    if isinstance(segment, ToolCallSegment):
        # Gramática válida, mas nome de ferramenta inexistente (`create_file`, `check_syntax`,
        # etc.) — o parser reconhece a tag, o REGISTRO real é quem rejeita.
        spec = _REGISTRY.get_spec(segment.name)
        assert spec is None, (
            f"esperava ferramenta '{segment.name}' NÃO registrada (achado real usa nomes "
            f"inexistentes) em {id_}, mas está registrada — o snippet malformado precisa usar "
            f"um nome que realmente não existe no registro"
        )
        traj.append_tool_result(
            segment.name,
            "error",
            {
                "code": error_codes.UNSUPPORTED_TOOL,
                "message": f"ferramenta desconhecida/desabilitada nesta fase: {segment.name}",
            },
        )
        return

    raise AssertionError(f"tipo de segmento inesperado em {id_}: {segment}")


def build_example(
    id_: str,
    domain: str,
    difficulty: str,
    user_request: str,
    think_before_attempt: str,
    malformed_snippet: str,
    think_after_correction: str,
    file_path: str,
    file_content: str,
    think_before_final: str,
    final_text: str,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_attempt}</think>")
        _attempt_malformed_tool_call(traj, malformed_snippet, id_)

        traj.append_raw(f"<think>{think_after_correction}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": file_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        checker_args = {
            "language": "python",
            "operation": "syntax_check",
            "files": [{"path": file_path, "content": file_content}],
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
            "task_type": "invalid_call_then_correction",
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
        id_="gen-createvalidate-ola_mundo",
        domain="tool_calling",
        difficulty="easy",
        user_request="Preciso de um script chamado ola_mundo.py que imprima 'Olá, mundo!' na tela. Depois de criar, confira se a sintaxe está correta.",
        think_before_attempt="Vou criar o arquivo com o conteúdo pedido.",
        malformed_snippet='<tool_call name="create_file">{name: "ola_mundo.py", content: "print(\'Olá, mundo!\')"}</tool_call>',
        think_after_correction="Essa chamada tem dois problemas: `create_file` não existe no registro de ferramentas deste projeto, e o corpo nem é JSON válido (as chaves precisam vir entre aspas duplas). A ferramenta certa é `write_file`, com JSON válido no corpo.",
        file_path="ola_mundo.py",
        file_content="print('Olá, mundo!')\n",
        think_before_final="O arquivo foi criado. Vou validar a sintaxe com o checker.",
        final_text="Criei ola_mundo.py com print('Olá, mundo!') e validei a sintaxe com o checker — está correta.",
    ),
    dict(
        id_="gen-createvalidate-soma_dois_numeros",
        domain="tool_calling",
        difficulty="easy",
        user_request="Crie um arquivo soma_dois_numeros.py que defina uma função somar(a, b) e imprima o resultado de somar(2, 3). Depois, valide a sintaxe.",
        think_before_attempt="Vou gerar o arquivo com a função pedida.",
        malformed_snippet='<tool_call name="save_file">{"path": "soma_dois_numeros.py", "content": "def somar(a, b):\\n    return a + b\\n\\nprint(somar(2, 3))\\n"}</tool_call>',
        think_after_correction="`save_file` não existe no registro de ferramentas deste projeto — a gramática da tag até está correta, mas o nome está errado. A ferramenta real para criar/escrever arquivo é `write_file`.",
        file_path="soma_dois_numeros.py",
        file_content="def somar(a, b):\n    return a + b\n\n\nprint(somar(2, 3))\n",
        think_before_final="O arquivo foi criado. Vou validar a sintaxe com o checker.",
        final_text="Criei soma_dois_numeros.py com a função somar(a, b) e o print(somar(2, 3)), e validei a sintaxe com o checker — está correta.",
    ),
    dict(
        id_="gen-createvalidate-verifica_par",
        domain="tool_calling",
        difficulty="easy",
        user_request="Escreva um script verifica_par.py que tenha uma função eh_par(n) e imprima eh_par(4). Depois de escrever, confira se a sintaxe está certa.",
        think_before_attempt="Vou criar o arquivo com a função pedida.",
        malformed_snippet="<tool_call name=\"create_file\">{'name': 'verifica_par.py', 'content': 'def eh_par(n): return n % 2 == 0'}</tool_call>",
        think_after_correction="Essa chamada usa aspas simples, que não são JSON válido (JSON exige aspas duplas), e `create_file` também não é uma ferramenta registrada. A ferramenta certa é `write_file`, com JSON estrito (aspas duplas) no corpo.",
        file_path="verifica_par.py",
        file_content="def eh_par(n):\n    return n % 2 == 0\n\n\nprint(eh_par(4))\n",
        think_before_final="O arquivo foi criado. Vou validar a sintaxe com o checker.",
        final_text="Criei verifica_par.py com a função eh_par(n) e o print(eh_par(4)), e validei a sintaxe com o checker — está correta.",
    ),
    dict(
        id_="gen-createvalidate-tabuada",
        domain="tool_calling",
        difficulty="easy",
        user_request="Gera um arquivo tabuada.py que imprima a tabuada do 7 (de 1 a 10). Confere a sintaxe depois.",
        think_before_attempt="Vou gerar o arquivo com a tabuada pedida.",
        malformed_snippet='<tool_call name="write_new_file">{"file_name": "tabuada.py", "code": "for i in range(1, 11):\\n    print(f\\"7 x {i} = {7 * i}\\")\\n"}</tool_call>',
        think_after_correction="`write_new_file` não existe — a ferramenta real chama-se `write_file`, e os argumentos certos são `path` e `content`, não `file_name`/`code`.",
        file_path="tabuada.py",
        file_content='for i in range(1, 11):\n    print(f"7 x {i} = {7 * i}")\n',
        think_before_final="O arquivo foi criado. Vou validar a sintaxe com o checker.",
        final_text="Criei tabuada.py imprimindo a tabuada do 7 de 1 a 10, e validei a sintaxe com o checker — está correta.",
    ),
    dict(
        id_="gen-createvalidate-saudacao_usuario",
        domain="tool_calling",
        difficulty="easy",
        user_request="Preciso de um script saudacao_usuario.py que peça o nome via input() e imprima 'Olá, <nome>!'. Depois de criar, confira a sintaxe.",
        think_before_attempt="Vou criar o arquivo com o conteúdo pedido.",
        malformed_snippet='<tool_call name="create_file">{"name": "saudacao_usuario.py", "content": "nome = input()\\nprint(f\'Olá, {nome}!\')",}</tool_call>',
        think_after_correction="Essa chamada tem uma vírgula sobrando antes do fechamento do objeto (JSON não aceita vírgula depois do último campo), e `create_file` também não existe. A ferramenta certa é `write_file`, com JSON válido e sem vírgula sobrando.",
        file_path="saudacao_usuario.py",
        file_content="nome = input()\nprint(f'Olá, {nome}!')\n",
        think_before_final="O arquivo foi criado. Vou validar a sintaxe com o checker.",
        final_text="Criei saudacao_usuario.py, que pede o nome via input() e imprime a saudação, e validei a sintaxe com o checker — está correta.",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for spec in EXAMPLES:
        example = build_example(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name} (tool_result de rejeição real + write_file/checker reais)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
