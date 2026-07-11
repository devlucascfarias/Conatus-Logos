#!/usr/bin/env python
"""Lote base do Gap 1 (docs/plan_dataset_expansion_error_recovery.md): recuperação de falha do
`checker` por infraestrutura (não `web_search`, que já é coberto por
`gen_tool_recovery_pilot.py`).

Achado real: pedido de função em Go, código correto de primeira, mas `checker` falhou por
`MISSING_DEPENDENCY: toolchain 'go' não disponível` na sessão do Colab. O modelo diagnosticou
certo que era falha de ambiente, mas ficou repetindo a MESMA chamada de `checker` até
`MAX_STEPS_EXCEEDED`, em vez de responder com o código já escrito e uma nota honesta.

Mecanismo real e determinístico usado aqui (mais estável que depender de uma toolchain ausente
específica do Colab): `checker` com `language="javascript"`/`"typescript"` retorna
`UNSUPPORTED_LANGUAGE` de verdade (`src/checker/core.py`), já que só há backend registrado
para Python e Go — nenhum erro é fabricado.

Cada exemplo: `<think>` -> `write_file` real -> `<think>` -> `checker` real (falha real com
`UNSUPPORTED_LANGUAGE`) -> `<think>` reconhecendo que é limitação do ambiente e decidindo NÃO
repetir a mesma chamada -> `<final>` honesto, com o código e a ressalva de que não foi validado
neste ambiente.

Uso:
    python scripts/gen_checker_infra_recovery_pilot.py
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


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


def build_example(
    id_, domain, difficulty, user_request, checker_language,
    think_before_write, file_path, file_content,
    think_before_checker, think_after_failure, final_text,
) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        traj.append_raw(f"<think>{think_before_write}</think>")
        write_result = _tool_call(traj, sandbox, "write_file", {"path": file_path, "content": file_content})
        assert write_result.passed, f"write_file falhou inesperadamente em {id_}: {write_result.to_json()}"

        traj.append_raw(f"<think>{think_before_checker}</think>")
        checker_args = {
            "language": checker_language,
            "operation": "compile_and_test",
            "files": [{"path": file_path, "content": file_content}],
        }
        checker_result = _tool_call(traj, sandbox, "checker", checker_args)
        assert not checker_result.passed, f"esperava UNSUPPORTED_LANGUAGE em {id_}, mas passou"
        assert checker_result.data["errors"][0]["code"] == "UNSUPPORTED_LANGUAGE", (
            f"esperava UNSUPPORTED_LANGUAGE em {id_}: {checker_result.to_json()}"
        )

        traj.append_raw(f"<think>{think_after_failure}</think>")
        traj.append_raw(f"<final>{final_text}</final>")
    finally:
        sandbox.cleanup()

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": checker_language,
            "difficulty": difficulty,
            "tools_used": ["write_file", "checker"],
            "num_steps": 2,
            "task_type": "checker_infra_unavailable",
            "source": "curated_manual_real_execution",
            "license": "synthetic-no-license-needed",
            "validation_status": "validated",
            "execution_performed": True,
            "split": "train",
            # D-checker-used-null-fix (docs/PLAN.md): campo OMITIDO (nunca `null`) quando não
            # se aplica — mesma convenção de `scripts/generate_dataset.py`'s `_example()`. Aqui
            # nenhum checker rodou com sucesso de verdade (é literalmente o ponto do exemplo:
            # UNSUPPORTED_LANGUAGE), então não existe versão de checker pra reportar.
            "expected_result": {"passed": False},
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
        id_="gen-checkerinfra-js-is-even",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Escreva uma função JavaScript que verifique se um número é par.",
        checker_language="javascript",
        think_before_write="Vou implementar a função pedida em JavaScript.",
        file_path="is_even.js",
        file_content="function isEven(n) {\n  return n % 2 === 0;\n}\n\nmodule.exports = { isEven };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "O checker não tem suporte para JavaScript neste ambiente (só Python e Go estão "
            "registrados) — não é um bug no código, é uma limitação de infraestrutura. Repetir "
            "a mesma chamada não vai mudar isso, então vou responder com o código já escrito, "
            "sendo honesto sobre não ter conseguido validar."
        ),
        final_text=(
            "Implementei isEven(n) em JavaScript:\n\n```javascript\nfunction isEven(n) {\n  "
            "return n % 2 === 0;\n}\n```\n\nEste ambiente não tem suporte de validação "
            "(`checker`) para JavaScript — só Python e Go são suportados aqui — então não "
            "consegui compilar/testar automaticamente. O código não foi validado."
        ),
    ),
    dict(
        id_="gen-checkerinfra-ts-sum",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Implemente uma função TypeScript que some dois números.",
        checker_language="typescript",
        think_before_write="Vou implementar a função pedida em TypeScript, com tipos explícitos.",
        file_path="sum.ts",
        file_content="function sum(a: number, b: number): number {\n  return a + b;\n}\n\nexport { sum };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "O checker não tem backend registrado para TypeScript neste ambiente — é uma "
            "limitação de infraestrutura, não um problema no código. Vou responder com o "
            "código já escrito em vez de insistir na mesma chamada."
        ),
        final_text=(
            "Implementei sum(a, b) em TypeScript:\n\n```typescript\nfunction sum(a: number, b: "
            "number): number {\n  return a + b;\n}\n```\n\nEste ambiente não tem suporte de "
            "validação (`checker`) para TypeScript — só Python e Go são suportados aqui — "
            "então não consegui compilar/testar automaticamente. O código não foi validado."
        ),
    ),
    dict(
        id_="gen-checkerinfra-js-reverse-string",
        domain="tratamento_erro",
        difficulty="easy",
        user_request="Crie uma função JavaScript que inverta uma string.",
        checker_language="javascript",
        think_before_write="Vou implementar a inversão de string em JavaScript.",
        file_path="reverse_string.js",
        file_content="function reverseString(text) {\n  return text.split('').reverse().join('');\n}\n\nmodule.exports = { reverseString };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "A falha veio de o checker não ter backend para JavaScript, não do código em si "
            "— é uma limitação deste ambiente. Insistir na mesma chamada não resolve, então "
            "vou responder com o código e ser honesto sobre a falta de validação."
        ),
        final_text=(
            "Implementei reverseString(text) em JavaScript:\n\n```javascript\nfunction "
            "reverseString(text) {\n  return text.split('').reverse().join('');\n}\n```\n\n"
            "Este ambiente não tem suporte de validação (`checker`) para JavaScript — só "
            "Python e Go são suportados aqui — então não consegui compilar/testar "
            "automaticamente. O código não foi validado."
        ),
    ),
    dict(
        id_="gen-checkerinfra-js-factorial",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Escreva uma função JavaScript que calcule o fatorial de um número.",
        checker_language="javascript",
        think_before_write="Vou implementar o fatorial em JavaScript, com recursão.",
        file_path="factorial.js",
        file_content="function factorial(n) {\n  if (n < 0) {\n    throw new Error('fatorial nao definido para negativos');\n  }\n  return n <= 1 ? 1 : n * factorial(n - 1);\n}\n\nmodule.exports = { factorial };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "O checker não suporta JavaScript neste ambiente — é uma limitação de "
            "infraestrutura, não um erro no código. Vou responder com o que já foi "
            "implementado em vez de repetir a mesma chamada sem efeito."
        ),
        final_text=(
            "Implementei factorial(n) em JavaScript, com tratamento para número negativo:\n\n"
            "```javascript\nfunction factorial(n) {\n  if (n < 0) {\n    throw new "
            "Error('fatorial nao definido para negativos');\n  }\n  return n <= 1 ? 1 : n * "
            "factorial(n - 1);\n}\n```\n\nEste ambiente não tem suporte de validação "
            "(`checker`) para JavaScript — só Python e Go são suportados aqui — então não "
            "consegui compilar/testar automaticamente. O código não foi validado."
        ),
    ),
    dict(
        id_="gen-checkerinfra-js-validate-email",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Implemente uma validação de e-mail em JavaScript.",
        checker_language="javascript",
        think_before_write="Vou implementar uma validação simples de e-mail em JavaScript.",
        file_path="validate_email.js",
        file_content="function isValidEmail(email) {\n  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n}\n\nmodule.exports = { isValidEmail };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "Essa falha é do ambiente (sem backend de JavaScript no checker), não do código. "
            "Não faz sentido repetir a mesma chamada — vou responder com o código já escrito e "
            "avisar que não pôde ser validado aqui."
        ),
        final_text=(
            "Implementei isValidEmail(email) em JavaScript:\n\n```javascript\nfunction "
            "isValidEmail(email) {\n  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n}"
            "\n```\n\nEste ambiente não tem suporte de validação (`checker`) para JavaScript — "
            "só Python e Go são suportados aqui — então não consegui compilar/testar "
            "automaticamente. O código não foi validado."
        ),
    ),
    dict(
        id_="gen-checkerinfra-ts-filter-evens",
        domain="tratamento_erro",
        difficulty="medium",
        user_request="Crie uma função TypeScript que filtre os números pares de uma lista.",
        checker_language="typescript",
        think_before_write="Vou implementar o filtro de números pares em TypeScript.",
        file_path="filter_evens.ts",
        file_content="function filterEvens(numbers: number[]): number[] {\n  return numbers.filter((n) => n % 2 === 0);\n}\n\nexport { filterEvens };\n",
        think_before_checker="Vou validar com o checker antes de responder.",
        think_after_failure=(
            "O checker não tem backend para TypeScript neste ambiente — limitação de "
            "infraestrutura, não bug no código. Vou responder com o que já foi implementado, "
            "sem repetir a mesma chamada."
        ),
        final_text=(
            "Implementei filterEvens(numbers) em TypeScript:\n\n```typescript\nfunction "
            "filterEvens(numbers: number[]): number[] {\n  return numbers.filter((n) => n % 2 "
            "=== 0);\n}\n```\n\nEste ambiente não tem suporte de validação (`checker`) para "
            "TypeScript — só Python e Go são suportados aqui — então não consegui "
            "compilar/testar automaticamente. O código não foi validado."
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
        print(f"OK: {out_path.name} (UNSUPPORTED_LANGUAGE real, tratado sem repetir chamada)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
