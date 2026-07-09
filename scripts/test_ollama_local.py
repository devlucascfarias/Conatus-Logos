#!/usr/bin/env python
"""Testa o harness real (`run_agent_loop`) contra um modelo servido localmente via Ollama —
usado para confirmar se a gramática `<think>/<tool_call>/<final>` sobrevive ao merge do LoRA
+ quantização GGUF (D-repetition-loop e adjacentes, mesmo espírito dos testes feitos no Colab,
mas rodando 100% local, sem GPU Python/transformers).

Pré-requisito: `ollama create logos-v2 -f Modelfile` já rodado e `ollama serve` ativo (ou o
serviço já rodando em background, como o Ollama normalmente deixa depois de instalado).

Uso:
    python scripts/test_ollama_local.py
    python scripts/test_ollama_local.py --model logos-v2 --host http://localhost:11434
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness import PRAXIS_SYSTEM_PROMPT, AgentLoopConfig, run_agent_loop  # noqa: E402
from src.inference.ollama_runner import OllamaModelRunner  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402

SCENARIOS = [
    (
        "direct_no_tool",
        "Quanto é 12 * 8?",
    ),
    (
        "simple_write_and_validate",
        "Crie um arquivo hello.py que imprime 'ola mundo' e valide a sintaxe.",
    ),
    (
        "cpf_validator",
        "Escreva uma função validate_cpf(cpf) que verifica se um CPF brasileiro tem 11 "
        "dígitos numéricos (sem validar os dígitos verificadores). Valide com um teste real.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logos-v2", help="nome do modelo no Ollama (padrão: logos-v2)")
    parser.add_argument("--host", default="http://localhost:11434", help="endereço do servidor Ollama")
    args = parser.parse_args()

    runner = OllamaModelRunner(model=args.model, host=args.host)
    policy = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
    registry = ToolExecutorRegistry()

    results = []
    for label, user_request in SCENARIOS:
        print(f"{'=' * 90}\n[{label}]\nPedido: {user_request}\n{'-' * 90}")
        sandbox = SandboxContext(policy=policy)
        try:
            result = run_agent_loop(
                user_request=user_request,
                system_prompt=PRAXIS_SYSTEM_PROMPT,
                model_runner=runner,
                tool_registry=registry,
                sandbox=sandbox,
                config=AgentLoopConfig(mode="dev"),
            )
            print(result.public_output)
            print(f"\n--- passos: {result.steps_taken} | forçado: {result.forced_final} ---")
            results.append((label, result.steps_taken, result.forced_final))
        except Exception as exc:
            print(f"ERRO ao rodar {label}: {exc!r}")
            results.append((label, None, "ERROR"))
        finally:
            sandbox.cleanup()
        print()

    print("=" * 90)
    print("Resumo:")
    for label, steps, forced in results:
        print(f"  {label}: passos={steps} forçado={forced}")


if __name__ == "__main__":
    main()
