#!/usr/bin/env python
"""Lote base da categoria "Shell real" (docs/plan_dataset_expansion_oop_shell.md, seção 2):
10 tarefas-base usando a ferramenta `shell` de verdade, restrita à allowlist de
`configs/sandbox_policy.yaml` (`ls`, `cat`, `grep`, `python`, `git status/diff/log`).

`ruff` está na allowlist mas não é dependência instalada neste ambiente (não está em
requirements.txt) — por isso não foi usado nesta rodada; ver nota em PLAN.md. `pytest` como
binário solto também não está no PATH deste ambiente (só o módulo, via `python -m pytest`,
que é como o próprio `checker` já invoca) — por isso a tarefa de rodar testes usa
`binary="python", args=["-m", "pytest", ...]`, igual ao `checker`.

Para as tarefas de `git`, o repositório é inicializado ANTES da trajetória via subprocess direto
(não pela ferramenta `shell` sandboxada, já que `git init` não está nos subcomandos permitidos,
só leitura: status/diff/log) — isso simula um repositório que já existia antes do turno do
agente, cenário real de uso.

Cada exemplo: uma sequência de `<think>` -> `write_file` e/ou `shell` reais -> `<final>`.
Nenhum `tool_result` é fabricado — tudo roda via `ToolExecutorRegistry`/`SandboxContext`.

Uso:
    python scripts/gen_shell_real_pilot.py
"""

from __future__ import annotations

import json
import subprocess
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


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, capture_output=True, check=True)


def _sanitize_shell_result(binary: str, result):
    """Mesmo ruído local de `pytest-asyncio` (visto em `gen_tool_recovery_pilot.py`) aparece
    quando `python -m pytest` roda direto pela ferramenta shell, não só pelo checker."""
    if binary == "python" and result.passed and result.data.get("stderr"):
        result.data["stderr"] = ""
    return result


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    if name == "shell":
        result = _sanitize_shell_result(args.get("binary"), result)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


def build_example(id_, domain, difficulty, user_request, setup, steps, final_text) -> dict:
    sandbox = SandboxContext(policy=_POLICY)
    try:
        if setup is not None:
            setup(sandbox.workspace)

        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)
        tools_used = []
        for step in steps:
            traj.append_raw(f"<think>{step['think']}</think>")
            result = _tool_call(traj, sandbox, step["tool"], step["args"])
            assert result.passed, (
                f"passo '{step['tool']}' falhou inesperadamente em {id_}: {result.to_json()}"
            )
            tools_used.append(step["tool"])

        traj.append_raw(f"<final>{final_text}</final>")
    finally:
        sandbox.cleanup()

    return {
        "metadata": {
            "id": id_,
            "domain": domain,
            "language": "shell",
            "difficulty": difficulty,
            "tools_used": tools_used,
            "num_steps": len(steps),
            "task_type": "shell_command",
            "source": "curated_manual_real_execution",
            "license": "synthetic-no-license-needed",
            "validation_status": "validated",
            "execution_performed": True,
            "split": "train",
            # D-checker-used-null-fix (docs/PLAN.md): omitido, não `null` — comandos de shell
            # não passam pelo `checker`, não existe versão de checker pra reportar aqui.
            "expected_result": {"passed": True},
            "execution_classification": "tested",
        },
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


def _setup_git_repo_with_commit(workspace: Path) -> None:
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "dev@example.com")
    _git(workspace, "config", "user.name", "Praxis Dev")
    (workspace / "readme.txt").write_text("projeto de exemplo\n", encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "commit inicial")


def _setup_git_repo_with_two_commits(workspace: Path) -> None:
    _setup_git_repo_with_commit(workspace)
    (workspace / "notes.txt").write_text("nota 1\n", encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "adiciona notas")


EXAMPLES = [
    dict(
        id_="gen-shell-ls-workspace",
        domain="shell_real",
        difficulty="easy",
        user_request="Crie um arquivo chamado config.txt e depois liste os arquivos do workspace para confirmar.",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o arquivo pedido antes de confirmar com a listagem.",
                args={"path": "config.txt", "content": "debug=false\n"},
            ),
            dict(
                tool="shell",
                think="Agora vou listar o workspace com ls para confirmar que o arquivo foi criado.",
                args={"binary": "ls", "args": ["-la"]},
            ),
        ],
        final_text=(
            "Criei config.txt e confirmei com 'ls -la' que ele está presente no workspace."
        ),
    ),
    dict(
        id_="gen-shell-cat-file",
        domain="shell_real",
        difficulty="easy",
        user_request="Crie um arquivo settings.ini com uma configuração de exemplo e depois mostre o conteúdo dele usando o terminal.",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o arquivo de configuração primeiro.",
                args={"path": "settings.ini", "content": "[app]\ntimeout=30\n"},
            ),
            dict(
                tool="shell",
                think="Vou usar cat para mostrar o conteúdo do arquivo diretamente pelo terminal, como pedido.",
                args={"binary": "cat", "args": ["settings.ini"]},
            ),
        ],
        final_text=(
            "Criei settings.ini com a configuração de exemplo e usei 'cat settings.ini' para "
            "mostrar o conteúdo pelo terminal, como pedido."
        ),
    ),
    dict(
        id_="gen-shell-grep-pattern",
        domain="shell_real",
        difficulty="medium",
        user_request="Crie um arquivo Python com uma função chamada calculate_total e depois use grep para confirmar que ela está definida no arquivo.",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o arquivo com a função pedida.",
                args={
                    "path": "totals.py",
                    "content": "def calculate_total(items):\n    return sum(items)\n",
                },
            ),
            dict(
                tool="shell",
                think="Vou usar grep para confirmar que a definição da função está no arquivo, em vez de só confiar na memória do que escrevi.",
                args={"binary": "grep", "args": ["def calculate_total", "totals.py"]},
            ),
        ],
        final_text=(
            "Criei totals.py com a função calculate_total e confirmei com 'grep \"def "
            "calculate_total\" totals.py' que a definição está presente no arquivo."
        ),
    ),
    dict(
        id_="gen-shell-pytest-run",
        domain="shell_real",
        difficulty="medium",
        user_request="Crie uma função is_even e um teste para ela, depois rode os testes direto pelo terminal (não preciso do checker, só do resultado do pytest).",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar a função pedida.",
                args={"path": "is_even.py", "content": "def is_even(n):\n    return n % 2 == 0\n"},
            ),
            dict(
                tool="write_file",
                think="Agora o teste correspondente, para rodar com pytest diretamente.",
                args={
                    "path": "test_is_even.py",
                    "content": (
                        "from is_even import is_even\n\n\n"
                        "def test_is_even():\n"
                        "    assert is_even(4) is True\n"
                        "    assert is_even(3) is False\n"
                    ),
                },
            ),
            dict(
                tool="shell",
                think="O pedido foi para rodar pytest direto pelo terminal, não pelo checker — vou usar python -m pytest, que é como o módulo pytest é executado neste ambiente.",
                args={"binary": "python", "args": ["-m", "pytest", "-q", "test_is_even.py"]},
            ),
        ],
        final_text=(
            "Criei is_even.py e test_is_even.py, e rodei os testes direto pelo terminal com "
            "'python -m pytest -q test_is_even.py' (sem passar pelo checker, como pedido). O "
            "teste passou."
        ),
    ),
    dict(
        id_="gen-shell-git-status-untracked",
        domain="shell_real",
        difficulty="medium",
        user_request="Crie um novo arquivo chamado draft.md no repositório e depois rode git status para ver o que mudou.",
        setup=_setup_git_repo_with_commit,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o arquivo pedido; como o repositório já existe, ele deve aparecer como não rastreado no git status.",
                args={"path": "draft.md", "content": "# rascunho\n"},
            ),
            dict(
                tool="shell",
                think="Vou rodar git status para confirmar que o novo arquivo aparece como untracked.",
                args={"binary": "git", "args": ["status"]},
            ),
        ],
        final_text=(
            "Criei draft.md e confirmei com 'git status' que ele aparece como arquivo não "
            "rastreado (untracked), já que ainda não foi adicionado ao commit."
        ),
    ),
    dict(
        id_="gen-shell-git-diff-modified",
        domain="shell_real",
        difficulty="medium",
        user_request="Atualize o conteúdo do readme.txt do repositório e depois mostre o diff da mudança.",
        setup=_setup_git_repo_with_commit,
        steps=[
            dict(
                tool="write_file",
                think="O readme.txt já existe e está commitado; vou sobrescrever o conteúdo dele para gerar uma mudança rastreável.",
                args={"path": "readme.txt", "content": "projeto de exemplo atualizado\n"},
            ),
            dict(
                tool="shell",
                think="Vou rodar git diff para mostrar exatamente o que mudou em relação ao commit anterior.",
                args={"binary": "git", "args": ["diff"]},
            ),
        ],
        final_text=(
            "Atualizei o conteúdo de readme.txt e usei 'git diff' para mostrar a mudança em "
            "relação à versão commitada anteriormente."
        ),
    ),
    dict(
        id_="gen-shell-git-log-history",
        domain="shell_real",
        difficulty="easy",
        user_request="Mostre o histórico de commits deste repositório de forma resumida.",
        setup=_setup_git_repo_with_two_commits,
        steps=[
            dict(
                tool="shell",
                think="Não preciso criar nada aqui, só consultar o histórico existente com git log --oneline.",
                args={"binary": "git", "args": ["log", "--oneline"]},
            ),
        ],
        final_text=(
            "Rodei 'git log --oneline' e mostrei o histórico resumido dos dois commits "
            "existentes no repositório."
        ),
    ),
    dict(
        id_="gen-shell-grep-recursive-todo",
        domain="shell_real",
        difficulty="medium",
        user_request="Crie dois arquivos com comentários TODO pendentes e depois use grep para encontrar todos os TODOs do projeto de uma vez.",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o primeiro arquivo com um TODO pendente.",
                args={"path": "module_a.py", "content": "# TODO: validar entrada do usuario\ndef process(x):\n    return x\n"},
            ),
            dict(
                tool="write_file",
                think="Agora o segundo arquivo, também com um TODO pendente.",
                args={"path": "module_b.py", "content": "# TODO: adicionar cache\ndef fetch(key):\n    return key\n"},
            ),
            dict(
                tool="shell",
                think="Vou usar grep recursivo para encontrar todos os TODOs do projeto de uma vez, em vez de abrir arquivo por arquivo.",
                args={"binary": "grep", "args": ["-r", "TODO", "."]},
            ),
        ],
        final_text=(
            "Criei module_a.py e module_b.py, cada um com um TODO pendente, e usei 'grep -r "
            "TODO .' para listar os dois de uma vez."
        ),
    ),
    dict(
        id_="gen-shell-ls-then-cat",
        domain="shell_real",
        difficulty="medium",
        user_request="Crie um arquivo de log chamado app.log dentro de uma pasta logs/, liste a pasta para confirmar e depois mostre o conteúdo do arquivo.",
        setup=None,
        steps=[
            dict(
                tool="write_file",
                think="Vou criar o arquivo dentro da subpasta logs/, que será criada automaticamente pelo write_file.",
                args={"path": "logs/app.log", "content": "app iniciado\n"},
            ),
            dict(
                tool="shell",
                think="Vou listar a pasta logs/ para confirmar que o arquivo foi criado ali.",
                args={"binary": "ls", "args": ["-la", "logs"]},
            ),
            dict(
                tool="shell",
                think="Agora vou mostrar o conteúdo do arquivo com cat, como pedido.",
                args={"binary": "cat", "args": ["logs/app.log"]},
            ),
        ],
        final_text=(
            "Criei logs/app.log, confirmei com 'ls -la logs' que ele está na pasta certa, e "
            "mostrei o conteúdo com 'cat logs/app.log'."
        ),
    ),
    dict(
        id_="gen-shell-git-status-clean",
        domain="shell_real",
        difficulty="easy",
        user_request="Verifique se há alguma mudança pendente para commitar neste repositório.",
        setup=_setup_git_repo_with_two_commits,
        steps=[
            dict(
                tool="shell",
                think="Não preciso alterar nada — só rodar git status para ver se a árvore de trabalho está limpa.",
                args={"binary": "git", "args": ["status"]},
            ),
        ],
        final_text=(
            "Rodei 'git status' e confirmei que não há mudanças pendentes — a árvore de "
            "trabalho está limpa, tudo já commitado."
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
        print(f"OK: {out_path.name} (shell real passou de verdade)")
        written += 1
    print(f"\n{written} exemplos gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
