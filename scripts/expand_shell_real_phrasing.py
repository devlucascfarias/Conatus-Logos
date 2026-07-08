#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Shell real"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4). Mesmo método das demais categorias:
8 variantes por tarefa-base, reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade (extraídos via `parse_segments`).

Diferente das outras categorias, aqui cada tarefa-base tem um número VARIÁVEL de passos (1 a 3
chamadas de ferramenta), então a reconstrução é genérica: só os segmentos `<think>` (um por
passo) e o `<final>` são substituídos pela variante; cada par `<tool_call>`/`<tool_result>` é
copiado verbatim, na ordem original.

Uso:
    python scripts/expand_shell_real_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_ROLE_TEMPLATES = {
    "write": [
        "Vou criar {detail} agora.",
        "Preciso escrever {detail} antes do próximo passo.",
        "Primeiro, vou gerar {detail}.",
        "Vou providenciar {detail} nesta etapa.",
        "Começo criando {detail}.",
        "Nesta etapa, crio {detail}.",
        "Vou registrar {detail} no workspace.",
        "Antes de seguir, preciso ter {detail} pronto.",
    ],
    "ls": [
        "Vou listar {detail} para confirmar o resultado.",
        "Agora uso ls para conferir {detail}.",
        "Vou checar {detail} com uma listagem de diretório.",
        "Para confirmar, listo {detail}.",
        "Uso o terminal para listar {detail}.",
        "Vou rodar ls e conferir {detail}.",
        "Confirmo {detail} listando o conteúdo do diretório.",
        "Listo {detail} para ter certeza do resultado.",
    ],
    "cat": [
        "Vou mostrar {detail} usando cat.",
        "Agora exibo {detail} pelo terminal.",
        "Uso cat para conferir {detail}.",
        "Vou ler {detail} diretamente pelo terminal.",
        "Mostro {detail} com o comando cat.",
        "Confirmo {detail} exibindo o arquivo no terminal.",
        "Vou checar {detail} com cat.",
        "Exibo {detail} para confirmar o conteúdo.",
    ],
    "grep": [
        "Vou usar grep para confirmar {detail}.",
        "Uso grep para localizar {detail}.",
        "Confirmo {detail} com uma busca via grep.",
        "Vou checar {detail} usando grep em vez de abrir o arquivo manualmente.",
        "Busco {detail} com grep para ter certeza.",
        "Uso o terminal (grep) para confirmar {detail}.",
        "Vou procurar {detail} com grep.",
        "Confirmo {detail} via grep, sem precisar abrir o arquivo.",
    ],
    "grep_r": [
        "Vou usar grep recursivo para encontrar {detail}.",
        "Uso grep -r para localizar {detail}.",
        "Busco {detail} recursivamente em todo o projeto.",
        "Vou varrer o projeto inteiro com grep -r para achar {detail}.",
        "Uso busca recursiva para encontrar {detail} de uma vez.",
        "Vou rodar grep -r para não precisar abrir arquivo por arquivo, buscando {detail}.",
        "Busco {detail} em todos os arquivos com grep recursivo.",
        "Confirmo {detail} com uma varredura recursiva via grep.",
    ],
    "pytest": [
        "Vou rodar os testes direto pelo terminal (python -m pytest), cobrindo {detail}.",
        "Uso python -m pytest para validar {detail} sem passar pelo checker.",
        "Vou executar {detail} com pytest via terminal, como pedido.",
        "Confirmo {detail} rodando pytest diretamente.",
        "Vou testar {detail} usando python -m pytest no terminal.",
        "Rodo os testes de {detail} direto, sem o checker.",
        "Uso o terminal para rodar pytest e validar {detail}.",
        "Vou confirmar {detail} executando pytest pelo terminal.",
    ],
    "git_status": [
        "Vou rodar git status para verificar {detail}.",
        "Confirmo {detail} com git status.",
        "Uso git status para checar {detail}.",
        "Vou conferir {detail} rodando git status.",
        "Verifico {detail} com o comando git status.",
        "Rodo git status para entender {detail}.",
        "Vou olhar {detail} usando git status.",
        "Confirmo {detail} pelo terminal, com git status.",
    ],
    "git_diff": [
        "Vou rodar git diff para mostrar {detail}.",
        "Confirmo {detail} com git diff.",
        "Uso git diff para ver exatamente {detail}.",
        "Vou conferir {detail} rodando git diff.",
        "Mostro {detail} com o comando git diff.",
        "Rodo git diff para visualizar {detail}.",
        "Vou checar {detail} usando git diff.",
        "Confirmo {detail} pelo terminal, com git diff.",
    ],
    "git_log": [
        "Vou consultar {detail} com git log --oneline.",
        "Confirmo {detail} usando git log.",
        "Uso git log --oneline para ver {detail}.",
        "Vou checar {detail} rodando git log.",
        "Mostro {detail} com o comando git log --oneline.",
        "Consulto {detail} pelo terminal, com git log.",
        "Vou olhar {detail} usando git log --oneline.",
        "Confirmo {detail} com uma consulta ao histórico via git log.",
    ],
}

_FINAL_TEMPLATES = [
    "Pronto: {summary}.",
    "Concluído — {summary}.",
    "{summary}, como pedido.",
    "Terminei: {summary}.",
    "Feito — {summary}.",
    "{summary}. Foi exatamente o que foi pedido.",
    "Resultado: {summary}.",
    "{summary}, confirmado pelo terminal.",
]

# base_id -> (lista de (role, detail), summary, lista de 8 pedidos)
_TASKS = {
    "gen-shell-ls-workspace": (
        [("write", "o arquivo config.txt"), ("ls", "o workspace")],
        "config.txt foi criado e confirmado com ls",
        [
            "Cria um arquivo config.txt e depois lista os arquivos do workspace pra confirmar.",
            "Pode criar config.txt e conferir com ls se ele apareceu?",
            "Quero um arquivo config.txt criado e confirmado por listagem do diretório.",
            "Preciso de config.txt criado, e depois quero ver a listagem do workspace.",
            "Crie config.txt e me mostre o resultado de ls no workspace.",
            "Me ajuda criando config.txt e depois listando o diretório pra confirmar?",
            "Um arquivo config.txt, por favor, com a listagem do workspace depois.",
            "Cria config.txt e usa ls para confirmar que está lá.",
        ],
    ),
    "gen-shell-cat-file": (
        [("write", "o arquivo settings.ini"), ("cat", "o conteúdo de settings.ini")],
        "settings.ini foi criado e o conteúdo mostrado com cat",
        [
            "Cria settings.ini com uma configuração de exemplo e mostra o conteúdo pelo terminal.",
            "Pode criar settings.ini e usar cat para mostrar o que tem dentro?",
            "Quero um settings.ini criado e exibido via terminal.",
            "Preciso de settings.ini com uma config de exemplo, e depois ver o conteúdo com cat.",
            "Crie settings.ini e me mostre o conteúdo usando o terminal.",
            "Me ajuda criando settings.ini e mostrando o conteúdo com cat?",
            "Um arquivo settings.ini, por favor, com o conteúdo exibido via cat.",
            "Cria settings.ini e usa cat para mostrar o que foi escrito.",
        ],
    ),
    "gen-shell-grep-pattern": (
        [("write", "o arquivo totals.py com a função calculate_total"), ("grep", "a definição de calculate_total em totals.py")],
        "totals.py foi criado e a função calculate_total confirmada com grep",
        [
            "Cria uma função calculate_total num arquivo Python e confirma com grep que ela está lá.",
            "Pode implementar calculate_total e usar grep pra confirmar a definição no arquivo?",
            "Quero calculate_total implementada e confirmada via grep.",
            "Preciso de uma função calculate_total, e depois quero confirmar com grep que existe.",
            "Crie calculate_total num arquivo e use grep para validar a definição.",
            "Me ajuda criando calculate_total e confirmando com grep que está no arquivo?",
            "Uma função calculate_total, por favor, confirmada com grep depois de criada.",
            "Cria calculate_total e usa grep para confirmar que a função foi definida.",
        ],
    ),
    "gen-shell-pytest-run": (
        [("write", "a função is_even"), ("write", "o teste de is_even"), ("pytest", "os testes de is_even.py")],
        "is_even.py e o teste foram criados e os testes rodados direto com python -m pytest",
        [
            "Cria uma função is_even com teste e roda os testes direto pelo terminal, sem usar o checker.",
            "Pode implementar is_even, escrever o teste e rodar pytest direto no terminal?",
            "Quero is_even implementada e testada via terminal, não pelo checker.",
            "Preciso de is_even com teste, e quero ver o resultado rodando pytest direto.",
            "Crie is_even e o teste correspondente, depois rode pytest pelo terminal.",
            "Me ajuda criando is_even, o teste, e rodando pytest direto no terminal?",
            "Uma função is_even com teste, por favor, validada rodando pytest direto.",
            "Cria is_even e o teste, e roda os testes direto pelo terminal (não pelo checker).",
        ],
    ),
    "gen-shell-git-status-untracked": (
        [("write", "o arquivo draft.md"), ("git_status", "o novo arquivo não rastreado")],
        "draft.md foi criado e confirmado como untracked via git status",
        [
            "Cria um arquivo draft.md no repositório e roda git status pra ver o que mudou.",
            "Pode criar draft.md e confirmar com git status que ele aparece como novo?",
            "Quero draft.md criado e o git status mostrando que está untracked.",
            "Preciso de draft.md criado, e quero ver o resultado do git status depois.",
            "Crie draft.md e rode git status pra confirmar a mudança no repositório.",
            "Me ajuda criando draft.md e confirmando com git status que ele é novo?",
            "Um arquivo draft.md, por favor, com git status confirmando que está untracked.",
            "Cria draft.md e usa git status para mostrar o estado do repositório.",
        ],
    ),
    "gen-shell-git-diff-modified": (
        [("write", "o conteúdo atualizado de readme.txt"), ("git_diff", "a mudança em readme.txt")],
        "readme.txt foi atualizado e a mudança confirmada via git diff",
        [
            "Atualiza o readme.txt do repositório e mostra o diff da mudança.",
            "Pode mudar o conteúdo de readme.txt e rodar git diff pra ver a diferença?",
            "Quero readme.txt atualizado e o diff mostrado pelo terminal.",
            "Preciso alterar readme.txt, e quero ver o git diff da mudança depois.",
            "Atualize o conteúdo de readme.txt e rode git diff pra confirmar.",
            "Me ajuda atualizando readme.txt e mostrando o diff da alteração?",
            "O readme.txt atualizado, por favor, com o diff mostrado depois.",
            "Muda o readme.txt e usa git diff para mostrar exatamente o que foi alterado.",
        ],
    ),
    "gen-shell-git-log-history": (
        [("git_log", "o histórico de commits do repositório")],
        "o histórico de commits foi mostrado com git log --oneline",
        [
            "Mostra o histórico de commits deste repositório de forma resumida.",
            "Pode rodar git log e mostrar os commits de forma compacta?",
            "Quero ver o histórico de commits resumido.",
            "Preciso ver os commits deste repositório, de forma enxuta.",
            "Mostre o log de commits resumido do repositório.",
            "Me ajuda mostrando o histórico de commits de forma compacta?",
            "O histórico de commits, por favor, de forma resumida.",
            "Roda git log --oneline e mostra o histórico do repositório.",
        ],
    ),
    "gen-shell-grep-recursive-todo": (
        [("write", "o primeiro TODO pendente (module_a.py)"), ("write", "o segundo TODO pendente (module_b.py)"), ("grep_r", "todos os TODOs do projeto de uma vez")],
        "dois TODOs foram criados e encontrados de uma vez com grep recursivo",
        [
            "Cria dois arquivos com TODOs pendentes e usa grep pra achar todos de uma vez.",
            "Pode criar dois TODOs em arquivos diferentes e buscar todos com grep recursivo?",
            "Quero dois TODOs criados e encontrados via grep -r.",
            "Preciso de dois arquivos com TODO, e quero ver todos encontrados de uma vez.",
            "Crie dois módulos com TODO pendente e use grep recursivo pra achar os dois.",
            "Me ajuda criando dois TODOs e buscando todos com grep de uma vez?",
            "Dois TODOs pendentes, por favor, encontrados com uma busca recursiva.",
            "Cria dois arquivos com TODO e usa grep -r pra listar todos os TODOs do projeto.",
        ],
    ),
    "gen-shell-ls-then-cat": (
        [("write", "o arquivo logs/app.log"), ("ls", "a pasta logs/"), ("cat", "o conteúdo de logs/app.log")],
        "logs/app.log foi criado, confirmado com ls e mostrado com cat",
        [
            "Cria um log app.log dentro de logs/, lista a pasta pra confirmar e mostra o conteúdo.",
            "Pode criar logs/app.log, confirmar com ls e depois mostrar o conteúdo com cat?",
            "Quero um arquivo de log criado, listado e depois mostrado pelo terminal.",
            "Preciso de logs/app.log criado, confirmado com ls, e visto com cat depois.",
            "Crie logs/app.log, liste a pasta logs/ e mostre o conteúdo do arquivo.",
            "Me ajuda criando o log, confirmando com ls e mostrando com cat?",
            "Um arquivo de log em logs/app.log, por favor, listado e depois exibido.",
            "Cria logs/app.log, confirma com ls e mostra o conteúdo com cat.",
        ],
    ),
    "gen-shell-git-status-clean": (
        [("git_status", "se há mudança pendente")],
        "confirmado com git status que a árvore de trabalho está limpa",
        [
            "Verifica se há alguma mudança pendente pra commitar neste repositório.",
            "Pode rodar git status e ver se está tudo commitado?",
            "Quero saber se a árvore de trabalho está limpa neste repositório.",
            "Preciso confirmar se não há mudança pendente de commit.",
            "Verifique com git status se está tudo em dia no repositório.",
            "Me ajuda checando se há algo pendente de commit?",
            "Uma checagem de mudanças pendentes, por favor, via git status.",
            "Roda git status e confirma se a árvore de trabalho está limpa.",
        ],
    ),
}


def main() -> None:
    count = 0
    for base_id, (roles, summary, request_variants) in _TASKS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        for k, (user_request, final_template) in enumerate(
            zip(request_variants, _FINAL_TEMPLATES), start=1
        ):
            # cada variante usa um "deslocamento" diferente dentro das 8 frases de cada role,
            # para não repetir sempre a mesma combinação em todas as tarefas-base
            raw_text = _rebuild_variant(original_raw_text, roles, final_template, summary, variant_index=k - 1)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para a categoria Shell real.")


def _rebuild_variant(original_raw_text: str, roles: list, final_template: str, summary: str, variant_index: int) -> str:
    segs = parse_segments(original_raw_text)
    n_steps = len(roles)
    expected_len = 3 * n_steps + 1
    if len(segs) != expected_len:
        raise AssertionError(f"forma inesperada: {len(segs)} segmentos, esperado {expected_len}")

    pieces = []
    for i, (role, detail) in enumerate(roles):
        tool_call_idx = 3 * i + 1
        tool_result_idx = 3 * i + 2
        think_variants = _ROLE_TEMPLATES[role]
        think_text = think_variants[(variant_index + i) % len(think_variants)].format(detail=detail)
        pieces.append(f"<think>{think_text}</think>")
        pieces.append(original_raw_text[segs[tool_call_idx].start : segs[tool_result_idx].end])

    pieces.append(f"<final>{final_template.format(summary=summary)}</final>")
    return "".join(pieces)


if __name__ == "__main__":
    main()
