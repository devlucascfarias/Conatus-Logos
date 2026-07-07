#!/usr/bin/env python
"""Expansão do dataset por paráfrase (PLAN.md seção 8.6, D-generalization-gap).

Um teste controlado (ver D-generalization-gap) confirmou que o adapter treinado executa a
trajetória canônica perfeitamente quando a frase do usuário é EXATAMENTE a do dataset de
treino, e falha em reformulações da mesma tarefa — o mecanismo de treino está correto
(D-train-prompt-mask, D-probe-system-prompt), o gargalo é diversidade de frase. Várias
categorias de trajetória multi-passo têm só 1-2 gabaritos de frase repetidos dezenas de vezes
com apenas o nome de arquivo trocado.

Este script gera paráfrases de `user_request` para exemplos já existentes em `data/train/` e
`data/validation/`, mantendo a MESMA trajetória (`raw_text`, portanto os mesmos
`tool_call`/`tool_result`/`<final>`) — só o texto do pedido muda. Isso é deliberadamente mais
barato que gerar tarefas novas: ataca exatamente a lacuna medida.

Regras (seção 8.3, isolamento de split):
- Toda paráfrase fica no MESMO split do exemplo-base (nunca atravessa train/validation).
- `id` derivado (`{id_original}-para{k}`), `metadata.source` marcado como
  `paraphrase_of:{id_original}` para rastreabilidade.
- Sem dependência de LLM externo — templates de frase escritos à mão, substituição
  determinística de variáveis (mesma filosofia de D-shell-v2: soluções autocontidas).

Uso:
    python scripts/expand_dataset_paraphrases.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_SPLIT_DIRS = ["train", "validation"]

# Cada task_type crítico/alto (seção 8.6) tem um regex de extração da variável `file` a partir
# do gabarito original, e uma lista de reformulações que preservam o mesmo significado. O
# gabarito original NÃO entra de novo na lista (já existe no dataset).
_PARAPHRASE_SPECS: dict[str, tuple[re.Pattern[str], list[str]]] = {
    "model_fixes_after_error": (
        re.compile(r"^Corrija (?P<file>\S+) até compilar\.$"),
        [
            "O arquivo {file} não compila — pode consertar?",
            "{file} está com erro de compilação, ajusta pra mim?",
            "Preciso que você conserte {file} até ele compilar sem erro.",
            "Dá uma olhada em {file}? Não está compilando.",
            "Corrige o {file}, ele não compila do jeito que está.",
            "Tem um problema de compilação em {file} — resolve isso.",
            "Ajuste {file} até que ele compile corretamente.",
        ],
    ),
    "checker_rejects_code": (
        re.compile(r"^Valide se (?P<file>\S+) compila\.$"),
        [
            "Confere pra mim se {file} compila sem erro?",
            "{file} compila? Roda a validação.",
            "Preciso saber se {file} está compilando corretamente.",
            "Pode checar a compilação de {file}?",
            "Roda o checker em {file} e me diz se compila.",
            "Verifique, por favor, se {file} compila sem problemas.",
            "{file} está ok pra compilar? Confirma aí.",
        ],
    ),
    "debugging": (
        re.compile(r"^Por que (?P<file>\S+) está lançando um erro\?$"),
        [
            "{file} está dando erro, você sabe por quê?",
            "Tem um erro em {file} — consegue descobrir a causa?",
            "Por que será que {file} não está funcionando?",
            "Investiga esse erro em {file} pra mim?",
            "O que está causando o erro em {file}?",
            "{file} quebrou. Por quê?",
            "Ajuda a entender por que {file} está falhando.",
        ],
    ),
    "code_explanation": (
        re.compile(r"^Explique o que o código em (?P<file>\S+) faz\.$"),
        [
            "O que {file} faz, exatamente?",
            "Me explica o que tem escrito em {file}.",
            "Pode descrever o que o código de {file} faz?",
            "Não entendi {file} — explica pra mim?",
            "Dá um resumo do que {file} faz.",
            "Qual é a lógica implementada em {file}?",
            "Explica o funcionamento de {file}.",
        ],
    ),
    "compiles_successfully": (
        re.compile(r"^Confirme que (?P<file>\S+) compila\.$"),
        [
            "{file} compila certinho?",
            "Confirma pra mim que {file} está compilando sem erro.",
            "Roda uma checagem em {file} e confirma que compila.",
            "Preciso de confirmação: {file} compila?",
            "Só confirma que {file} compila, por favor.",
            "Valida que {file} está ok pra compilar.",
            "Dá uma checada se {file} compila sem problema.",
        ],
    ),
    "language_migration": (
        re.compile(r"^Reescreva (?P<file>\S+) em Go\.$"),
        [
            "Passa {file} para Go, por favor.",
            "Preciso de uma versão em Go de {file}.",
            "Converte {file} de Python pra Go.",
            "Pode migrar {file} para a linguagem Go?",
            "Traduz {file} para Go.",
            "Faz a portabilidade de {file} para Go.",
            "Migra o conteúdo de {file} pra Go pra mim?",
        ],
    ),
}


def _iter_split_files(data_dir: Path) -> list[tuple[str, Path]]:
    result = []
    for split in _SPLIT_DIRS:
        split_dir = data_dir / split
        for path in sorted(split_dir.glob("*.json")):
            result.append((split, path))
    return result


def expand(data_dir: Path, dry_run: bool = False) -> dict[str, int]:
    stats = {"eligible_task_type": 0, "no_template_match": 0, "generated": 0, "skipped_task_type": 0}
    existing_texts_by_split: dict[str, set[str]] = {split: set() for split in _SPLIT_DIRS}

    all_files = _iter_split_files(data_dir)
    for split, path in all_files:
        example = json.loads(path.read_text(encoding="utf-8"))
        existing_texts_by_split[split].add(example["trajectory"]["user_request"])

    for split, path in all_files:
        example = json.loads(path.read_text(encoding="utf-8"))
        task_type = example["metadata"]["task_type"]

        spec = _PARAPHRASE_SPECS.get(task_type)
        if spec is None:
            stats["skipped_task_type"] += 1
            continue
        stats["eligible_task_type"] += 1

        extract_re, templates = spec
        match = extract_re.match(example["trajectory"]["user_request"])
        if match is None:
            stats["no_template_match"] += 1
            continue
        file_var = match.group("file")

        original_id = example["metadata"]["id"]
        for k, template in enumerate(templates, start=1):
            new_request = template.format(file=file_var)
            if new_request in existing_texts_by_split[split]:
                continue  # dedup (seção 8.4) — não deveria ocorrer, mas por segurança

            new_example = json.loads(json.dumps(example))  # deep copy
            new_id = f"{original_id}-para{k}"
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"paraphrase_of:{original_id}"
            new_example["trajectory"]["user_request"] = new_request

            out_path = path.parent / f"{new_id}.json"
            if not dry_run:
                out_path.write_text(
                    json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            existing_texts_by_split[split].add(new_request)
            stats["generated"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="só mostra estatísticas, não escreve arquivos")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data"
    stats = expand(data_dir, dry_run=args.dry_run)

    print(f"Exemplos-base elegíveis (task_type crítico/alto): {stats['eligible_task_type']}")
    print(f"  sem match de template (frase fora do gabarito esperado, pulados): {stats['no_template_match']}")
    print(f"Exemplos gerados: {stats['generated']}" + (" (dry-run, nada escrito)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
