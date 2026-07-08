#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Módulos realistas"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4). Mesmo método das demais categorias:
8 variantes por tarefa-base, reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade.

Uso:
    python scripts/expand_realistic_modules_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_EXPECTED_SHAPE = ("think", "tool_call", "tool_result", "think", "tool_call", "tool_result", "final")

_VARIANTS = [
    dict(
        think_before_write="Vou estruturar {summary} como um módulo com funções relacionadas, cobrindo o comportamento pedido de ponta a ponta.",
        think_before_final="O módulo está pronto. Vou validar antes de responder.",
        final="Implementei {summary}. A validação confirma que compila e o teste passa.",
    ),
    dict(
        think_before_write="Para {summary}, o mais direto é um módulo com as funções relacionadas, sem complicar com estruturas desnecessárias.",
        think_before_final="Pronto o código; falta rodar a validação antes de dar a resposta final.",
        final="{summary} ficou pronto e passou na validação sem problemas.",
    ),
    dict(
        think_before_write="Vou modelar {summary} como funções de módulo, já que não há necessidade de estado persistente entre chamadas aqui.",
        think_before_final="Com o código pronto, o próximo passo é validar antes de fechar a resposta.",
        final="Criei {summary}. Validado: compila e o teste passa.",
    ),
    dict(
        think_before_write="{summary} é o tipo de tarefa de engenharia com múltiplas funções relacionadas — vou implementar dessa forma.",
        think_before_final="Falta só confirmar que o código está correto antes de responder ao usuário.",
        final="{summary} está implementado. A checagem confirma que está tudo funcionando.",
    ),
    dict(
        think_before_write="Antes de escrever qualquer coisa, decido a estrutura: funções de módulo cobrindo {summary}.",
        think_before_final="Código pronto. Rodo a validação agora para ter certeza antes de responder.",
        final="Modelei {summary}, já validado e funcionando corretamente.",
    ),
    dict(
        think_before_write="Vou implementar {summary} com cuidado nos casos de borda que esse tipo de tarefa costuma ter.",
        think_before_final="Antes de finalizar, preciso rodar a validação para confirmar que está tudo certo.",
        final="{summary} foi implementado. Rodei a validação e confirmei que passa.",
    ),
    dict(
        think_before_write="{summary} é justamente o tipo de módulo que precisa de funções bem definidas e testáveis separadamente.",
        think_before_final="Terminei a implementação; agora preciso validar antes de responder ao usuário.",
        final="Pronto: {summary} implementado, e já validado com sucesso.",
    ),
    dict(
        think_before_write="Vou escrever {summary} pensando no caso de uso real, não só no caminho feliz.",
        think_before_final="Só falta validar o código antes de considerar a tarefa concluída.",
        final="Implementei {summary}. A validação passou sem erros.",
    ),
]

_REQUESTS = {
    "gen-module-inventory": [
        "Preciso de um módulo de controle de estoque com adicionar, remover e listar itens com pouco estoque.",
        "Pode implementar um sistema simples de estoque com essas três operações?",
        "Quero funções para gerenciar estoque: adicionar, remover e listar itens baixos.",
        "Implementa um controle de estoque em Python com essas operações.",
        "Cria um módulo que gerencie quantidade de itens em estoque.",
        "Estou precisando de um controle de estoque básico com essas três funções.",
        "Me ajuda com um módulo de estoque que liste itens abaixo de um limite?",
        "Um módulo de inventário, por favor, com add/remove/low_stock.",
    ],
    "gen-module-cnpj-validator": [
        "Preciso de uma função que valide CNPJ de verdade, calculando os dígitos verificadores.",
        "Pode implementar a validação de CNPJ seguindo a regra oficial dos dois dígitos?",
        "Quero uma função is_valid_cnpj com o cálculo real dos dígitos verificadores.",
        "Implementa a validação de CNPJ em Python com os pesos corretos.",
        "Cria uma função que confirme se um CNPJ é válido pelo algoritmo oficial.",
        "Estou precisando validar CNPJ, não só o formato, mas os dígitos verificadores.",
        "Me ajuda com uma validação de CNPJ que use os pesos certos para cada dígito?",
        "Uma função is_valid_cnpj, por favor, com o cálculo oficial completo.",
    ],
    "gen-module-imc-calculator": [
        "Preciso calcular o IMC de uma pessoa e classificar o resultado.",
        "Pode implementar cálculo e classificação de IMC?",
        "Quero funções calculate_imc e classify_imc.",
        "Implementa um cálculo de IMC com as faixas de classificação padrão.",
        "Cria um módulo que calcule IMC e diga a faixa (peso normal, sobrepeso etc).",
        "Estou precisando calcular e classificar o IMC de alguém.",
        "Me ajuda com uma função que calcule IMC e outra que classifique o resultado?",
        "Um módulo de IMC, por favor, com cálculo e classificação separados.",
    ],
    "gen-module-national-holidays": [
        "Preciso verificar se uma data é feriado nacional brasileiro de data fixa.",
        "Pode implementar uma checagem de feriados nacionais (só os de data fixa)?",
        "Quero uma função is_national_holiday para os feriados fixos do Brasil.",
        "Implementa uma verificação de feriado nacional em Python.",
        "Cria um módulo que diga se uma data é feriado e qual o nome dele.",
        "Estou precisando checar feriados nacionais de data fixa.",
        "Me ajuda com uma função que identifique feriados nacionais brasileiros fixos?",
        "Um módulo de feriados nacionais, por favor, cobrindo as datas fixas.",
    ],
    "gen-module-currency-format": [
        "Preciso formatar valores em Real e percentuais no padrão brasileiro.",
        "Pode implementar formatação de moeda em R$ com vírgula decimal?",
        "Quero funções format_brl e format_percentage no padrão brasileiro.",
        "Implementa uma formatação de valor em reais e percentual em Python.",
        "Cria um módulo de formatação de moeda e percentual brasileiros.",
        "Estou precisando formatar R$ e % seguindo o padrão brasileiro de vírgula.",
        "Me ajuda com uma formatação de moeda brasileira e de percentual?",
        "Um módulo de formatação, por favor, com R$ e percentual no padrão BR.",
    ],
    "gen-module-password-generator": [
        "Preciso de uma função que gere senhas aleatórias, mas reproduzíveis com uma seed.",
        "Pode implementar um gerador de senha que aceite seed para testes?",
        "Quero uma função generate_password determinística via seed.",
        "Implementa um gerador de senha aleatória com controle de reprodutibilidade.",
        "Cria uma função que gere senha aleatória de tamanho configurável.",
        "Estou precisando de um gerador de senha testável (mesma seed, mesmo resultado).",
        "Me ajuda com um gerador de senha que use random.Random para reprodutibilidade?",
        "Uma função generate_password, por favor, com seed opcional.",
    ],
    "gen-module-slug-generator": [
        "Preciso de uma função que transforme texto em português em slug para URL.",
        "Pode implementar slugify tratando acentos corretamente?",
        "Quero uma função slugify que remova acentos e normalize o texto.",
        "Implementa uma geração de slug a partir de texto acentuado.",
        "Cria uma função que converta título em slug amigável para URL.",
        "Estou precisando gerar slugs a partir de texto com acentuação.",
        "Me ajuda com uma função slugify que lide com texto em português?",
        "Uma função slugify, por favor, tratando acentos e pontuação.",
    ],
    "gen-module-csv-line-parser": [
        "Preciso de uma função que faça parse de uma linha CSV com campos entre aspas.",
        "Pode implementar um parser de CSV simples que trate delimitador dentro de aspas?",
        "Quero uma função parse_csv_line que lide com campos citados.",
        "Implementa um parser de linha CSV em Python, sem usar a biblioteca csv.",
        "Cria uma função que separe uma linha CSV respeitando aspas.",
        "Estou precisando de um parser de CSV que não quebre com vírgula dentro de aspas.",
        "Me ajuda com um parser de linha CSV que trate campos entre aspas?",
        "Uma função parse_csv_line, por favor, com suporte a delimitador customizado.",
    ],
    "gen-module-user-registry": [
        "Preciso de um cadastro de usuários que impeça e-mail duplicado.",
        "Pode implementar registro e busca de usuário por e-mail?",
        "Quero funções register_user e find_user_by_email.",
        "Implementa um cadastro simples de usuários em Python.",
        "Cria um módulo que registre usuários e valide e-mail único.",
        "Estou precisando de um registro de usuários com busca por e-mail.",
        "Me ajuda com um cadastro que rejeite e-mail já existente?",
        "Um módulo de cadastro de usuários, por favor, com validação de duplicidade.",
    ],
    "gen-module-discount-calculator": [
        "Preciso de um cálculo de desconto simples e desconto progressivo por quantidade.",
        "Pode implementar desconto por faixa de quantidade comprada?",
        "Quero funções apply_discount e apply_bulk_discount.",
        "Implementa um cálculo de desconto progressivo em Python.",
        "Cria um módulo que aplique desconto conforme a quantidade.",
        "Estou precisando calcular desconto simples e desconto por volume.",
        "Me ajuda com um cálculo de desconto que varie pela quantidade comprada?",
        "Um módulo de desconto, por favor, com faixas progressivas.",
    ],
    "gen-module-unit-converter": [
        "Preciso converter entre quilômetros/milhas e quilogramas/libras.",
        "Pode implementar conversões de unidade de distância e peso?",
        "Quero funções km_to_miles, miles_to_km e kg_to_lb.",
        "Implementa um conversor de unidades em Python.",
        "Cria um módulo que converta distância e peso entre sistemas.",
        "Estou precisando converter km para milhas e kg para libras.",
        "Me ajuda com conversões entre unidades métricas e imperiais?",
        "Um módulo de conversão de unidades, por favor, com essas três funções.",
    ],
    "gen-module-text-statistics": [
        "Preciso calcular a frequência de palavras num texto e achar a mais comum.",
        "Pode implementar estatísticas simples de texto?",
        "Quero funções word_frequency e most_common_word.",
        "Implementa uma contagem de frequência de palavras em Python.",
        "Cria um módulo que analise a frequência de palavras de um texto.",
        "Estou precisando saber qual palavra aparece mais num texto.",
        "Me ajuda com uma contagem de frequência de palavras e a mais comum?",
        "Um módulo de estatística de texto, por favor, com essas duas funções.",
    ],
    "gen-module-todo-list": [
        "Preciso de uma lista de tarefas com adicionar, concluir e listar pendentes.",
        "Pode implementar um todo list simples com essas três operações?",
        "Quero funções add_task, complete_task e pending_tasks.",
        "Implementa uma lista de tarefas em Python.",
        "Cria um módulo de gerenciamento de tarefas com pendências.",
        "Estou precisando de uma lista de tarefas com status de conclusão.",
        "Me ajuda com uma lista de tarefas que filtre as pendentes?",
        "Um módulo de todo list, por favor, com add/complete/pending.",
    ],
    "gen-module-phone-formatter": [
        "Preciso formatar um telefone celular brasileiro no padrão (DD) 9XXXX-XXXX.",
        "Pode implementar uma formatação de telefone que valide a quantidade de dígitos?",
        "Quero uma função format_phone para números de celular brasileiros.",
        "Implementa uma formatação de telefone em Python, ignorando caracteres não numéricos.",
        "Cria uma função que formate telefone celular no padrão brasileiro.",
        "Estou precisando formatar um número de telefone com DDD e validação.",
        "Me ajuda com uma função que formate telefone brasileiro e valide os dígitos?",
        "Uma função format_phone, por favor, no padrão (DD) 9XXXX-XXXX.",
    ],
    "gen-module-grade-calculator": [
        "Preciso calcular a média de notas de um aluno e classificar o resultado.",
        "Pode implementar cálculo de média e classificação (aprovado/recuperação/reprovado)?",
        "Quero funções calculate_average e classify_result.",
        "Implementa um cálculo de média escolar com classificação do resultado.",
        "Cria um módulo que calcule média de notas e diga a situação do aluno.",
        "Estou precisando calcular média de notas e saber se o aluno passou.",
        "Me ajuda com um cálculo de média que classifique aprovado, recuperação ou reprovado?",
        "Um módulo de cálculo de notas, por favor, com média e classificação.",
    ],
}

_SUMMARIES = {
    "gen-module-inventory": "o módulo inventory_management com add_stock, remove_stock e low_stock_items",
    "gen-module-cnpj-validator": "is_valid_cnpj(cnpj), com o cálculo oficial dos dois dígitos verificadores",
    "gen-module-imc-calculator": "o módulo imc_calculator com calculate_imc e classify_imc",
    "gen-module-national-holidays": "o módulo national_holidays com is_national_holiday e holiday_name",
    "gen-module-currency-format": "o módulo currency_format com format_brl e format_percentage",
    "gen-module-password-generator": "generate_password(length, seed), reproduzível via seed",
    "gen-module-slug-generator": "slugify(text), normalizando acentos e pontuação",
    "gen-module-csv-line-parser": "parse_csv_line(line, delimiter=','), tratando campos entre aspas",
    "gen-module-user-registry": "o módulo user_registry com register_user e find_user_by_email",
    "gen-module-discount-calculator": "o módulo discount_calculator com apply_discount e apply_bulk_discount",
    "gen-module-unit-converter": "o módulo unit_converter com km_to_miles, miles_to_km e kg_to_lb",
    "gen-module-text-statistics": "o módulo text_statistics com word_frequency e most_common_word",
    "gen-module-todo-list": "o módulo todo_list com add_task, complete_task e pending_tasks",
    "gen-module-phone-formatter": "format_phone(digits), no padrão brasileiro (DD) 9XXXX-XXXX",
    "gen-module-grade-calculator": "o módulo grade_calculator com calculate_average e classify_result",
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, final6: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think0}</think>"
        + original_raw_text[segs[1].start : segs[2].end]
        + f"<think>{think3}</think>"
        + original_raw_text[segs[4].start : segs[5].end]
        + f"<final>{final6}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        slots = {"summary": _SUMMARIES[base_id]}

        for k, (user_request, variant) in enumerate(zip(request_variants, _VARIANTS), start=1):
            think0 = variant["think_before_write"].format(**slots)
            think3 = variant["think_before_final"].format(**slots)
            final6 = variant["final"].format(**slots)

            raw_text = _rebuild_raw_text(original_raw_text, think0, think3, final6)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para a categoria Módulos realistas.")


if __name__ == "__main__":
    main()
