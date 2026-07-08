#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Tratamento de erro"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4). Mesmo método de
`scripts/expand_oop_classes_phrasing.py`: 8 variantes por tarefa-base, reaproveitando os
fragmentos `<tool_call>`/`<tool_result>` já executados de verdade.

Uso:
    python scripts/expand_error_handling_phrasing.py
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
        think_before_write="O comportamento esperado aqui é tratar o erro explicitamente, não deixar a exceção vazar sem contexto — vou cobrir {summary}.",
        think_before_final="O tratamento de erro está no lugar certo. Vou validar antes de responder.",
        final="Implementei {summary}, tratando o erro de forma explícita. A validação confirma que compila e o teste passa.",
    ),
    dict(
        think_before_write="Para {summary}, o caminho certo é capturar a exceção específica em vez de deixar o programa quebrar sem contexto.",
        think_before_final="Código pronto, com o tratamento de erro implementado — falta validar.",
        final="Cobri {summary} com tratamento de erro explícito, e a validação passou sem problemas.",
    ),
    dict(
        think_before_write="Vou implementar {summary}, cuidando para que o erro tratado seja específico e não esconda outros problemas inesperados.",
        think_before_final="Com o código pronto, o próximo passo é rodar a validação.",
        final="A implementação cobre {summary}, tratando o erro com uma exceção específica. Validado: compila e o teste passa.",
    ),
    dict(
        think_before_write="Aqui o ponto central é o tratamento de erro — vou implementar {summary} garantindo que a exceção certa seja capturada ou levantada.",
        think_before_final="Falta só confirmar que o comportamento de erro está correto antes de responder.",
        final="Implementei {summary}, com o tratamento de erro correto. A checagem confirma que está tudo funcionando.",
    ),
    dict(
        think_before_write="Antes de codificar, já decido a estratégia de erro: para {summary}, preciso capturar/levantar a exceção certa, com mensagem útil.",
        think_before_final="Código pronto. Rodo a validação agora para confirmar antes de responder.",
        final="Cobri {summary} com o tratamento de erro adequado — validado e funcionando corretamente.",
    ),
    dict(
        think_before_write="Vou tratar isso com cuidado: {summary} precisa de uma exceção específica, não um catch genérico que esconderia outros bugs.",
        think_before_final="Antes de finalizar, preciso rodar a validação para confirmar que está tudo certo.",
        final="{summary} está implementado com o tratamento de erro correto. Rodei a validação e confirmei que passa.",
    ),
    dict(
        think_before_write="{summary} é justamente o tipo de caso que precisa de tratamento de erro explícito, não uma checagem solta.",
        think_before_final="Terminei a implementação; agora preciso validar antes de responder ao usuário.",
        final="Pronto: {summary} implementado com tratamento de erro explícito, e já validado com sucesso.",
    ),
    dict(
        think_before_write="Vou escrever isso cuidando do caso de erro desde o início — {summary} só funciona bem se a exceção certa for tratada.",
        think_before_final="Só falta validar o código antes de considerar a tarefa concluída.",
        final="Implementei {summary}, com tratamento de erro explícito. A validação passou sem erros.",
    ),
]

_REQUESTS = {
    "gen-error-safe-divide": [
        "Preciso de uma função que divida dois números e retorne None em vez de quebrar quando o divisor for zero.",
        "Pode implementar uma divisão segura que não lance exceção na divisão por zero?",
        "Quero uma função safe_divide que trate divisão por zero retornando None.",
        "Implementa uma divisão que não quebre o programa quando o denominador for zero.",
        "Cria uma função que capture ZeroDivisionError e retorne None.",
        "Estou precisando de uma divisão segura que não propague exceção ao dividir por zero.",
        "Me ajuda com uma função de divisão que trate o caso de denominador zero?",
        "Uma função safe_divide, por favor, que retorne None em vez de lançar erro.",
    ],
    "gen-error-parse-int-safe": [
        "Preciso de uma função que converta string para inteiro sem quebrar se o texto não for numérico.",
        "Pode implementar uma conversão segura que retorne um valor padrão em caso de erro?",
        "Quero uma função que trate ValueError ao converter texto para int.",
        "Implementa uma conversão de string para int que não lance exceção.",
        "Cria uma função parse_int_safe com valor padrão configurável.",
        "Estou precisando de uma conversão de texto para número que não quebre com entrada inválida.",
        "Me ajuda com uma função que capture erro de conversão e use um valor padrão?",
        "Uma função parse_int_safe, por favor, que trate erro de conversão.",
    ],
    "gen-error-validate-age": [
        "Preciso de uma função que valide idade, levantando erro específico para negativa e para acima de 150.",
        "Pode implementar uma validação de idade com mensagens de erro distintas para cada regra?",
        "Quero uma função validate_age que rejeite idade negativa ou implausível.",
        "Implementa uma validação de idade que levante ValueError com a razão específica.",
        "Cria uma função que garanta que a idade está entre 0 e 150.",
        "Estou precisando validar idade com mensagens de erro claras para cada caso inválido.",
        "Me ajuda com uma função que valide idade e diga exatamente qual regra falhou?",
        "Uma função validate_age, por favor, com mensagens de erro específicas.",
    ],
    "gen-error-config-lookup": [
        "Preciso de uma função que busque uma chave de configuração e levante erro customizado se não existir.",
        "Pode implementar uma busca em dicionário que relance KeyError com mensagem melhor?",
        "Quero uma função que trate chave ausente numa configuração com uma mensagem clara.",
        "Implementa uma busca de configuração que preserve a causa original do erro.",
        "Cria uma função get_config_value que use raise...from para encadear o erro.",
        "Estou precisando de uma função que dê um erro mais claro quando a chave de config não existe.",
        "Me ajuda com uma busca de configuração que relance KeyError com mensagem customizada?",
        "Uma função get_config_value, por favor, com erro customizado para chave ausente.",
    ],
    "gen-error-safe-list-get": [
        "Preciso de uma função que acesse um índice de lista sem quebrar se ele não existir.",
        "Pode implementar um acesso seguro a lista que retorne um valor padrão?",
        "Quero uma função que trate IndexError ao acessar uma lista.",
        "Implementa um acesso a lista que não lance exceção em índice inválido.",
        "Cria uma função safe_list_get com valor padrão configurável.",
        "Estou precisando de um acesso a lista que não quebre com índice fora do intervalo.",
        "Me ajuda com uma função que capture IndexError e use um valor padrão?",
        "Uma função safe_list_get, por favor, que trate índice inválido.",
    ],
    "gen-error-validate-email": [
        "Preciso de uma função que valide o formato de um e-mail e diga exatamente o que está errado.",
        "Pode implementar uma validação de e-mail com mensagens de erro específicas?",
        "Quero uma função validate_email que rejeite formatos inválidos com razão clara.",
        "Implementa uma validação de e-mail que cheque @ e domínio separadamente.",
        "Cria uma função que valide e-mail e levante ValueError com a causa exata.",
        "Estou precisando validar e-mails com mensagens de erro específicas para cada problema.",
        "Me ajuda com uma função que valide e-mail e diga qual parte está errada?",
        "Uma função validate_email, por favor, com mensagens de erro específicas.",
    ],
    "gen-error-parse-brl": [
        "Preciso de uma função que converta 'R$ 1.234,56' para float, tratando formato inválido.",
        "Pode implementar uma conversão de valor em reais para float com erro claro em caso de falha?",
        "Quero uma função parse_brl que trate o formato brasileiro de moeda.",
        "Implementa uma conversão de string em R$ para float, relançando erro com mensagem clara.",
        "Cria uma função que converta valores em reais formatados para número.",
        "Estou precisando converter texto no formato R$ para float, com tratamento de erro.",
        "Me ajuda com uma função que converta 'R$ 1.234,56' e trate formato inválido?",
        "Uma função parse_brl, por favor, com tratamento de erro para formato inválido.",
    ],
    "gen-error-retry-operation": [
        "Preciso de uma função que tente executar algo várias vezes e levante o último erro se todas falharem.",
        "Pode implementar um mecanismo de retry que propague a última exceção após esgotar tentativas?",
        "Quero uma função retry_operation que capture erros e tente de novo até um limite.",
        "Implementa uma lógica de retry que relance o último erro se nada funcionar.",
        "Cria uma função que tente novamente uma operação até um número máximo de vezes.",
        "Estou precisando de um retry que guarde o último erro e o relance no final.",
        "Me ajuda com uma função de retry que levante a última exceção após esgotar tentativas?",
        "Uma função retry_operation, por favor, com relançamento do último erro.",
    ],
    "gen-error-password-strength": [
        "Preciso de uma função que valide a força de uma senha com mensagens de erro específicas.",
        "Pode implementar uma validação de senha que diga exatamente qual regra falhou?",
        "Quero uma função que rejeite senha fraca com razão clara (tamanho, maiúscula, dígito).",
        "Implementa uma validação de senha que levante ValueError específico por regra.",
        "Cria uma função validate_password_strength com mensagens de erro por regra.",
        "Estou precisando validar senha com feedback específico sobre o que está faltando.",
        "Me ajuda com uma função que valide senha e diga qual critério não foi atendido?",
        "Uma função validate_password_strength, por favor, com mensagens específicas por regra.",
    ],
    "gen-error-safe-json-parse": [
        "Preciso de uma função que tente fazer parse de JSON sem quebrar se for inválido.",
        "Pode implementar um parse seguro de JSON que retorne None em caso de erro?",
        "Quero uma função que trate json.JSONDecodeError ao converter uma string.",
        "Implementa um parse de JSON que não lance exceção para texto inválido.",
        "Cria uma função safe_json_parse que retorne None em JSON malformado.",
        "Estou precisando de um parse de JSON que não quebre com entrada inválida.",
        "Me ajuda com uma função que capture erro de parse de JSON e retorne None?",
        "Uma função safe_json_parse, por favor, que trate JSON inválido.",
    ],
    "gen-error-insufficient-funds": [
        "Preciso de uma exceção customizada para saldo insuficiente e uma função de saque que a use.",
        "Pode implementar um erro específico (não genérico) para saque maior que o saldo?",
        "Quero uma exceção InsufficientFundsError e uma função withdraw que a levante.",
        "Implementa um saque que levante um erro customizado quando não há saldo suficiente.",
        "Cria uma exceção própria para representar saldo insuficiente numa função de saque.",
        "Estou precisando de um erro customizado para o caso de saque maior que o saldo.",
        "Me ajuda com uma exceção customizada para saldo insuficiente numa função de saque?",
        "Uma exceção InsufficientFundsError, por favor, usada numa função withdraw.",
    ],
    "gen-error-validate-date-br": [
        "Preciso de uma função que valide uma data no formato DD/MM/AAAA, tratando formato ou data inválida.",
        "Pode implementar uma validação de data brasileira que rejeite datas inexistentes como 31/02?",
        "Quero uma função validate_date_br com mensagem clara para data inválida.",
        "Implementa uma validação de data no formato DD/MM/AAAA usando datetime.",
        "Cria uma função que confirme se uma data no formato brasileiro é válida.",
        "Estou precisando validar datas no formato DD/MM/AAAA, incluindo datas que não existem.",
        "Me ajuda com uma função que valide data brasileira e trate erro de formato?",
        "Uma função validate_date_br, por favor, com tratamento de erro claro.",
    ],
    "gen-error-division-pipeline": [
        "Preciso de uma função que processe uma lista de divisões, usando try/except/else/finally.",
        "Pode implementar um pipeline de divisões que conte sucessos e tentativas usando else/finally?",
        "Quero uma função que trate divisão por zero numa lista de pares e conte o resultado.",
        "Implementa um processamento de divisões que use try/except/else/finally corretamente.",
        "Cria uma função que registre sucessos e tentativas ao processar várias divisões.",
        "Estou precisando processar pares de divisão contando quantos deram certo.",
        "Me ajuda com um pipeline de divisões que use else e finally do jeito certo?",
        "Uma função process_divisions, por favor, usando try/except/else/finally.",
    ],
    "gen-error-validate-positive-number": [
        "Preciso de uma função que valide se um valor é número positivo, com erros distintos para tipo e valor.",
        "Pode implementar uma validação que levante TypeError para tipo errado e ValueError para negativo?",
        "Quero uma função validate_positive_number que distinga erro de tipo e erro de valor.",
        "Implementa uma validação numérica que trate tipo inválido e valor negativo separadamente.",
        "Cria uma função que garanta que um valor é numérico e positivo.",
        "Estou precisando validar número positivo com exceções diferentes por tipo de problema.",
        "Me ajuda com uma função que valide número positivo com TypeError e ValueError distintos?",
        "Uma função validate_positive_number, por favor, com erros distintos por tipo de problema.",
    ],
    "gen-error-stock-check": [
        "Preciso de uma exceção customizada para estoque insuficiente e uma função de compra que a use.",
        "Pode implementar uma função de compra que levante erro customizado sem vazar detalhe técnico interno?",
        "Quero uma exceção OutOfStockError usada numa função purchase.",
        "Implementa uma função de compra que trate item inexistente e quantidade insuficiente.",
        "Cria uma exceção própria para representar falta de estoque numa função de compra.",
        "Estou precisando de um erro customizado para quando o estoque não é suficiente.",
        "Me ajuda com uma exceção customizada para estoque insuficiente que esconda a causa técnica?",
        "Uma exceção OutOfStockError, por favor, usada numa função purchase.",
    ],
}

_SUMMARIES = {
    "gen-error-safe-divide": "safe_divide(a, b), retornando None em vez de lançar exceção na divisão por zero",
    "gen-error-parse-int-safe": "parse_int_safe(text, default=0), com valor padrão em caso de conversão inválida",
    "gen-error-validate-age": "validate_age(age), com mensagens distintas para idade negativa e acima de 150",
    "gen-error-config-lookup": "get_config_value(config, key), relançando KeyError com mensagem customizada",
    "gen-error-safe-list-get": "safe_list_get(items, index, default=None), tratando índice inexistente",
    "gen-error-validate-email": "validate_email(email), com mensagens específicas para cada formato inválido",
    "gen-error-parse-brl": "parse_brl(value), convertendo o formato R$ para float com erro claro em falha",
    "gen-error-retry-operation": "retry_operation(func, retries=3), relançando o último erro após esgotar tentativas",
    "gen-error-password-strength": "validate_password_strength(password), com mensagem específica por regra",
    "gen-error-safe-json-parse": "safe_json_parse(text), retornando None para JSON inválido",
    "gen-error-insufficient-funds": "a exceção InsufficientFundsError e a função withdraw(balance, amount)",
    "gen-error-validate-date-br": "validate_date_br(date_str), validando o formato DD/MM/AAAA",
    "gen-error-division-pipeline": "process_divisions(pairs), usando try/except/else/finally",
    "gen-error-validate-positive-number": "validate_positive_number(value), distinguindo TypeError de ValueError",
    "gen-error-stock-check": "a exceção OutOfStockError e a função purchase(stock, item, quantity)",
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

    print(f"{count} variantes geradas para a categoria Tratamento de erro.")


if __name__ == "__main__":
    main()
