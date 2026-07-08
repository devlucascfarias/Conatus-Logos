#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Padrões Python"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4). Mesmo método das demais categorias:
8 variantes por tarefa-base, reaproveitando os fragmentos `<tool_call>`/`<tool_result>` já
executados de verdade.

Uso:
    python scripts/expand_python_patterns_phrasing.py
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
        think_before_write="O padrão certo aqui é {pattern} — vou estruturar {summary} dessa forma.",
        think_before_final="A implementação usa {pattern} corretamente. Vou validar antes de responder.",
        final="Implementei {summary} usando {pattern}. A validação confirma que compila e o teste passa.",
    ),
    dict(
        think_before_write="Para {summary}, {pattern} é a ferramenta certa da linguagem — vou modelar assim.",
        think_before_final="Pronto o código com {pattern}; falta rodar a validação antes de responder.",
        final="Cobri {summary} com {pattern}, e a validação passou sem problemas.",
    ),
    dict(
        think_before_write="Vou usar {pattern} para {summary}, já que é o jeito idiomático de resolver esse tipo de problema em Python.",
        think_before_final="Com o código pronto usando {pattern}, o próximo passo é validar.",
        final="A implementação de {summary} usa {pattern} — validado: compila e o teste passa.",
    ),
    dict(
        think_before_write="{pattern} é a estrutura certa aqui — vou implementar {summary} em cima disso.",
        think_before_final="Falta só confirmar que o código com {pattern} está correto antes de responder.",
        final="{summary} está implementado com {pattern}. A checagem confirma que está tudo funcionando.",
    ),
    dict(
        think_before_write="Antes de escrever qualquer coisa, já decido pela estrutura: {pattern}, porque {summary} pede exatamente isso.",
        think_before_final="Código pronto usando {pattern}. Rodo a validação agora para confirmar antes de responder.",
        final="Modelei {summary} com {pattern} — validado e funcionando corretamente.",
    ),
    dict(
        think_before_write="Vou usar {pattern} aqui, não uma solução mais verbosa, porque {summary} se encaixa bem nesse padrão.",
        think_before_final="Antes de finalizar, preciso rodar a validação para confirmar que está tudo certo.",
        final="{summary} foi implementado com {pattern}. Rodei a validação e confirmei que passa.",
    ),
    dict(
        think_before_write="{summary} é justamente o tipo de caso que pede {pattern} em vez de uma solução mais manual.",
        think_before_final="Terminei a implementação com {pattern}; agora preciso validar antes de responder ao usuário.",
        final="Pronto: {summary} implementado com {pattern}, e já validado com sucesso.",
    ),
    dict(
        think_before_write="Vou escrever isso usando {pattern} desde o início, já que {summary} depende dessa estrutura para funcionar bem.",
        think_before_final="Só falta validar o código com {pattern} antes de considerar a tarefa concluída.",
        final="Implementei {summary} com {pattern}. A validação passou sem erros.",
    ),
]

_REQUESTS = {
    "gen-pattern-count-calls": [
        "Preciso de um decorator que conte quantas vezes uma função foi chamada.",
        "Pode implementar um contador de chamadas usando decorator?",
        "Quero um decorator count_calls que exponha o número de chamadas.",
        "Implementa um decorator que rastreie quantas vezes a função rodou.",
        "Cria um decorator para contar invocações de uma função.",
        "Estou precisando de um jeito de contar chamadas de função via decorator.",
        "Me ajuda com um decorator que conte as chamadas de uma função?",
        "Um decorator count_calls, por favor, com contador acessível de fora.",
    ],
    "gen-pattern-memoize": [
        "Preciso de um decorator que faça cache do resultado de uma função por argumentos.",
        "Pode implementar memoização como decorator?",
        "Quero um decorator memoize que evite recomputar resultados já vistos.",
        "Implementa um cache de resultados via decorator em Python.",
        "Cria um decorator que armazene resultados por argumento.",
        "Estou precisando de memoização para uma função pura.",
        "Me ajuda com um decorator de cache baseado nos argumentos da função?",
        "Um decorator memoize, por favor, que evite chamadas repetidas.",
    ],
    "gen-pattern-retry-decorator": [
        "Preciso de um decorator com parâmetro que tente novamente uma função em caso de erro.",
        "Pode implementar um decorator retry configurável com número máximo de tentativas?",
        "Quero um decorator que reexecute uma função até um limite de tentativas.",
        "Implementa uma fábrica de decorator para retry com max_attempts.",
        "Cria um decorator retry(max_attempts) que relance o último erro.",
        "Estou precisando de retry automático via decorator parametrizado.",
        "Me ajuda com um decorator que tente novamente e relance o último erro?",
        "Um decorator retry, por favor, com número de tentativas configurável.",
    ],
    "gen-pattern-require-positive": [
        "Preciso de um decorator que valide se os argumentos numéricos são positivos.",
        "Pode implementar um decorator que rejeite valores não positivos antes de rodar a função?",
        "Quero um decorator require_positive que valide os args antes da execução.",
        "Implementa uma validação de argumentos positivos via decorator.",
        "Cria um decorator que garanta que os números passados são positivos.",
        "Estou precisando de um decorator de validação de argumentos positivos.",
        "Me ajuda com um decorator que valide argumentos positivos antes de chamar a função?",
        "Um decorator require_positive, por favor, que valide antes de executar.",
    ],
    "gen-pattern-log-calls": [
        "Preciso de um decorator que registre argumentos e resultado de cada chamada.",
        "Pode implementar um decorator de log de chamadas de função?",
        "Quero um decorator log_calls que guarde um histórico de chamadas.",
        "Implementa um decorator que registre args, kwargs e resultado.",
        "Cria um decorator que mantenha um log das chamadas de uma função.",
        "Estou precisando de um decorator que registre cada chamada com seus dados.",
        "Me ajuda com um decorator que registre argumentos e resultado de cada chamada?",
        "Um decorator log_calls, por favor, com histórico acessível de fora.",
    ],
    "gen-pattern-fibonacci-generator": [
        "Preciso de um generator que produza os primeiros n números de Fibonacci.",
        "Pode implementar Fibonacci como generator, sem montar a lista toda de uma vez?",
        "Quero um generator fibonacci_sequence que produza a sequência sob demanda.",
        "Implementa a sequência de Fibonacci usando yield.",
        "Cria um generator que gere Fibonacci preguiçosamente.",
        "Estou precisando de Fibonacci como generator em Python.",
        "Me ajuda com um generator que produza Fibonacci sob demanda?",
        "Um generator fibonacci_sequence, por favor, usando yield.",
    ],
    "gen-pattern-chunked-generator": [
        "Preciso de um generator que divida um iterável em blocos de tamanho fixo.",
        "Pode implementar chunking como generator, incluindo o bloco parcial final?",
        "Quero um generator chunked que produza listas de tamanho fixo.",
        "Implementa uma divisão em blocos usando yield.",
        "Cria um generator que agrupe itens de um iterável em chunks.",
        "Estou precisando dividir uma sequência em blocos de tamanho n.",
        "Me ajuda com um generator que produza blocos de tamanho fixo de um iterável?",
        "Um generator chunked, por favor, cobrindo o bloco parcial final.",
    ],
    "gen-pattern-countdown-generator": [
        "Preciso de um generator que conte de n até 1.",
        "Pode implementar uma contagem regressiva como generator?",
        "Quero um generator countdown que produza os números decrescentes.",
        "Implementa uma contagem regressiva usando yield.",
        "Cria um generator que conte de trás para frente até 1.",
        "Estou precisando de uma contagem regressiva preguiçosa em Python.",
        "Me ajuda com um generator countdown que pare em 1?",
        "Um generator countdown, por favor, usando yield.",
    ],
    "gen-pattern-flatten-generator": [
        "Preciso de um generator que achate uma lista de listas aninhada.",
        "Pode implementar flatten usando yield from para qualquer profundidade?",
        "Quero um generator flatten que lide com aninhamento em vários níveis.",
        "Implementa um achatamento de listas aninhadas usando generator.",
        "Cria um generator recursivo que produza os itens de uma estrutura aninhada.",
        "Estou precisando achatar listas aninhadas de forma preguiçosa.",
        "Me ajuda com um generator flatten que use yield from recursivamente?",
        "Um generator flatten, por favor, cobrindo aninhamento em qualquer profundidade.",
    ],
    "gen-pattern-take-generator": [
        "Preciso de um generator infinito de números naturais e uma função que pegue os primeiros n.",
        "Pode implementar um generator infinito com uma função take para limitar quantos valores extrair?",
        "Quero um generator natural_numbers e uma função take(generator, n).",
        "Implementa um generator infinito controlado por uma função auxiliar take.",
        "Cria um generator que nunca termina sozinho, mais uma função que extrai n valores.",
        "Estou precisando de números naturais como generator infinito, com controle externo de quantos pegar.",
        "Me ajuda com um generator infinito e uma função take que extraia os primeiros valores?",
        "Um generator natural_numbers, por favor, com uma função take associada.",
    ],
    "gen-pattern-suppress-error": [
        "Preciso de um context manager que suprima exceções de um tipo específico.",
        "Pode implementar algo parecido com contextlib.suppress, do zero?",
        "Quero um context manager suppress_error que ignore certos tipos de exceção.",
        "Implementa um context manager que suprima erros configuráveis via __exit__.",
        "Cria um context manager que engula exceções de tipos específicos.",
        "Estou precisando de uma forma de suprimir exceção específica com with.",
        "Me ajuda com um context manager que suprima só os tipos de erro configurados?",
        "Um context manager suppress_error, por favor, deixando outros erros propagarem.",
    ],
    "gen-pattern-temporary-override": [
        "Preciso de um context manager que sobrescreva temporariamente uma chave de dicionário.",
        "Pode implementar uma sobrescrita temporária que restaure o valor original ao sair?",
        "Quero um context manager temporary_override para configuração.",
        "Implementa um context manager que restaure ou remova a chave ao sair do bloco.",
        "Cria um context manager que altere um valor de config só durante o bloco with.",
        "Estou precisando sobrescrever uma configuração temporariamente e restaurar depois.",
        "Me ajuda com um context manager que restaure o dicionário ao estado original?",
        "Um context manager temporary_override, por favor, cobrindo chave nova e chave existente.",
    ],
    "gen-pattern-transaction-contextmanager": [
        "Preciso de um context manager de transação com begin/commit/rollback.",
        "Pode implementar isso com @contextmanager, registrando commit ou rollback?",
        "Quero um context manager transaction que trate sucesso e falha diferente.",
        "Implementa uma transação simulada usando @contextmanager.",
        "Cria um context manager que faça rollback automático em caso de exceção.",
        "Estou precisando simular uma transação com commit e rollback.",
        "Me ajuda com um context manager de transação que relance a exceção após o rollback?",
        "Um context manager transaction, por favor, usando @contextmanager.",
    ],
    "gen-pattern-timer-contextmanager": [
        "Preciso de um context manager Timer que meça o tempo decorrido de forma testável.",
        "Pode implementar um Timer que aceite uma fonte de tempo customizada?",
        "Quero um context manager Timer com relógio injetável para os testes.",
        "Implementa uma medição de tempo com context manager, sem depender do relógio real no teste.",
        "Cria um Timer que guarde o tempo decorrido no atributo elapsed.",
        "Estou precisando medir tempo de execução de forma determinística e testável.",
        "Me ajuda com um context manager Timer que permita injetar um clock falso?",
        "Um context manager Timer, por favor, com clock injetável.",
    ],
    "gen-pattern-resource-pool-contextmanager": [
        "Preciso de um context manager que registre aquisição e liberação de um recurso.",
        "Pode implementar isso com @contextmanager, garantindo liberação mesmo com erro?",
        "Quero um context manager acquire_resource que sempre libere o recurso no final.",
        "Implementa um gerenciamento de recurso com finally garantindo a liberação.",
        "Cria um context manager que registre 'acquired'/'released' num pool.",
        "Estou precisando garantir que um recurso seja liberado mesmo se der erro no bloco.",
        "Me ajuda com um context manager que libere o recurso mesmo em caso de exceção?",
        "Um context manager acquire_resource, por favor, usando @contextmanager.",
    ],
}

_PATTERNS = {
    "gen-pattern-count-calls": "um decorator",
    "gen-pattern-memoize": "um decorator com cache (memoização)",
    "gen-pattern-retry-decorator": "uma fábrica de decorator parametrizado",
    "gen-pattern-require-positive": "um decorator de validação",
    "gen-pattern-log-calls": "um decorator com registro de chamadas",
    "gen-pattern-fibonacci-generator": "um generator",
    "gen-pattern-chunked-generator": "um generator",
    "gen-pattern-countdown-generator": "um generator",
    "gen-pattern-flatten-generator": "um generator recursivo com yield from",
    "gen-pattern-take-generator": "um generator infinito",
    "gen-pattern-suppress-error": "um context manager baseado em classe",
    "gen-pattern-temporary-override": "um context manager baseado em classe",
    "gen-pattern-transaction-contextmanager": "um context manager com @contextmanager",
    "gen-pattern-timer-contextmanager": "um context manager baseado em classe",
    "gen-pattern-resource-pool-contextmanager": "um context manager com @contextmanager",
}

_SUMMARIES = {
    "gen-pattern-count-calls": "count_calls, expondo o total de chamadas no atributo calls",
    "gen-pattern-memoize": "memoize, cacheando resultados por argumentos",
    "gen-pattern-retry-decorator": "retry(max_attempts), relançando o último erro ao esgotar tentativas",
    "gen-pattern-require-positive": "require_positive, validando argumentos numéricos antes de executar",
    "gen-pattern-log-calls": "log_calls, registrando args/kwargs/resultado de cada chamada",
    "gen-pattern-fibonacci-generator": "fibonacci_sequence(n), produzindo a sequência sob demanda",
    "gen-pattern-chunked-generator": "chunked(iterable, size), incluindo o bloco parcial final",
    "gen-pattern-countdown-generator": "countdown(n), contando de n até 1",
    "gen-pattern-flatten-generator": "flatten(nested), achatando qualquer nível de aninhamento",
    "gen-pattern-take-generator": "natural_numbers() e take(generator, n)",
    "gen-pattern-suppress-error": "suppress_error, suprimindo só os tipos configurados",
    "gen-pattern-temporary-override": "temporary_override, restaurando o valor original ao sair",
    "gen-pattern-transaction-contextmanager": "transaction, com commit em sucesso e rollback em erro",
    "gen-pattern-timer-contextmanager": "Timer, com relógio injetável para testes determinísticos",
    "gen-pattern-resource-pool-contextmanager": "acquire_resource, garantindo liberação mesmo com erro",
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

        slots = {"pattern": _PATTERNS[base_id], "summary": _SUMMARIES[base_id]}

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

    print(f"{count} variantes geradas para a categoria Padrões Python.")


if __name__ == "__main__":
    main()
