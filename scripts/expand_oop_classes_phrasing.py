#!/usr/bin/env python
"""Expansão de diversidade de frase para a categoria "Classes/OOP"
(docs/plan_dataset_expansion_oop_shell.md, seções 2 e 4).

Gera 8 variantes de frase por tarefa-base de `gen_oop_classes_pilot.py`, reaproveitando os
MESMOS fragmentos `<tool_call>`/`<tool_result>` já executados de verdade (extraídos via
`parse_segments`, nunca reescritos) — nenhuma execução nova é necessária.

Seguindo o aviso crítico do plano: cada uma das 8 variantes usa uma estrutura de frase
genuinamente diferente (ordem de cláusulas, grau de formalidade, direta vs. indireta), não só
troca de sinônimo isolado — o mesmo método que já corrigiu D-generalization-gap e
D-tool-pilots-phrasing.

Forma fixa de cada trajetória-base (7 segmentos, em ordem):
    think, tool_call, tool_result, think, tool_call, tool_result, final
Só os 3 segmentos de prosa (índices 0, 3, 6) são substituídos por variante; os 4 de
tool_call/tool_result (índices 1, 2, 4, 5) são copiados verbatim do original.

Uso:
    python scripts/expand_oop_classes_phrasing.py
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

# 8 variantes de estrutura de frase, genuinamente diferentes entre si (não só sinônimo).
_VARIANTS = [
    dict(
        think_before_write="O pedido pede uma classe de verdade, com estado e métodos — vou estruturar {summary} dessa forma, não como funções soltas.",
        think_before_final="A classe está implementada. Antes de responder, preciso confirmar que compila e passa nos testes.",
        final="Implementei a classe {class_name}, cobrindo {summary}. A validação confirma que compila e os testes passam.",
    ),
    dict(
        think_before_write="Para atender {summary}, uma classe com estado interno faz mais sentido do que funções isoladas — é assim que vou modelar.",
        think_before_final="Pronto o código; falta rodar a validação antes de dar a resposta final.",
        final="A classe {class_name} ficou pronta, cobrindo {summary}, e passou na validação sem problemas.",
    ),
    dict(
        think_before_write="Vou modelar isso como uma classe, já que {summary} envolve estado que precisa persistir entre chamadas de método.",
        think_before_final="Com a classe escrita, o próximo passo é validar antes de fechar a resposta.",
        final="Criei {class_name} como uma classe, com métodos para {summary}. Validado: compila e os testes passam.",
    ),
    dict(
        think_before_write="Uma classe é a estrutura certa aqui — vou implementar {class_name} cobrindo {summary} com estado guardado no objeto.",
        think_before_final="Falta só confirmar que o código está correto antes de responder ao usuário.",
        final="{class_name} está implementada como classe, cobrindo {summary}. A checagem confirma que está tudo funcionando.",
    ),
    dict(
        think_before_write="Antes de escrever qualquer coisa, já decido pela estrutura: classe, porque {summary} exige guardar estado entre chamadas.",
        think_before_final="Código pronto. Rodo a validação agora para ter certeza antes de responder.",
        final="Modelei {class_name} como classe, com métodos cobrindo {summary} — validado e funcionando corretamente.",
    ),
    dict(
        think_before_write="Vou usar uma classe Python aqui, não funções soltas, porque o comportamento pedido ({summary}) depende de estado interno.",
        think_before_final="Antes de finalizar, preciso rodar a validação para confirmar que está tudo certo.",
        final="A classe {class_name} foi implementada cobrindo {summary}. Rodei a validação e confirmei que passa.",
    ),
    dict(
        think_before_write="{summary} é justamente o tipo de comportamento que pede uma classe com estado — vou implementar {class_name} assim.",
        think_before_final="Terminei a implementação; agora preciso validar antes de responder ao usuário.",
        final="Pronto: {class_name} implementada como classe, cobrindo {summary}, e já validada com sucesso.",
    ),
    dict(
        think_before_write="Vou escrever isso como classe — {class_name}, com métodos para {summary} — já que funções soltas perderiam o estado entre chamadas.",
        think_before_final="Só falta validar o código antes de considerar a tarefa concluída.",
        final="Implementei {class_name} (classe) cobrindo {summary}. A validação passou sem erros.",
    ),
]

_REQUESTS = {
    "gen-oop-calculator": [
        "Preciso de uma classe Calculator em Python com soma, subtração, multiplicação, divisão e fatorial.",
        "Pode implementar uma classe de calculadora com essas cinco operações: soma, subtração, multiplicação, divisão e fatorial?",
        "Quero uma classe que faça soma, subtração, multiplicação, divisão e fatorial.",
        "Implementa uma calculadora orientada a objetos com essas cinco operações.",
        "Cria uma classe Calculator com métodos de soma, subtração, multiplicação, divisão e fatorial.",
        "Estou precisando de uma classe Python que reúna soma, subtração, multiplicação, divisão e fatorial.",
        "Me ajuda a criar uma classe de calculadora com as quatro operações básicas mais fatorial?",
        "Uma classe Calculator, por favor, com soma, subtração, multiplicação, divisão e fatorial.",
    ],
    "gen-oop-stack": [
        "Preciso de uma classe Stack com push, pop, peek e is_empty.",
        "Pode implementar uma pilha (Stack) com essas quatro operações?",
        "Quero uma classe de pilha com push, pop, peek e verificação de vazio.",
        "Implementa uma Stack em Python com push/pop/peek/is_empty.",
        "Cria uma classe que funcione como pilha (LIFO) com push, pop, peek e is_empty.",
        "Estou precisando de uma pilha orientada a objetos com essas operações.",
        "Me ajuda com uma classe Stack que tenha push, pop, peek e is_empty?",
        "Uma classe Stack, por favor, com as operações clássicas de pilha.",
    ],
    "gen-oop-queue": [
        "Preciso de uma classe Queue com enqueue, dequeue e is_empty.",
        "Pode implementar uma fila (Queue) com essas três operações?",
        "Quero uma classe de fila FIFO com enqueue, dequeue e verificação de vazio.",
        "Implementa uma Queue em Python com enqueue/dequeue/is_empty.",
        "Cria uma classe que funcione como fila com enqueue, dequeue e is_empty.",
        "Estou precisando de uma fila orientada a objetos com essas operações.",
        "Me ajuda com uma classe Queue que tenha enqueue, dequeue e is_empty?",
        "Uma classe Queue, por favor, com as operações clássicas de fila.",
    ],
    "gen-oop-bank_account": [
        "Preciso de uma classe BankAccount com depósito, saque e saldo, sem permitir saldo negativo.",
        "Pode implementar uma conta bancária que bloqueie saque maior que o saldo?",
        "Quero uma classe de conta bancária com deposit, withdraw e balance.",
        "Implementa uma BankAccount em Python que impeça saldo negativo.",
        "Cria uma classe de conta que controle depósito e saque com validação de saldo.",
        "Estou precisando de uma classe que modele uma conta bancária simples.",
        "Me ajuda com uma classe BankAccount que valide saldo antes do saque?",
        "Uma classe BankAccount, por favor, com deposit, withdraw e balance protegidos.",
    ],
    "gen-oop-rectangle": [
        "Preciso de uma classe Rectangle com área, perímetro e verificação se é quadrado.",
        "Pode implementar um retângulo com esses três cálculos?",
        "Quero uma classe Rectangle com area, perimeter e is_square.",
        "Implementa uma classe de retângulo em Python com essas operações.",
        "Cria uma classe que calcule área, perímetro e se o retângulo é um quadrado.",
        "Estou precisando de uma classe Rectangle com esses três métodos.",
        "Me ajuda com uma classe de retângulo que tenha area, perimeter e is_square?",
        "Uma classe Rectangle, por favor, com área, perímetro e is_square.",
    ],
    "gen-oop-circle": [
        "Preciso de uma classe Circle com área e circunferência.",
        "Pode implementar um círculo com esses dois cálculos?",
        "Quero uma classe Circle com area e circumference.",
        "Implementa uma classe de círculo em Python com essas operações.",
        "Cria uma classe que calcule área e circunferência de um círculo.",
        "Estou precisando de uma classe Circle com area e circumference.",
        "Me ajuda com uma classe de círculo que calcule área e circunferência?",
        "Uma classe Circle, por favor, com área e circunferência.",
    ],
    "gen-oop-temperature_converter": [
        "Preciso de uma classe que converta Celsius para Fahrenheit e Kelvin.",
        "Pode implementar um conversor de temperatura orientado a objetos?",
        "Quero uma classe TemperatureConverter com to_fahrenheit e to_kelvin.",
        "Implementa uma classe de conversão de temperatura em Python.",
        "Cria uma classe que guarde uma temperatura em Celsius e converta para as outras escalas.",
        "Estou precisando de um conversor de temperatura como classe.",
        "Me ajuda com uma classe que converta Celsius para Fahrenheit e Kelvin?",
        "Uma classe TemperatureConverter, por favor, com conversão para Fahrenheit e Kelvin.",
    ],
    "gen-oop-counter": [
        "Preciso de uma classe Counter com increment, decrement e reset.",
        "Pode implementar um contador orientado a objetos com essas operações?",
        "Quero uma classe Counter com increment, decrement, reset e uma propriedade value.",
        "Implementa uma classe de contador em Python.",
        "Cria uma classe que incremente, decremente e reinicie um valor guardado.",
        "Estou precisando de um contador como classe, com value acessível.",
        "Me ajuda com uma classe Counter que tenha increment, decrement e reset?",
        "Uma classe Counter, por favor, com essas três operações e uma propriedade value.",
    ],
    "gen-oop-linked_list": [
        "Preciso de uma lista encadeada simples em Python com append, prepend e conversão para lista.",
        "Pode implementar uma linked list com essas três operações?",
        "Quero uma classe LinkedList com append, prepend e to_list.",
        "Implementa uma lista encadeada orientada a objetos em Python.",
        "Cria uma estrutura de lista encadeada com append, prepend e conversão pra lista comum.",
        "Estou precisando de uma linked list simples como classe.",
        "Me ajuda com uma lista encadeada que tenha append, prepend e to_list?",
        "Uma LinkedList, por favor, com append, prepend e to_list.",
    ],
    "gen-oop-matrix": [
        "Preciso de uma classe Matrix que envolva uma lista de listas, com soma e transposição.",
        "Pode implementar uma classe de matriz com essas duas operações?",
        "Quero uma classe Matrix com add e transpose.",
        "Implementa uma classe de matriz em Python que valide dimensões na soma.",
        "Cria uma classe que represente uma matriz com soma e transposição.",
        "Estou precisando de uma classe Matrix com add e transpose.",
        "Me ajuda com uma classe de matriz que some e transponha?",
        "Uma classe Matrix, por favor, com soma e transposição.",
    ],
    "gen-oop-shopping_cart": [
        "Preciso de uma classe ShoppingCart com adicionar item, remover item e total em R$.",
        "Pode implementar um carrinho de compras que formate o total em Real?",
        "Quero uma classe ShoppingCart com add_item, remove_item e total formatado em R$.",
        "Implementa um carrinho de compras em Python com total em reais.",
        "Cria uma classe que gerencie itens de um carrinho e calcule o total em R$.",
        "Estou precisando de uma classe de carrinho de compras com total em Real.",
        "Me ajuda com uma classe ShoppingCart que calcule o total formatado como R$?",
        "Uma classe ShoppingCart, por favor, com total em formato de moeda brasileira.",
    ],
    "gen-oop-employee_payroll": [
        "Preciso de uma classe Employee que calcule o salário líquido com bônus percentual.",
        "Pode implementar uma classe de funcionário com cálculo de salário e bônus?",
        "Quero uma classe Employee com net_salary considerando um bônus percentual.",
        "Implementa uma classe de folha de pagamento simples em Python.",
        "Cria uma classe que calcule o salário líquido a partir do salário base e um bônus.",
        "Estou precisando de uma classe Employee com cálculo de salário líquido.",
        "Me ajuda com uma classe que calcule salário com bônus percentual?",
        "Uma classe Employee, por favor, com net_salary e bônus percentual.",
    ],
    "gen-oop-cpf_validator": [
        "Preciso de uma classe que valide CPF calculando os dígitos verificadores de verdade.",
        "Pode implementar um validador de CPF orientado a objetos, com o cálculo oficial?",
        "Quero uma classe CPFValidator com is_valid que calcule os dígitos verificadores.",
        "Implementa uma validação de CPF em Python seguindo a regra oficial dos dígitos.",
        "Cria uma classe que confirme se um CPF é válido pelo algoritmo do módulo 11.",
        "Estou precisando de um validador de CPF como classe, não só checagem de formato.",
        "Me ajuda com uma classe CPFValidator que calcule os dígitos verificadores?",
        "Uma classe CPFValidator, por favor, com o cálculo real dos dígitos verificadores.",
    ],
    "gen-oop-vector2d": [
        "Preciso de uma classe Vector2D com soma, subtração e magnitude, usando sobrecarga de operadores.",
        "Pode implementar um vetor 2D onde + e - funcionem entre instâncias?",
        "Quero uma classe Vector2D com __add__, __sub__ e magnitude.",
        "Implementa um vetor bidimensional em Python com sobrecarga de operadores.",
        "Cria uma classe Vector2D que suporte soma e subtração via operadores.",
        "Estou precisando de um Vector2D com magnitude e operadores sobrecarregados.",
        "Me ajuda com uma classe de vetor 2D que use __add__ e __sub__?",
        "Uma classe Vector2D, por favor, com operadores sobrecarregados e magnitude.",
    ],
    "gen-oop-playlist": [
        "Preciso de uma classe Playlist com adicionar música, remover música e avançar faixa de forma circular.",
        "Pode implementar uma playlist que volte para a primeira faixa ao chegar no fim?",
        "Quero uma classe Playlist com add_song, remove_song e next_track circular.",
        "Implementa uma playlist em Python com navegação circular entre faixas.",
        "Cria uma classe que gerencie músicas e avance a faixa atual de forma circular.",
        "Estou precisando de uma classe Playlist com next_track voltando ao início.",
        "Me ajuda com uma classe Playlist que tenha navegação circular de faixas?",
        "Uma classe Playlist, por favor, com add_song, remove_song e next_track circular.",
    ],
}

_CLASS_NAMES = {
    "gen-oop-calculator": "Calculator",
    "gen-oop-stack": "Stack",
    "gen-oop-queue": "Queue",
    "gen-oop-bank_account": "BankAccount",
    "gen-oop-rectangle": "Rectangle",
    "gen-oop-circle": "Circle",
    "gen-oop-temperature_converter": "TemperatureConverter",
    "gen-oop-counter": "Counter",
    "gen-oop-linked_list": "LinkedList",
    "gen-oop-matrix": "Matrix",
    "gen-oop-shopping_cart": "ShoppingCart",
    "gen-oop-employee_payroll": "Employee",
    "gen-oop-cpf_validator": "CPFValidator",
    "gen-oop-vector2d": "Vector2D",
    "gen-oop-playlist": "Playlist",
}

_SUMMARIES = {
    "gen-oop-calculator": "soma, subtração, multiplicação, divisão e fatorial",
    "gen-oop-stack": "push, pop, peek e is_empty de uma pilha",
    "gen-oop-queue": "enqueue, dequeue e is_empty de uma fila",
    "gen-oop-bank_account": "depósito e saque com proteção contra saldo negativo",
    "gen-oop-rectangle": "área, perímetro e verificação de quadrado",
    "gen-oop-circle": "área e circunferência de um círculo",
    "gen-oop-temperature_converter": "conversão de Celsius para Fahrenheit e Kelvin",
    "gen-oop-counter": "increment, decrement e reset de um contador",
    "gen-oop-linked_list": "append, prepend e conversão para lista de uma lista encadeada",
    "gen-oop-matrix": "soma e transposição de uma matriz",
    "gen-oop-shopping_cart": "adicionar/remover item e total formatado em R$",
    "gen-oop-employee_payroll": "cálculo de salário líquido com bônus percentual",
    "gen-oop-cpf_validator": "validação de CPF pelo cálculo real dos dígitos verificadores",
    "gen-oop-vector2d": "soma e subtração via operadores, além de magnitude",
    "gen-oop-playlist": "navegação circular entre faixas de uma playlist",
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

        slots = {"class_name": _CLASS_NAMES[base_id], "summary": _SUMMARIES[base_id]}

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

    print(f"{count} variantes geradas para a categoria Classes/OOP.")


if __name__ == "__main__":
    main()
