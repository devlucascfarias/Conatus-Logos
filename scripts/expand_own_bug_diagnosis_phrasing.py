#!/usr/bin/env python
"""Expansão de diversidade de frase para o Gap 6 (diagnóstico de bug autoral + edição real na
retentativa) (docs/plan_dataset_expansion_own_bug_diagnosis.md). Mesmo método das expansões
anteriores: variantes por tarefa-base reaproveitando os fragmentos `<tool_call>`/
`<tool_result>` já executados de verdade.

Cuidado deliberado (achado em D-hypothesis-revision-expansion): cada variante usa uma frase
de diagnóstico (`think2`) GENUINAMENTE distinta por tarefa-base, não um slot único
reaproveitado entre as 4 variantes — diferente do que aconteceu em
`expand_multifile_diagnosis_phrasing.py` (que reaproveita a MESMA `_DIAGNOSES[base_id]` nas 4
variantes), esta expansão evita repetir esse padrão.

Forma fixa de cada trajetória-base (10 segmentos):
    think, tool_call, tool_result,   (write_file, código com NameError real)
    think, tool_call, tool_result,   (checker — falha real, NameError)
    think, tool_call, tool_result,   (checker — correção real, passa)
    final

Uso:
    python scripts/expand_own_bug_diagnosis_phrasing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.parsers import parse_segments  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_EXPECTED_SHAPE = (
    "think", "tool_call", "tool_result",
    "think", "tool_call", "tool_result",
    "think", "tool_call", "tool_result",
    "final",
)

_REQUESTS = {
    "gen-ownbug-todo-list": [
        "Crie uma classe TodoList com add_task e complete_task, levantando TaskNotFoundError se a tarefa não existir.",
        "Pode implementar TodoList com add_task/complete_task? complete_task deve levantar TaskNotFoundError pra tarefa inexistente.",
        "Preciso de uma TodoList com add_task e complete_task, com uma exceção própria pra tarefa não encontrada.",
        "Implementa TodoList: add_task adiciona, complete_task marca como concluída ou levanta TaskNotFoundError.",
    ],
    "gen-ownbug-event-bus": [
        "Implemente EventBus com subscribe e publish, chamando todos os callbacks inscritos.",
        "Pode implementar um EventBus com subscribe/publish, onde publish notifica todos os inscritos?",
        "Preciso de um EventBus simples: subscribe registra um callback, publish dispara pra todos.",
        "Implementa EventBus: subscribe(callback) e publish(event) chamando cada callback registrado.",
    ],
    "gen-ownbug-parse-duration": [
        "Escreva parse_duration convertendo textos tipo '5s'/'2m'/'1h' para segundos.",
        "Pode implementar parse_duration que aceite segundos, minutos e horas (5s, 2m, 1h)?",
        "Preciso converter durações em texto (5s, 2m, 1h) pra segundos com parse_duration.",
        "Implementa parse_duration cobrindo as unidades s, m e h.",
    ],
    "gen-ownbug-to-cents": [
        "Escreva to_cents convertendo um valor monetário em texto pra centavos, usando Decimal.",
        "Pode implementar to_cents que converta '1.50' em 150 centavos, com Decimal pra precisão?",
        "Preciso de to_cents pra valores monetários em texto virarem centavos inteiros, via Decimal.",
        "Implementa to_cents usando Decimal pra evitar erro de ponto flutuante.",
    ],
}

_SUMMARIES = {
    "gen-ownbug-todo-list": "TodoList",
    "gen-ownbug-event-bus": "EventBus",
    "gen-ownbug-parse-duration": "parse_duration",
    "gen-ownbug-to-cents": "to_cents",
}

# 4 aberturas (think1/think_before_first_checker) genuinamente distintas por tarefa-base.
_OPENINGS_BY_TASK = {
    "gen-ownbug-todo-list": [
        dict(
            think1="A estrutura mais natural aqui é representar cada tarefa da lista como um objeto Task próprio, com nome e status — vou seguir esse desenho em TodoList.",
            think_before_first_checker="Antes de responder, vou cobrir sucesso e o caso de erro com um teste real.",
        ),
        dict(
            think1="Vou escrever TodoList agora, representando cada tarefa como um objeto Task com nome e status.",
            think_before_first_checker="Com a implementação pronta, vou escrever o teste e rodar a validação.",
        ),
        dict(
            think1="Vou implementar TodoList conforme pedido, guardando cada tarefa num objeto Task.",
            think_before_first_checker="Agora escrevo o teste correspondente e valido tudo de uma vez.",
        ),
        dict(
            think1="Vou codificar TodoList usando um objeto Task pra representar cada tarefa da lista.",
            think_before_first_checker="Vou escrever o teste em seguida e rodar tudo junto pelo checker.",
        ),
    ],
    "gen-ownbug-event-bus": [
        dict(
            think1="Faz sentido encapsular cada inscrição num objeto Subscriber, pra deixar espaço pra extensões futuras (filtro de evento, por exemplo) — vou seguir esse desenho.",
            think_before_first_checker="Antes de responder, vou validar publish/subscribe com um teste real.",
        ),
        dict(
            think1="Vou escrever EventBus agora, encapsulando cada callback registrado num objeto Subscriber.",
            think_before_first_checker="Com a implementação pronta, vou escrever o teste e rodar a validação.",
        ),
        dict(
            think1="Vou implementar EventBus conforme pedido, usando Subscriber pra representar cada inscrição.",
            think_before_first_checker="Agora escrevo o teste correspondente e valido tudo de uma vez.",
        ),
        dict(
            think1="Vou codificar EventBus guardando cada callback num Subscriber.",
            think_before_first_checker="Vou escrever o teste em seguida e rodar tudo junto pelo checker.",
        ),
    ],
    "gen-ownbug-parse-duration": [
        dict(
            think1="A abordagem mais limpa é separar o número da unidade e delegar a conversão de unidade pra uma função auxiliar isolada — vou seguir esse desenho.",
            think_before_first_checker="Antes de responder, vou validar as três unidades (s/m/h) com um teste real.",
        ),
        dict(
            think1="Vou escrever parse_duration agora, separando o número da unidade e delegando a conversão pra um helper.",
            think_before_first_checker="Com a implementação pronta, vou escrever o teste e rodar a validação.",
        ),
        dict(
            think1="Vou implementar parse_duration conforme pedido, usando uma função auxiliar pra traduzir a unidade em segundos.",
            think_before_first_checker="Agora escrevo o teste correspondente e valido tudo de uma vez.",
        ),
        dict(
            think1="Vou codificar parse_duration extraindo o número e delegando a unidade pra uma função auxiliar.",
            think_before_first_checker="Vou escrever o teste em seguida e rodar tudo junto pelo checker.",
        ),
    ],
    "gen-ownbug-to-cents": [
        dict(
            think1="Valor monetário com float é receita pra erro de arredondamento, então vou usar Decimal na conversão desde o início.",
            think_before_first_checker="Antes de responder, vou validar a conversão com um teste real.",
        ),
        dict(
            think1="Vou escrever to_cents agora, usando Decimal pra evitar imprecisão de ponto flutuante em valor monetário.",
            think_before_first_checker="Com a implementação pronta, vou escrever o teste e rodar a validação.",
        ),
        dict(
            think1="Vou implementar to_cents conforme pedido, convertendo com Decimal pra manter precisão.",
            think_before_first_checker="Agora escrevo o teste correspondente e valido tudo de uma vez.",
        ),
        dict(
            think1="Vou codificar to_cents usando Decimal pra lidar com valor monetário sem erro de float.",
            think_before_first_checker="Vou escrever o teste em seguida e rodar tudo junto pelo checker.",
        ),
    ],
}

# 4 diagnósticos (think2) genuinamente distintos por tarefa-base — não um slot compartilhado.
_DIAGNOSES_BY_TASK = {
    "gen-ownbug-todo-list": [
        "O checker aponta 'NameError: name Task is not defined'. Task não existe em lugar nenhum do arquivo — como não é uma classe de nenhuma biblioteca, o problema não é import faltando, é DEFINIÇÃO faltando. Vou escrevê-la antes de TodoList.",
        "'NameError: name Task is not defined' — não escrevi a classe Task em lugar nenhum. Não adianta importar nada, porque Task não é de nenhuma biblioteca; falta eu DEFINIR essa classe no próprio arquivo.",
        "O traceback aponta 'Task is not defined'. Task nunca existiu neste código — não é um import faltando, é uma definição faltando. Vou escrever a classe Task antes de usá-la.",
        "Falhou com 'NameError: name Task is not defined'. Referenciei Task sem nunca ter escrito essa classe — a correção é DEFINIR Task, não importar nada.",
    ],
    "gen-ownbug-event-bus": [
        "O checker aponta 'NameError: name Subscriber is not defined'. Essa classe não existe em nenhum lugar do arquivo — não é um problema de import, é DEFINIÇÃO faltando. Vou escrevê-la guardando o callback recebido.",
        "'NameError: name Subscriber is not defined' — não escrevi essa classe em lugar nenhum. Subscriber não vem de nenhuma biblioteca pra importar; falta eu DEFINIR ela.",
        "O traceback aponta 'Subscriber is not defined'. Essa classe nunca existiu no código — a correção é escrever a definição de Subscriber, não um import.",
        "Falhou com 'NameError: name Subscriber is not defined'. Referenciei Subscriber sem defini-la — vou criar a classe agora, guardando o callback recebido.",
    ],
    "gen-ownbug-parse-duration": [
        "O checker aponta 'NameError: name _parse_unit is not defined'. Essa função auxiliar nunca chegou a ser escrita — não é import faltando (não vem de biblioteca nenhuma), é DEFINIÇÃO faltando. Vou escrevê-la com o mapeamento de unidades.",
        "'NameError: name _parse_unit is not defined' — essa função auxiliar nunca foi escrita. Não tem import que resolva isso; preciso DEFINIR _parse_unit com o mapeamento s/m/h.",
        "O traceback mostra '_parse_unit is not defined'. Chamei um helper que nunca cheguei a implementar — vou escrever essa função antes de usá-la em parse_duration.",
        "Falhou com 'NameError: name _parse_unit is not defined'. Esqueci de escrever a função auxiliar de conversão de unidade — vou definir _parse_unit agora.",
    ],
    "gen-ownbug-to-cents": [
        "O checker aponta 'NameError: name Decimal is not defined'. Desta vez é o oposto do padrão comum: Decimal já existe na biblioteca padrão, não precisa ser definida — só faltou importar de decimal. Vou adicionar o import.",
        "'NameError: name Decimal is not defined' — Decimal é da biblioteca padrão (módulo decimal), não precisa ser escrita, só importada. Esqueci o 'from decimal import Decimal'.",
        "O traceback aponta 'Decimal is not defined'. Diferente de uma classe própria faltando, aqui Decimal já existe em decimal — só faltou importar. Vou adicionar o import.",
        "Falhou com 'NameError: name Decimal is not defined'. Não preciso definir nada — Decimal vem do módulo padrão decimal, só esqueci de importar. Vou corrigir o import.",
    ],
}

_FINALS_BY_TASK = {
    "gen-ownbug-todo-list": [
        "TodoList com add_task e complete_task ficou pronta. A primeira validação apontou 'NameError: name Task is not defined' — eu usava Task(name) sem tê-la escrito em lugar nenhum; adicionei a classe (name e completed) e agora compila e o teste passa.",
        "TodoList está pronta. O primeiro teste falhou porque eu referenciava Task sem tê-la definido — corrigi definindo a classe Task, e agora está tudo validado.",
        "Depois de identificar que Task nunca tinha sido definida (não era um import faltando), corrigi escrevendo a classe. A validação confirma que TodoList compila e o teste passa.",
        "TodoList implementada. O bug encontrado foi Task nunca definida — corrigi criando a classe com name e completed. Validação confirma que compila e o teste passa.",
    ],
    "gen-ownbug-event-bus": [
        "EventBus com subscribe e publish ficou pronta. A primeira validação apontou 'NameError: name Subscriber is not defined' — eu usava Subscriber(callback) sem tê-la escrito em lugar nenhum; adicionei a classe guardando o callback e agora compila e o teste passa.",
        "EventBus está pronta. O primeiro teste falhou porque eu referenciava Subscriber sem tê-la definido — corrigi definindo a classe, e agora está tudo validado.",
        "Depois de identificar que Subscriber nunca tinha sido definida, corrigi escrevendo a classe guardando o callback. A validação confirma que EventBus compila e o teste passa.",
        "EventBus implementada. O bug encontrado foi Subscriber nunca definida — corrigi criando a classe. Validação confirma que compila e o teste passa.",
    ],
    "gen-ownbug-parse-duration": [
        "parse_duration(text) ficou pronta. A primeira validação apontou 'NameError: name _parse_unit is not defined' — eu chamava essa função auxiliar sem tê-la escrito; adicionei _parse_unit com o mapeamento s/m/h e agora compila e o teste passa.",
        "parse_duration está pronta. O primeiro teste falhou porque eu chamava _parse_unit sem tê-la escrito — corrigi definindo a função, e agora está tudo validado.",
        "Depois de identificar que _parse_unit nunca tinha sido escrita, corrigi definindo o mapeamento de unidades. A validação confirma que parse_duration compila e o teste passa.",
        "parse_duration implementada. O bug encontrado foi _parse_unit nunca definida — corrigi escrevendo o helper. Validação confirma que compila e o teste passa.",
    ],
    "gen-ownbug-to-cents": [
        "to_cents(value) com Decimal ficou pronta. A primeira validação apontou 'NameError: name Decimal is not defined' — eu tinha esquecido o import; adicionei 'from decimal import Decimal' e agora compila e o teste passa.",
        "to_cents está pronta. O primeiro teste falhou porque eu esqueci de importar Decimal — corrigi adicionando o import, e agora está tudo validado.",
        "Depois de identificar que faltava importar Decimal (não definir nada, já que é da biblioteca padrão), corrigi o import. A validação confirma que to_cents compila e o teste passa.",
        "to_cents implementada. O bug encontrado foi o import de Decimal faltando — corrigi adicionando 'from decimal import Decimal'. Validação confirma que compila e o teste passa.",
    ],
}


def _rebuild_raw_text(original_raw_text: str, think0: str, think3: str, think6: str, final9: str) -> str:
    segs = parse_segments(original_raw_text)
    shape = tuple(s.kind for s in segs)
    if shape != _EXPECTED_SHAPE:
        raise AssertionError(f"forma inesperada de trajetória: {shape}")

    return (
        f"<think>{think0}</think>"
        + original_raw_text[segs[1].start : segs[2].end]
        + f"<think>{think3}</think>"
        + original_raw_text[segs[4].start : segs[5].end]
        + f"<think>{think6}</think>"
        + original_raw_text[segs[7].start : segs[8].end]
        + f"<final>{final9}</final>"
    )


def main() -> None:
    count = 0
    for base_id, request_variants in _REQUESTS.items():
        path = OUT_DIR / f"{base_id}.json"
        example = json.loads(path.read_text(encoding="utf-8"))
        original_raw_text = example["trajectory"]["raw_text"]

        openings = _OPENINGS_BY_TASK[base_id]
        diagnoses = _DIAGNOSES_BY_TASK[base_id]
        finals = _FINALS_BY_TASK[base_id]

        for k, user_request in enumerate(request_variants, start=1):
            idx = k - 1
            opening = openings[idx]

            think0 = opening["think1"]
            think3 = opening["think_before_first_checker"]
            think6 = diagnoses[idx]
            final9 = finals[idx]

            raw_text = _rebuild_raw_text(original_raw_text, think0, think3, think6, final9)

            new_id = f"{base_id}-var{k}"
            new_example = json.loads(json.dumps(example))
            new_example["metadata"]["id"] = new_id
            new_example["metadata"]["source"] = f"phrasing_variant_of:{base_id}"
            new_example["trajectory"]["user_request"] = user_request
            new_example["trajectory"]["raw_text"] = raw_text

            out_path = OUT_DIR / f"{new_id}.json"
            out_path.write_text(json.dumps(new_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"{count} variantes geradas para o Gap 6 (diagnóstico de bug autoral).")


if __name__ == "__main__":
    main()
