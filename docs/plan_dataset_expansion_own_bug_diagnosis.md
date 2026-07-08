# Plano: diagnóstico correto de bug autoral + edição real na retentativa (Gap 6)

Status: **executado em 2026-07-08** (ver `D-own-bug-diagnosis-expansion` em `docs/PLAN.md`,
seção 2). 20 exemplos gerados (4 base + 16 variantes), na mesma leva do Gap 5 — retreino ainda
pendente de decisão.

## 1. Motivação

Achado real no reteste de `D-repetition-loop-confirmed` (2026-07-08, depois de confirmar que
`repetition_penalty=1.15` resolveu a degeneração de repetição de token): o mesmo prompt
(`ood_combo_todo_exception`) parou de entrar em loop de token, mas revelou uma falha diferente,
em duas partes:

1. **Diagnóstico de causa raiz errado num bug de autoria própria**: o modelo escreveu
   `TodoList.add_task` chamando `Task(name)`, mas nunca definiu a classe `Task` em lugar
   nenhum. `NameError: name 'Task' is not defined` — real. O `<think>` seguinte concluiu
   "ele esqueceu de importar a classe Task da própria biblioteca" — um diagnóstico sem
   sentido (não existe `Task` para importar; ela precisa ser DEFINIDA, não importada).
2. **Retentativa idêntica, sem correção real**: apesar do `<think>` dizer "vou corrigir só o
   teste", o `<tool_call>` de `checker` seguinte tinha exatamente o mesmo conteúdo (byte a
   byte) do anterior — nenhuma edição de verdade foi aplicada. O harness bloqueou isso via
   deduplicação de chamadas (`seen_calls` → `MAX_STEPS_EXCEEDED` como erro de ferramenta), o
   que evitou um loop infinito, mas o modelo levou 2 tentativas idênticas bloqueadas antes de
   desistir.

O lado positivo, que não motivaria por si só uma correção: o `<final>` foi honesto ("não
consegui validar esse código... a ferramenta retornou 'Max steps exceeded'"), sem fabricação.
Mas a combinação "diagnóstico errado + retentativa que não muda nada" é um padrão que vale
ensinar explicitamente, distinto de tudo já coberto:

- **Gap 2b** (`multi_file_root_cause_diagnosis`): o bug está num arquivo PRÉ-EXISTENTE dado
  pelo cenário (normalmente o teste), não em código que o próprio modelo acabou de escrever
  do zero. Aqui o bug nasce na primeira implementação do próprio modelo.
- **Gap 4** (`hypothesis_revision_after_failure`): a segunda tentativa É uma correção real,
  só que baseada numa teoria errada — o código muda de verdade, só que continua incorreto.
  Aqui a segunda tentativa não muda NADA — não é uma teoria errada aplicada, é ausência de
  aplicação.
- **Gap 5** (`docs/plan_dataset_expansion_confabulation_gap.md`): o problema lá é inventar uma
  narrativa de correção que nunca aconteceu, quando não havia bug nenhum. Aqui o bug é real
  desde o início — o problema é diagnosticar errado E não aplicar a correção prometida.

Categoria específica de bug escolhida — `NameError` por referência a um nome nunca definido
(classe auxiliar esquecida, função auxiliar esquecida, ou import esquecido de uma biblioteca
padrão) — porque é o padrão exato observado no caso real, e é um erro comum e genuíno de
quem escreve código rápido: referenciar algo que "deveria existir" antes de defini-lo.

## 2. Categoria e escala-alvo

| Categoria | Tarefas-base | Variantes de frase | Total aprox. |
|---|---|---|---|
| Diagnóstico de bug autoral + edição real na retentativa | 4 | 4-5 | ~20-24 |

Mesmo porte dos gaps anteriores (2a/2b/3/4/5) — comportamento específico, não categoria de
domínio.

## 3. Desenho de cada tarefa-base

Diferente de Gap 2b (bug em arquivo pré-existente) e de Gap 4 (segunda tentativa real mas com
teoria errada), aqui a trajetória é:

1. **`write_file`** com a implementação inicial contendo o `NameError` real (referência a um
   nome nunca definido — classe auxiliar, função auxiliar ou import de biblioteca padrão).
2. **`checker`** real, `compile_and_test`, falha de verdade com `NameError` no traceback.
3. **`<think>` de diagnóstico**: precisa nomear EXATAMENTE o que falta — "`X` nunca foi
   definido/importado neste arquivo" — sem inventar uma causa alternativa (import faltando
   quando na verdade é definição faltando, ou vice-versa). Vocabulário buscado: "`Task` nunca
   foi definida em lugar nenhum — preciso definir essa classe, não é um problema de import";
   vocabulário evitado: qualquer causa que não corresponda ao traceback real.
4. **`write_file`/`checker`** de correção: o conteúdo PRECISA diferir do anterior de forma
   real (adiciona a definição/import que faltava) — nunca um reenvio idêntico. Verificado por
   construção: o script de geração compara os dois payloads e falha se forem iguais.
5. **`checker`** real passa. `<final>` honesto reportando a causa real e a correção aplicada.

Tarefas-base propostas (todas verificadas por execução Python direta — ver seção 7):

- `TodoList`/`Task`: `add_task` referencia `Task(name)` nunca definida. Correção: definir
  `class Task` com `name`/`completed`.
- `EventBus`/`Subscriber`: `subscribe` referencia `Subscriber(callback)` nunca definida.
  Correção: definir `class Subscriber` guardando o callback.
- `parse_duration`/`_parse_unit`: referencia uma função auxiliar `_parse_unit(unit)` nunca
  definida. Correção: definir o helper com o mapeamento de unidades.
- `to_cents`: usa `Decimal` sem `from decimal import Decimal`. Correção: adicionar o import
  faltando — variante deliberada (import de biblioteca padrão esquecido, não uma classe
  auxiliar não definida), para o gap não virar "sempre é uma classe faltando".

## 4. Regras de conteúdo

- Mesmas regras já estabelecidas: identificadores em inglês, tom de dev profissional,
  `execution_performed: True`, nenhum `tool_result` fabricado.
- **Regra nova, central desta categoria**: o `<think>` de diagnóstico precisa citar o nome
  exato que falta e classificar corretamente se é "falta definir" ou "falta importar" —
  nunca confundir os dois (o erro observado no caso real foi exatamente essa confusão).
- **Segunda regra nova**: o payload do `write_file`/`checker` de correção precisa ser
  verificavelmente diferente do payload da tentativa que falhou — checado por comparação de
  string no próprio script de geração antes de aceitar a tarefa-base.

## 5. AVISO CRÍTICO — mesmo cuidado de sempre

Mesmo risco documentado nas expansões anteriores: variantes de frase por tarefa-base
precisam ter estrutura genuinamente diferente, escritas à mão, reaproveitando fragmentos de
`<tool_call>`/`<tool_result>` já executados via `parse_segments`. Rodar
`scripts/check_phrasing_similarity.py` antes de aceitar.

## 6. Método de execução (quando for a hora de executar)

1. Escrever cada tarefa-base com as DUAS rodadas reais de `checker` (falha real por
   `NameError`, sucesso real após a correção), via `ToolExecutorRegistry`/`SandboxContext`.
   Assert no próprio script de geração: payload de correção ≠ payload da falha.
2. Gerar variantes de frase reaproveitando os fragmentos já executados.
3. Rodar suite de testes + checagem de vazamento + checagem de similaridade.
4. Documentar como nova decisão em `docs/PLAN.md`.
5. Commit + push — sem retreinar ainda (mesma decisão de consolidação do Gap 5).

## 7. Checklist antes de começar a execução

- [x] Confirmar que os quatro `NameError` propostos (seção 3) são reais e que as correções
      propostas realmente resolvem — verificado com execução Python direta em 2026-07-08
      (`TodoList`/`Task`, `EventBus`/`Subscriber`, `parse_duration`/`_parse_unit`,
      `to_cents`/`Decimal`, todos com falha real confirmada e correção real confirmada).
- [ ] Escrever as 4 tarefas-base, validando cada uma (2 rodadas reais + payload de correção
      verificavelmente diferente do payload da falha) antes de passar para a próxima
- [ ] Gerar variantes de frase (4-5 por tarefa-base), com checagem de similaridade
- [ ] Rodar suite de testes + checagem de vazamento
- [ ] Atualizar `docs/PLAN.md`
- [ ] Commit + push
