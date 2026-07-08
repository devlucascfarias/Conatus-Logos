# Plano: revisão de hipótese após segunda falha real (Gap 4)

Status: **planejado, não iniciado**. Combinado em conversa em 2026-07-08, depois de revisar o
CoT do modelo pós-retreino nos casos Go e Sudoku (`docs/PLAN.md` seção 2:
`D-oop-expansion-round3-ood-tests`) e de consolidar o plano de recuperação de erro
(`D-error-recovery-expansion`). Créditos de Colab limitados até a renovação — geração/validação
local acontece agora, junto do resto, para um único retreino cobrir tudo.

## 1. Motivação

O pior sinal observado até agora: nos casos Go (checker falha por `MISSING_DEPENDENCY`) e
Sudoku (JSON malformado por escaping), o modelo repetiu a MESMA chamada várias vezes até
`MAX_STEPS_EXCEEDED`, sem nunca mudar de abordagem. O Gap 2a/2b (já gerados em
`D-error-recovery-expansion`) ensinam "observar o erro real → diagnosticar certo → corrigir" —
mas em TODOS esses exemplos o modelo acerta o diagnóstico já na segunda tentativa. Nenhum
exemplo ensina o que fazer quando a PRIMEIRA correção (baseada numa teoria plausível) também
falha de verdade — ou seja, nenhum ensina a reconhecer que a própria hipótese anterior estava
errada e que insistir nela não vai funcionar.

Isso é distinto de "diagnosticar certo": é a habilidade de usar uma segunda observação de
falha como evidência de que a primeira teoria era ruim, não só repetir a mesma lógica com
variação cosmética. Para um agente que vai virar CLI de uso prolongado, esse é o padrão que
mais importa — falhas raramente vêm limpas numa tentativa só.

## 2. Categoria e escala-alvo

| Categoria | Tarefas-base | Variantes de frase | Total aprox. |
|---|---|---|---|
| Revisão de hipótese após 2ª falha real | 4 | 4-5 | ~20-24 |

Escala pequena de propósito — é um comportamento único e bem específico, não uma categoria de
domínio. Mesmo porte dos Gaps 2a/2b da leva anterior.

## 3. Desenho de cada tarefa-base (3 rodadas reais de `checker`)

Diferente de todos os exemplos gerados até agora (que resolvem em no máximo 2 chamadas de
`checker`), aqui cada tarefa-base precisa de exatamente 3, todas reais:

1. **Rodada 1 (falha real)**: código com um bug genuíno. `checker` real falha (assert/erro real).
2. **Rodada 2 (falha real, teoria errada)**: o `<think>` propõe uma correção baseada numa
   hipótese PLAUSÍVEL mas ERRADA sobre a causa — não uma correção aleatória ou boba, uma que
   um desenvolvedor razoável tentaria primeiro. O código corrigido segundo essa teoria é
   escrito de verdade, o `checker` roda de verdade, e **falha de novo** (mesma categoria de
   problema, ainda incorreto) — nunca fabricado, a falha precisa ser genuína.
3. **Reconhecimento explícito**: o `<think>` seguinte precisa dizer, em tom neutro e factual,
   que a correção anterior NÃO resolveu o problema real — que a hipótese estava errada — antes
   de propor a causa correta. Vocabulário evitado: "a ferramenta"/"o ambiente" (não é disso que
   se trata aqui); vocabulário buscado: "minha correção anterior não resolveu, o problema
   real é outro".
4. **Rodada 3 (sucesso real)**: a causa correta, `checker` real passa.

Tarefas-base propostas (todas com bug real de lógica, não de sintaxe — para não sobrepor com
o Gap 2a):

- `sum_up_to(n)`: soma de 1 até n inclusive. Bug real: `range(1, n)` exclui n. Teoria errada
  plausível: "deve começar de 0" → `range(0, n)` ainda exclui n, soma continua errada (falha
  real de novo, resultado numericamente diferente mas ainda incorreto). Causa real: faltava
  `+1` no fim do range, não no início.
- `is_in_range(value, low, high)`: verificação de intervalo inclusivo. Bug real:
  `low < value < high` exclui os dois limites. Teoria errada plausível: "só o limite superior
  está errado" → corrige só um lado, o teste com o limite inferior ainda falha. Causa real:
  os dois lados precisavam de `<=`.
- `last_n_items(items, n)`: retornar os ÚLTIMOS n itens. Bug real: `items[:n]` retorna os
  PRIMEIROS n. Teoria errada plausível: "preciso inverter o slice" → `items[n:]` (pega do
  índice n até o fim, não os últimos n) — ainda errado para a maioria dos casos, falha real de
  novo. Causa real: `items[-n:]`.
- `dedup_keep_order(items)`: remover duplicatas preservando a ordem original. Bug real:
  `list(set(items))` remove duplicatas mas perde a ordem original. Teoria errada plausível:
  "o problema é a ordem não ser determinística, preciso ordenar" → `sorted(set(items))` —
  remove duplicatas de forma determinística, mas ainda não é a ordem ORIGINAL pedida (falha
  real de novo). Causa real: rastrear itens já vistos com um `set` auxiliar enquanto percorre
  a lista original, preservando a ordem em que cada item apareceu pela primeira vez.
  (Confirmado com execução real: `list(set(...))` e `sorted(set(...))` dão o mesmo resultado
  errado para os dados de teste escolhidos — ainda assim são teorias genuinamente diferentes,
  e ambas falham de verdade contra a ordem original esperada.)

Em todos os quatro, a "teoria errada" precisa ser uma correção real, escrita e executada de
verdade — nunca uma falha fabricada em texto.

## 4. Regras de conteúdo

- Mesmas regras já estabelecidas: identificadores em inglês, tom de dev profissional,
  `execution_performed: True`, nenhum `tool_result` fabricado.
- **O ponto central é o vocabulário de revisão**: o `<think>` da rodada 3 precisa demonstrar
  reconhecimento explícito de que a hipótese da rodada 2 estava errada, não só "vou tentar de
  novo" genérico. Ex.: "Minha correção anterior (mudar o início do range) não resolveu — o
  resultado ainda está errado, então a teoria estava errada. O problema real está no fim do
  range, que precisa incluir n."
- **Diferença deliberada em relação ao Gap 2a/2b**: lá o modelo acerta na 2ª tentativa; aqui,
  ele erra na 2ª e acerta na 3ª — é isso que ensina a "pensar depois de observar o erro" de
  forma repetida, não só uma vez.

## 5. AVISO CRÍTICO — mesmo cuidado de sempre

Mesmo risco documentado nas expansões anteriores: variantes de frase por tarefa-base
precisam ter estrutura genuinamente diferente, escritas à mão, reaproveitando fragmentos de
`<tool_call>`/`<tool_result>` já executados via `parse_segments`. Rodar
`scripts/check_phrasing_similarity.py` antes de aceitar.

## 6. Método de execução

1. Escrever cada tarefa-base com os TRÊS códigos reais (bug original, correção com teoria
   errada, correção certa) e um teste real que rode as três vezes via
   `ToolExecutorRegistry`/`SandboxContext` — as duas primeiras rodadas PRECISAM falhar de
   verdade (assert no script de geração), a terceira precisa passar de verdade.
2. Gerar variantes de frase reaproveitando os fragmentos já executados.
3. Rodar suite de testes + checagem de vazamento + checagem de similaridade.
4. Documentar como nova decisão em `docs/PLAN.md`.
5. Commit + push — SEM retreinar ainda (retreino é decisão separada, já em pauta pra logo
   depois de consolidar tudo).

## 7. Checklist antes de começar

- [x] Confirmar que os quatro bugs propostos (seção 3) realmente produzem falha real nas
      rodadas 1 e 2 — verificado com execução Python direta antes de escrever os exemplos
      completos (`sum_up_to`, `is_in_range`, `last_n_items` e `dedup_keep_order`, este último
      trocado no lugar de `count_vowels` depois que a teoria errada original acabou sendo uma
      correção completa, não parcial, no teste escolhido)
- [ ] Escrever as 4 tarefas-base, validando cada uma (3 rodadas reais) antes de passar para a
      próxima
- [ ] Gerar variantes de frase (4-5 por tarefa-base), com checagem de similaridade
- [ ] Rodar suite de testes + checagem de vazamento
- [ ] Atualizar `docs/PLAN.md`
- [ ] Commit + push
