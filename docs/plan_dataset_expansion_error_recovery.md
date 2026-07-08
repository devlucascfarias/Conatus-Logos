# Plano: expansão pequena — recuperação de falha de `checker`/infra e diagnóstico de causa raiz

Status: **executado em 2026-07-08** (mesmo dia do planejamento, ver `D-error-recovery-expansion`
em `docs/PLAN.md`, seção 2). 100 exemplos gerados (30+20+20+30), todos com execução real,
suite/similaridade/vazamento limpos. Este arquivo permanece como registro do plano original; o
resultado real está consolidado na decisão do PLAN.md.

## 1. Motivação

Dois gaps reais e reprodutíveis, achados em teste manual com o modelo real (não probe
sintético), nenhum deles envolvendo fabricação — mas ambos afetando a qualidade da
recuperação de erro:

**Gap 1 — `checker` falha por infraestrutura, modelo não sabe seguir em frente.**
Caso real: pedido de função em Go, código correto de primeira, mas `checker` retornou
`MISSING_DEPENDENCY: toolchain 'go' não disponível` nesta sessão. O modelo diagnosticou
CORRETAMENTE que era falha de ambiente, mas em vez de responder com o código já escrito e uma
nota honesta sobre a limitação (o padrão já ensinado para `web_search` falhando —
`D-tool-recovery-pilot`), ficou repetindo a mesma chamada de `checker` até
`MAX_STEPS_EXCEEDED`. Só ensinamos essa recuperação para `web_search`, nunca para o `checker`
em si falhando por motivo de ambiente/dependência.

**Gap 2 — diagnóstico de causa raiz em bug do próprio código, mal coberto em dois pontos
específicos (investigado, não é ausência total):**

- **2a — JSON malformado em código mais longo/complexo.** Caso real: pedido de validador de
  Sudoku, o código gerado teve uma sequência de escape inválida (`\` solto antes de espaço)
  quebrando o JSON do `tool_call`. O modelo repetiu a mesma chamada malformada 6 vezes,
  culpando "a ferramenta de escrita"/"o ambiente", nunca corrigindo o problema de verdade.
  Investigação: existem 10 exemplos de `task_type: invalid_call_then_correction` no dataset,
  mas todos são typos triviais de uma linha (`extra_bracket`, `missing_comma`, `bool_typo`...),
  **nenhum executado de verdade** (`execution_performed: False` nos 10), 400-700 caracteres de
  conteúdo. O código do Sudoku tinha ~15 linhas com loop aninhado e regex — complexidade nunca
  exercitada nesse task_type.
- **2b — diagnóstico de qual arquivo tem o bug real, quando são vários.** Caso real: pedido de
  jogo em pygame, o teste gerado esquecia de importar `main` de `pong_game.py` — bug real no
  arquivo de TESTE, não na implementação. O modelo resubmeteu repetidamente o
  `pong_game.py` (que nunca teve o problema), culpando "o ambiente"/"a ferramenta", sem nunca
  examinar o arquivo de teste. Investigação: existem 20 exemplos de `task_type: test_fails`,
  mas **nenhum tem mais de 1 passo** — todos ensinam só "reportar falha honesta quando o teste
  falha genuinamente", nenhum ensina diagnosticar QUAL dos múltiplos arquivos enviados ao
  `checker` tem o defeito real e corrigir só esse.

**Gap 3 — tool-choice equivocado em pergunta curta/direta (adicionado a esta leva por ser
igualmente barato e os créditos de Colab estarem escassos).** No probe #20 da suíte de 25
(`docs/PLAN.md`, testes pós-retreino), a pergunta "responda em uma frase: para que serve o
comando import?" levou o modelo a chamar uma ferramenta sem necessidade — provavelmente a
palavra "comando" puxou associação com shell/tool. `probe_direct_vs_tool_choice` já existe como
categoria, mas o dataset original tem poucos exemplos de perguntas CURTAS e diretas com
palavras que soam a ferramenta (comando, rodar, executar) sem exigir uma de verdade.

## 2. Categorias e escala-alvo

| Categoria | Tarefas-base | Variantes de frase | Total aprox. |
|---|---|---|---|
| Recuperação de falha de `checker` por infra (Gap 1) | 5-6 | 4-5 | ~25-30 |
| JSON malformado em código complexo, execução real (Gap 2a) | 4-5 | 4-5 | ~20-25 |
| Diagnóstico de arquivo com bug real entre vários (Gap 2b) | 4-5 | 4-5 | ~20-25 |
| Tool-choice em pergunta curta com palavra-armadilha (Gap 3) | 5-6 | 4-5 | ~20-25 |
| **Total** | **~19-22** | — | **~85-105** |

Escala bem menor que a expansão de ontem (85 tarefas-base/765 exemplos) — de propósito: são
quatro comportamentos estreitos e específicos, não categorias inteiras de domínio. Referência de
proporção: os pilotos de recuperação de `web_search` (`D-tool-recovery-pilot`/
`D-tool-success-pilot`) foram 5 tarefas-base × 4-5 variantes ≈ 25-40 exemplos cada, e
resolveram um gap do mesmo porte.

## 3. Regras de conteúdo

- **Todo `tool_result` vem de execução real** (`ToolExecutorRegistry`/`SandboxContext`), sem
  exceção — inclusive para o Gap 2a, que na versão antiga (`invalid_call_then_correction`)
  nunca foi executado de verdade. Isso corrige uma dívida técnica do dataset original, não só
  cobre o gap novo.
- **Gap 1 — mecanismo real e determinístico confirmado**: Go está instalado nesta máquina
  (não reproduz a falha do Colab, que é específica daquela sessão), mas o `checker` tem
  `UNSUPPORTED_LANGUAGE` real (`src/checker/core.py`) para qualquer linguagem aceita pelo
  schema (`javascript`, `typescript`) sem backend implementado (`src/checker/backends/` só tem
  Python e Go). Chamar `checker` com `language: "javascript"` produz esse erro de verdade, sem
  fabricar nada — e é mais estável como padrão de treino do que depender de uma ausência de
  toolchain específica do Colab, que pode mudar. Ensina o padrão geral "o `checker` não suporta
  isso neste ambiente, não é bug no código" — mesma lição, mecanismo mais robusto.
- **Gap 2a — o bug precisa surgir organicamente do código real**, não ser um typo injetado
  artificialmente como nos 10 exemplos antigos. Vou escrever código complexo o bastante
  (regex, múltiplas linhas, strings com aspas/backslash) onde um erro de escaping é plausível,
  gerar a falha real via `ToolExecutorRegistry`, e então corrigir de verdade — nunca fabricar o
  erro nem a correção.
- **Gap 2b — dois arquivos reais, bug real em só um deles.** Preciso escrever cenários onde
  `write_file` cria dois arquivos (implementação + teste, ou dois módulos), um dos dois tem um
  defeito genuíno, o `checker` real aponta o erro, e a trajetória ensina a examinar o traceback
  para identificar QUAL arquivo/linha está envolvido antes de decidir o que corrigir — nunca
  assumir que é sempre a implementação.
- **Tom e identificadores**: mesmas regras da expansão de ontem — identificadores de código em
  inglês, tom de "ferramenta de dev profissional", sem gíria pesada.
- **Vocabulário de recuperação**: a prosa de `<think>`/`<final>` precisa ser neutra e factual
  ("o JSON da minha chamada estava malformado, vou corrigir" / "o bug está no arquivo de
  teste, não na implementação — vou corrigir o teste"), evitando a linguagem de
  "culpar a ferramenta/ambiente" que causou o comportamento problemático nos dois casos reais.
- **Gap 3 — perguntas curtas com palavra-armadilha, sem ferramenta nenhuma.** Cada exemplo é
  `task_type: direct_answer` (`num_steps: 0`, sem `tool_call`), igual ao padrão já existente de
  `probe_direct_vs_tool_choice`. O pedido usa palavras que soam a ação de ferramenta ("comando",
  "rodar", "executar", "arquivo") mas pede só explicação conceitual — o `<think>` precisa
  reconhecer explicitamente que, apesar da palavra, não há nada para escrever/validar/rodar de
  verdade.

## 4. AVISO CRÍTICO — mesmo cuidado de sempre com repetição de template

Mesmo risco já documentado duas vezes (`D-generalization-gap`, `D-tool-pilots-phrasing`) e
prevenido na expansão de ontem (`plan_dataset_expansion_oop_shell.md`): variantes de frase
precisam ter estrutura genuinamente diferente, não só troca de sinônimo — escrever à mão,
reaproveitar fragmentos de `<tool_call>`/`<tool_result>` já executados via `parse_segments`
(não reexecutar por variante), e rodar `scripts/check_phrasing_similarity.py` antes de aceitar
o lote.

## 5. Método de execução

1. **Antes de escrever qualquer coisa**: confirmar como simular a falha de infra do Gap 1 de
   forma real e determinística (checar disponibilidade de toolchains localmente).
2. Escrever cada tarefa-base com código real, executar via `ToolExecutorRegistry`/
   `SandboxContext` — para os Gaps 2a/2b, isso inclui deixar a falha REAL acontecer primeiro
   (não simular o erro em texto) e só then corrigir de verdade.
3. Gerar variantes de frase reaproveitando os fragmentos já executados (`parse_segments`),
   como em todos os scripts de ontem.
4. Sanitizar `stderr`/`stdout` de ruído local antes de aceitar (mesmo cuidado de sempre — já
   achamos um leak real em `aug-testfail-absolute.json`, do dataset ORIGINAL, não de hoje,
   durante a investigação deste plano — worth corrigir separadamente, fora deste plano).
5. Rodar suite de testes completa + checagem de vazamento + checagem de similaridade antes de
   aceitar o lote.
6. Documentar como nova(s) decisão(ões) em `docs/PLAN.md`.
7. Só depois: retreinar.

## 6. Nota à parte (fora do escopo deste plano)

Durante a investigação, achei um vazamento de caminho Windows real em
`data/train/aug-testfail-absolute.json` (dataset original, não gerado hoje) — o stderr do
aviso de depreciação do `pytest-asyncio` vazou um caminho local
(`C:\Users\Pichau\AppData\Local\Programs\Python\...`) sem sanitização. Não é escopo desta
expansão, mas deveria ser corrigido em algum momento (grep no dataset inteiro por esse padrão,
não só nos exemplos novos).

## 7. Checklist antes de começar

- [x] Confirmar mecanismo real e determinístico para o Gap 1 — `checker` com
      `language: "javascript"` produz `UNSUPPORTED_LANGUAGE` de verdade (ver seção 3)
- [ ] Escrever as ~14-16 tarefas-base, uma categoria por vez, validando cada uma (execução
      real, incluindo a falha real antes da correção) antes de passar para a próxima
- [ ] Gerar variantes de frase por tarefa-base (4-5 cada), com checagem de similaridade
- [ ] Rodar suite de testes + checagem de vazamento (incluindo grep pelo leak achado na nota
      acima, se decidirmos corrigir junto)
- [ ] Atualizar `docs/PLAN.md`
- [ ] Commit + push
- [ ] Só então: retreinar no Colab
