# Plano: confabulação de correção inexistente (Gap 5)

Status: **planejado, não executado** (ver `D-confabulation-gap-plan` em `docs/PLAN.md`,
seção 2). Execução fica para depois da decisão de consolidação/retreino em pauta.

## 1. Motivação

Achado real durante o teste manual pós-treino da leva anterior (rodada de gaps 1-4, ver
`D-oop-expansion-round3-ood-tests` e a análise da conversa em 2026-07-08): pedi ao modelo
para "descobrir a causa raiz e corrigir" um teste supostamente falhando
(`gap2b_diagnosis_bug_in_source`). O código que o próprio modelo escreveu já estava correto
desde a primeira execução real — o `checker` passou de primeira, sem nenhuma segunda
tentativa, sem nenhuma edição. Mesmo assim, o `<final>` afirmou:

> "test_calc_utils.py estava testando o comportamento errado de add (a soma estava sendo
> subtraída). Corrigi só o teste."

Isso nunca aconteceu na trajetória — não existiu nenhuma subtração, nenhuma segunda escrita,
nenhuma correção. O modelo inventou uma narrativa de diagnóstico e correção plausível para
"preencher" a expectativa implícita no pedido do usuário ("encontre e corrija o bug"), mesmo
quando a evidência real (o próprio histórico de tool_calls) não sustenta essa história.

Isso é distinto de tudo que já foi coberto:
- **Fabricação de `tool_result`** (`contains_fabricated_tool_result`, já bloqueada pelo
  harness): o modelo inventa uma tag `<tool_result>` que nunca foi produzida por uma
  ferramenta real. Aqui as tags são todas reais — o problema está na PROSA do `<final>`,
  que descreve eventos que as tags reais não sustentam.
- **Revisão de hipótese** (Gap 4, `D-hypothesis-revision-expansion`): ensina a reconhecer
  quando uma correção real e genuína falhou de novo. Aqui não há nem falha real nem segunda
  tentativa — o problema é o modelo AGIR como se tivesse havido uma quando não houve.
- **Diagnóstico multi-arquivo** (Gap 2b, `D-error-recovery-expansion`): ensina a apontar o
  arquivo certo quando existe um bug real. Aqui a lição é o oposto: reconhecer quando NÃO
  existe bug real e dizer isso, em vez de inventar um.

Para um agente de uso prolongado em CLI, esse é um risco sério de confiança: um usuário que
lê "corrigi o bug de subtração" vai acreditar que havia um bug de subtração e que ele foi
corrigido — nenhuma das duas coisas é verdade. É uma mentira sutil, não detectável pelos
checks de fabricação de tool_result existentes, porque a mentira não está em nenhuma tag
estruturada — está no texto livre que o usuário lê.

## 2. Categoria e escala-alvo

| Categoria | Tarefas-base | Variantes de frase | Total aprox. |
|---|---|---|---|
| Confabulação de correção inexistente | 4 | 4-5 | ~20-24 |

Mesmo porte dos Gaps 2a/2b/3/4 — comportamento único e específico, não uma categoria de
domínio.

## 3. Desenho de cada tarefa-base

Diferente de tudo gerado até agora, aqui o pedido do usuário **afirma** (de forma honesta,
não maliciosa) que existe um bug — mas o código correto, escrito e executado de verdade,
passa de primeira. A lição não é sobre o código (que já está certo), é sobre o `<think>`/
`<final>`: reconhecer que a trajetória real não contém nenhuma correção e reportar isso com
precisão, em vez de inventar uma.

Formato de cada tarefa-base (1 única rodada real de `checker`, deliberadamente — o ponto é
que só existe UMA execução, não duas):

1. **Pedido do usuário**: menciona um bug específico e plausível (ex.: "acho que
   `is_palindrome` está contando espaços errado", "o `count_vowels` parece estar contando
   maiúscula duas vezes"), pedindo para encontrar e corrigir.
2. **`<think>` inicial**: reconhece a alegação do usuário, decide implementar a função
   corretamente do zero (ou verificar o código, se fornecido) e validar com teste real antes
   de assumir qualquer coisa sobre a causa.
3. **`write_file`** com a implementação correta (verificada previamente por execução Python
   direta — ver seção 7) + teste real cobrindo o caso que o usuário mencionou.
4. **`checker`** real, `compile_and_test`, passa de primeira — **sem segunda rodada**.
5. **`<think>` antes do final**: precisa reconhecer explicitamente que a validação passou na
   primeira execução, sem nenhuma correção ter sido de fato necessária — vocabulário
   buscado: "os testes passaram já na primeira execução real, não fiz nenhuma correção";
   vocabulário evitado: qualquer menção a uma causa, diagnóstico ou correção que não
   aparece nas chamadas de ferramenta reais da trajetória.
6. **`<final>`**: honesto e calibrado — relata que a implementação passou nos testes reais
   sem que nenhum bug tenha sido encontrado ou corrigido, e reconhece a possibilidade
   (sem afirmar) de que o problema relatado pelo usuário não esteja coberto pelos casos de
   teste usados. Nunca declara "corrigi X" quando X nunca foi alterado.

Tarefas-base propostas (todas com implementação correta verificada por execução Python
direta antes de escrever o exemplo completo — ver seção 7):

- `is_palindrome(s)`: usuário alega que espaços/maiúsculas quebram a checagem. Implementação
  normaliza caixa e espaços corretamente desde o início; teste cobre exatamente o caso
  citado (`"A ra ra"`) e passa de primeira.
- `count_vowels(s)`: usuário alega contagem duplicada em maiúsculas. Implementação usa
  `.lower()` corretamente; teste cobre string com maiúsculas e passa de primeira.
- `average(nums)`: usuário alega que lista vazia quebra o cálculo (`ZeroDivisionError`).
  Implementação já trata lista vazia (`if nums else 0.0`); teste cobre `average([])` e passa
  de primeira.
- `is_prime(n)`: usuário alega que `n=1` é classificado como primo. Implementação já trata
  `n < 2` como não-primo; teste cobre `is_prime(1)` e passa de primeira.

## 4. Regras de conteúdo

- Mesmas regras já estabelecidas: identificadores em inglês, tom de dev profissional,
  `execution_performed: True`, nenhum `tool_result` fabricado.
- **Regra nova, central desta categoria**: o texto do `<final>` (e do `<think>` que o
  precede) só pode descrever eventos que de fato aparecem nas chamadas de ferramenta reais
  da trajetória — nenhuma menção a uma causa raiz, diagnóstico ou correção que não tem
  `tool_call`/`tool_result` correspondente. Esta é a regra que a rodada de teste manual
  violou e que este gap existe para corrigir.
- Evitar overclaiming no sentido oposto também: o `<final>` não deve declarar com certeza
  "não existe bug nenhum" — a validação real só cobre os casos testados, não todo o espaço
  de entrada. O tom correto é "os testes reais que rodei passaram sem correção" mais um
  reconhecimento calibrado de que pode haver um caso não coberto.

## 5. AVISO CRÍTICO — mesmo cuidado de sempre

Mesmo risco documentado nas expansões anteriores: variantes de frase por tarefa-base
precisam ter estrutura genuinamente diferente, escritas à mão, reaproveitando fragmentos de
`<tool_call>`/`<tool_result>` já executados via `parse_segments`. Rodar
`scripts/check_phrasing_similarity.py` antes de aceitar.

## 6. Método de execução (quando for a hora de executar)

1. Verificar por execução Python direta que cada implementação proposta realmente passa nos
   testes propostos sem qualquer alteração (feito para as 4 acima — seção 7).
2. Escrever cada tarefa-base com a ÚNICA rodada real de `checker`, via
   `ToolExecutorRegistry`/`SandboxContext`.
3. Gerar variantes de frase reaproveitando os fragmentos já executados.
4. Rodar suite de testes + checagem de vazamento + checagem de similaridade.
5. Documentar como nova decisão em `docs/PLAN.md`.
6. Commit + push — sem retreinar ainda (decisão de retreino é separada e será tomada depois
   de consolidar tudo, incluindo os resultados da leva de gaps 1-4 já em treino).

## 7. Checklist antes de começar a execução

- [x] Confirmar que as quatro implementações propostas (seção 3) realmente passam de
      primeira, sem nenhum bug real — verificado com execução Python direta em 2026-07-08
      (`is_palindrome`, `count_vowels`, `average`, `is_prime`, todas ok).
- [ ] Escrever as 4 tarefas-base, validando cada uma (1 rodada real, passa de primeira)
      antes de passar para a próxima
- [ ] Gerar variantes de frase (4-5 por tarefa-base), com checagem de similaridade
- [ ] Rodar suite de testes + checagem de vazamento
- [ ] Atualizar `docs/PLAN.md`
- [ ] Commit + push
