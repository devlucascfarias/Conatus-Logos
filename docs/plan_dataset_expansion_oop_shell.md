# Plano: expansão do dataset — OOP, erros, padrões, módulos realistas, busca-em-inglês, shell

Status: **planejado, não iniciado**. Combinado em conversa em 2026-07-08, execução prevista para
depois (não fazer no mesmo dia do planejamento). Este arquivo é o registro de intenção — ao
executar, seguir isto e depois consolidar o resultado real em `docs/PLAN.md` (decisão nova,
como já foi feito para D-tool-recovery-pilot/D-tool-success-pilot/D-tool-pilots-phrasing).

## 1. Motivação

Teste real no adapter atual: pedido de "uma classe Python com funções de soma, subtração,
multiplicação, divisão e fatorial" resultou em funções soltas (nenhuma `class`), **sem chamar
`checker`** antes do `<final>`, e o `<final>` afirmou falsamente "criei a classe". O dataset
atual (1.629 exemplos) tem cobertura forte de algoritmos isolados (função pura + teste), mas
quase nenhuma de:
- Estruturas orientadas a objeto (classes com múltiplos métodos, estado interno)
- Tratamento de erro explícito (`try/except`, `raise`)
- Padrões de Python mais avançados (decorator, generator, context manager)
- Módulos com formato mais "realista" de tarefa de engenharia (não só uma função pura)
- `web_search` bem-sucedida retornando conteúdo em INGLÊS, ainda assim produzindo
  código real e resposta em português (viu-se falhar 2x: pedido de FFT e de jogo em pygame —
  o modelo respondeu em inglês e nunca chamou `write_file`/`checker`)

## 2. Categorias e escala-alvo

| Categoria | Tarefas-base | Variantes de frase | Total aprox. |
|---|---|---|---|
| Classes/OOP | 15 | 8-9 | ~120-135 |
| Tratamento de erro | 15 | 8-9 | ~120-135 |
| Padrões Python (decorator/generator/context manager) | 15 | 8-9 | ~120-135 |
| Módulos realistas (inventário, validação, utilitários) | 15 | 8-9 | ~120-135 |
| Busca bem-sucedida em inglês → código real em português | 15 | 8-9 | ~120-135 |
| Shell real (allowlist: `pytest`, `ruff`, `git status/diff/log`, `ls`, `cat`, `grep`) | 10 | 8-9 | ~80-90 |
| **Total** | **85** | — | **~700-750** |

Referência de proporção: a expansão por paráfrase que já funcionou (D-generalization-gap
resolvido) foi ~179 tarefas-base × ~6 variantes ≈ 1.036 exemplos, cobrindo categorias já
existentes. Esta rodada é do mesmo porte relativo, mas com tarefas-base **novas** (não
reaproveitadas), o que exige mais cuidado de autoria — ver seção 4.

## 3. Regras de conteúdo (decididas em conversa)

- **Identificadores de código sempre em inglês** (nomes de função/classe/variável) — boa
  prática, independente do idioma da conversa. Só usar português no código se o usuário pedir
  isso explicitamente no `user_request` (não é o caso padrão aqui).
- **Tom de "ferramenta de dev profissional"** — português natural e correto, contrações comuns
  ("pra", "não dá") são aceitáveis com moderação, mas **sem gíria/informalidade excessiva**
  (nada de "beleza", regionalismos fortes). É uma ferramenta profissional, não um assistente
  descontraído.
- **Identidade brasileira sutil no DOMÍNIO da tarefa**, não nos identificadores nem em gíria:
  CPF/CNPJ (validação, dígito verificador), formatação de valor em Real (`R$ 1.234,56`), data
  no formato DD/MM/AAAA, IMC, feriados nacionais. Usar em parte das tarefas (não todas — não
  forçar brasilidade em toda tarefa).
- **`web_search` fora de escopo para JavaScript/frontend nesta rodada** — o `checker` só tem
  backend real para Python e Go (`src/checker/backends/`); gerar exemplos de JS validados de
  verdade exigiria construir um backend novo primeiro (tarefa de engenharia separada, maior,
  fora do escopo desta expansão de dataset).
- **Categoria "busca bem-sucedida em inglês"**: incluir um `<think>` EXPLÍCITO reconhecendo o
  descompasso de idioma antes do `<final>`, por exemplo: *"O conteúdo encontrado está em
  inglês, mas a conversa é em português — vou responder no idioma do usuário mesmo assim."*
  Não deixar essa regra implícita só na diferença de idioma entre o conteúdo da busca e o
  `<final>`; tornar a decisão parte do raciocínio treinado.

## 4. AVISO CRÍTICO — não repetir o erro do template (2ª vez já corrigido)

Isto já aconteceu **duas vezes** nesta geração do projeto:
1. `D-generalization-gap`: categorias inteiras do dataset original tinham 1-2 frases de pedido
   repetidas dezenas de vezes — o modelo decorou a forma da frase em vez de generalizar.
2. `D-tool-pilots-phrasing`: os 10 exemplos-base dos pilotos de ferramenta tinham
   `<think>`/`<final>` de fechamento quase idênticos entre si — o mesmo problema, na prosa de
   resolução em vez do pedido.

Ao gerar as variantes de frase desta rodada:
- **Não usar uma única string-template com `.format()` reaproveitada em todas as variantes**
  do mesmo jeito que causou o problema antes — cada variante de `<think>`/`<final>` precisa ter
  estrutura de frase genuinamente diferente (ordem de cláusulas, grau de formalidade, forma
  direta vs. indireta), não só troca de sinônimo isolado.
- Antes de aceitar o lote, rodar uma checagem de similaridade grosseira (ex.: normalizar e
  comparar os `<think>`/`<final>` de todas as variantes de uma mesma tarefa-base) para confirmar
  que não ficaram quase idênticos entre si.
- Preferir escrever as variantes "à mão" (like `D-tool-recovery-pilot`/`D-tool-success-pilot`),
  não gerar por template mecânico simples — mais trabalhoso, mas é exatamente o método que já
  funcionou bem nas duas correções anteriores.

## 5. Método de execução (reaproveitar o que já validado)

1. Escrever cada tarefa-base com código Python real e um teste real (`assert`), cobrindo o
   comportamento esperado.
2. Executar de verdade via `ToolExecutorRegistry`/`SandboxContext` (`write_file` + `checker`,
   `compile_and_test`) — nunca fabricar `tool_result`. Para a categoria shell, executar de
   verdade via `run_shell`/o executor de `shell`, só com binários da allowlist
   (`configs/sandbox_policy.yaml`).
3. Para a categoria de busca: usar `MockSearchBackend` com fixtures reais coletadas via busca
   de verdade (mesmo padrão de `data/fixtures/web_search/`) — nunca fabricar conteúdo de busca.
4. Sanitizar `stderr`/`stdout` de ruído específico desta máquina local antes de aceitar (já
   aconteceu uma vez: aviso de depreciação do `pytest-asyncio`, que não é dependência do
   projeto, vazando caminho de arquivo Windows — ver `D-tool-recovery-pilot`).
5. Gerar as variantes de frase reaproveitando os MESMOS fragmentos `<tool_call>`/`<tool_result>`
   já executados (extração via `parse_segments`, como em `expand_tool_pilots_phrasing.py`) —
   nenhuma re-execução necessária por variante, só a prosa ao redor muda.
6. Rodar a suite de testes completa (`pytest`) e a verificação automatizada de vazamento de
   detalhe interno no `<final>` antes de aceitar o lote.
7. Documentar como nova(s) decisão(ões) em `docs/PLAN.md` (seção 2, tabela de decisões) e
   atualizar a lista de probes/riscos se aplicável.
8. Só depois de tudo isso, retreinar.

## 6. Checklist antes de começar (amanhã)

- [ ] Confirmar que a chave `PRAXIS_OLLAMA_SEARCH_API_KEY` não precisa estar configurada
      localmente (usamos `MockSearchBackend`, não a API real, para gerar dataset)
- [ ] Confirmar allowlist de binários do shell ainda é a mesma (`configs/sandbox_policy.yaml`)
      antes de escrever os 10 exemplos de shell
- [ ] Escrever as 85 tarefas-base, uma categoria por vez, validando cada uma antes de passar
      para a próxima (não escrever tudo e validar só no final)
- [ ] Gerar as variantes de frase por tarefa-base (8-9 cada), com checagem de similaridade
- [ ] Rodar suite de testes + checagem de vazamento
- [ ] Atualizar `docs/PLAN.md`
- [ ] Commit + push
- [ ] Só então: retreinar no Colab
