# Plano: disciplina de `<think>` antes de agir (e a questão `write_file` vs `create_file`)

Status: **planejado, não executado.** Escrito depois do primeiro treino real na L4, quando as
transcrições dos probes mostraram o modelo indo direto pro `<tool_call>` (com nome/sintaxe
errados) sem nenhum `<think>` visível antes.

## 1. Motivação (achado real, medido)

Nas transcrições dos probes rodadas no Colab (adapter L4 real), **em nenhuma** o modelo emitiu
`<think>`, mesmo em `mode="dev"` (que exibe raciocínio). Ele pula direto pra ação. Isso importa
porque as falhas observadas (`create_file` no lugar de `write_file`, sintaxe self-closing,
`<tool_use>` no lugar de `<tool_call>`) são exatamente o tipo de erro que um passo de raciocínio
antes de agir tende a evitar: "qual é a ferramenta certa aqui? como é a sintaxe dela?".

Medição real no dataset atual (2460 exemplos de treino, contagem por script, não estimativa):

| Fatia | Contagem | % |
|---|---|---|
| Exemplos com `<think>` em algum ponto | 1513 | 61,5% |
| Exemplos que chamam ferramenta com **zero** `<think>` antes | 947 | 38,5% |

Os 947 sem `<think>` **não** são respostas diretas (todos os 109 exemplos `direct_answer` têm
`<think>`). São todos exemplos `aug-*`/`gen-*`/`seed-*` gerados por script, concentrados em
categorias grandes de uso de ferramenta:

| task_type | exemplos sem `<think>` |
|---|---|
| checker_rejects_code | 253 |
| code_explanation | 217 |
| compiles_successfully | 169 |
| language_migration | 118 |
| multi_tool_call | 77 |
| (outros: debugging, test_fails, complexity_analysis, etc.) | ~113 |

Todos vêm dos builders em massa de `scripts/generate_dataset.py` (funções `build_expanded_examples`
etc.), que concatenam `_tool_call(...)` diretamente, sem nenhum `<think>` na frente. Exemplo real
(`aug-multi-go-Abs.json`):

```
<tool_call name="write_file">{...}</tool_call><tool_result .../><tool_call name="checker">{...}</tool_call><tool_result .../><final>Arquivo criado e validado com sucesso.</final>
```

Conclusão: mais de um terço do dataset ensina, na prática, que "chamar ferramenta direto, sem
pensar antes" é o padrão normal, inclusive em `multi_tool_call` (o tipo que mais se parece com o
cenário que falhou nos probes). Isso é um candidato forte, e mensurável, pra explicar por que o
`<think>` some na inferência.

## 2. Restrição crítica de consistência (ler antes de qualquer mudança)

`PRAXIS_SYSTEM_PROMPT` (`src/harness/system_prompt.py`) é a fonte única de verdade e está
**embutido, byte a byte, no campo `system_prompt` de todos os 2460 exemplos**, e também no
`agent.SystemPrompt` do CLI Go (`cli/internal/agent/agent.go`). O próprio docstring do módulo
avisa: divergir esse texto entre treino e inferência recria o mismatch que `D-train-prompt-mask`
corrigiu.

Consequência para este plano: **se mexermos no texto do system prompt (Frente A), TODOS os 2460
exemplos e o CLI Go precisam ser atualizados em lockstep** — a mesma disciplina já aplicada na
troca de identidade Praxis→Logos-3 (`D-conatus-logos3-identity`, wave 2). Não é uma edição de um
arquivo só.

## 3. Duas frentes (independentes, podem ser feitas juntas ou uma só)

### Frente A — reforçar o system prompt

Texto atual:

> "Você é Logos-3, um agente de engenharia de software da família de modelos Conatus. Raciocine
> em `<think>`, use `<tool_call name="...">` quando precisar de uma ferramenta, e responda ao
> usuário só dentro de `<final>`."

Problema: "Raciocine em `<think>`" é uma menção passiva, sem "sempre" nem "antes de agir". Proposta
(a confirmar com o usuário, exige regenerar os 2460 `system_prompt` + CLI Go):

> "...**Sempre** comece pensando em `<think>` antes de qualquer `<tool_call>` ou `<final>`: decida
> ali qual ferramenta usar e confira o nome e a sintaxe dela. Use `<tool_call name="...">` quando
> precisar de uma ferramenta, e responda ao usuário só dentro de `<final>`."

Trade-off honesto: reforçar o prompt **sozinho** provavelmente ajuda pouco se 38,5% do dataset
continua demonstrando o contrário — o comportamento aprendido tende a ganhar do texto de
instrução. Frente A faz sentido principalmente **combinada** com a Frente B, não isolada.

### Frente B — reintroduzir `<think>` nos ~947 exemplos gerados (o trabalho de verdade)

Não é edição manual de 947 arquivos: é **alterar os builders em massa de
`scripts/generate_dataset.py`** para injetar um `<think>` genuíno antes da primeira ação, e
regenerar de forma determinística.

Regra central de qualidade (segue a spec de estilo já acordada — memória
`project-next-dataset-think-style`, `feedback-dataset-scenario-realism`): o `<think>` precisa ser
**real e proporcional**, nunca um carimbo vazio. Um `<think>Vou usar a ferramenta.</think>`
repetido 947 vezes ensinaria um ritual oco, não raciocínio — seria um problema novo no lugar do
antigo. O `<think>` de cada categoria verbaliza a intenção real, em primeira pessoa:

| task_type | `<think>` proposto (curto, mas real) |
|---|---|
| multi_tool_call (criar+validar) | "Preciso criar o arquivo e depois confirmar que compila. Vou usar `write_file` pra escrever e `checker` pra validar." |
| checker_rejects_code | "O usuário quer saber se `X` compila. Vou rodar o `checker` no arquivo e reportar o que ele encontrar." |
| code_explanation | "Antes de explicar, preciso ler o conteúdo real de `X` com `read_file` em vez de supor o que ele faz." |
| compiles_successfully | "Vou compilar `X` com o `checker` pra confirmar que está válido antes de responder." |
| language_migration | "Vou traduzir a lógica de Python pra Go preservando o comportamento, e validar a versão Go com o `checker`." |

Profundidade proporcional: essas tarefas são simples, então o `<think>` é curto (1 frase). O que
importa é que ele **nomeie a ferramenta certa e a intenção** — exatamente o passo que, ausente,
deixa o modelo chutar `create_file`/`save_file`.

Restrições de conteúdo já estabelecidas que se aplicam: sem hífen/travessão no texto gerado
(tell de texto de IA que o usuário rejeita), identificadores em inglês, nenhum `tool_result`
fabricado (os que já existem vêm de execução real e são preservados).

## 4. A questão `write_file` vs `create_file` (decisão)

O usuário perguntou se não seria melhor adotar `create_file`, "que já é a orientação do modelo".
Recomendação: **manter `write_file`**, pelos seguintes motivos concretos:

1. **O modelo não convergiu pra `create_file`.** Nas transcrições reais ele inventou também
   `save_file`, `write_new_file`, `validate_code`, `check_syntax`, `file_get_content`,
   `file_list_contents`, e trocou a tag inteira (`<tool_use>`). Não é preferência por um nome — é
   chute de um nome plausível quando falta reforço. Renomear pra `create_file` não impede o
   próximo chute (`make_file`, `new_file`).
2. **Semântica.** `write_file` cria E sobrescreve (usado em fluxos de correção, ex.
   `gen_tool_recovery_pilot.py`, onde o mesmo arquivo é reescrito após erro do checker).
   "create" seria mais estreito e menos preciso.
3. **Custo/superfície.** Trocar tocaria o registro de ferramentas
   (`configs/tools_registry.yaml`), o executor, o CLI Go, e os ~2460 exemplos que já usam
   `write_file` — tudo pra perseguir um sintoma do mesmo problema de reforço que a Frente B já
   ataca pela raiz.

A causa real do erro de nome não é o nome estar "errado" — é reforço insuficiente + ausência do
`<think>` que confirmaria o nome antes de chamar. A Frente B trata as duas coisas de uma vez.

## 5. Método de execução (quando for a hora)

1. **Decidir o escopo com o usuário:** só Frente B, ou A+B juntas? (A sozinha não recomendada.)
2. Se incluir Frente A: editar `PRAXIS_SYSTEM_PROMPT` + `agent.go`, e regenerar o campo
   `system_prompt` em todos os 2460 exemplos (edição estrutural do JSON, como na troca de
   identidade — não replace cego). Confirmar byte-identidade treino/inferência.
3. Frente B: alterar os builders de `generate_dataset.py` pra injetar o `<think>` proposto por
   categoria. Regerar. Como os `<tool_call>`/`<tool_result>` vêm de execução real, a regeneração
   re-executa o checker de verdade (mesma garantia de sempre: nada fabricado).
4. Rodar o pipeline de validação existente: `structural_validate`/`validate_metadata`/
   `validate_trajectory`, checagem de vazamento, checagem de similaridade de frase, dedup de id,
   verificação de bytes UTF-8, e a checagem de pontuação (sem hífen/travessão).
5. Re-medir a proporção: alvo é derrubar os 38,5% "sem `<think>`" para perto de zero **nas
   categorias de uso de ferramenta** (respostas diretas legítimas que não precisam de `<think>`
   antes de um `<final>` conceitual continuam válidas — não forçar `<think>` onde não agrega).
6. Suite Python completa verde.
7. Documentar como decisão em `docs/PLAN.md`.
8. Commit + push. Retreino é decisão separada do usuário.

## 6. Riscos

- **`<think>` templatizado:** o maior risco. Se os `<think>` injetados forem repetitivos demais
  entre exemplos da mesma categoria, o modelo aprende o template literal, não o hábito de
  raciocinar. Mitigação: variar a redação por tarefa-base dentro de cada categoria (não uma
  string fixa), e rodar `check_phrasing_similarity.py` sobre os `<think>` gerados, não só sobre os
  `<final>`.
- **Regressão de outras métricas:** mudar 947 trajetórias muda a distribuição de treino. A única
  forma real de confirmar que ajuda (e não só desloca o problema) é retreinar e rodar os probes de
  novo — custo de créditos que o usuário controla. O plano entrega o dataset pronto; a validação
  final depende do retreino.
- **Frente A sem Frente B:** reforçar só o texto e não os dados pode dar falsa sensação de
  correção. Por isso a recomendação é A+B ou só B, nunca só A.

## 7. Checklist antes de começar a execução

- [ ] Usuário decide escopo: Frente B sozinha, ou A+B.
- [ ] Se A: novo texto do system prompt confirmado com o usuário.
- [ ] `<think>` proposto por categoria revisado (curto, real, variado, sem hífen/travessão).
- [ ] Editar builders de `generate_dataset.py` (Frente B) e, se A, o system prompt + CLI Go +
      regeneração dos 2460 `system_prompt`.
- [ ] Regerar com re-execução real do checker.
- [ ] Pipeline de validação + vazamento + similaridade (inclusive sobre `<think>`) + dedup +
      UTF-8 + pontuação.
- [ ] Re-medir proporção "sem `<think>`" nas categorias de ferramenta (alvo ~0).
- [ ] Suite Python completa.
- [ ] `docs/PLAN.md` atualizado.
- [ ] Commit + push. (Retreino: decisão do usuário.)
