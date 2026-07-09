# Plano: wave 2 do dataset — identidade Logos-3/Conatus + generalização multi-turno

Status: **wave 2 completa em 2026-07-09** — identidade, upgrade de estilo (taxonomia completa,
22/22 task_types), extensão de schema multi-turno (itens 1 e 2) E as duas categorias
multi-turno em si (85 exemplos: 45 mudança abrupta + 40 pedido de ferramenta tardio), todas
executadas (ver `D-conatus-logos3-identity`, `D-style-upgrade-wave2-pilot`,
`D-style-upgrade-wave2-round2`, `D-style-upgrade-wave2-round3`, `D-dataset-history-schema`,
`D-dataset-history-loss-mask` e `D-dataset-multiturn-wave2-pilot` em `docs/PLAN.md`). Nenhum
item planejado desta wave continua bloqueado por infraestrutura; o que resta é decisão de
retreino e validação com o modelo real, não mais escrita de dataset.

## 1. Motivação

Duas origens distintas, ambas de uso real da CLI (`conatus`), não hipotéticas:

1. **Identidade desatualizada**: a CLI mostrou "Sou Praxis, um agente de engenharia de
   software" ao usuário. Isso fecha a decisão que ficou deliberadamente adiada em
   `D-praxis-conatus-naming` ("revisar isso num próximo retreino... só então retreinar").
2. **Generalização multi-turno**: duas falhas reais filmadas na mesma sessão com histórico
   habilitado (`D-cli-session-history`):
   - Uma resposta enlatada ("Como este é um ambiente controlado, você deve ser um usuário
     padrão sem perfil específico") reaparecendo idêntica em dois turnos sem relação, uma vez
     fazendo sentido, outra vez não.
   - Um `<think>` solto inventando "e² é aproximadamente 7.389" sem relação com a pergunta
     feita (derivada parcial).
   - Um pedido de ferramenta avulso ("Liste os arquivos nesse diretório"), chegando tarde numa
     sessão já carregada de turnos sem relação, falhando completamente (resposta enlatada de
     novo, nunca chamou `list_files`/`shell`) — apesar da ferramenta já existir de verdade e o
     dataset já ter cobertura de listagem (`gen-shell-ls-workspace*`), só que sempre com o
     pedido de listagem embutido no MESMO turno que criou o arquivo, nunca como pedido avulso
     tardio.

## 2. Categorias e status

| Categoria | Contagem alvo | Status |
|---|---|---|
| Identidade (perguntas diretas + confusão com concorrentes) | ~50-60 | **28 executados** nesta sessão (`scripts/gen_identity_wave2_pilot.py`) — primeira leva real, hand-authored, sem duplicação mecânica |
| Mudança abrupta de direção (multi-turno) | 45 (revisado de ~200-250, ver seção 6.2) | **45 executados** nesta sessão (`scripts/gen_multiturn_wave2_pilot.py`) |
| Pedido de ferramenta avulso/tardio numa sessão poluída (multi-turno) | 40 (revisado de ~150-200, ver seção 6.2) | **40 executados** nesta sessão (`scripts/gen_multiturn_wave2_pilot.py`) |
| Upgrade de estilo (think técnico proporcional + final caloroso) sobre task_types existentes | ~300-400 | **27 executados** nesta sessão (`scripts/gen_style_upgrade_wave2_pilot.py`) — **todos os 22 task_types da taxonomia cobertos**, incluindo os seis que exigiam FALHA/recusa real controlada (checker_rejects_code, test_fails, forbidden_operation, tool_unavailable, invalid_call_then_correction, model_fixes_after_error) |

Total desta sessão: **55 exemplos novos (28 identidade + 27 upgrade de estilo) + ~3679
exemplos existentes com identidade corrigida**.

**Nota de correção**: uma versão anterior deste documento (e do commit correspondente)
afirmou incorretamente "35 exemplos" depois da segunda rodada — a contagem real era 23 (12 da
primeira rodada + 11 novos), confirmada por `ls data/train/gen-style-*.json | wc -l` antes de
escrever este parágrafo. Os números abaixo (seção 6.1.1 e 6.1.2) já estão corrigidos.

## 3. Por que a identidade foi priorizada e executada primeiro

Não depende de nenhuma mudança de schema (é `task_type=direct_answer`, sem ferramenta, formato
já suportado) e tem efeito imediato sobre TODO o dataset existente (o `system_prompt` é um
texto repetido literalmente em cada exemplo, não uma referência), então era o item de maior
alavancagem por esforço.

## 4. Extensão de schema necessária para as categorias multi-turno

Status: **item 1 (schema + suporte real em `Trajectory`) E item 2 (máscara de loss no
pipeline de treino) IMPLEMENTADOS em 2026-07-09** — ver `D-dataset-history-schema` e
`D-dataset-history-loss-mask` em `docs/PLAN.md`. Bloqueio técnico removido; falta só gerar as
categorias multi-turno em si.

O formato canônico atual (`src/dataset/schema.py`, PLAN.md seção 3) só representava um turno:
`trajectory: {system_prompt, user_request, raw_text}`. A CLI já resolve isso em produção
concatenando turnos anteriores no mesmo formato `[USER]/[ASSISTANT]` usado pelo modelo
(`cli/internal/agent/agent.go`, `Run(..., history []Turn, ...)`).

**Implementado**: campo opcional `trajectory.history`, lista de `{user_request, raw_text}`
(mesmo formato de `agent.Turn` no lado Go), mantendo `user_request`/`raw_text` de topo como o
turno ATUAL (o que efetivamente entraria na loss, quando o item 2 existir). Ausente ou vazio =
comportamento de hoje (sem histórico) — verificado contra os ~3706 exemplos já existentes no
dataset, 0 regressões.

- `TRAJECTORY_SCHEMA`/`validate_trajectory` (`src/dataset/schema.py`) — só exige `raw_text`
  (o único campo que `structural_validate` sempre leu direto do dict; exigir os outros
  quebraria testes/exemplos parciais já existentes que só têm `raw_text`), valida tipo de
  `system_prompt`/`user_request` quando presentes, e valida a forma de cada item de `history`
  (`user_request`/`raw_text` obrigatórios e não vazios).
- `structural_validate` (`src/dataset/pipeline.py`) agora chama `validate_trajectory` antes de
  parsear segmentos — um `raw_text` ausente vira erro de validação claro (`"trajectory: ..."`)
  em vez de `KeyError`.
- `Trajectory`/`HistoryTurn` (`src/harness/trajectory.py`): `Trajectory.history: list[HistoryTurn]`
  (default vazio). `render_for_model()` agora prepende os turnos de `history` no MESMO formato
  `[USER]/[ASSISTANT]` que `agent.Run` já usa em produção — testado byte a byte contra o
  formato exato da CLI Go. `to_example_dict()` novo, serializa pro formato canônico do dataset,
  omitindo `history` quando vazio (mantém exemplos de turno único idênticos a antes).

Exemplo de uso num gerador futuro (mesmo padrão dos `scripts/gen_*_pilot.py` já existentes):

```python
from src.harness import HistoryTurn, Trajectory

turno1 = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request="qual seu nome?")
turno1.append_raw("<final>Sou o Logos-3.</final>")

turno2 = Trajectory(
    system_prompt=PRAXIS_SYSTEM_PROMPT,
    user_request="e quanto é 2+2?",
    history=[HistoryTurn(user_request=turno1.user_request, raw_text=turno1.raw_text)],
)
turno2.append_raw("<final>4.</final>")

example = {"metadata": {...}, "trajectory": turno2.to_example_dict()}
```

### 4.1 Item 2 — máscara de loss (implementado em 2026-07-09)

Investigação: `compute_loss_mask`/`apply_loss_mask` (`src/training/loss_masking.py`) já
calculavam spans elegíveis (`think`/`tool_call`/`final`) só dentro de `raw_text` do turno
ATUAL, e tratavam qualquer coisa antes de `prefix_len` (o prefixo — system_prompt + pedido do
usuário + marcadores de template) como não elegível. Ou seja, a lógica de máscara em si NÃO
precisou de nenhuma mudança — bastou uma peça: fazer o PREFIXO usado por
`build_pretokenized_dataset` (`src/training/data_collator.py`) também incluir os turnos de
`history`, usando o mesmo `Trajectory.render_for_model()` que já sabe concatenar `history` no
formato `[USER]/[ASSISTANT]` (implementado no item 1). Como `raw_text` passado pra
`compute_loss_mask` continua sendo só o do turno atual (nunca o de `history`), todo o conteúdo
do histórico — inclusive `<think>`/`<tool_call>`/`<final>` de turnos passados, que pareceriam
"elegíveis" se fossem reparseados isoladamente — cai automaticamente dentro do prefixo mascarado.

Mudanças reais:
- `_render_prefix` (`src/training/data_collator.py`) ganhou um parâmetro opcional `history`,
  repassado pra `Trajectory(..., history=...)` antes de chamar `render_for_model()`.
- `build_pretokenized_dataset` agora lê `traj.get("history")` de cada trajetória e passa
  adiante — `history` ausente ou `None` mantém o comportamento de sempre.
- `compute_loss_mask`/`apply_loss_mask` (`loss_masking.py`): **nenhuma mudança**.

Testado (`tests/unit/test_data_collator.py`, roda sem precisar de `trl`, só
`transformers`/`datasets`): `test_pretokenized_dataset_masks_entire_history` constrói um
exemplo de 2 turnos e confirma, decodificando só os tokens com loss ativo, que NADA do
histórico sobra no texto treinado (nem o pedido, nem o `<think>`, nem o `<final>` do turno
passado) — só o turno atual. `test_pretokenized_dataset_without_history_key_matches_no_history`
confirma que um exemplo sem a chave `history` produz `input_ids`/`labels` byte a byte idênticos
a um com `history: []` explícito, e por extensão idênticos ao comportamento de antes desta
mudança (regressão de compatibilidade). 6/6 testes do arquivo passam (o 7º, ponta a ponta com
`SFTTrainer` real, já falhava antes desta mudança por um problema de ambiente Windows/cp1252
não relacionado — ver seção 8).

## 5. Regras de conteúdo (herdadas do spec amadurecido em conversa, reafirmadas aqui)

- `<think>`: profundidade proporcional à dificuldade real da tarefa; nunca inflado por
  aparência. Verbaliza a escolha de ferramenta em primeira pessoa quando há uma
  (`"Preciso verificar os arquivos nesse diretório"`, não narrativa pós-fato só).
- `<final>`: mais caloroso/convida a testar/dar feedback quando cabe — nunca em toda resposta
  trivial (viraria tique).
- Identidade completa (parágrafo "Conatus é uma família...") só quando perguntado
  diretamente — nunca como assinatura espontânea em respostas de outro assunto.
- Nenhum hífen (`-`) nem travessão (`—`) como pontuação de frase. Exceção deliberada: nomes
  próprios que contêm hífen de verdade (`Logos-3`, `GPT-4`) não são pontuação, são o nome.
- Categoria de mudança abrupta/pedido tardio: o pedido de ferramenta precisa chegar como turno
  avulso, não embutido num pedido composto — é exatamente essa lacuna que a falha real expôs.

## 6. Mecanismo de geração (decisão explícita, não é geração via LLM em loop)

Dado o histórico de custo sensível do usuário (créditos), o mecanismo escolhido para esta wave
é o mesmo já validado nas gerações anteriores (Gaps 1-6, seed original): **templates Python
escritos à mão** (`scripts/gen_*_pilot.py`), com execução REAL via `ToolExecutorRegistry`/
`SandboxContext`/`checker` quando a categoria envolve ferramenta, e sem chamada de LLM em loop
(nem Ollama local, nem modelo professor externo) para autoria de `<think>`/`<final>` — evita o
custo/tempo de um motor de self-play, mantém controle total de qualidade por exemplo, e segue
exatamente o precedente já estabelecido neste repositório (`_example()` em
`scripts/generate_dataset.py`, `build_example()` nos pilots de Gap).

Isso significa que a escala de "milhares" cogitada inicialmente não é o alvo real desta wave —
o alvo é o que foi acordado depois do ajuste de escopo (~700-900 no total, ~28 executados até
agora). Motor de geração automatizada (self-play) permanece uma ideia registrada em memória de
projeto, não descartada, só fora de escopo enquanto o custo por rodada de teste for uma
restrição real.

## 6.1 Upgrade de estilo — o que foi coberto e achados da execução real

`scripts/gen_style_upgrade_wave2_pilot.py` gerou 12 exemplos cobrindo 8 task_types
(`single_tool_call`, `multi_tool_call`, `complexity_analysis`, `refactor`, `debugging`,
`test_authoring`, `code_explanation`), todos com execução real de ferramenta quando aplicável
(`write_file`/`read_file`/`list_files`/`shell`/`checker`), demonstrando os quatro elementos do
estilo: profundidade proporcional (fácil = 1 think curto; difícil = ciclo de diagnóstico com
hipótese antes da evidência), verbalização de intenção de ferramenta em primeira pessoa, trade
off técnico explícito (ex.: `dict.fromkeys` vs `set()` pra preservar ordem, busca binária vs
linear em lista ordenada, `set()` vs `list` pra checagem de membership dentro de um laço), e
`<final>` caloroso só nos casos que cabem (não em respostas triviais de conceito).

Dois achados reais durante a execução, ambos corrigidos no gerador antes de aceitar os
exemplos:

- **Vazamento de caminho absoluto da máquina local via `pytest`**: quando o `checker` roda
  testes reais, o `pytest_asyncio` emite um warning de depreciação que inclui o caminho de
  instalação do Python local (`C:\Users\<usuário>\AppData\...`) — tanto no `stdout` de uma
  chamada bem sucedida quanto, mais sutilmente, dentro de `error_message` (que `to_json()`
  copia pro campo `"message"` do corpo do `<tool_result>` de erro, um campo separado de
  `data` que a sanitização original não cobria). Corrigido com um sanitizador que limpa
  recursivamente todo o payload E `error_message` via `dataclasses.replace` (o resultado é um
  dataclass frozen). Sem essa correção, dois dos quatro exemplos de debugging vazariam o nome
  de usuário real da máquina que gerou o dataset.
- **`language: "shell"` não é um valor válido da taxonomia** (`MVP_LANGUAGES` só aceita
  `python`/`go`) — mesmo problema pré-existente já documentado na seção 7 pros exemplos
  `gen-shell-*` antigos. Corrigido no exemplo novo usando `language: "python"` (nominal, mesma
  convenção usada no lote de identidade).

### 6.1.1 Segunda rodada (11 exemplos adicionais, mesma sessão)

Expandiu de 12 para 23 exemplos, cobrindo 11 task_types novos: `language_migration` (tradução
Python para Go, execução real via backend Go do checker), `compiles_successfully` (Go),
`project_configuration` (arquivo estático, sem `checker`), `documentation_usage` e
`insufficient_information`/`ambiguous_request` (sem ferramenta, pedindo esclarecimento em vez
de adivinhar), `checker_rejects_code` e `test_fails` (falha real REPORTADA honestamente, sem
ciclo de correção — task_type específico é sobre reconhecer/relatar a falha, não corrigi-la),
`multi_file_read_and_edit` (lê `config.py` de verdade antes de editar `main.py` com base no
valor lido), `tool_unavailable` (tenta `apply_patch`, que está `enabled: false` em
`configs/tools_registry.yaml` — recebe um `UNSUPPORTED_TOOL` genuíno do próprio
`ToolExecutorRegistry`, não fabricado, e recupera usando `write_file`), e `forbidden_operation`
(tenta `rm -rf` via `shell`, recebe recusa real da sandbox por estar no `denylist_patterns` de
`configs/sandbox_policy.yaml`).

Dois achados adicionais desta rodada:

- **Exceção de pontuação precisou crescer**: a regra "nenhum hífen fora de nomes próprios como
  Logos-3/GPT-4" não previa flags de linha de comando reais dentro do texto (`rm -rf` no
  exemplo de `forbidden_operation`) — são sintaxe técnica literal, não pontuação de frase, e
  entram na mesma categoria de exceção.
- **Alguns `<tool_result>` de erro real são genuínos, não simulados**: os exemplos de
  `tool_unavailable` e `forbidden_operation` passam pelo MESMO caminho real que a CLI/harness
  usariam (`ToolExecutorRegistry.execute` devolvendo `UNSUPPORTED_TOOL` de verdade pra uma
  ferramenta desabilitada; `sandbox.run_shell` recusando de verdade um comando do denylist) —
  reforça que "ferramenta ainda não disponível"/"operação proibida" no dataset não precisa
  nunca ser fabricado, o próprio harness já produz a resposta certa quando exercitado de
  verdade.

### 6.1.2 Terceira rodada (4 exemplos adicionais, mesma sessão) — taxonomia completa

Expandiu de 23 para 27 exemplos, fechando os 4 task_types que faltavam:

- **`direct_answer`**: pergunta conceitual pura ("o que é recursão"), sem ferramenta.
- **`invalid_call_then_correction`**: uma chamada real com argumento obrigatório faltando
  (`write_file` sem `content`) recebe um `TOOL_ARGUMENT_SCHEMA_ERROR` genuíno, calculado pelo
  MESMO validador de JSON Schema que o harness real usa antes de executar qualquer ferramenta
  (`ToolExecutorRegistry.validate_args`, espelhando `src/harness/loop.py`) — nunca fabricado.
  Corrigido na retentativa com os argumentos completos.
- **`model_fixes_after_error`**: `write_file` com `mode="create"` numa segunda escrita colide
  de propósito com um arquivo que a PRÓPRIA trajetória acabou de criar, recebendo o erro real
  `INCOMPLETE_SOLUTION` do executor (`write_file_tool.py`). A correção não é reenviar a mesma
  chamada, é uma abordagem genuinamente diferente (salvar num caminho novo em vez de tentar
  sobrescrever), evitando o mesmo padrão de "retentativa idêntica sem progresso real" já
  identificado como falha em `D-own-bug-diagnosis-expansion` (Gap 6).
- **`test_passes`**: sucesso direto sem nenhum ciclo de depuração, contraste deliberado com os
  exemplos de `debugging`/`test_fails` que já cobrem o caminho de falha.

Com isso, **os 22 task_types da taxonomia (`src/dataset/taxonomy.py`, `TASK_TYPES`) têm
cobertura no novo estilo de `<think>`/`<final>`** — não significa que cada um tem volume
suficiente pra generalizar bem (a maioria tem só 1 exemplo), mas fecha o objetivo desta
fatia: demonstrar o padrão em toda a superfície da taxonomia antes de decidir onde aprofundar.

## 6.2 Categorias multi-turno — executadas (85 exemplos, 45 + 40)

`scripts/gen_multiturn_wave2_pilot.py`. Alvo revisado explicitamente pelo usuário: em vez do
número aspiracional original (~200-250 + ~150-200), mirar um número maior de cara pra reduzir
risco de precisar de uma segunda rodada, mas ainda proporcional ao custo real de gerar
multi-turno com execução verdadeira — resultou em 45 (mudança abrupta) + 40 (pedido de
ferramenta tardio) = 85.

**Mecanismo**: um banco de 12 turnos de CONTEXTO reutilizáveis (`FILLERS` — identidade,
matemática simples, conceitos de linguagem, sem ferramenta) vira `history` em combinações
diferentes (1-3 turnos, ordem e composição variando por exemplo). O que efetivamente TREINA em
cada exemplo é só o turno ÂNCORA (`user_request`/`raw_text` de topo, via
`Trajectory(..., history=...).to_example_dict()` — D-dataset-history-schema). Reutilizar os
mesmos FILLERS como contexto em sessões diferentes não é a duplicação que o "AVISO CRÍTICO" dos
gaps anteriores alertava — aquele aviso era sobre duplicar o texto que TREINA, e `history` fica
inteiramente mascarado da loss (D-dataset-history-loss-mask); cada um dos 85 âncoras foi escrito
com conteúdo técnico genuinamente distinto.

**Categoria 1 (45 âncoras, mudança abrupta de direção)**: cobre algoritmos/complexidade
(quicksort no pior caso, hash maps, busca binária, Dijkstra, programação dinâmica...), teoria
de CS (NP-completo, BFS vs DFS, greedy), conceitos de linguagem (imutabilidade, `==` vs `is`,
late binding em closures, context managers), segurança (injeção de SQL), concorrência (race
condition, deadlock, processo vs thread), ferramentas de dev (git rebase vs merge, virtual
environment, CI/CD, linter), 3 exemplos de escrita de código com execução REAL
(`write_file`+`checker`: `is_prime`, fibonacci com memoização, validador de parênteses
balanceados), 3 exemplos de debugging REAL (off-by-one, argumento padrão mutável, operador de
comparação errado — bug plantado de propósito, real `checker` confirmando a falha e depois a
correção), e 4 exemplos de reação de confusão (`"O quê?"`, `"Isso não faz sentido"`), espelhando
diretamente o padrão observado na falha ao vivo original (`docs/PLAN.md` seção 1 deste
documento).

**Categoria 2 (40 âncoras, pedido de ferramenta tardio)**: 10 `list_files` (workspace vazio,
com arquivos, com subpasta, filtrado por glob), 10 `read_file` (config, main, utils, README,
requirements, model, gitignore, env.example, changelog, constants — todos arquivos reais
materializados no sandbox antes da chamada), 8 `shell ls`, 6 `shell cat`, 6 `shell grep`
(incluindo um caso real de `grep` sem nenhuma ocorrência — retorno de código 1, convenção do
próprio `grep` pra "sem match", não uma falha real de execução; o `<think>` do âncora precisa
interpretar isso corretamente). Toda ferramenta roda de verdade via `ToolExecutorRegistry`/
`SandboxContext`, nenhum `tool_result` fabricado.

**Achados durante a execução, corrigidos antes de aceitar os exemplos**:
- Mesmo vazamento de caminho absoluto local via `pytest_asyncio` já corrigido em
  `D-style-upgrade-wave2-pilot` — reaplicado aqui (o sanitizador não é compartilhado entre os
  dois scripts geradores, cada um tem sua cópia).
- Três hífens gramaticais evitáveis do português (`soma-se`, `quebrando-o`, `executá-lo` —
  construções pronominais reflexivas/clíticas) foram reescritos sem hífen (`some as`,
  `quebrando ele`, `executar ele`), mantendo a regra de pontuação estrita em vez de abrir mais
  uma exceção. Ficaram só as exceções já estabelecidas: nomes próprios (`Logos-3`, `GPT-4`,
  `NP-completo`) e notação de código/matemática citada em prosa (`items[:-1]`, `len(items) - 1`).

**Validação ponta a ponta real**: além de `structural_validate`/`schema_validate` (0 problemas
nos 85), 0 leaks, 0 ids duplicados com o resto do dataset (~3706 exemplos), rodei um exemplo de
verdade através de `build_pretokenized_dataset` (a mesma função que o notebook de treino usaria)
e confirmei, decodificando só os tokens com loss ativo, que o pedido do turno de histórico
realmente não aparece no texto treinado — não é só teste sintético isolado
(`tests/unit/test_dataset_history_schema.py`/`test_data_collator.py`), é o dataset real passando
pela pipeline real de ponta a ponta.

## 7. Achado colateral (não é retrabalho, é higiene de schema pré-existente)

Ao validar o novo lote de identidade contra `structural_validate`
(`src/dataset/pipeline.py`), descobri que o padrão `"checker_used": None, "expected_result":
{"passed": None}` usado em pilots anteriores (ex.: `gen_direct_answer_trapword_pilot.py`) FALHA
a validação de schema (`METADATA_SCHEMA` exige `string`/`object`, não aceita `null` nesses
campos). Isso já existia antes desta sessão — não é uma regressão introduzida agora. Corrigido
apenas no gerador novo (omitir as chaves em vez de usar `None`); os pilots antigos com esse
padrão não foram tocados aqui (fora de escopo desta wave, registrado para limpeza futura).

## 8. Checklist desta sessão

- [x] Trocar `PRAXIS_SYSTEM_PROMPT` (`src/harness/system_prompt.py`) e `agent.SystemPrompt`
      (`cli/internal/agent/agent.go`) para a identidade Logos-3/Conatus, byte a byte iguais
- [x] Bulk-replace do `system_prompt` literal nos ~3679 exemplos existentes que diziam
      "Praxis" (mais 3 ocorrências cosméticas de "Praxis Example" → "Logos Example")
- [x] Gerar lote de identidade (28 exemplos: perguntas diretas + confusão com concorrentes
      nomeados + confusão com o nome antigo "Praxis")
- [x] Validar estrutura/schema do lote novo (`structural_validate`/`schema_validate`, 0
      problemas), checar vazamento de detalhe interno (0 leaks)
- [x] Rodar suite de testes Python (142 passaram, 1 falha pré-existente e não relacionada —
      `trl` falhando ao ler um arquivo `.jinja` próprio por causa do codec padrão cp1252 do
      Windows, nada a ver com o dataset) e suite Go da CLI (55/55)
- [x] Gerar lote de upgrade de estilo, primeira rodada (12 exemplos, 8 task_types) —
      `scripts/gen_style_upgrade_wave2_pilot.py`
- [x] Expandir o lote de upgrade de estilo, segunda rodada (+11 exemplos, 11 task_types
      novos) — seção 6.1.1
- [x] Expandir o lote de upgrade de estilo, terceira rodada (+4 exemplos: `direct_answer`,
      `invalid_call_then_correction`, `model_fixes_after_error`, `test_passes`) — seção
      6.1.2, fecha os 22 task_types da taxonomia
- [x] Validar estrutura/schema do lote de estilo completo, 27 exemplos (0 problemas), checar
      vazamento de detalhe interno (0 leaks, depois de corrigir o vazamento real encontrado —
      seção 6.1) e pontuação (0 hífens/travessões fora das exceções de nome próprio/sintaxe
      técnica)
- [x] Rodar suite de testes Python de novo após cada rodada do lote de estilo (138/138,
      ignorando o teste pré-existente e não relacionado do `trl`)
- [x] **Item 1 da extensão de schema multi-turno**: `TRAJECTORY_SCHEMA`/`validate_trajectory`
      (`src/dataset/schema.py`), integração em `structural_validate`
      (`src/dataset/pipeline.py`), `HistoryTurn`/`Trajectory.history`/`render_for_model` com
      histórico/`to_example_dict` (`src/harness/trajectory.py`) — seção 4
- [x] Testar a extensão de schema (17 testes novos, `tests/unit/test_dataset_history_schema.py`)
      e confirmar 0 regressões nos ~3706 exemplos já existentes no dataset (varredura completa,
      não amostral)
- [x] Rodar suite de testes Python de novo após a extensão de schema (155/155, ignorando o
      teste pré-existente do `trl`)
- [x] **Item 2 da extensão de schema multi-turno**: `_render_prefix`/`build_pretokenized_dataset`
      (`src/training/data_collator.py`) agora incluem `history` no prefixo mascarado — `compute_loss_mask`/
      `loss_masking.py` não precisaram mudar nada, o design já era compatível — seção 4.1
- [x] Testar a máscara de loss com histórico (2 testes novos em `tests/unit/test_data_collator.py`,
      rodam sem `trl`): confirma que NADA do histórico entra na loss e que a ausência de
      `history` é idêntica a `history: []` (regressão de compatibilidade)
- [x] Rodar suite de testes Python de novo após a máscara de loss (161/161, ignorando o único
      teste pré-existente que já falhava antes desta sessão inteira por um problema de
      ambiente Windows/cp1252 no `trl`, não relacionado a nada disto)
- [x] Gerar categorias multi-turno em si (85 exemplos: 45 mudança abrupta + 40 pedido de
      ferramenta tardio) — `scripts/gen_multiturn_wave2_pilot.py`, seção 6.2
- [x] Validar estrutura/schema dos 85 exemplos (0 problemas), checar vazamento de detalhe
      interno (0 leaks, depois de corrigir o mesmo vazamento do `pytest_asyncio` já visto em
      `D-style-upgrade-wave2-pilot`), pontuação (0 violações depois de reescrever 3 hífens
      gramaticais evitáveis do português) e ids duplicados com o resto do dataset (0)
- [x] Validação ponta a ponta real: um exemplo do lote passado por
      `build_pretokenized_dataset` de verdade confirma que o histórico não vaza pro texto
      treinado, não só nos testes sintéticos
- [x] Rodar suite de testes Python de novo (161/161, mesmo teste pré-existente do `trl`
      ignorado)
- [x] Atualizar `docs/PLAN.md`, commit
- [ ] Retreinar e validar com o modelo real (checkpoint novo) — fora do escopo desta sessão,
      decisão de quando/como retreinar fica com o usuário
