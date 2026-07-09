# Plano: wave 2 do dataset — identidade Logos-3/Conatus + generalização multi-turno

Status: **identidade, upgrade de estilo (taxonomia completa, 22/22 task_types) e item 1 da
extensão de schema multi-turno executados em 2026-07-09** (ver `D-conatus-logos3-identity`,
`D-style-upgrade-wave2-pilot`, `D-style-upgrade-wave2-round2`, `D-style-upgrade-wave2-round3` e
`D-dataset-history-schema` em `docs/PLAN.md`). Categorias multi-turno em si (geração dos
exemplos) ainda **pendentes** — o schema já suporta `history`, mas falta o item 2 (máscara de
loss no pipeline de treino) antes de gerar essas categorias de verdade.

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
| Mudança abrupta de direção (multi-turno) | ~200-250 | Pendente — requer extensão de schema (seção 4) |
| Pedido de ferramenta avulso/tardio numa sessão poluída (multi-turno) | ~150-200 | Pendente — mesma dependência |
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

Status: **item 1 (schema + suporte real em `Trajectory`) IMPLEMENTADO em 2026-07-09** — ver
`D-dataset-history-schema` em `docs/PLAN.md`. **Item 2 (máscara de loss no pipeline de treino)
continua PENDENTE** — não foi tocado, não foi nem investigado ainda.

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

**Trabalho ainda pendente antes de gerar as categorias multi-turno (item 2)**: a máscara de
loss (o notebook/pipeline de treino) precisa saber ignorar TODO o conteúdo de `history` (é
contexto, não algo que o modelo deveria aprender a reproduzir fresco) e mascarar dentro do
turno atual exatamente como já faz hoje (só `think`/`tool_call`/`final` gerados pelo modelo,
nunca `tool_result`). Isso ainda não foi tocado nem investigado — é o próximo passo.

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
- [ ] **Item 2 da extensão de schema multi-turno**: máscara de loss no pipeline/notebook de
      treino pra ignorar `history` — **PENDENTE, não investigado ainda, é o próximo passo**
- [ ] Categorias multi-turno em si (mudança abrupta, pedido tardio) — pendente, depende do
      item 2 acima
- [x] Atualizar `docs/PLAN.md`, commit
