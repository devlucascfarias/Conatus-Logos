# Plano: wave 2 do dataset — identidade Logos-3/Conatus + generalização multi-turno

Status: **identidade e upgrade de estilo executados em 2026-07-09** (ver
`D-conatus-logos3-identity` e `D-style-upgrade-wave2-pilot` em `docs/PLAN.md`). Categorias
multi-turno ainda **pendentes** — dependem de uma extensão de schema que este documento desenha
mas não implementa ainda.

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
| Upgrade de estilo (think técnico proporcional + final caloroso) sobre task_types existentes | ~300-400 | **35 executados** nesta sessão (`scripts/gen_style_upgrade_wave2_pilot.py`) — 18 dos 22 task_types da taxonomia cobertos, incluindo quatro que exigiam FALHA real controlada (checker_rejects_code, test_fails, forbidden_operation, tool_unavailable). Faltam: `direct_answer`, `invalid_call_then_correction`, `model_fixes_after_error`, `test_passes` |

Total desta sessão: **63 exemplos novos (28 identidade + 35 upgrade de estilo) + ~3679
exemplos existentes com identidade corrigida**.

## 3. Por que a identidade foi priorizada e executada primeiro

Não depende de nenhuma mudança de schema (é `task_type=direct_answer`, sem ferramenta, formato
já suportado) e tem efeito imediato sobre TODO o dataset existente (o `system_prompt` é um
texto repetido literalmente em cada exemplo, não uma referência), então era o item de maior
alavancagem por esforço.

## 4. Extensão de schema necessária para as categorias multi-turno (desenhada, não implementada)

O formato canônico atual (`src/dataset/schema.py`, PLAN.md seção 3) só representa um turno:
`trajectory: {system_prompt, user_request, raw_text}`. A CLI já resolve isso em produção
concatenando turnos anteriores no mesmo formato `[USER]/[ASSISTANT]` usado pelo modelo
(`cli/internal/agent/agent.go`, `Run(..., history []Turn, ...)`), mas o dataset de treino não
tem como representar isso ainda.

Proposta: campo opcional `trajectory.history`, lista de `{user_request, raw_text}` (mesmo
formato de `agent.Turn` no lado Go), mantendo `user_request`/`raw_text` de topo como o turno
ATUAL (o que efetivamente entra na loss). Ausente ou vazio = comportamento de hoje (sem
histórico). Isso espelha exatamente `Trajectory.render_for_model`/`agent.Run`: o prompt
completo vira `system_prompt + history_concatenada + "[USER]\n{user_request}\n[ASSISTANT]\n{raw_text}"`.

**Trabalho pendente antes de gerar essas categorias**: a máscara de loss (o notebook/pipeline
de treino) precisa saber ignorar TODO o conteúdo de `history` (é contexto, não algo que o
modelo deveria aprender a reproduzir fresco) e mascarar dentro do turno atual exatamente como
já faz hoje (só `think`/`tool_call`/`final` gerados pelo modelo, nunca `tool_result`). Isso não
foi tocado nesta sessão — é um pré-requisito real antes da próxima leva.

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

### 6.1.1 Segunda rodada (23 exemplos adicionais, mesma sessão)

Expandiu de 12 para 35 exemplos, cobrindo 11 task_types novos: `language_migration` (tradução
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
- [x] Expandir o lote de upgrade de estilo, segunda rodada (+23 exemplos, 11 task_types
      novos, incluindo os quatro que exigiam falha real controlada) — seção 6.1.1
- [x] Validar estrutura/schema do lote de estilo completo, 35 exemplos (0 problemas), checar
      vazamento de detalhe interno (0 leaks, depois de corrigir o vazamento real encontrado —
      seção 6.1) e pontuação (0 hífens/travessões fora das exceções de nome próprio/sintaxe
      técnica)
- [x] Rodar suite de testes Python de novo após cada rodada do lote de estilo (138/138,
      ignorando o teste pré-existente e não relacionado do `trl`)
- [ ] Categorias multi-turno (mudança abrupta, pedido tardio) — pendente, depende da extensão
      de schema da seção 4 e do trabalho de máscara de loss no pipeline de treino
- [ ] Completar os 4 task_types restantes do upgrade de estilo (`direct_answer`,
      `invalid_call_then_correction`, `model_fixes_after_error`, `test_passes`) — próxima
      fatia mais barata, mesmo mecanismo já validado
- [x] Atualizar `docs/PLAN.md`, commit
