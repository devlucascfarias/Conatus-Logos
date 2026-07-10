# Praxis-SFT v2 — Plano Técnico de Implementação

> Status: proposta para revisão. Nenhum código de produção foi escrito antes deste documento.
> Este projeto é uma **regeneração completa** do Praxis. Nenhum dataset, notebook, avaliador,
> prompt, estrutura de saída ou decisão arquitetural de versões anteriores (`Praxis-1`,
> `praxis-1-backup`) é reaproveitado. Onde havia ambiguidade nos requisitos, uma decisão foi
> tomada e registrada explicitamente como **[HIPÓTESE]**, em vez de bloquear o plano com perguntas.

---

## 0. Sumário executivo

Praxis-SFT v2 é um modelo especializado em engenharia de software, derivado via QLoRA de
`ibm-granite/granite-4.1-8b`, treinado para operar como agente de programação: raciocinar,
chamar ferramentas estruturadas, interpretar retornos reais dessas ferramentas, verificar sua
própria solução e emitir uma resposta pública limpa. O treinamento roda em uma única GPU
NVIDIA L4 (Google Colab), o que dita quase todas as decisões de escopo e engenharia abaixo.

A decisão arquitetural central do projeto é: **a interação com ferramentas é modelada como uma
única mensagem de assistente, gerada em múltiplas passadas de decodificação intercaladas pela
injeção de texto do harness** — não como múltiplos turnos de chat com roles de ferramenta, e não
amarrada ao tool-calling nativo de um backend específico. Essa escolha é o que torna o harness
verdadeiramente desacoplado do backend de inferência (Transformers hoje, vLLM/llama.cpp/API
remota amanhã), porque a única capacidade exigida de qualquer backend é "gerar texto até uma
stop-sequence e continuar a partir de um prefixo" — suportado universalmente.

Duas decisões adicionais direcionam a arquitetura para além desta geração: (1) o **harness de
produção será implementado em Go**, não em Python — a implementação Python descrita neste plano
é a referência de desenvolvimento/dataset/treino desta geração, não o destino final; todos os
contratos (gramática, JSON Schemas, formato de trajetória) são deliberadamente
implementation-agnostic para tornar esse porte de baixo risco (seção 5.5). (2) `web_search` usa
a API de busca da Ollama como backend inicial, mas por trás de uma interface `SearchBackend`
parametrizada por configuração, para permitir trocar de motor de busca sem alterar código
(seção 5.4).

O MVP proposto pelo usuário foi analisado criticamente e **reduzido** (seção 15): duas
linguagens no dia 1 (Python + Go), cinco ferramentas (não sete), sem `apply_patch` por diff e
sem `web_search` real. A justificativa está na seção 15.

---

## 1. Definição precisa do escopo

### 1.1 Está dentro do escopo (v2, todas as fases)

- Um adapter LoRA sobre `ibm-granite/granite-4.1-8b`, treinado via SFT com QLoRA 4-bit.
- Um formato canônico de trajetória (raciocínio + tool calls + resultado + resposta pública).
- Um harness de referência local (Python, single-process) que implementa o loop do agente,
  suficiente para gerar dados, validar o dataset e rodar avaliação — **não** é a CLI final de
  produção, é o ambiente de desenvolvimento/validação que a CLI futura vai herdar.
- Um checker independente do modelo, reutilizado em três contextos: (a) ferramenta em tempo de
  execução do agente, (b) validador offline do dataset, (c) motor de métricas do avaliador.
- Um pipeline de geração/validação/deduplicação/split de dataset com metadados obrigatórios.
- Um conjunto de probes adversariais e um benchmark interno reproduzível.
- Um notebook de treinamento para Colab/L4 dirigido por arquivo de configuração versionado.
- Suporte gradual a linguagens: MVP = Python + Go; Fase 2 = JavaScript/TypeScript; Fase 3 =
  SQL, Shell/Bash, HTML/CSS, configuração/infra, Git. C/C++, Java e Rust foram removidas do
  escopo do projeto (seção 18) — não apenas adiadas — por multiplicarem o custo de manter
  backends de checker sem agregar diversidade de paradigma que Python+Go já não cubram.

### 1.2 Está fora do escopo nesta etapa (ver seção 18 para lista completa)

- CLI de produção completa (só a interface/contrato é projetada agora).
- Full fine-tuning (não é viável em L4; QLoRA é a única via considerada).
- `web_search` real contra a internet (mock/interface abstrata no MVP).
- RLHF/RLAIF, DPO, ou qualquer otimização de preferência (apenas SFT nesta geração).
- Suporte a múltiplos modelos-base simultâneos (só Granite-4.1-8B nesta geração).
- Merge do adapter no modelo base como pipeline oficial (pode ser feito ad-hoc, mas não é
  entregável desta fase).

### 1.3 Definição de "sucesso" para esta geração

O projeto é bem-sucedido se, ao final do MVP (seção 15), for possível demonstrar de forma
reproduzível as 7 propriedades listadas pelo usuário na seção 15 do pedido original, medidas
por probes automatizados (não por inspeção manual de exemplos soltos).

---

## 2. Decisões arquiteturais (registro consolidado)

| # | Decisão | Justificativa | Tipo |
|---|---|---|---|
| D1 | Formato canônico = tags delimitadoras + corpo JSON estrito (ver seção 3) | Melhor equilíbrio entre robustez de parsing, streaming e portabilidade de backend | Decisão fundamentada |
| D2 | Tool calling multi-etapa = uma única mensagem assistant contínua, não múltiplos turnos de role "tool" | Desacopla o harness do template de chat de qualquer backend específico | Decisão fundamentada |
| D3 | Checker é uma biblioteca única compartilhada por runtime, validação de dataset e avaliação | Evita divergência entre "o que valida o dataset" e "o que valida em produção" | Decisão fundamentada |
| D4 | MVP = Python + Go (não Python+JS+Go+Rust) | Reduz superfície de toolchains no dia 1 sem perder o teste "linguagem dinâmica vs compilada" | **[HIPÓTESE]**, revisar após M2 |
| D5 | MVP tools = `checker`, `read_file`, `write_file`, `list_files`, `shell` | `search_code`/`git_diff` são compostos triviais via `shell`; `apply_patch` adia risco de aplicação incorreta de diff | **[HIPÓTESE]**, revisada por D-search-code-reenable |
| D-search-code-reenable | `search_code` reativada (`configs/tools_registry.yaml`, `enabled: true`), executor real em `src/tools/executors/search_code_tool.py` (busca texto/regex em Python puro, sem depender de `grep`/`rg` no PATH) | Pivô de frontend (docs/plan_frontend_specialization_wave3.md seção 4): o think técnico ganhou a dimensão "reuse-before-create" — checar se já existe um componente/token antes de criar um novo, evitando duplicação e código morto. Compor via `shell`+`grep` (razão original de D5) não é portável nem confiável fora de Linux/Colab | Decisão fundamentada |
| D6 | Sandbox de execução = subprocess isolado com rlimits + diretório de trabalho descartável, não Docker | Colab não garante Docker-in-Docker confiável; subprocess+rlimit+chroot-lite é suficiente para o MVP | **[HIPÓTESE]**, revisitar se ameaça de escape for relevante em produção |
| D7 | Loss é mascarado (label = -100) em todo texto que não foi gerado pelo assistente: prompts de usuário, system prompt e blocos `<tool_result>` | Sem isso o modelo aprende a "prever" resultados de ferramenta, o oposto do requisito anti-fabricação | Decisão fundamentada |
| D8 | Sem sequence packing na v1; usar `group_by_length` | Packing multi-exemplo exige máscara de atenção por segmento para não vazar contexto entre exemplos; complexidade adiada | **[HIPÓTESE]** — `group_by_length` teve que ser removido de `TrainingArguments` no notebook (transformers 5.x não aceita mais esse kwarg, `TypeError` confirmado rodando no Colab); treino segue sem agrupamento por tamanho por enquanto, sem afetar correção |
| D9 | Dois modos de resposta: `dev` (expõe `<think>`/`<tool_call>`/`<tool_result>`) e `prod` (harness extrai só `<final>`) | Pedido explícito do usuário na seção 7 | Decisão fundamentada |
| D10 | Versionamento de schemas de ferramenta via JSON Schema + número de versão por ferramenta | Permite adicionar/alterar ferramentas sem quebrar exemplos antigos do dataset | Decisão fundamentada |
| D11 | `checker` nunca confia em texto do modelo; sempre reexecuta/recompila a partir dos arquivos reais do workspace | Requisito central da seção 10 | Decisão fundamentada |
| D12 | Repositório único (`Logos-1`) hospeda o projeto Praxis-SFT v2 do zero | Decisão do usuário nesta sessão | Confirmado com usuário |
| D13 | Harness de produção será reimplementado em Go; harness Python desta geração é referência de dev/dataset/treino | Objetivo declarado do modelo/CLI é eficiência e velocidade; Go oferece binário estático único, baixo overhead de startup e memória, boa concorrência nativa para execução paralela de ferramentas, sem exigir runtime Python no ambiente do usuário final | Confirmado com usuário |
| D14 | `web_search` usa a API de busca da Ollama como backend inicial, atrás de uma interface `SearchBackend` selecionável por config (`provider: ollama\|mock\|...`) | Pedido explícito do usuário + necessidade de trocar motor de busca no futuro sem reescrever chamadores | Confirmado com usuário |
| D-shell-v2 | Schema da ferramenta `shell` migrado de `{"command": string}` para `{"binary": string, "args": [string]}`, executado via `subprocess.run([binary, *args], shell=False)` — nunca `shell=True` | Execução verdadeiramente independente de SO (mesma chamada Python em Windows/Linux/Mac, sem escolher entre `cmd.exe`/`bash`); fecha classe de risco de injeção via metacaracteres de shell; modelo aprende um único dialeto de comando (argv), não dois por SO | Confirmado com usuário |
| D-sfttrainer-v2 | Dataset é pré-tokenizado (com máscara de loss por segmento, D7) via `build_pretokenized_dataset` ANTES de entregar ao `SFTTrainer`; `TrajectoryDataCollator` virou um collator de padding puro, não tokeniza mais nada | `trl.SFTTrainer` (>= 1.5) tokeniza o dataset internamente procurando uma coluna `"text"` (`KeyError: 'text'` confirmado no Colab) A MENOS QUE o dataset já tenha `input_ids` — nesse caso pula sua própria tokenização. É esse mecanismo que usamos para garantir que é a NOSSA máscara de loss que vale, não uma lógica genérica do trl que não conhece a gramática canônica | Confirmado rodando no Colab + testado localmente de ponta a ponta (`tests/unit/test_data_collator.py`) contra um `SFTTrainer` real |
| D-bestcheckpoint | `TrainingArguments` ganha `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False` | Primeiro treino real (8B, 90 passos) mostrou a `validation loss` atingindo o mínimo no passo 60 (fim da 2ª época) e subindo levemente nos passos 75/90 — overfitting leve no fim do treino num dataset pequeno. Sem isso, o adapter salvo seria sempre o do ÚLTIMO passo, não o de melhor generalização | Confirmado com dado real de treino no Colab |
| D-oom-eval | `TransformersModelRunner.from_loaded(model, tokenizer)` — construtor alternativo que reaproveita um modelo/tokenizer JÁ carregados em memória, sem chamar `from_pretrained` de novo | Carregar uma segunda cópia do modelo para avaliação (células 14/16) enquanto o `model` do treino ainda ocupa quase toda a VRAM da L4 causava `OutOfMemoryError` — confirmado rodando no Colab. Reaproveitar o mesmo objeto em memória (já com os melhores pesos, via `load_best_model_at_end`) elimina o carregamento redundante | Confirmado rodando no Colab; testado localmente (`tests/unit/test_transformers_runner.py::test_from_loaded_reuses_existing_model_without_reloading`) |
| D-oom-eval-2 | Célula de avaliação (14) libera explicitamente `trainer` (`del trainer`, `model.zero_grad(set_to_none=True)`, `gc.collect()`, `torch.cuda.empty_cache()`) antes de gerar | Mesmo já reaproveitando o `model` via `from_loaded` (D-oom-eval), o objeto `trainer` (otimizador `paged_adamw_8bit`, gradientes, estados intermediários) continua ocupando quase toda a VRAM depois de `trainer.train()` retornar — o PyTorch caching allocator não libera isso sozinho. `OutOfMemoryError` durante a própria geração (desquantização 4-bit do bitsandbytes), confirmado no Colab | Confirmado rodando no Colab. **Não** testado localmente de ponta a ponta (exige pressão de memória real de GPU pós-treino que o ambiente de dev, sem GPU, não reproduz) — o padrão (liberar otimizador + `empty_cache`) é prática padrão bem estabelecida do PyTorch, mas fica registrado como não 100% verificado por mim antes de chegar ao usuário |
| D-usecache | `TransformersModelRunner.generate()` passa `use_cache=True` explicitamente em toda chamada a `model.generate()` | `gradient_checkpointing_enable()` (treino) costuma deixar `model.config.use_cache=False`; sem o cache de atenção, cada token gerado recalcula a sequência inteira do zero — mais lento E consumindo cada vez mais VRAM ao longo de uma única geração (VRAM subindo até quase o teto da L4 durante os probes, confirmado no Colab). Passar `use_cache=True` explicitamente na chamada sempre sobrepõe o que ficou configurado no `model.config` pelo treino | Testado localmente com modelo real (CPU) — comportamento correto preservado; o ganho de memória/velocidade em si só se confirma rodando de novo no Colab |
| D-maxtokens | `AgentLoopConfig` ganha `max_tokens_per_step: int = 256`, encaminhado em toda chamada `model_runner.generate(..., max_tokens=config.max_tokens_per_step)` | Mesmo com `use_cache=True` (D-usecache), um modelo que não emite a stop-sequence de forma limpa gera até o teto padrão do backend (1024) em cada um dos até `max_steps` passos — o contexto acumulado pode estourar VRAM no *prefill* de uma chamada seguinte (`OutOfMemoryError` de ~6.67 GiB numa única operação de atenção, confirmado no Colab). 256 é generoso para as trajetórias curtas do dataset desta geração | Testado localmente (`tests/integration/test_agent_loop.py::test_max_tokens_per_step_is_forwarded_to_model_runner`) — confirma que o valor chega até o `ModelRunner`, não que resolve o OOM real (só se confirma rodando de novo no Colab) |
| D-train-prompt-mask | `build_pretokenized_dataset` recebe a trajetória COMPLETA (`{system_prompt, user_request, raw_text}`), não só `raw_text`; monta o mesmo prefixo de `Trajectory.render_for_model()` (harness/trajectory.py) e passa `prefix_len` para `compute_loss_mask` mascarar tudo antes de `raw_text` (além do `<tool_result>`, D7) | **Bug crítico** encontrado ao investigar por que o adapter treinado (3 épocas, 90 passos, 3/3 probes OK) ignorava o pedido do usuário em prompts fora dos 3 probes testados: alucinava `<final>` sem nenhum `<tool_call>` em 3 de 5 formulações variadas, e degenerava num loop de repetição de `<think>` sem fechar após um `<tool_result status="error">` na 4ª. Causa raiz: a célula 22 do notebook só passava `ex["trajectory"]["raw_text"]` para `build_pretokenized_dataset` — o dataset de treino nunca incluía `system_prompt`/`user_request`, então o modelo nunca aprendeu o formato de prompt que `run_agent_loop`/`Trajectory.render_for_model() `de fato usa na inferência (`{system_prompt}\n\n[USER]\n{user_request}\n\n[ASSISTANT]\n`). O treino de 90 passos que passou nos 3 probes ainda assim funcionou nesses casos por coincidência de formulação, não porque o mecanismo estava correto — **invalida os adapters treinados antes desta correção**, exige retreino completo do zero | Testado localmente com um tokenizer real (`tests/unit/test_data_collator.py::test_pretokenized_dataset_masks_system_prompt_and_user_request`) — confirma que o texto do prefixo nunca aparece nos tokens com loss ativo; o comportamento correto do modelo treinado só se confirma retreinando no Colab |
| D-probe-system-prompt | `PRAXIS_SYSTEM_PROMPT` centralizado em `src/harness/system_prompt.py` (exportado por `src.harness`), usado por `scripts/generate_dataset.py` (geração do dataset) E pelos 4 probes (`src/evaluation/probes/*.py`) — fonte única de verdade | Depois de D-train-prompt-mask, retreinar e rodar os 3 probes no Colab deu uma REGRESSÃO (3/3 `PASS`→`FAIL`, muito mais lentos: 135–384s vs 4–75s antes). Causa: todos os 4 probes chamavam `run_agent_loop(user_request, "system", ...)` — `"system"` era um placeholder literal (herdado do padrão usado em `tests/integration/test_agent_loop.py`, onde é inofensivo porque `ScriptedModelRunner` ignora o conteúdo do prompt), nunca o prompt real do dataset. Antes de D-train-prompt-mask isso não importava (modelo não treinava com `system_prompt` de jeito nenhum); depois, virou um mismatch de formato NOVO, só que na avaliação, não no treino — o adapter treinado pode estar correto, era o teste que usava um prefixo que o modelo nunca viu. Célula 34 do notebook tinha uma variante parecida (prompt mais curto, faltando a instrução de `<think>`/`<tool_call>`/`<final>`), também corrigida | Testado localmente (`tests/unit/test_probes.py::test_probe_sends_real_praxis_system_prompt_not_placeholder`, com um runner que grava o prompt recebido) — confirma que os 3 probes testáveis mandam o prompt real; a melhora efetiva do resultado dos probes só se confirma rodando de novo no Colab |
| D-generalization-gap | Nenhuma mudança de código — decisão é diagnóstica: teste controlado no mesmo adapter (pós D-train-prompt-mask + D-probe-system-prompt) confirmou que o mecanismo de treino está correto; o gargalo restante é diversidade de frases no dataset. Motiva a expansão por paráfrase (seção 8.6) em vez de mais correções de pipeline | Mesmo adapter, duas condições: (1) frase EXATA de um exemplo do dataset (`aug-authoring-add.json`, task_type `test_authoring`) → trajetória perfeita (`write_file`→`checker`→`<final>` genuíno, 3 passos, não forçado, conteúdo do arquivo idêntico ao do exemplo de treino); (2) 5 frases parafraseadas da mesma família de tarefa (criar arquivo + validar) → 5/5 `MAX_STEPS_EXCEEDED`, nenhum `<tool_call>` bem formado. Auditoria de `data/train`+`data/validation` (556 exemplos) por `task_type` mostra a causa provável: várias categorias de múltiplos passos têm só 1–2 gabaritos de frase distintos repetidos dezenas de vezes (`model_fixes_after_error`: 41 exemplos/1 template; `checker_rejects_code`: 41/1; `debugging`: 16/1; `code_explanation`: 32/2; `compiles_successfully`: 26/2; `language_migration`: 23/2) — o modelo decorou a forma da frase, não generalizou a gramática de `<tool_call>` para reformulações | Confirmado com teste real no Colab, mesmo adapter, duas frases controladas; contagem de templates por `task_type` gerada localmente a partir dos arquivos reais em `data/` |
| D-generalization-gap-resolved | Nenhuma mudança de código — confirmação de que a expansão por paráfrase (seção 8.6, 556→1.592 exemplos) resolveu a lacuna medida em D-generalization-gap | Retreino completo com o dataset expandido (mínimo de validation loss 0.0864 no passo 165/252, contra 0.0919/75 e 0.1310/60 das rodadas anteriores — melhora monotônica a cada correção). As mesmas 5 frases parafraseadas que davam 0/5 sucesso antes agora dão **5/5** — `write_file`→`checker`→`<final>` genuíno, arquivo real no workspace, sem alucinação. Sinal mais forte: no prompt 4, o modelo bateu no mesmo erro de schema já visto antes (`checker operation="run"` sem `entrypoint`) e **se autocorrigiu** na tentativa seguinte — evidência de generalização real (recuperação de erro aprendida), não memorização de trajetória | Confirmado com teste real no Colab, mesmo conjunto de 5 frases usado para diagnosticar o problema original, adapter retreinado com o dataset expandido |
| D-probe-paraphrase-generalization | Novo probe `probe_paraphrase_generalization` (`src/evaluation/probes/`) — roda a MESMA tarefa (criar arquivo + validar) em 5 formulações diferentes e falha se o `<final>` alega sucesso mas o arquivo esperado não existe de verdade no workspace do sandbox; registrado em `scripts/run_eval.py` (modo demo) e na célula de probes do notebook (modo real) | Os 5 prompts que expuseram D-generalization-gap viviam só numa célula manual do notebook — não rodavam automaticamente, e nenhum dos 4 probes originais checa o estado real do workspace contra a alegação do `<final>` (só a estrutura da trajetória), então uma regressão futura como essa passaria despercebida até alguém testar manualmente de novo (quase aconteceu duas vezes nesta geração: D-train-prompt-mask e D-probe-system-prompt). Formaliza a checagem como parte permanente da avaliação | Testado localmente (`tests/unit/test_probes.py`) e validado ponta a ponta via `python scripts/run_eval.py --demo` (9/9 probes, incluindo os 5 cenários de paráfrase) |
| D-cuda-fragmentation | Célula 4 do notebook (verificação de GPU, primeira célula que roda em toda sessão, antes de qualquer import de torch) define `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` | `OutOfMemoryError` confirmado no Colab numa sessão longa de avaliação manual (muitas chamadas de `generate()` seguidas: probes, prompts variados, testes ad-hoc): pediu 9.64 GiB, só 4.62 GiB livres, mas **9.63 GiB "reservados mas não alocados"** pelo caching allocator do PyTorch — fragmentação de memória acumulada ao longo da sessão, não falta real de VRAM. Só funciona se definida antes da inicialização do CUDA nesse processo — daí ficar na primeira célula executável, não perto de onde o problema aparece | Testado localmente (import não falha em ambiente sem GPU — `os.environ.setdefault` é uma chamada segura mesmo fora do Colab); o efeito real sobre fragmentação só se confirma numa sessão longa nova no Colab |
| D-tool-recovery-pilot | 5 exemplos novos (`data/train/gen-toolrecovery-*.json`, `task_type: tool_unavailable`) gerados por `scripts/gen_tool_recovery_pilot.py` — ensinam recuperação de erro de ferramenta: `web_search` falha de verdade (sem `PRAXIS_OLLAMA_SEARCH_API_KEY` neste ambiente, reproduzindo o bug real), o modelo reconhece que é falha de ambiente (não do código), implementa a tarefa direto sem depender da busca, valida com `checker` de verdade, e responde um `<final>` limpo — sem copiar detalhe interno (nome de variável de ambiente, referência a seção do PLAN.md) | Endereça o risco registrado (seção 17): um teste real (pedido de código de FFT) mostrou o modelo desistindo e vazando a mensagem de erro interna verbatim no `<final>` quando `web_search` falhava. Diferente da expansão por paráfrase (D-generalization-gap-resolved), aqui a trajetória é NOVA — por isso, seguindo a decisão de usar "destilação de raciocínio" só de forma cirúrgica (não em massa), todo `<tool_call>` foi executado de verdade via `ToolExecutorRegistry`/`SandboxContext` (nunca fabricado por texto); um `stderr` de aviso de depreciação do `pytest-asyncio` (dependência só da minha máquina local, não do projeto) vazando um caminho de arquivo Windows foi encontrado e sanitizado antes de aceitar o lote | Testado localmente: os 5 `checker`s passaram de verdade (asserts no próprio script de geração), suite completa de testes (135) sem regressão, e verificação automatizada de que nenhum `<final>` contém string de detalhe interno (`PRAXIS_OLLAMA`, `PLAN.md`, caminho local). Efeito real sobre o comportamento do modelo treinado só se confirma retreinando |
| D-runner-quantization | `TransformersModelRunner.__init__` ganha `quantization_config: Optional[Any] = None`, repassado para `AutoModelForCausalLM.from_pretrained(..., quantization_config=quantization_config)` — o chamador decide o valor (normalmente o mesmo `BitsAndBytesConfig` 4-bit do treino), o módulo não depende de `TrainConfig` | Achado ao planejar um teste numa sessão NOVA do Colab (sem treinar, só recarregando o adapter salvo): o construtor direto (`TransformersModelRunner(model, adapter_path=..., device="cuda")`, o "fast path" recomendado quando não há `model` já em memória via `from_loaded`) carregava o Granite-4.1-8B **sem quantização de 4-bit** — precisão cheia de um modelo de 8B não cabe nos 24GB da L4 (`OutOfMemoryError` esperado já no carregamento, antes de qualquer geração). Isso nunca apareceu antes porque toda avaliação anterior reaproveitava (`from_loaded`) um `model` que já tinha sido carregado em 4-bit pela célula de treino — o "fast path" nunca tinha sido exercitado numa sessão realmente do zero | Testado localmente (`tests/unit/test_transformers_runner.py`, `quantization_config=None` por padrão não quebra o caminho existente) — o efeito real de VRAM só se confirma carregando de verdade com `BitsAndBytesConfig` numa sessão nova do Colab |
| D-tool-success-pilot | 5 exemplos novos (`data/train/gen-toolsuccess-*.json`, `task_type: multi_tool_call`) gerados por `scripts/gen_tool_success_pilot.py` — complementam D-tool-recovery-pilot cobrindo o caminho de SUCESSO da busca: `web_search` funciona (via `MockSearchBackend` com fixtures reais coletadas por busca de verdade, congeladas em `data/fixtures/web_search/`), o modelo sintetiza o que encontrou, implementa com `write_file` e valida com `checker` de verdade, e responde um `<final>` limpo em português | Depois de D-runner-quantization e D-cuda-fragmentation resolverem os bugs de infra, um teste real (pedido de FFT) com `web_search` **funcionando de verdade pela primeira vez** (chave configurada) revelou uma lacuna nova: o modelo pesquisou, achou a doc certa, mas nunca chamou `write_file`/`checker` — só respondeu com texto em inglês terminando em "Example code follows." sem nenhum código. O dataset (incluindo D-tool-recovery-pilot) só tinha exemplos de busca FALHANDO; nenhum de busca bem-sucedida ainda assim exigindo produzir e validar um artefato real | Testado localmente: os 5 `checker`s passaram de verdade (asserts no script de geração), suite completa (135 testes) sem regressão, verificação automatizada de que nenhum `<final>` tem texto em inglês. Efeito real sobre o comportamento do modelo só se confirma retreinando |
| D-tool-pilots-phrasing | 40 variantes novas (`scripts/expand_tool_pilots_phrasing.py`, 4 por exemplo-base × 10 exemplos-base de D-tool-recovery-pilot/D-tool-success-pilot) — reaproveitam os MESMOS fragmentos `<tool_call>`/`<tool_result>` já executados de verdade (extraídos via `parse_segments`, nunca reescritos), variando só `user_request` e a prosa de `<think>`/`<final>` ao redor | Revisão própria antes do retreino: os 10 exemplos-base tinham `<think>`/`<final>` de fechamento quase idênticos entre si (ex.: "A implementação está pronta. Vou validar com o checker." repetido nos 5 de cada categoria) — o mesmo padrão de gabarito repetido que causou D-generalization-gap, dessa vez na prosa de resolução em vez do pedido do usuário. Corrigido ANTES de gastar um ciclo de retreino descobrindo isso de novo | Testado localmente: comparação byte a byte confirma que os fragmentos de `tool_call`/`tool_result` são idênticos ao original em cada variante; verificação automatizada de que nenhum `<final>` das 40 variantes vaza detalhe interno; suite completa (135 testes) sem regressão |
| D-oop-error-patterns-modules-search-shell-expansion | Expansão planejada em `docs/plan_dataset_expansion_oop_shell.md` e executada em 2026-07-08: 765 exemplos novos (85 tarefas-base × 8-9 variantes de frase, hand-written) em 6 categorias — Classes/OOP (135, `gen-oop-*`, `task_type: class_implementation`), Tratamento de erro (135, `gen-error-*`, `task_type: error_handling`), Padrões Python — decorator/generator/context manager (135, `gen-pattern-*`, `task_type: python_pattern`), Módulos realistas (135, `gen-module-*`, `task_type: realistic_module`, com domínio brasileiro sutil: CNPJ, IMC, R$, feriados nacionais, telefone), Busca-em-inglês (175 com o piloto anterior somado, `gen-ensearch-*`, `task_type: multi_tool_call`, 15 novas fixtures reais em `data/fixtures/web_search/` coletadas via busca de verdade, incluindo o caso real que motivou a categoria — bolinha em pygame) e Shell real (90, `gen-shell-*`, `task_type: shell_command`, `scripts/gen_shell_real_pilot.py`/`scripts/expand_shell_real_phrasing.py`) | Endereça a lacuna descrita em `plan_dataset_expansion_oop_shell.md` seção 1: pedido real de "classe Python com soma/subtração/multiplicação/divisão/fatorial" resultou em funções soltas (nenhuma `class`), sem chamar `checker`, com `<final>` alegando falsamente que a classe foi criada; e os dois casos reais de busca bem-sucedida em inglês (FFT, pygame) onde o modelo respondeu em inglês sem produzir código. Todo `tool_result` vem de execução real (`ToolExecutorRegistry`/`SandboxContext`; `MockSearchBackend` com fixtures coletadas via `WebSearch` de verdade; `shell` real contra um repositório git inicializado por script de setup, já que `git init` não é um subcomando permitido — só leitura). Identificadores de código em inglês em todas as categorias; tom de "ferramenta de dev profissional" nos textos em português; categoria de busca inclui `<think>` explícito reconhecendo o descompasso de idioma antes do `<final>` | Testado localmente: os 85 `checker`s/passos de shell das tarefas-base passaram de verdade (asserts nos scripts de geração); checagem automatizada de similaridade (`scripts/check_phrasing_similarity.py`) não encontrou repetição de template real (só falsos positivos esperados: `user_request` da 1ª variante naturalmente próxima do original, e prosa curta em 2 tarefas de shell de 1 passo); checagem de vazamento (`scripts/check_leak_new_batch.py`) confirma 0 ocorrências de detalhe interno nas 765 trajetórias; suite completa (134/135 — a falha restante é `ImportError` do `trl` por um problema de codec do Windows nesta máquina, préexistente e não relacionado a esta mudança). `ruff` está na allowlist do shell mas não é dependência instalada neste ambiente (não está em `requirements.txt`), por isso não apareceu nas 10 tarefas de shell desta rodada — gap conhecido, não bloqueante. Efeito real sobre o comportamento do modelo (classes de verdade, tratamento de erro, disciplina de `checker`, resposta em português mesmo com busca em inglês) só se confirma retreinando no Colab |
| D-retrain-oop-expansion-results | Retreino completo em 2026-07-08 com o dataset expandido (2.155 exemplos, `D-oop-error-patterns-modules-search-shell-expansion`), 3 épocas/405 passos, melhor checkpoint no passo 120 (`eval_loss=0.0951`) — abaixo do platô já esperado, sem melhora depois disso. Suite de 25 probes (baseline + 10 paráfrases + 7 direct-vs-tool + 5 anti-fabricação): 23/25 (92%), incluindo 5/5 em anti-fabricação mesmo sob prompts adversariais diretos ("finja que criou", "diga que passou sem rodar"). Teste manual dos 5 casos que motivaram a expansão (classe Calculator, FFT, pygame, validate_age, shell/pytest): a classe Calculator agora sai correta (`class`, métodos certos, `ValueError` nos casos de erro) e o caso FFT teve sucesso ponta a ponta idêntico ao padrão treinado (busca falha → reconhece ambiente → implementa → valida → `<final>` honesto). Em nenhum dos 5 casos o modelo alegou sucesso falso, mesmo quando `forced_final` bateu 3 vezes | As duas falhas do probe suite (1 paráfrase com gíria pesada — "printa"/"beleza"/"tá válido" — e 1 caso direto muito curto com a palavra "comando") são esperadas: fora do registro de tom que decidimos treinar (seção 3 do plano de expansão), não indicam regressão. Mais relevante: no teste manual do caso Calculator, o modelo **tentou fabricar um `<tool_result>`** depois de duas falhas reais de truncamento no `checker` — e o harness bloqueou isso em tempo real (`TOOL_RESULT_FABRICATION`), a primeira vez que essa defesa disparou numa situação orgânica (não um probe sintético). A causa da truncagem foi identificada: `max_tokens_per_step=256` é pequeno demais para uma chamada de `checker` que embute uma classe inteira + arquivo de teste inteiro como JSON (payload muito maior que as trajetórias curtas da geração anterior de dataset) — motivou `D-maxtokens-oop`. Também reveladas duas lacunas comportamentais reais (não de fabricação): diagnóstico errado de causa-raiz num teste pygame com bug de import próprio (culpou "ambiente" em vez do próprio código gerado) e ambiguidade de framework de teste no caso shell (`unittest` com sintaxe de `pytest`, combinação inválida) — ambas geraram loop de chamada repetida detectado e cortado com segurança pelo `seen_calls`, nunca falso sucesso | Testado localmente com o modelo real carregado no Colab (não probe sintético) — trajetória completa inspecionada manualmente para os 5 casos motivadores; suite de 25 probes rodou sem exceção de infraestrutura (`infrastructure_failure_rate: 0.0`) |
| D-maxtokens-oop | `AgentLoopConfig.max_tokens_per_step` sobe de `256` para `512` (`src/harness/loop.py`) | Confirmado em `D-retrain-oop-expansion-results`: uma chamada de `checker` com uma classe inteira + teste inteiro embutidos no JSON (categoria OOP da expansão de hoje) truncava no meio em 256 tokens (`UNTERMINATED_TAG`), levando a falhas repetidas e uma tentativa real de fabricação de `<tool_result>` (bloqueada pelo harness). 512 dá margem para esses payloads maiores sem reabrir o risco de OOM que motivou o teto original (`D-maxtokens`) — é um aumento moderado, não a remoção do teto | Testado localmente: suite completa (126/126, excluindo o teste do `trl` já sinalizado como pré-existente) sem regressão, incluindo `tests/integration/test_agent_loop.py::test_max_tokens_per_step_is_forwarded_to_model_runner`. Efeito real sobre o caso Calculator (se o `checker` completa sem truncar) só se confirma retestando no Colab |
| D-maxtokens-oop-confirmed | Nenhuma mudança de código — confirmação de que `D-maxtokens-oop` (512) resolveu o caso Calculator; os 5 casos motivadores foram reexecutados no Colab (modelo real, `checkpoint-120`) com `max_tokens_per_step=512` hardcoded na chamada | Calculator: agora 3 passos, sem truncamento, sem tentativa de fabricação, `checker` passa limpo, `<final>` honesto — a causa raiz estava correta. FFT e validate_age seguem 100% (sem mudança, não dependiam do teto de tokens). pygame: ainda `forced_final`, mas por um motivo NOVO e mais profundo — o teste chamou `main()` (que contém `while running: ...` só encerrado por um evento `QUIT` real) direto num ambiente headless, travando até o timeout do `checker` (`TIMEOUT`, 15s); o autodiagnóstico do modelo ("falhou por timeout, não por causa do código") está **errado** desta vez — é um problema real de testabilidade (código com loop bloqueante chamado direto num teste), não coberto por nenhuma categoria do dataset atual, e diferente do bug de import visto na rodada anterior (que não se repetiu). Shell: repetiu o MESMO erro de antes (`python -m unittest` com teste estilo `pytest`, combinação inválida) — confirma que essa lacuna é de fato independente do teto de tokens, não resolvida pelo `D-maxtokens-oop` | Testado localmente com o modelo real no Colab (não probe sintético), mesma inspeção manual de trajetória completa da rodada anterior. Nenhuma tentativa de fabricação em nenhum dos 5 casos, mesmo nos que deram `forced_final` |
| D-oop-expansion-round2-tests | Nenhuma mudança de código — 8 casos novos testados manualmente no Colab (modelo real, `checkpoint-120`, `max_tokens_per_step=512`), cobrindo categorias ainda não exercitadas na rodada anterior: OOP (Stack, e uma paráfrase do Calculator com frase bem distante da de treino — "Meu chefe pediu uma calculadora orientada a objetos... Pode fazer?"), decorator (memoize), context manager (temporary_override), módulo realista com domínio BR (IMC, CNPJ), exceção customizada (OutOfStockError) e shell (criar README + listar workspace) | 8/8 completaram com sucesso (`checker` passou em todos os casos que validavam código; nenhum `forced_final`), zero tentativa de fabricação. Dois sinais fortes de generalização real, não decoreba: (1) a paráfrase do Calculator, com frase informal e estrutura nunca vista, produziu a classe correta e validada; (2) no caso de shell, o modelo alucinou um binário inexistente (`"workspace list"`), a allowlist do sandbox bloqueou corretamente (`UNSAFE_COMMAND`), e o modelo se recuperou sozinho usando `ls -la` — defesa em profundidade funcionando e recuperação de erro bem-sucedida numa combinação não vista em treino. Um hiccup pontual de geração malformada no caso do decorator (`<think>` mal fechado, capturado como `UNTERMINATED_TAG`) se autocorrigiu na tentativa seguinte sem cair em loop. Ressalva: a maioria dos 8 casos são variações próximas de tarefas-base literalmente treinadas hoje (mesma família, frase diferente) — só a paráfrase do Calculator e o binário alucinado no shell são sinais fortes de generalização fora da frase treinada; os demais confirmam retenção sólida da categoria, não generalização a território novo | Testado localmente com o modelo real no Colab (não probe sintético), inspeção manual de trajetória completa dos 8 casos |
| D-oop-expansion-round3-ood-tests | Nenhuma mudança de código — 9 casos deliberadamente FORA de qualquer tarefa-base do dataset (domínio novo: classe Book/biblioteca, validação de CEP, generator de números primos, juros compostos, combinação OOP+exceção customizada nunca treinada junta, troca para Go, pedido de API FastAPI fora de escopo, validador de Sudoku, pergunta conceitual direta) | 7/9 sucesso completo com `checker` passando, incluindo domínios nunca vistos no dataset (Book, CEP, primos, juros compostos, LibraryCatalog+BookNotFoundError) — forte evidência de generalização de COMPORTAMENTO (não memorização de domínio). Achado mais importante: **FastAPI funcionou de ponta a ponta** (app real + `TestClient` real + `checker` passou de verdade) — confirma a hipótese discutida em conversa de que a disciplina agente (write_file→checker→validar) generaliza para bibliotecas nunca treinadas, apoiando-se no conhecimento de pré-treino do Granite base para a sintaxe específica. Duas falhas reais, ambas informativas: (1) **Go** — código correto gerado de primeira, mas `checker` falhou por `MISSING_DEPENDENCY: toolchain 'go' não disponível` nesta sessão do Colab (falha de infraestrutura, não do modelo); o modelo diagnosticou CORRETAMENTE que era problema de ambiente, mas ao contrário do padrão aprendido para `web_search` (seguir sem a busca), aqui só repetiu a mesma chamada de `checker` até `MAX_STEPS_EXCEEDED` — nunca aprendemos a ensinar "se o CHECKER falhar por infra, dê a resposta final com o código já escrito, explicando a limitação", só ensinamos isso para `web_search`; (2) **Sudoku** — o código gerado tinha uma sequência de escape inválida (`\` solto antes de espaços, quebrando o JSON do `tool_call`), e o modelo, repetidamente, culpou "a ferramenta de escrita"/"o ambiente" em vez de reconhecer que o próprio código tinha um defeito de formatação — o mesmo padrão de má atribuição de causa raiz visto no caso pygame da rodada 1. Achado secundário: no caso LibraryCatalog, o modelo pulou `write_file` e foi direto para `checker` com o conteúdo embutido — validou de verdade, mas o arquivo nunca foi persistido no workspace (`Arquivos no workspace: []`), diferente do padrão sempre-escreve-depois-valida ensinado em todas as tarefas-base. Nenhuma fabricação em nenhum dos 9 casos, mesmo nos dois que deram `forced_final` | Testado localmente com o modelo real no Colab (não probe sintético), inspeção manual de trajetória completa dos 9 casos. Os dois gaps (recuperação de falha de `checker`/infra, má atribuição de causa raiz em código próprio malformado) são candidatos concretos para uma futura rodada de dataset, mas não são risco de fabricação — em ambos os casos o modelo terminou honesto (`MAX_STEPS_EXCEEDED`), nunca alegou sucesso falso |
| D-error-recovery-gap-investigation | Nenhuma mudança de código — investigação dos dois gaps de `D-oop-expansion-round3-ood-tests`, documentada em `docs/plan_dataset_expansion_error_recovery.md`: não é ausência total de exemplo, é insuficiência de profundidade em dois `task_type` já existentes | `invalid_call_then_correction` (10 exemplos) já ensina "JSON malformado → corrigir e tentar de novo", mas só com typos triviais de uma linha, **nenhum executado de verdade** (`execution_performed: False` nos 10), 400-700 caracteres — o bug real do Sudoku (regex + loop aninhado, ~15 linhas) é uma complexidade nunca exercitada nesse task_type. `test_fails` (20 exemplos) ensina só "reportar falha honesta quando o teste falha genuinamente" — **nenhum dos 20 tem mais de 1 passo**, ou seja, nenhum ensina diagnosticar QUAL arquivo (implementação vs. teste) tem o bug real quando o `checker` recebe múltiplos arquivos — exatamente o gap do caso pygame (bug estava no teste, modelo resubmeteu a implementação). Achado incidental fora de escopo: `data/train/aug-testfail-absolute.json` (dataset original) tem um vazamento real de caminho Windows no stderr do aviso de depreciação do pytest-asyncio, não sanitizado — registrado para correção futura, não faz parte deste plano | Investigação feita via grep/contagem direta nos arquivos de `data/train`, resultados reproduzíveis |
| D-error-recovery-expansion | Plano executado em 2026-07-08 (mesmo dia do planejamento — créditos de Colab limitados antes da renovação, geração/validação local consolidada numa leva só): 100 exemplos novos em 4 categorias — Recuperação de falha de `checker` por infra (30, `gen-checkerinfra-*`, `task_type: checker_infra_unavailable`, usando `checker(language="javascript"/"typescript")` → `UNSUPPORTED_LANGUAGE` real, mecanismo mais estável que depender de uma toolchain ausente específica do Colab), JSON malformado em código complexo (20, `gen-jsonrecovery-*`, `task_type: tool_call_json_recovery`, primeira chamada processada pela MESMA função `src.parsers.parse_last_segment` usada pelo harness em produção — `TOOL_CALL_PARSE_ERROR` real, não fabricado — seguida de correção e `checker` reais), diagnóstico de arquivo com bug real entre vários (20, `gen-multifile-*`, `task_type: multi_file_root_cause_diagnosis`, bug real só no arquivo de teste — import faltando, nome errado, argumento a mais, variável com typo — `checker` real aponta o erro, trajetória ensina identificar o arquivo certo antes de corrigir) e tool-choice em pergunta curta com palavra-armadilha (30, `gen-trapword-*`, `task_type: direct_answer`, sem ferramenta, perguntas com "comando"/"rodar"/"executar"/"arquivo" que só pedem conceito) | Endereça os 3 gaps comportamentais achados nos testes pós-retreino de hoje (`D-oop-expansion-round3-ood-tests`): recuperação de falha do `checker` por infraestrutura sem repetir a mesma chamada (só tínhamos esse padrão para `web_search`); diagnóstico de causa raiz em JSON malformado de código real mais complexo (os 10 exemplos antigos eram triviais e nunca executados); diagnóstico de qual arquivo tem o bug quando o `checker` recebe vários (nenhum exemplo anterior tinha mais de 1 passo em `test_fails`); e o miss pontual do probe #20 (tool-choice em pergunta curta com "comando"). Vocabulário de recuperação deliberadamente neutro em todos os casos ("o JSON da minha chamada estava malformado" / "o bug está no arquivo de teste"), evitando a linguagem de culpar "a ferramenta"/"o ambiente" observada nos casos reais | Testado localmente: todas as 20 tarefas-base passaram com execução real (`UNSUPPORTED_LANGUAGE` real via dispatch do checker, `TOOL_CALL_PARSE_ERROR` real via `parse_last_segment`, bugs reais de teste capturados pelo `checker`); checagem de similaridade (`scripts/check_phrasing_similarity.py`) não encontrou repetição de template nova (só o mesmo padrão de falso positivo já aceito em rodadas anteriores); checagem de vazamento (`scripts/check_leak_new_batch.py`) confirma 0 ocorrências em 866 trajetórias; suite completa (134/134, excluindo o teste do `trl` já sinalizado como pré-existente). Efeito real sobre o comportamento do modelo só se confirma retreinando no Colab |
| D-hypothesis-revision-expansion | Plano executado em 2026-07-08 (`docs/plan_dataset_expansion_hypothesis_revision.md`), consolidado na mesma leva por créditos de Colab limitados: 20 exemplos novos (`gen-hyprevision-*`, `task_type: hypothesis_revision_after_failure`) em 4 tarefas-base (`sum_up_to`, `is_in_range`, `last_n_items`, `dedup_keep_order`), cada uma com TRÊS rodadas reais de `checker` — falha real, correção com teoria plausível que TAMBÉM falha de verdade, e só na terceira rodada a causa correta e o sucesso real | Motivado pelo pior sinal do CoT observado hoje (casos Go e Sudoku, `D-oop-expansion-round3-ood-tests`): o modelo repetiu a MESMA chamada até `MAX_STEPS_EXCEEDED` sem nunca revisar a própria hipótese. Diferente dos Gaps 2a/2b (que sempre acertam na 2ª tentativa), aqui a trajetória ensina explicitamente "minha correção anterior não resolveu — a hipótese estava errada" antes de propor a causa real. Os quatro bugs/teorias-erradas/causas-corretas foram verificados com execução Python direta antes de escrever os exemplos completos, para garantir que as duas falhas fossem genuínas e não coincidissem por acidente em sucesso prematuro (`count_vowels` foi descartado por esse motivo, substituído por `dedup_keep_order`) | Testado localmente: as 4 tarefas-base passaram com as duas falhas reais + sucesso real confirmados via asserts no script de geração; **achado próprio durante a checagem de similaridade**: a primeira versão das variantes de frase repetia a MESMA frase de diagnóstico (`think2`/`think3`) em todas as 4 variantes de cada tarefa-base (só a abertura/fechamento variava) — exatamente o erro já documentado em `D-generalization-gap`/`D-tool-pilots-phrasing`, pego e corrigido ANTES de aceitar o lote, reescrevendo `think2`/`think3` com 4 formulações genuinamente distintas por tarefa-base; checagem de vazamento confirma 0 ocorrências em 886 trajetórias; suite completa (134/134). Efeito real sobre o comportamento do modelo só se confirma retreinando |
| D-gap1-4-manual-retest | Nenhuma mudança de código nesta entrada — 6 cenários testados manualmente no Colab contra o adapter retreinado com a leva `D-error-recovery-expansion`/`D-hypothesis-revision-expansion` (2.275 exemplos): infra de linguagem não suportada (Rust), reteste literal do caso Go (`MISSING_DEPENDENCY`) que motivou o Gap 4, reteste do caso Sudoku (JSON/escaping), diagnóstico multi-arquivo OOD, trap-word OOD e revisão de hipótese OOD. Duas células novas e permanentes adicionadas ao notebook (célula 16b, este teste; célula 18, exportação do adapter/logs para `outputs-logos-1-v2/` no Drive) | **O pior sinal documentado (`D-oop-expansion-round3-ood-tests`, caso Go em loop até `MAX_STEPS_EXCEEDED`) está resolvido**: o mesmo tipo de falha (`MISSING_DEPENDENCY`) agora é reconhecido em 3 passos, sem repetição, com `<final>` honesto — efeito real do Gap 4 confirmado, não só teórico. Rust (Gap 1) e trap-word (Gap 3) também limpos. Sudoku e a revisão de hipótese OOD não reproduziram a condição de falha original (o modelo evitou o JSON problemático / acertou de primeira) — inconclusivo, não é regressão. **Achado novo e mais sério** no cenário de diagnóstico multi-arquivo: pedi para "descobrir a causa raiz e corrigir" um teste supostamente falhando, mas o código que o próprio modelo escreveu passou de primeira (nunca existiu bug real, falha de desenho do teste, não do modelo) — mesmo assim o `<final>` inventou uma narrativa de diagnóstico e correção ("a soma estava sendo subtraída... corrigi só o teste") que não tem nenhuma base nas chamadas de ferramenta reais da trajetória. Motivou o plano do Gap 5 (`docs/plan_dataset_expansion_confabulation_gap.md`, `D-confabulation-gap-plan`) | Testado com o modelo real no Colab (não probe sintético), trajetória completa em `mode="dev"` inspecionada manualmente para os 6 casos |
| D-confabulation-gap-plan | Nenhuma mudança de dataset ainda — só o plano em `docs/plan_dataset_expansion_confabulation_gap.md` (Gap 5, ~20-24 exemplos, 4 tarefas-base: `is_palindrome`, `count_vowels`, `average`, `is_prime`, cada uma com uma ÚNICA rodada real de `checker` que passa de primeira, sem bug real) | Endereça o achado de `D-gap1-4-manual-retest`: o modelo confabulou uma correção que nunca aconteceu na trajetória real, quando o usuário alegou um bug que na verdade não existia. Categoria nova, distinta de fabricação de `tool_result` (tags reais aqui, o problema é a prosa do `<final>`) e distinta de revisão de hipótese (Gap 4, que pressupõe uma falha real acontecendo) | As 4 implementações propostas foram verificadas por execução Python direta em 2026-07-08 — todas passam de primeira, confirmando que a condição-base da tarefa (sucesso genuíno sem qualquer correção) é reproduzível. Execução completa (tarefas-base + variantes + checagens + retreino) fica para depois da consolidação em andamento — decisão deliberada de não gastar mais créditos de Colab até decidir o que entra no próximo retreino |
| D-oop-round4-ood-tests | Nenhuma mudança de código nesta entrada — 8 cenários novos testados manualmente no Colab (modelo real, mesmo adapter de `D-gap1-4-manual-retest`), 100% fora de qualquer tarefa-base treinada: OOP (CardDeck), exceção customizada (Wallet/InsufficientFundsError), decorator (retry_on_failure), módulo BR (validate_pix_key), shell (git log resumido), trap-word (versão do Python), combinação OOP+exceção nunca treinada junta (TodoList/TaskNotFoundError), e diagnóstico com bug REAL no arquivo de código (não no teste, ao contrário do padrão de Gap 2b treinado) | 7/8 sucesso limpo, com dois sinais fortes de generalização de comportamento (não memorização): (1) `ood_module_pix_key` — o modelo gerou um `<tool_call>` com JSON genuinamente malformado (barra invertida solta) e se recuperou usando exatamente o vocabulário do Gap 2a ("o JSON da minha última chamada estava malformado... erro meu de formatação, não da ferramenta"), a primeira vez que esse gap foi observado se ativando organicamente, não fabricado por um teste desenhado pra isso; (2) `ood_genuine_bug_diagnosis` — bug real no arquivo de código, diagnosticado e corrigido corretamente (`split(' ')`→`split()`), confirmando que a habilidade de diagnóstico generaliza além do padrão "bug sempre no teste" visto em todos os exemplos-base de Gap 2b, e que quando o bug É real o modelo não confabula (ao contrário do achado de `D-gap1-4-manual-retest`, onde não havia bug nenhum). **1 falha real, de natureza nova**: `ood_combo_todo_exception` — o `<think>` de diagnóstico entrou em loop de repetição de token da mesma frase ("não ValueError...") até `UNTERMINATED_TAG`, repetido em 4 tentativas seguidas até `MAX_STEPS_EXCEEDED`. Sem fabricação (harness terminou honesto), mas motivou `D-repetition-loop` (achado de causa raiz: decodificação gulosa sem `repetition_penalty`) | Testado com o modelo real no Colab (não probe sintético), trajetória completa em `mode="dev"` inspecionada manualmente para os 8 casos |
| D-repetition-loop | `TransformersModelRunner.__init__`/`from_loaded` ganham `repetition_penalty: float = 1.15` (`src/inference/transformers_runner.py`), aplicado em `generate()` junto com `do_sample=False`. Dois testes ajustados (`test_generate_stops_exactly_at_stop_sequence`, `test_from_loaded_reuses_existing_model_without_reloading`): a substring de parada usada nos asserts (`"ct"`) parou de aparecer na saída determinística do modelo de pesos aleatórios (`hf-internal-testing/tiny-random-gpt2`) com a penalidade ativa — trocada por `"if"`, que continua confiável | Causa raiz do achado em `D-oop-round4-ood-tests` (`ood_combo_todo_exception`): decodificação gulosa (`do_sample=False`) sem nenhuma penalidade de repetição não tem mecanismo pra escapar de um loop assim que o modelo começa a repetir uma frase — repetir vira, token a token, a opção de maior probabilidade. 1.15 é um valor moderado de propósito: alto o suficiente pra quebrar esse tipo de loop, baixo o suficiente pra não penalizar repetição legítima de código (indentação, `self.`, `def `, chaves de teste repetidas) | Testado localmente: suite completa (134/134, excluindo o teste do `trl` já sinalizado como pré-existente) — os únicos dois testes afetados foram os que dependiam da substring exata de parada no modelo de teste de pesos aleatórios, corrigidos trocando a substring, não a lógica. Efeito real sobre o caso de loop (`ood_combo_todo_exception`) só se confirma reexecutando o mesmo cenário no Colab com o Granite real |
| D-repetition-loop-confirmed | Nenhuma mudança de código — reteste real no Colab de `ood_combo_todo_exception` (mesmo prompt, mesmo adapter, `repetition_penalty=1.15` ativo via `git pull` + `importlib.reload` do módulo, sem reiniciar o runtime) | **Confirmado: a degeneração de repetição de token (`UNTERMINATED_TAG` em cascata) desapareceu por completo** — zero ocorrência no reteste. Em vez disso, surgiu uma falha de natureza DIFERENTE, mais branda: o modelo escreveu código próprio com um bug real (referenciou `Task(name)` sem nunca definir a classe `Task`), diagnosticou a causa ERRADA ("esqueceu de importar Task" — não faz sentido, a classe nunca existiu), e reenviou um `<tool_call>` de `checker` **byte a byte idêntico** ao anterior sem aplicar nenhuma correção real, apesar de dizer "vou corrigir só o teste". A rede de segurança de deduplicação (`seen_calls`) bloqueou a repetição (`MAX_STEPS_EXCEEDED` como erro de ferramenta, não fabricado) nas duas tentativas seguintes, e o modelo desistiu de forma **honesta**: o `<final>` disse explicitamente "não consegui validar esse código" e apresentou o código com a ressalva clara de que não foi validado — nenhuma alegação de sucesso falso. Trocou um problema de decodificação (resolvido) por um problema de raciocínio/diagnóstico já observado antes nesta sessão ("atribuição de causa confiante-mas-errada"), mas sem regressão na disciplina anti-fabricação — candidato de anotação para uma futura rodada de dataset (não urgente, comportamento final continua seguro), não uma correção de código pendente | Testado com o modelo real no Colab (não probe sintético), trajetória completa em `mode="dev"` inspecionada manualmente |
| D-own-bug-diagnosis-plan | Nenhuma mudança de dataset ainda — só o plano em `docs/plan_dataset_expansion_own_bug_diagnosis.md` (Gap 6, ~20-24 exemplos, 4 tarefas-base: `TodoList`/`Task`, `EventBus`/`Subscriber`, `parse_duration`/`_parse_unit`, `to_cents`/`Decimal`, cada uma com um `NameError` real por nome nunca definido/importado no próprio código recém-escrito, seguido de diagnóstico correto e uma correção verificavelmente diferente da tentativa que falhou) | Endereça o achado de `D-repetition-loop-confirmed`: diagnóstico de causa raiz errado num bug de AUTORIA PRÓPRIA (não num arquivo pré-existente como Gap 2b) combinado com uma retentativa que não aplica nenhuma correção real (payload idêntico ao anterior, bloqueado pelo `seen_calls`). Distinto de Gap 4 (lá a segunda tentativa é uma correção real baseada em teoria errada — aqui não há correção nenhuma) e de Gap 5 (lá não existe bug nenhum — aqui o bug é genuíno desde o início) | As 4 combinações bug/correção propostas foram verificadas por execução Python direta em 2026-07-08 — todas produzem `NameError` real na versão com bug e passam de verdade na versão corrigida. Execução completa (tarefas-base + variantes + checagens + retreino) fica para a mesma leva futura do Gap 5 — decisão deliberada de não gastar mais créditos de Colab até decidir o que entra no próximo retreino |
| D-confabulation-gap-expansion | Plano executado em 2026-07-08 (mesmo dia do planejamento, `docs/plan_dataset_expansion_confabulation_gap.md`): 20 exemplos novos (`gen-confab-*`, `task_type: no_bug_found_report`) em 4 tarefas-base (`is_palindrome`, `count_vowels`, `average`, `is_prime`), cada uma com UMA ÚNICA rodada real de `checker` que passa de primeira — o usuário alega um bug que na verdade não existe, e o `<final>` reporta isso com precisão, sem inventar diagnóstico ou correção | Endereça o achado de `D-gap1-4-manual-retest`: o modelo confabulou uma correção que nunca aconteceu quando não havia bug real. Vocabulário central: `<think>` antes do `<final>` reconhece explicitamente que a validação passou sem correção real, e o `<final>` nunca alega ter corrigido algo que não foi alterado — mas também não overclaima "não existe bug nenhum", reconhecendo que pode haver um caso não coberto pelo teste usado | Testado localmente: as 4 tarefas-base passaram de primeira via execução real (`ToolExecutorRegistry`/`SandboxContext`), confirmado por assert no próprio script de geração; **achado próprio durante a checagem de similaridade, aplicando a lição de `D-hypothesis-revision-expansion`**: a primeira versão da variante 1 de cada tarefa reaproveitava quase verbatim o texto de `think`/`final` da própria tarefa-base (prosa 0.92-0.99, sinalizado pela checagem) — corrigido reescrevendo as 4 variante-1 com frases genuinamente distintas antes de aceitar o lote; reverificado, 0 pares de alta similaridade novos (só o padrão de falso positivo já aceito: pedido alto/prosa baixa). Checagem de vazamento confirma 0 ocorrências em 926 trajetórias; suite completa (134/134). Efeito real sobre o comportamento do modelo só se confirma retreinando |
| D-own-bug-diagnosis-expansion | Plano executado em 2026-07-08 (mesmo dia do planejamento, `docs/plan_dataset_expansion_own_bug_diagnosis.md`): 20 exemplos novos (`gen-ownbug-*`, `task_type: own_bug_diagnosis`) em 4 tarefas-base (`TodoList`/`Task`, `EventBus`/`Subscriber`, `parse_duration`/`_parse_unit`, `to_cents`/`Decimal`), cada uma com um `NameError` real (nome nunca definido ou nunca importado no próprio código recém-escrito), diagnóstico correto classificando definição-faltando vs. import-faltando, e uma correção verificavelmente diferente do payload que falhou (assert no próprio script de geração compara os dois payloads) | Endereça o achado de `D-repetition-loop-confirmed`: diagnóstico de causa raiz errado num bug de autoria própria, combinado com uma retentativa que reenviava o mesmo payload sem aplicar correção nenhuma. As 4 tarefas-base cobrem deliberadamente os dois subtipos (classe/função auxiliar faltando DEFINIR nas 3 primeiras, import de biblioteca padrão faltando na 4ª) pra o gap não virar "sempre é uma classe faltando" | Testado localmente: as 4 tarefas-base confirmaram `NameError` real na primeira rodada e sucesso real na segunda, com assert de que os payloads diferem; mesmo achado da checagem de similaridade que `D-confabulation-gap-expansion` (variante 1 reaproveitando texto quase verbatim da tarefa-base em `think`/diagnóstico/`final`) — corrigido da mesma forma, reverificado sem novos pares de alta similaridade. Checagem de vazamento confirma 0 ocorrências em 926 trajetórias; suite completa (134/134). Efeito real sobre o comportamento do modelo só se confirma retreinando |
| D-local-inference-export | Adapter mergeado (`merge_and_unload` do peft, fp16) + convertido para GGUF (`llama.cpp convert_hf_to_gguf.py`) + quantizado Q4_K_M (`llama-quantize`), servido localmente via Ollama (`logos-v2`, GPU AMD 5700 XT/RDNA1, 8GB VRAM) — primeiro teste real de inferência de produção fora do Colab/Transformers | Motivado pela prioridade declarada de tokens/s: decisão de deployment é cliente-servidor (CLI fala HTTP com um servidor de inferência local, escalando depois para GPU dedicada), motor de inferência escolhido é `llama.cpp`/Ollama via backend Vulkan (ROCm não suporta RDNA1 de forma confiável no Windows), f16 sem quantização descartado por não caber em 8GB VRAM (~17.6GB de pesos). Dois problemas reais resolvidos no processo: (1) `tokenizer.save_pretrained()` gravou um `tokenizer_config.json` com `"tokenizer_class": "TokenizersBackend"`, não reconhecido pelo `AutoTokenizer` de nenhuma versão de `transformers` disponível via pip — corrigido copiando os arquivos de tokenizer ORIGINAIS do Granite base direto do hub por cima (merge de LoRA não altera o tokenizer, então os arquivos pristinos são corretos); (2) um `!comando` de shell que falha dentro de uma célula do Colab não levanta exceção Python, então o `print` de sucesso seguinte rodava mesmo com a conversão tendo falhado de verdade — corrigido adicionando `assert Path(...).exists()` depois de cada etapa de exportação, em vez de confiar na mensagem impressa | Velocidade medida com o modelo real: 45-47 tokens/s de geração (comparável/levemente melhor que os 42 tok/s do Granite base puro Q4_K_M, confirmando que merge+quantização não custou performance). Tamanho final do GGUF confirmado por `assert`: 5.12 GB. Comportamento (gramática `<think>/<tool_call>/<final>`) AINDA NÃO validado neste teste — `ollama run` usa o chat template padrão da Ollama, não o formato cru que o harness realmente usa (D2); motivou `D-ollama-model-runner` |
| D-ollama-model-runner | Novo `OllamaModelRunner` (`src/inference/ollama_runner.py`) implementando o mesmo contrato `ModelRunner.generate(prompt, stop, max_tokens)` de `TransformersModelRunner`, batendo na API HTTP local da Ollama com `"raw": true` (ignora o chat template do `Modelfile`, manda o texto cru exatamente como `Trajectory.render_for_model()` produz). Parada por stop-sequence feita no lado do cliente via streaming (checa o sufixo do texto acumulado a cada pedaço), porque o parâmetro nativo `options.stop` da Ollama TRUNCA a tag de fechamento fora do texto — o harness precisa dela incluída (mesmo contrato de `TransformersModelRunner`, testado em `test_generate_stops_exactly_at_stop_sequence`). Script `scripts/test_ollama_local.py` roda `run_agent_loop` real contra um modelo Ollama local (3 cenários: resposta direta, write_file+checker simples, validador de CPF) | Necessário porque `ollama run logos-v2` (D-local-inference-export) testa o template de chat padrão da Ollama, não o formato de treino do harness — sem isso, não dá pra saber se a gramática `<think>/<tool_call>/<final>` sobreviveu ao merge/quantização, só a velocidade bruta. É também, na prática, a implementação real do "Model Runner como cliente HTTP" que `D13` já tinha definido como arquitetura de produção — não é código descartável | Testado localmente com `requests.post` mockado (6 testes novos, `tests/unit/test_ollama_runner.py`): confirma modo `raw`/prompt sem modificação, tag de parada incluída no texto retornado (não truncada como o comportamento nativo da Ollama faria), `stop_reason` correto nos 3 casos (`stop_sequence`/`max_tokens`/`eos`), host customizável, propagação de erro HTTP. Suite completa: 140/140 (134 anteriores + 6 novos). Validação end-to-end real (rodar `scripts/test_ollama_local.py` contra o `logos-v2` de verdade) ainda pendente — próximo passo |
| D-ollama-temperature-zero | `OllamaModelRunner` ganha `temperature: float = 0.0` no construtor, propagado em `options.temperature` de toda chamada — antes, a Ollama usava seu default de amostragem (tipicamente 0.8), não configurado explicitamente | Confirmado num teste real (`scripts/test_ollama_local.py`, cenário `simple_write_and_validate`): a MESMA solicitação, rodada duas vezes, produziu trajetórias diferentes — numa vez o modelo chamou `checker`/`syntax_check` corretamente, na outra inventou `shell`/`python -m syntaxcheck` (módulo inexistente). Sem determinismo, fica impossível separar "a quantização/merge mudou o comportamento" de "essa rodada específica teve sorte/azar de amostragem" — todos os testes de comportamento feitos no Colab usavam decodificação gulosa (`do_sample=False`) por esse exato motivo; o runner da Ollama precisava do equivalente pra a comparação ser válida | Testado localmente: 2 novos testes confirmam que `temperature=0.0` é o padrão e que pode ser sobrescrito explicitamente. Suite completa: 142/142. Efeito real (se a inconsistência observada desaparece com temperatura zero) só se confirma reexecutando `scripts/test_ollama_local.py` |
| D-local-inference-validated | Nenhuma mudança de código — reteste de `scripts/test_ollama_local.py` com `temperature=0.0` (pós `D-ollama-temperature-zero`) contra `logos-v2` (adapter mergeado + Q4_K_M, servido via Ollama na AMD 5700 XT local) | **Confirma o objetivo desta etapa inteira**: com determinismo restaurado, os 3 cenários reproduzem de forma estável — `direct_no_tool` e `simple_write_and_validate` idênticos à primeira rodada boa, e `cpf_validator` completou com `compile_and_test` real (teste com 3 casos escrito e executado de verdade, não só `syntax_check`), `<final>` honesto. A gramática `<think>/<tool_call>/<final>` sobreviveu integralmente ao merge do LoRA + conversão GGUF + quantização Q4_K_M — nenhuma fabricação, nenhum desvio de formato em nenhum dos 3 cenários. Velocidade (45-47 tok/s, `D-local-inference-export`) e comportamento agora validados juntos, não só a velocidade isolada. **Achado real e reproduzível, não urgente**: em `direct_no_tool`, o modelo consistentemente tenta chamar uma ferramenta `calculator` inexistente no registro do projeto antes de se recuperar e responder direto — repetiu-se identicamente com temperatura zero, então não é ruído de amostragem; a recuperação é sempre honesta (nunca fabrica um resultado de calculadora), mas o ideal seria responder direto de cara para aritmética simples, sem tentar ferramenta nenhuma primeiro (como os exemplos de `direct_answer`/trap-word do Gap 3 já ensinam para outros casos). Candidato de dataset para uma futura rodada, não bloqueante. Detalhe inofensivo observado: o `stderr` do `checker` em `cpf_validator` vazou um aviso de depreciação do `pytest-asyncio` da máquina local (artefato de ambiente já documentado, não do modelo) — não vazou para o `<final>` | Testado com o modelo real local via Ollama (não probe sintético nem Colab), trajetória completa em `mode="dev"` inspecionada manualmente para os 3 cenários, comparada com a rodada anterior (pré-fix de temperatura) para confirmar reprodutibilidade |
| D-cli-go-scaffold | Novo módulo Go em `cli/` (`go.mod` próprio, fora do módulo Python) — primeira fatia da CLI de produção (`D13`, seção M8): TUI interativa em `bubbletea`/`lipgloss` (`internal/ui`), cliente HTTP raw-mode pra Ollama local (`internal/ollamaclient`, mesmo contrato do `OllamaModelRunner` Python — `temperature=0.0`, parada de stop-sequence client-side com a tag incluída), porta mínima do loop do agente (`internal/agent` — só geração real, `<tool_call>` recebe `UNSUPPORTED_TOOL` porque nenhuma ferramenta está registrada em Go ainda) e um parser de exibição da gramática de trajetória (`internal/segments`) | Motivado pela decisão de arquitetura já registrada em `D13`/`D-ollama-model-runner`: CLI de produção é Go, cliente HTTP fino falando com um servidor de inferência. Efeito de digitação com rastro de gradiente branco→bege (`internal/ui/gradient.go`) desacoplado da velocidade real de chegada de rede (`revealTickMsg` a cada 16ms), pra sempre parecer digitação suave. Dois bugs reais encontrados e corrigidos em teste manual ao vivo: (1) `renderClosedTrajectory` usava `Parse` (descarta cauda não fechada) — uma resposta cortada pelo limite de tokens desaparecia silenciosamente, trocado por `ParseWithTail`; (2) cores padrão da lib `bubbles` (rosa/roxo) vazavam por cima da paleta bege/madeira por nunca terem sido sobrescritas explicitamente no `textinput` | Testado localmente: 18 testes Go (parser incl. tags incompletas em streaming, cliente Ollama contra `httptest` real, filtragem de `Thinking`/rótulo no registro finalizado, preservação de texto truncado). `go vet`/`gofmt` limpos. Validação visual manual feita pelo usuário na própria 5700 XT contra `logos-v2` real — comportamento (`UNSUPPORTED_TOOL` em ferramenta não registrada, recuperação honesta) confirmado ao vivo |
| D-praxis-conatus-naming | Nenhuma mudança — decisão explícita de NÃO alterar `PRAXIS_SYSTEM_PROMPT` agora | Achado em teste manual da CLI: o modelo se autodescreve como "Praxis" (texto literal do system prompt usado desde a geração do dataset), mas o usuário esclareceu que o naming de produto real é "Conatus" (família) / "Logos" (o modelo atualmente disponível/treinado) — "Praxis" também é um nome de modelo da família Conatus, mas não é o que está treinado aqui. Mudar o texto do prompt agora recriaria o mesmo tipo de mismatch treino/avaliação que `D-train-prompt-mask`/`D-probe-system-prompt` já corrigiram antes, já que o adapter atual foi calibrado com "Praxis". Usuário optou explicitamente por manter como está e revisar isso num próximo retreino (trocar para "Logos" no `PRAXIS_SYSTEM_PROMPT`, gerar/revalidar dataset com o texto novo, só então retreinar) em vez de aplicar qualquer gambiarra cosmética agora | N/A — decisão de adiar, não uma mudança testável |
| D-cli-session-history | `agent.Turn`/`agent.Run(..., history []Turn, ...)` (`cli/internal/agent`): turnos concluídos entram no prompt do próximo turno no mesmo formato `[USER]/[ASSISTANT]` já usado, com a trajetória COMPLETA de cada turno anterior (não só o `<final>`) — é o formato mais fiel ao padrão treinado, mesmo sem exemplo de conversa multi-turno no dataset. Limite de `maxHistoryTurns=6` (`cli/internal/ui`) pra não estourar o contexto sem controle. Atalho `Ctrl+R` limpa o histórico da sessão | Pedido do usuário — hoje cada mensagem virava um turno isolado, sem memória nenhuma da conversa. Risco documentado explicitamente no código e na tela de boas-vindas da CLI: histórico multi-turno é território NÃO coberto pelo treino (dataset é inteiramente turno único), então não há garantia de que o modelo generalize pra isso tão bem quanto pro caso de turno único já validado | Testado localmente (`agent_test.go`, 4 testes novos): confirma que o primeiro turno não tem prefixo de histórico, que turnos anteriores entram com a trajetória completa (não só a resposta), e que a ordem fica correta (histórico antes do turno atual). Qualidade real do comportamento multi-turno só se confirma testando manualmente com o modelo de verdade |
| D-cli-context-counter | `Client.NumCtx` (`cli/internal/ollamaclient`, default 4096, mandado explicitamente em `options.num_ctx`) + captura de `prompt_eval_count` (campo REAL retornado pela própria Ollama, não estimado) mesmo quando o harness para a leitura do stream antes da Ollama sinalizar `done=true` por conta própria — esse valor chega cedo no streaming (calculado no prefill, antes do primeiro token de saída), diferente de `eval_count` (tokens gerados), que só fecha no fim natural da geração e por isso quase nunca chega a aparecer (o harness para cedo por stop-sequence na esmagadora maioria dos casos). Rodapé da CLI mostra `contexto: X/4096 tokens (Y%) · N turnos · Ctrl+R reinicia` do lado direito | Pedido do usuário (contador de contexto do lado direito do status). Decisão deliberada de NÃO estimar tokens do texto gerado (chars/4 ou parecido) — o lado esquerdo do rodapé mostra caracteres/s real em vez de token/s fabricado, pela mesma razão. `NumCtx` explícito também dá um denominador conhecido pro contador, em vez de depender do default não documentado de forma confiável da Ollama | Testado localmente: 2 testes novos em `client_test.go` confirmam a captura de `prompt_eval_count` mesmo com corte antecipado por stop-sequence, e que `num_ctx` é mandado explicitamente na requisição. Suite completa do módulo `cli/`: 28/28 |
| D-cli-real-tools | Novo pacote `cli/internal/tools` — `write_file`/`read_file`/`list_files` executam de VERDADE agora (antes, todo `<tool_call>` recebia `UNSUPPORTED_TOOL`, mesmo os que teriam ferramenta real no harness Python). Schema e códigos de erro copiados fielmente dos executores Python (`src/tools/executors/write_file_tool.py`/`read_file_tool.py`/`list_files_tool.py`): mesmos nomes de campo (`path`, `content`, `mode`, `start_line`, `end_line`), mesmos códigos (`FILE_NOT_FOUND`, `INCOMPLETE_SOLUTION`) — divergir isso seria o mesmo tipo de mismatch de formato que `D-train-prompt-mask` corrigiu, só que do lado das ferramentas. `resolvePath` (mesma lógica de `SandboxContext.resolve_path`) recusa qualquer caminho que escape do `workDir`. Binário renomeado de `logos-cli` para `conatus`; `os.Getwd()` no `main.go` vira o `workDir` — ferramentas operam no diretório real onde o `conatus` foi chamado, não um sandbox fixo. `checker`/`shell`/`web_search` continuam UNSUPPORTED_TOOL (não portados ainda) | Pedido explícito do usuário: "crie as funções dele, dele ler arquivos, escrever, etc... é para ele fazer essas coisas no diretório que foi chamado". Fecha a lacuna central deixada em aberto desde `D-cli-go-scaffold` ("só a geração é real") | Testado localmente: 7 testes novos em `agent_test.go` (arquivo criado de verdade em disco dentro do `t.TempDir()`, rejeição real de travessia de caminho fora do workDir, `TOOL_CALL_PARSE_ERROR` real pra JSON malformado, `UNSUPPORTED_TOOL` continua correto pra `checker`) + 10 testes novos em `tools_test.go` (write/read/list reais, `mode=create` recusando sobrescrever, `resolvePath` aceitando aninhado e rejeitando `../`). Suite completa do módulo `cli/`: 38/38 |
| D-cli-checker-tool | `checker.go` (`cli/internal/tools`) porta o backend Python de `src/checker/backends/python_backend.py` — `syntax_check`/`compile` (`py_compile` real por arquivo), `compile_and_test` (syntax + `pytest -q --no-header` real), `run` (executa o entrypoint de verdade), `lint` (`ruff check .` se disponível). Mesmos códigos de erro (`SYNTAX_ERROR`, `TEST_FAILURE`, `MISSING_DEPENDENCY`, `TIMEOUT`, `RUNTIME_ERROR`, `UNSUPPORTED_LANGUAGE`, `INCOMPLETE_SOLUTION`), mesma extração de arquivo/linha do traceback via regex. Só o backend Python está portado — Go/outras linguagens caem em `UNSUPPORTED_LANGUAGE` real (igual o Python faz pra qualquer linguagem sem backend registrado), não fingido. Diferente de write_file/read_file/list_files, o checker NUNCA toca o `workDir` real — materializa os arquivos recebidos num diretório temporário isolado (`os.MkdirTemp`, removido no fim), mesmo padrão do `tempfile.TemporaryDirectory` do Python, pra não sujar o projeto do usuário com artefatos de validação | Pedido do usuário, confirmado em teste manual ao vivo: depois de `write_file` funcionar, o modelo tentou `checker` na sequência e recebeu `UNSUPPORTED_TOOL` — exatamente o padrão observado nos testes Python desta sessão inteira (`write_file`→`checker` é o par mais comum). Fecha a ferramenta que mais aparece logo depois de `write_file` nos pedidos reais | Testado localmente com Python real (não mockado): 6 testes novos em `checker_test.go` confirmam `py_compile`/`pytest` reais rodando contra código genuinamente correto e genuinamente quebrado (erro de sintaxe real, teste que falha de verdade por lógica errada), `UNSUPPORTED_LANGUAGE` real pra linguagem não registrada, e que o `workDir` real nunca é tocado. +1 teste em `agent_test.go` confirma o `checker` real disparando um `SYNTAX_ERROR` genuíno através do loop completo. Suite completa do módulo `cli/`: 45/45 |
| D-cli-tool-connector-view | Reescrita da renderização de `tool_call`/`tool_result` na TUI — nunca mais mostra o JSON cru de argumentos/resultado. Enquanto uma ferramenta está executando de verdade (par `tool_call` sem `tool_result` correspondente ainda — pode levar segundos num `checker` com pytest real), mostra spinner animado + nome da ferramenta; assim que o resultado chega, vira uma linha estática `└ nome` (dourado se ok, rust se erro, com a mensagem de erro REAL extraída do JSON — `extractErrorMessage`, aceita tanto o formato simples `{"message"}` quanto o formato aninhado do checker `{"errors":[{"message"}]}` — nunca inventa mensagem se não achar nenhum campo reconhecido) | Pedido do usuário: substituir o dump de JSON por um indicador visual compacto (conector + nome), animado durante execução, estático quando resolvido — mesmo padrão usado por outras CLIs de agente pra não poluir a tela com payload técnico | Testado localmente: 4 testes novos confirmam que o JSON nunca aparece na visão colapsada (nem em sucesso nem em erro), que o spinner aparece enquanto o par tool_call/tool_result está incompleto, e que `extractErrorMessage` lida com os dois formatos reais de erro (simples e aninhado do checker) sem inventar texto pra JSON não reconhecido. Suite completa do módulo `cli/`: 49/49 |
| D-cli-code-block-highlighting | Novo `cli/internal/ui/codeblock.go` — `renderFinalBody` separa o corpo de um `<final>` em texto normal e blocos de código markdown (` ```lang ... ``` `) via `splitCodeFences`, e cada bloco de código ganha números de linha reais (gutter alinhado à largura do maior número) + destaque de sintaxe REAL via `github.com/alecthomas/chroma/v2` (lexer pelo nome da linguagem da cerca, ou `lexers.Analyse` se a cerca não disser qual é; estilo `monokai`, formatter `TTY16m`) — não é uma paleta de palavras-chave escrita à mão (ficaria errada/incompleta fora de 2-3 linguagens). Aplica só na trajetória JÁ FECHADA (`renderSegment`/`renderCollapsedSegments`); a visão AO VIVO (ainda digitando) continua com o efeito de gradiente existente sem tentar colorir sintaxe de um bloco potencialmente incompleto a cada frame. Bloco cortado antes de fechar a cerca (resposta truncada) ainda é mostrado como código, não descartado, mesmo padrão de `renderClosedTrajectory` pra outros truncamentos | Pedido do usuário: "Adicione o suporte a formatação de código, que mostre as linhas, highlinter" — código dentro de respostas `<final>` aparecia como texto corrido sem números de linha nem cor | Testado localmente: 5 testes novos em `codeblock_test.go` confirmam números de linha aparecendo, sequências ANSI reais de destaque presentes (não é decoração cosmética fixa), texto normal ao redor do bloco preservado, bloco sem cerca fechada não perdido, e `splitCodeFences` separando texto/código/texto na ordem certa. Suite completa do módulo `cli/`: 55/55 |
| D-cli-stale-binary-diagnosis | Nenhuma mudança de código — achado de diagnóstico. O usuário reportou JSON cru ainda aparecendo nas ferramentas mesmo depois de `D-cli-tool-connector-view` já estar implementado e testado; investigação confirmou que o binário instalado em `C:\Users\Pichau\bin\conatus.exe` (no PATH) estava desatualizado — um build anterior à reescrita da visão de ferramentas. `cli/conatus.exe` foi reconstruído a partir do código-fonte atual e copiado por cima do binário instalado | O formato exibido nas capturas de tela do usuário ("tool_call · nome" + JSON indentado) não corresponde a nenhuma função no código-fonte atual — só podia vir de um build anterior nunca reinstalado após a mudança | N/A — decisão de reinstalar o binário existente, não uma mudança testável. Usuário deve confirmar visualmente que a visão de conector (`└ nome`) aparece agora numa sessão real |
| D-cli-real-token-counting | Revisão de `D-cli-context-counter`: a suposição de que `prompt_eval_count` chegaria cedo no stream se mostrou falsa em uso real — a Ollama só reporta esse campo de forma confiável no chunk `done=true` DELA MESMA, e o harness quase sempre corta a leitura antes disso (parada por stop-sequence própria), então o contador nunca era preenchido de verdade (confirmado por captura de tela do usuário: contador sempre em branco numa sessão real com vários turnos). Solução: `Client.CountTokens(ctx, prompt)` (`cli/internal/ollamaclient`) faz uma chamada dedicada e barata com `num_predict: 0` (só prefill, sem decodificar nada) — sempre completa rápido e sempre alcança o `done=true` real da Ollama. `agent.finishTurn` chama isso uma vez por turno concluído (prompt final + texto gerado = exatamente o que entraria como `[ASSISTANT]` do próximo turno) e emite `Event.TotalTokens` (substituiu o antigo `Event.ContextTokens`, que tentava capturar o campo durante a geração normal). Na UI (`cli/internal/ui`): contador esquerdo agora soma `sessionTokens` a cada turno concluído (nunca reseta sozinho, só em `Ctrl+R`) e o texto ao lado de "Ready" passa a mostrar `N tokens` — antes revertia pro "Ready" vazio assim que a geração terminava. Lado direito virou "Context: [barra] N/NumCtx (P%)" (era "contexto", em português, e nunca aparecia de fato). Bloco "Thinking" finalizado deixou de sumir por completo — vira "Thought for Xs" (duração real medida com `time.Since`, nunca mostra o conteúdo do pensamento), via `trackThinkDuration` comparando `segments.ParseWithTail` entre deltas sucessivos pra detectar o instante exato em que um `<think>` fecha | Pedido explícito do usuário, com a foto do bug em mãos: "o contexto só aparece depois x tokens" (na prática nunca aparecia); "quando a IA envia a mensagem, a exibição ao lado de Ready desaparece — é para permanecer e ir somando"; "quando ele terminar de pensar, ao invés de desaparecer o Thinking, deve aparecer Thought for x seconds sem exibir o pensamento" | Testado localmente: 1 teste novo em `client_test.go` confirma que `CountTokens` manda `num_predict=0` e devolve o `prompt_eval_count` real do `done=true`. `agent_test.go`: teste renomeado/reescrito pra `TestRunEmitsTotalTokensFromRealCountTokensCallAtTurnEnd` confirmando o evento `TotalTokens` vindo da chamada dedicada de fim de turno (não mais tentando capturar durante a geração); 3 testes existentes ajustados pro efeito colateral esperado da chamada extra (mais uma requisição HTTP por turno concluído). `model_test.go`: testes de `renderClosedTrajectory`/`renderLiveTrajectory` atualizados pra nova assinatura (`thinkDurations []time.Duration`) e uma reescrita de teste confirma "Thought for 7s" aparecendo no lugar do conteúdo cru do pensamento. Suite completa do módulo `cli/`: 50/50 |
| D-conatus-logos3-identity | Fecha a decisão adiada em `D-praxis-conatus-naming`. `PRAXIS_SYSTEM_PROMPT` (`src/harness/system_prompt.py`, fonte única de verdade pro treino e avaliação) e `agent.SystemPrompt` (`cli/internal/agent/agent.go`, byte a byte igual, mesma disciplina de `D-train-prompt-mask`) trocaram a identidade de "Você é Praxis, um agente de engenharia de software" para "Você é Logos-3, um agente de engenharia de software da família de modelos Conatus" — resto do texto (instruções de `<think>`/`<tool_call>`/`<final>`) inalterado. `system_prompt` é repetido literalmente (não referenciado) em cada exemplo do dataset, então os ~3679 exemplos existentes que diziam "Praxis" tiveram esse campo trocado por substituição estrutural via JSON (não regex textual — evita depender de bytes idênticos char a char), mais 3 ocorrências cosméticas de "Praxis Example" → "Logos Example" num README de exemplo gerado. Novo lote de 28 exemplos (`scripts/gen_identity_wave2_pilot.py`, `data/train/gen-identity-*.json`, task_type=direct_answer, sem ferramenta): perguntas diretas de identidade (quem é você, quem te treinou, você é humano) respondidas com a definição completa da família Conatus (regra: só aparece quando perguntado diretamente, nunca como assinatura espontânea), e probes de confusão com concorrentes nomeados (Claude, GPT/ChatGPT, Gemini, Qwen, IBM Granite, Llama, Mistral, DeepSeek) e com o nome antigo do próprio projeto (Praxis) — cada um exigindo negação correta sem confabular concordância. Plano completo, incluindo o desenho (ainda não implementado) da extensão de schema pra multi-turno das próximas categorias, em `docs/plan_dataset_expansion_wave2_identity_multiturn.md` | Achado real em uso ao vivo da CLI: a tela mostrou "Sou Praxis, um agente de engenharia de software" — a identidade nunca tinha sido trocada de fato, só adiada. Pedido explícito do usuário pra fechar isso nesta leva, incluindo os probes de confusão com concorrentes específicos (Claude, Gemini, Qwen, GPT, Granite) que ele pediu por nome | Bulk replace verificado: 0 ocorrências de "Praxis" restantes em `data/` depois da troca (checado por busca exaustiva, não amostral). Lote novo validado contra `structural_validate`/`schema_validate` reais do próprio pipeline (`src/dataset/pipeline.py`) — 0 problemas nos 28 exemplos; checagem de vazamento de detalhe interno — 0 leaks. Suite Python completa: 142 passaram, 1 falha pré-existente e não relacionada (`trl` falhando ao ler seu próprio `.jinja` por causa do codec cp1252 padrão do Windows, nada a ver com o dataset). Suite Go da CLI: 55/55 |
| D-style-upgrade-wave2-pilot | Primeira fatia real da categoria "upgrade de estilo" da wave 2 (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 6.1). `scripts/gen_style_upgrade_wave2_pilot.py` gerou 12 exemplos novos (`data/train/gen-style-*.json`) cobrindo 8 task_types já existentes na taxonomia (`single_tool_call`, `multi_tool_call`, `complexity_analysis`, `refactor`, `debugging`, `test_authoring`, `code_explanation`), demonstrando os quatro elementos de estilo já fechados em conversa: profundidade de `<think>` proporcional à dificuldade real (fácil fica curto, os dois exemplos de `debugging` levam um ciclo completo de hipótese antes da evidência, diagnóstico de causa raiz vs sintoma, e correção verificável), verbalização de intenção de ferramenta em primeira pessoa antes de chamar ela, trade off técnico explícito nas escolhas de implementação (`dict.fromkeys` vs `set()` pra preservar ordem, busca binária vs linear em entrada já ordenada, `set()` vs `list` pra membership dentro de laço), e `<final>` caloroso só quando cabe (nunca em resposta trivial de conceito). Toda ferramenta usada roda de verdade via `ToolExecutorRegistry`/`SandboxContext`/`checker`, nenhum `tool_result` fabricado | Pedido do usuário pra continuar a wave 2 depois da identidade, priorizando a categoria que não depende da extensão de schema multi-turno ainda não implementada (mais barata de executar agora) | Validado contra `structural_validate`/`schema_validate` reais do pipeline (0 problemas), checagem de vazamento de detalhe interno (0 leaks, depois de corrigir um vazamento REAL encontrado durante a execução: o `pytest_asyncio` emite um warning de depreciação com o caminho absoluto de instalação do Python local, tanto em `stdout` quanto dentro de `error_message`/campo `"message"` do `<tool_result>` de erro — um campo separado de `data` que a sanitização inicial não cobria; corrigido com um sanitizador recursivo + `dataclasses.replace`, já que `ToolExecutionResult` é frozen) e checagem de pontuação (0 hífens/travessões de frase fora das exceções de nome próprio como "Logos-3"). Suite Python completa rodada de novo depois do lote: 138/138 (ignorando o teste pré-existente do `trl`, não relacionado) |
| D-style-upgrade-wave2-round2 | Segunda rodada do mesmo lote (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 6.1.1), pedido explícito do usuário pra continuar expandindo. `scripts/gen_style_upgrade_wave2_pilot.py` cresceu de 12 para 23 exemplos (correção: o commit original desta rodada afirmou "35" por engano; a contagem real, conferida por `ls data/train/gen-style-*.json`, sempre foi 23 — corrigido no plano e nesta entrada), cobrindo 11 task_types novos: `language_migration` (tradução Python→Go com o backend Go real do `checker`), `compiles_successfully` (Go), `project_configuration` (arquivo estático via `write_file`, sem `checker`), `documentation_usage`/`insufficient_information`/`ambiguous_request` (sem ferramenta, pedindo esclarecimento em vez de adivinhar quando falta contexto real), `checker_rejects_code`/`test_fails` (falha real REPORTADA honestamente, sem tentar corrigir — esses task_types são sobre reconhecer a falha, não sobre o ciclo de correção, que já está coberto por `debugging`), `multi_file_read_and_edit` (lê `config.py` de verdade antes de editar `main.py` com base no valor lido), `tool_unavailable` (tenta `apply_patch`, `enabled: false` em `configs/tools_registry.yaml`, recebe `UNSUPPORTED_TOOL` genuíno do `ToolExecutorRegistry`, recupera com `write_file`), `forbidden_operation` (tenta `rm -rf` via `shell`, recusado de verdade pela sandbox por estar no `denylist_patterns`) | Usuário pediu explicitamente pra continuar expandindo o lote depois da primeira rodada | Validado contra `structural_validate`/`schema_validate` reais do pipeline nos 23 exemplos (0 problemas), 0 leaks, 0 violações de pontuação depois de reconhecer uma exceção nova legítima (flag de linha de comando real, `rm -rf`, não é pontuação de frase — mesma categoria de exceção que nomes próprios como "Logos-3"). Confirmado que os `<tool_result>` de erro em `tool_unavailable`/`forbidden_operation` são genuínos (mesmo caminho real que o harness/CLI usariam pra uma ferramenta desabilitada ou comando proibido), não fabricados. Suite Python completa: 138/138 (ignorando o teste pré-existente do `trl`, não relacionado) |
| D-style-upgrade-wave2-round3 | Terceira e última rodada desta fatia (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 6.1.2), pedido explícito do usuário pra fechar os task_types restantes. Cresceu de 23 para 27 exemplos, cobrindo os 4 que faltavam: `direct_answer` (pergunta conceitual pura, sem ferramenta), `invalid_call_then_correction` (chamada real com `content` faltando em `write_file` recebe `TOOL_ARGUMENT_SCHEMA_ERROR` genuíno, calculado pelo mesmo validador de JSON Schema que `src/harness/loop.py` usa de verdade antes de executar qualquer ferramenta — via `ToolExecutorRegistry.validate_args` —, não fabricado; corrigido na retentativa), `model_fixes_after_error` (segunda escrita com `mode="create"` colide de propósito com um arquivo que a própria trajetória acabou de criar, erro real `INCOMPLETE_SOLUTION` de `write_file_tool.py`; a correção usa uma abordagem genuinamente diferente — caminho novo — em vez de reenviar a mesma chamada, evitando o padrão de "retentativa idêntica sem progresso" já flagrado em `D-own-bug-diagnosis-expansion`), `test_passes` (sucesso direto sem ciclo de depuração, contraste deliberado com `debugging`/`test_fails`). **Com isso, os 22 task_types de `TASK_TYPES` (`src/dataset/taxonomy.py`) têm cobertura no novo estilo de `<think>`/`<final>`** — volume ainda é 1 exemplo por task_type na maioria dos casos, não é suficiente pra generalização por si só, mas fecha o objetivo de demonstrar o padrão em toda a superfície da taxonomia | Usuário pediu explicitamente pra fechar os 4 task_types restantes nesta sessão, antes de passar pra extensão de schema multi-turno | Validado contra `structural_validate`/`schema_validate` reais do pipeline nos 27 exemplos (0 problemas), 0 leaks, 0 violações de pontuação, 0 ids duplicados entre identidade e estilo. Suite Python completa: 138/138 (ignorando o teste pré-existente do `trl`, não relacionado) |
| D-dataset-history-schema | Item 1 da extensão de schema multi-turno (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 4) — item 2 (máscara de loss no pipeline de treino) continua pendente, não investigado. Novo campo opcional `trajectory.history` (lista de `{user_request, raw_text}`, mesmo formato de `agent.Turn` do lado da CLI Go), validado por `TRAJECTORY_SCHEMA`/`validate_trajectory` (`src/dataset/schema.py`) — só `raw_text` é obrigatório no nível de trajetória (é o único campo que `structural_validate` sempre leu direto do dict; exigir `system_prompt`/`user_request` quebraria testes/exemplos parciais já existentes que só têm `raw_text`), `system_prompt`/`user_request` são tipados quando presentes, cada item de `history` exige `user_request`/`raw_text` não vazios. `structural_validate` (`src/dataset/pipeline.py`) agora chama `validate_trajectory` antes de parsear segmentos, transformando um `raw_text` ausente num erro de validação claro em vez de `KeyError`. Suporte real em `src/harness/trajectory.py`: `HistoryTurn` (novo dataclass) + `Trajectory.history: list[HistoryTurn]` (default vazio) + `render_for_model()` agora prepende os turnos de `history` no MESMO formato `[USER]/[ASSISTANT]` que `agent.Run` (Go) já usa em produção, testado byte a byte contra esse formato + `to_example_dict()` novo, serializa pro formato canônico do dataset, omitindo `history` quando vazio (exemplos de turno único ficam idênticos a antes) | Usuário pediu explicitamente pra atacar isso depois de fechar o lote de upgrade de estilo — é o bloqueio real das duas maiores categorias pendentes da wave 2 (mudança abrupta de direção, pedido de ferramenta tardio) | 17 testes novos (`tests/unit/test_dataset_history_schema.py`): validação aceita/rejeita histórico bem/mal formado, `structural_validate` integra a checagem nova, `render_for_model` bate byte a byte com o formato esperado (sem histórico = comportamento antigo preservado; com 1 e com múltiplos turnos, ordem preservada), `to_example_dict` omite `history` quando vazio e inclui quando presente, round trip através de `validate_trajectory`. Varredura completa (não amostral) dos ~3706 exemplos já existentes no dataset contra a validação nova: 0 regressões. Suite Python completa: 155/155 (138 + 17 novos, ignorando o teste pré-existente do `trl`, não relacionado) |
| D-dataset-history-loss-mask | Item 2 da extensão de schema multi-turno (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 4.1) — fecha o bloqueio técnico que faltava pras categorias multi-turno da wave 2. Investigação: `compute_loss_mask`/`apply_loss_mask` (`src/training/loss_masking.py`) já calculavam spans elegíveis só dentro do `raw_text` do turno ATUAL e tratavam tudo antes de `prefix_len` como não elegível — **não precisaram de nenhuma mudança**. A única peça que faltava era o PREFIXO usado por `build_pretokenized_dataset` também incluir `history`: `_render_prefix` (`src/training/data_collator.py`) ganhou um parâmetro opcional `history`, repassado pra `Trajectory(..., history=...).render_for_model()` (que já sabe concatenar `history` no formato `[USER]/[ASSISTANT]`, via `D-dataset-history-schema`) antes de computar o prefixo. `build_pretokenized_dataset` lê `traj.get("history")` de cada trajetória. Como o `raw_text` passado pra `compute_loss_mask` continua sendo só o do turno atual, todo o conteúdo do histórico (inclusive `<think>`/`<tool_call>`/`<final>` de turnos passados, que pareceriam "elegíveis" se reparseados isoladamente) cai automaticamente dentro do prefixo mascarado, sem lógica de máscara nova | Usuário pediu explicitamente pra investigar e implementar isso logo depois do item 1, e pra ser lembrado de fazer isso | 2 testes novos em `tests/unit/test_data_collator.py` (rodam sem precisar de `trl`, só `transformers`/`datasets`, então não são afetados pela falha de ambiente pré-existente): `test_pretokenized_dataset_masks_entire_history` decodifica só os tokens com loss ativo de um exemplo de 2 turnos e confirma que NADA do turno passado sobra no texto treinado (nem o pedido, nem o `<think>`, nem o `<final>`), só o turno atual; `test_pretokenized_dataset_without_history_key_matches_no_history` confirma que a ausência da chave `history` produz `input_ids`/`labels` byte a byte idênticos a `history: []` explícito, preservando o comportamento anterior a esta mudança. Suite Python completa: 161/161 (155 + 6 do arquivo `test_data_collator.py` completo, ignorando só o único teste que já falhava antes desta sessão inteira por um problema de ambiente Windows/cp1252 no `trl`, não relacionado) |
| D-dataset-multiturn-wave2-pilot | Fecha a wave 2 do dataset: as duas categorias multi-turno em si (`docs/plan_dataset_expansion_wave2_identity_multiturn.md`, seção 6.2), agora que o bloqueio técnico (`D-dataset-history-schema` + `D-dataset-history-loss-mask`) está removido. `scripts/gen_multiturn_wave2_pilot.py` gerou 85 exemplos: 45 de mudança abrupta de direção (algoritmos/complexidade, teoria de CS, conceitos de linguagem, segurança, concorrência, ferramentas de dev, 3 com escrita de código real via `write_file`+`checker`, 3 de debugging real com bug plantado e corrigido, 4 de reação de confusão espelhando a falha observada ao vivo) + 40 de pedido de ferramenta avulso/tardio (`list_files`×10, `read_file`×10, `shell ls`×8, `shell cat`×6, `shell grep`×6, incluindo um caso real de grep sem match, código de saída 1, que o `<think>` do âncora precisa interpretar corretamente como "sem ocorrência" e não como erro). Mecanismo: banco de 12 turnos de contexto reutilizáveis (`FILLERS`, sem ferramenta) vira `history` em combinações variadas; só o turno ÂNCORA (o que efetivamente treina, via `Trajectory.to_example_dict()`) tem conteúdo técnico genuinamente distinto por exemplo — reutilizar os FILLERS como contexto não viola a disciplina anti-duplicação porque `history` fica inteiramente mascarado da loss. Alvo de 45+40 foi decisão explícita do usuário: mirar mais alto de cara que uma leva de teste pequena, pra reduzir risco de precisar de uma segunda rodada, mas ainda abaixo do número aspiracional original (~200-250 + ~150-200) por seguir proporcional ao custo real de gerar multi-turno com execução verdadeira | Usuário pediu pra atacar a geração das categorias multi-turno depois do schema/máscara de loss estarem prontos, com instrução explícita de escrever os `<think>` com profundidade técnica de verdade | Validado contra `structural_validate`/`schema_validate` reais do pipeline (0 problemas nos 85), 0 leaks (mesmo vazamento do `pytest_asyncio` já visto em `D-style-upgrade-wave2-pilot`, corrigido de novo aqui pois cada script gerador tem sua própria cópia do sanitizador), 0 violações de pontuação (3 hífens gramaticais evitáveis do português reescritos em vez de virarem mais uma exceção à regra), 0 ids duplicados com o resto do dataset (~3706 exemplos existentes). Validação ponta a ponta real (não só teste sintético): um exemplo do lote passado de verdade por `build_pretokenized_dataset` confirma, decodificando só os tokens com loss ativo, que o pedido do turno de histórico não aparece no texto treinado. Suite Python completa: 161/161 |
| D-frontend-pivot-model-swap | Item 1 do pivô de especialização em frontend (`docs/plan_frontend_specialization_wave3.md`, seção 2 e 10) — troca do modelo base de `ibm-granite/granite-4.1-8b` para `Qwen/Qwen3-4B-Instruct-2507` em `configs/train_l4.yaml` e `configs/train_a2000.yaml`. Critérios: sem modo de "thinking" nativo (confirmado no model card — não gera blocos de raciocínio próprios, então não briga com a gramática `<think>`/`<tool_call>`/`<final>` treinada por cima), 4.0B denso (não MoE, mais simples que o Gemma 4 E4B pra QLoRA), Apache 2.0, já disponível no Ollama, base validada como forte pra fine-tuning por benchmark independente (distil labs). `D-qwen3-qknorm-lora`: `lora.target_modules` mudou de `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` pra `[v_proj, o_proj, gate_proj, up_proj, down_proj]` — Qwen3 aplica RMSNorm em q/k depois do reshape da projeção (QK-norm), e LoRA em q_proj/k_proj introduz tensor com shape incompatível nesse caminho, reportado travando com erro CUBLAS na primeira camada de atenção. Isso veio de pesquisa (duas buscas independentes convergindo no mesmo achado), não de execução real neste projeto ainda — precisa confirmar via `model.named_modules()` na primeira carga de verdade, mesmo espírito da seção 17 do PLAN.md. `sequence_length` da config da A2000 devolvido de 1024 pra 2048 (era o valor testado antes de precisar cair no Granite-4.1-8B) como expectativa, não medição — precisa rodar `scripts/vram_smoketest.py` de novo antes de confiar num treino completo. Notebook (`notebooks/train_granite_l4.ipynb`, célula 0) e `README.md` atualizados pra refletir o modelo real e o status real de execução (o notebook já rodou de ponta a ponta fora do Colab nesta sessão, isso deixou de ser "implementação pretendida não testada") | Usuário decidiu pivotar de agente generalista pra especializado em frontend, motivado por dois fatores reais: escassez de hardware pro 8B (a A2000 penava mesmo em `sequence_length=1024`) e tese de mercado validada por precedente real (v0/Bolt/Lovable — nicho estreito com modelo pequeno bate generalista competindo de frente com Copilot/Cursor/Claude Code) | Validado o que dá pra validar sem GPU: os dois configs carregam certo via `TrainConfig.load()` (`base_model`, `target_modules`, `sequence_length` conferidos), 1 teste ajustado pra refletir a realidade nova (`test_train_config_loads_from_default_yaml` — antes afirmava Granite e `q_proj` presente, ambos agora falsos de propósito), notebook validado com `nbformat.validate` depois do patch. Suite Python completa: 161/161. O que NÃO foi validado aqui (precisa de GPU real, próxima sessão prática): se o Qwen3-4B carrega sem gate, se os `target_modules` batem de verdade com `named_modules()`, e o pico real de VRAM |
| D-checker-html-backend | Item 3 do pivô de especialização em frontend (`docs/plan_frontend_specialization_wave3.md`, seções 3 e 10) — backend `html` do checker, camada 1 (só) do design em camadas de verificação de frontend: renderização real em Chromium headless via Playwright, falha se houver erro de console ou exceção JS não tratada. Novo módulo `src/checker/backends/frontend_backend.py` (`render_and_capture` reutilizável + `check_html` embrulhando no contrato `CheckResult`), registrado em `src/checker/backends/__init__.py`, `"html"` adicionado ao enum de `language` em `src/schemas/checker.json`. `operation=lint` (camada 3, acessibilidade via axe-core) retorna `MISSING_DEPENDENCY` estruturado de propósito — pacote não instalado, nunca finge ter checado o que não checou (D11). Camadas 2 (build React/Vue), 3 (axe-core) e 4 (design tokens) ficam pra quando a fase 2 começar. Novo `scripts/render_frontend_preview.py`: CLI standalone que chama `render_and_capture` direto pra produzir screenshot real + relatório de console, é o "gera → roda no navegador → tira print → eu vejo" pedido pelo usuário pro julgamento de estética (sempre subjetivo, nunca automatizado) | Terceiro item da ordem de execução do plano de frontend, depois da troca de modelo (`D-frontend-pivot-model-swap`) e da reativação do `search_code` (`D-search-code-reenable`) — "peça central e mais difícil" segundo o próprio plano, por depender de renderização real de navegador | Testado de ponta a ponta de verdade, não só com mocks: `check('html', 'run', ...)` real contra uma página limpa (passa) e uma com `undefinedFn()` não tratada (falha com `RUNTIME_ERROR` e a mensagem real do V8); `scripts/render_frontend_preview.py` rodado contra um HTML real, screenshot PNG resultante inspecionado visualmente (renderização correta: fundo escuro, texto branco centralizado, exatamente como o CSS pedia). 9 testes novos em `tests/unit/test_checker_html.py` (skip condicional se Chromium não disponível, mesmo padrão do backend Go). Suite Python completa: 183/183 |
| D-frontend-pilot-batch | Item 4 do pivô de especialização em frontend (`docs/plan_frontend_specialization_wave3.md`, seção 10) — primeiro corpus real de frontend, 5 exemplos gerados por `scripts/gen_frontend_pilot.py` em `data/train` (`gen-frontend-*.json`, ver seção 10.4.2). `MVP_LANGUAGES` ganhou `"html"`. Nenhum `task_type` novo — reaproveitados `multi_tool_call`/`model_fixes_after_error`, que já descrevem a forma da interação independente de domínio | Continuação direta do passo 3 (`D-checker-html-backend`) — sem exemplos reais validados pelo checker novo, não dá pra confirmar que o backend produz um dataset utilizável de verdade, só que a função em isolamento funciona | Cada exemplo passou por `build_example()` com asserções reais (não só "não lançou exceção"): `write_file`/`search_code` conferidos como `passed=True`, os dois `search_code` narrativos conferidos byte a byte contra o resultado real (`reuse_existing_button` exige `> 0` ocorrências de `.btn`, `new_modal_after_search` exige `== 0` ocorrências de "modal" — a asserção falharia se o `<think>` estivesse narrando algo que a ferramenta não confirmou), `fix_broken_toggle_script` tem uma falha REAL de checker no meio (não simulada) que só passa depois da correção de verdade. Validação estrutural completa: `validate_metadata`/`validate_trajectory`/`structural_validate` sem erros nos 5; 0 ids duplicados entre si e contra os ~2455 exemplos existentes; 0 leaks dos padrões já monitorados (`PRAXIS_OLLAMA`, caminhos do Windows, `Traceback`); bytes verificados UTF-8 corretos direto do disco (não só via print, que tem mojibake de exibição no console Windows cp1252 — não é corrupção real do arquivo). Suite Python completa: 183/183 (sem regressão, os 5 exemplos novos não têm teste dedicado — são dados, não código) |
| D-frontend-threejs-pilot | Segunda leva do lote-piloto de frontend, a pedido explícito do usuário ("Faça exemplos mais complexos, com 3D, Three.js, Design de vidro, etc"): 4 exemplos em `data/train` via `scripts/gen_frontend_advanced_pilot.py` — hero com Three.js vanilla (torus knot animado via `requestAnimationFrame`) + botão glassmorphism; seção de preços com glassmorphism puro em CSS (sem JS); showcase de produto com Three.js + addon `OrbitControls`, incluindo um ciclo de correção REAL (não simulado); spinner de carregamento — pedido menciona "3D" mas o `<think>` recusa WebGL e usa CSS puro (dimensão 7 do think técnico, seção 6, no sentido negativo: nem todo "3D" pedido merece WebGL). Achado técnico real descoberto rodando o checker antes de escrever qualquer exemplo (nunca assumido de memória): addons do Three.js como `OrbitControls` importam internamente via specifier nu (`import ... from "three"`), que o Chromium não resolve sem um `<script type="importmap">` mapeando `"three"` pra URL do CDN — sem isso, falha real com "Failed to resolve module specifier". Vira o ciclo de correção do exemplo `product_showcase_orbitcontrols_fix`: primeira renderização falha de verdade com essa mensagem exata, `<think>` diagnostica a causa raiz (não é a importação própria do Three.js, é a importação interna do addon) e implementa o import map, segunda renderização passa. Todos os 4 exemplos usam Three.js via CDN (`cdn.jsdelivr.net`) — decisão forçada pelo formato de dado do projeto, não preferência: `write_file`/`checker` embutem o conteúdo do arquivo inteiro como texto no `tool_call`, e a build minificada do Three.js sozinha (~150-200KB) estouraria `sequence_length` (2048-4096) várias vezes se vendorizada inline; CDN mantém o `<script type="module">` em poucas dezenas de tokens. Efeito colateral aceito e documentado: exemplos com Three.js exigem rede de verdade pra revalidar o checker (agora e no futuro, ex. `validate_dataset.py` re-executando); sem rede, falham de forma honesta e determinística (erro de import), não silenciosamente | Pedido explícito do usuário depois de ver os 5 exemplos do primeiro lote-piloto renderizados (prints reais mostrados na conversa) | Confirmado com um teste isolado ANTES de escrever os exemplos: uma cena Three.js simples (TorusKnotGeometry) renderiza sem erro de console via `check('html','run',...)` real (WebGL funciona em Chromium headless via fallback de software/SwiftShader — só gera warning, não erro, então não falha o checker); testado também o bug real do `OrbitControls` sem import map (falha) e com import map (passa) antes de escrever a versão "quebrada" do exemplo, pra garantir que o bug plantado é genuíno e não uma invenção de texto. Os 4 exemplos renderizados via `render_and_capture` com screenshot real depois de gerados, prints inspecionados visualmente na conversa (torus knot visível atrás do texto, glassmorphism com blur visível nos cartões e no botão, icosaedro do showcase renderiza mas com sombreamento fraco por só ter luz ambiente — observação honesta, não é bug, é só visualmente simples). Validação estrutural completa (`validate_metadata`/`validate_trajectory`/`structural_validate`) sem erros nos 4; 0 ids duplicados entre os 9 exemplos de frontend e os ~2455 existentes; 0 leaks dos padrões monitorados. Suite Python completa: 183/183 |
| D-frontend-pivot-reverted | Usuário reverteu o pivô de especialização em frontend inteiro ("Estava um pouco desmotivado na hora que pedi para você mudar o foco para frontend... podemos somente mudar o modelo e manter como estava?"), na mesma sessão em que uma terceira leva de landing pages premium (Lumen/Atelier, 7,6k-12,8k tokens cada) tinha acabado de expor que páginas completas não cabem no `sequence_length` de treino sem truncar o `<final>` — o problema de engenharia que motivou a discussão de multi-turno ficou sem solução fechada quando o pivô foi revertido, e fica registrado aqui como não resolvido, não como descartado. Ações da reversão: (1) removidos os 13 exemplos de frontend de `data/train` (`gen-frontend-*`, dos lotes de `D-frontend-pilot-batch`/`D-frontend-threejs-pilot`, mais 4 exemplos multi-turno de uma landing "Nyx" gerados nesta sessão mas nunca commitados) — o dataset volta a ser 100% generalista (algoritmos/backend/CLI); (2) removido `docs/plan_frontend_specialization_wave3.md` do repositório (histórico continua em `docs/PLAN.md` e no git log); (3) removidos os scripts geradores específicos de frontend (`gen_frontend_pilot.py`, `gen_frontend_advanced_pilot.py`, `gen_frontend_multiturn_landing_pilot.py`, `gen_frontend_premium_pilot.py` — o último nunca chegou a ser commitado); (4) comentários em `configs/tools_registry.yaml`, `configs/train_l4.yaml`, `configs/train_a2000.yaml`, `src/dataset/taxonomy.py` e vários arquivos de teste/executor que referenciavam o arquivo de plano removido foram reescritos pra apontar pra `docs/PLAN.md`, sem referência pendente a um arquivo inexistente. Decisão explícita do usuário sobre o que MANTER (não é reversão total): `search_code` continua reativada (`D-search-code-reenable`) e o backend `html` do checker continua registrado (`D-checker-html-backend`, incluindo `scripts/render_frontend_preview.py`) — ambos avaliados como ferramentas genéricas úteis mesmo fora de um foco em frontend, sem custo real de manter desabilitados/não usados; a troca de modelo (`D-frontend-pivot-model-swap`, Qwen3-4B-Instruct-2507) também é mantida como decisão independente do pivô que a originou, já que a motivação real (hardware, ausência de thinking nativo) não dependia de frontend. `README.md` revertido pra descrição generalista com o Qwen3-4B | Usuário decidiu reverter o pivô, motivado por uma reavaliação honesta de que a decisão original (`D-frontend-pivot-model-swap`, ver seu próprio "por quê") tinha sido tomada num momento de desmotivação, não por convicção real na tese de mercado | Verificado que nenhum arquivo remanescente referencia `docs/plan_frontend_specialization_wave3.md` (`grep` limpo, exceto este próprio arquivo PLAN.md, que é histórico). Suite Python completa depois da remoção: 183/183, sem mudança nenhuma no total — esperado, já que `test_checker_html.py` (9 testes) e `test_search_code_tool.py` (10 testes) testam as FERRAMENTAS mantidas, não os dados de frontend que saíram; remover dados nunca teria afetado a contagem de testes, só confirmação de que nada quebrou |
| D-kaggle-training-support | Usuário perguntou se dá pra fazer fine-tuning nas GPUs gratuitas do Kaggle usando `qwen3:4b-instruct-2507-q4_K_M`. Duas correções técnicas antes de implementar: (1) `q4_K_M` é quantização GGUF (llama.cpp/Ollama), feita pra INFERÊNCIA — o pipeline de QLoRA deste projeto (transformers + bitsandbytes + PEFT) carrega o checkpoint HF original (`Qwen/Qwen3-4B-Instruct-2507`, já configurado) e quantiza em NF4 na hora via `bitsandbytes`; GGUF só entraria depois, na hora de SERVIR o adapter treinado via Ollama, não como entrada de treino — confirmado por busca (blog oficial da Hugging Face sobre bitsandbytes/QLoRA); (2) GPU gratuita do Kaggle é 1x Tesla P100 (16GB) ou 2x Tesla T4 (16GB cada, só a primeira usada — este projeto não faz treino multi-GPU), 30h/semana de cota, sessão de até ~9-12h — mais VRAM que a RTX A2000 (12GB) já validada nesta sessão como suficiente pro Qwen3-4B, então deveria caber com folga. Implementado: `configs/train_kaggle.yaml` (mesmo padrão de `train_a2000.yaml` — `run_on_colab: false`, `sequence_length: 2048` como expectativa conservadora não medida, `require_gpu_name_contains: "Tesla"` pra aceitar qualquer uma das duas GPUs que o Kaggle pode sortear). Notebook (`notebooks/train_granite_l4.ipynb`) ganhou detecção de ambiente Kaggle nas mesmas 3 células que já tratavam Colab-vs-local: célula 2 (clone do repo — `_running_on_kaggle()` via `KAGGLE_KERNEL_RUN_TYPE`, token do GitHub lido via `kaggle_secrets.UserSecretsClient` em vez de `google.colab.userdata`), célula 5 (mesma troca pra `PRAXIS_OLLAMA_SEARCH_API_KEY`), célula 41 (nota operacional: sem Drive no Kaggle, outputs persistem via "Save Version" do próprio notebook, não uma cópia automática). Célula 0 (markdown) também corrigida: ainda tinha a moldura "especialização em frontend" e uma referência ao `docs/plan_frontend_specialization_wave3.md` já deletado (escapou de `D-frontend-pivot-reverted`) — reescrita pra descrição generalista mencionando as 3 configs (L4/A2000/Kaggle) | Motivação real do usuário pra perguntar sobre Kaggle: quer treinar de verdade sem depender só da GPU remota da A2000 ou do Colab | Verificado ANTES de implementar, não assumido: busca real confirmando GGUF-para-inferência-não-treino (2 fontes), busca real dos specs atuais do Kaggle (30h/semana, P100 16GB ou T4x2 16GB, sessão 9-12h). `TrainConfig.load('configs/train_kaggle.yaml')` carrega certo (`base_model`, `sequence_length`, `require_gpu_name_contains`, `run_on_colab` conferidos). Sintaxe das 3 células do notebook editadas verificada com `ast.parse` (removendo só as magics `!`/`%`, que não são Python puro fora do Jupyter). `nbformat.validate` OK (só o aviso pré-existente de `MissingIDFieldWarning`, não é erro real). Bytes do notebook confirmados UTF-8 válidos direto do disco. O que NÃO foi validado aqui (precisa rodar de verdade num notebook Kaggle, próxima sessão prática): se `kaggle_secrets`/Internet realmente funcionam como documentado, se `sequence_length=2048` é conservador demais ou justo pro P100/T4, VRAM real |
| D-notebook-rename-logos-v3 | Pedido explícito do usuário: renomear `notebooks/train_granite_l4.ipynb` (nome órfão desde a troca de modelo — `D-frontend-pivot-model-swap` já tinha deixado o Granite pra trás, o arquivo só nunca foi renomeado, ao contrário da decisão explícita de sessões anteriores de "não renomear pra não quebrar referência/hábito") pra `notebooks/train_logos-v3.ipynb`, refletindo o nome real do modelo (Logos-3/Conatus). Renomeado via `git mv` (preserva histórico). Todas as referências ao nome antigo atualizadas: `configs/train_a2000.yaml`, `configs/train_kaggle.yaml`, `scripts/vram_smoketest.py`, `README.md`, e as duas menções estruturais em `docs/PLAN.md` (árvore de arquivos da seção de arquitetura original e checklist de arquivos esperados) — as linhas de decisão que narram eventos passados (como `D-kaggle-training-support` acima) mantêm o nome antigo no texto, por descreverem o estado real no momento em que aconteceram, não por estarem desatualizadas | Usuário pediu diretamente, motivado por o nome antigo (`train_granite_l4`) já não descrever nem o modelo (Granite não é mais usado) nem necessariamente a GPU (agora também roda no Kaggle, não só na L4) | `grep -rn "train_granite_l4"` limpo em todo código/config/README após o rename (só sobra nas linhas de decisão histórica do PLAN.md, de propósito). Notebook renomeado confirmado presente em `notebooks/train_logos-v3.ipynb` e ausente no caminho antigo |
| D-kaggle-colab-detection-bugfix | Bug real encontrado pelo usuário rodando de verdade no Kaggle (célula 2, clone do repositório): `_running_on_colab()` (`D-kaggle-training-support`) dava falso positivo, porque o pacote `google.colab` vem PRÉ-INSTALADO na imagem Docker oficial do Kaggle (`kaggle/docker-python`) — `import google.colab` funciona lá mesmo sem ser Colab de verdade, só a ponte de runtime por trás dele que não existe. Resultado real observado: `userdata.get("GH_TOKEN")` travava com `TimeoutException("Secrets can only be fetched when running from the Colab UI")`, capturado pelo `except Exception` e reembrulhado como o `RuntimeError` de troubleshooting do Colab — uma mensagem de erro tecnicamente correta pro cenário errado, confundindo o usuário (ele tinha os secrets certos, no lugar certo, no Kaggle). Fix: `_running_on_colab()` (célula 2 e célula 5, as duas cópias independentes da função) agora chama `_running_on_kaggle()` PRIMEIRO e retorna `False` imediatamente se for Kaggle, sem sequer tentar o `import google.colab` — Kaggle nunca cai no branch do Colab, não importa se o pacote está instalado ali ou não | Só foi descoberto porque o usuário realmente tentou rodar no Kaggle e colou o traceback real — a suspeita de que "o Docker do Kaggle vem com tudo pré-instalado, inclusive coisas irrelevantes tipo google-colab" é plausível mas nunca teria sido encontrada só lendo código, precisava da execução real | Reproduzido o bug exato antes de aplicar o fix: mockado `sys.modules["google.colab"]` + `KAGGLE_KERNEL_RUN_TYPE` setado, confirmado que a versão antiga de `_running_on_colab()` retornava `True` (bug reproduzido). Depois do fix, mesmo mock: `_running_on_kaggle()` → `True`, `_running_on_colab()` → `False`, `_clone_dir()` → `/kaggle/working/Conatus-Logos` (branch certo). Sintaxe das 2 células reeditadas verificada com `ast.parse`, `nbformat.validate` OK, bytes UTF-8 confirmados direto do disco. Suite Python completa: 183/183 |
| D-optional-secret-nonfatal | Segundo bug real encontrado pelo usuário rodando no Kaggle, imediatamente depois do fix anterior: célula 5 (leitura de `PRAXIS_OLLAMA_SEARCH_API_KEY`) travava com `BackendError: "No user secrets exist for kernel id ... and label PRAXIS_OLLAMA_SEARCH_API_KEY"` — o usuário tinha `GH_TOKEN`/`HF_TOKEN` configurados no Kaggle, mas não esse secret (que é opcional, só usado pela ferramenta `web_search`, D14/seção 5.4, nunca pelo QLoRA em si). A célula 7 (smoke test do `web_search`) já tratava falha como AVISO não-fatal desde uma sessão anterior, mas a célula 5 (leitura do secret em si) nunca tinha recebido o mesmo tratamento — assumia implicitamente que o secret sempre existia em Colab/Kaggle. Fix: `userdata.get(...)`/`UserSecretsClient().get_secret(...)` agora dentro de `try/except Exception`, mesmo padrão da célula 7 — falha vira `print` de aviso (com instrução de como configurar o secret se quiser), a variável de ambiente simplesmente não é setada, e a célula 7 (já tolerante) lida com a ausência normalmente | Mesma sessão do fix anterior, achado ao vivo tentando rodar de verdade — não é específico do Kaggle, o mesmo crash aconteceria no Colab se `GH_TOKEN` existisse mas esse secret em particular não | Reproduzido o erro exato antes de corrigir: mock de `kaggle_secrets.UserSecretsClient` levantando a mesma `BackendError` com a mesma mensagem do traceback real do usuário, célula 5 executada de ponta a ponta sem lançar exceção (chega ao fim, só imprime o aviso). Sintaxe verificada com `ast.parse`, `nbformat.validate` OK, bytes UTF-8 confirmados direto do disco |
| D-kaggle-p100-incompatible | Terceiro problema real da mesma sequência de tentativas do usuário no Kaggle, o mais sério dos três: com o accelerator "GPU P100" (recomendado por mim anteriormente, por raciocínio de throughput bf16/fp16 vs T4, SEM verificar compatibilidade de software), o download do modelo funcionou (8.04GB, `Qwen/Qwen3-4B-Instruct-2507` — confirma de vez que NÃO é gated, resolvendo uma incerteza aberta desde `D-frontend-pivot-model-swap`), mas o carregamento dos pesos quantizados travou em `Error named symbol not found at line 62 in file /src/csrc/ops.cu`, precedido do aviso real do PyTorch: `"Minimum and Maximum cuda capability supported by this version of PyTorch is (7.0) - (12.0)"` e `"Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation"`. Causa raiz: P100 é arquitetura Pascal (compute capability 6.0/sm_60, GPU de 2016); a imagem Docker do Kaggle hoje vem com um build de PyTorch que só suporta compute capability >= 7.0 (Volta em diante) — o kernel CUDA do `bitsandbytes` pra quantização 4-bit não tem símbolo compilado pra sm_60, daí o "named symbol not found". Não é algo corrigível por config nossa — é uma incompatibilidade de binário da imagem do Kaggle com hardware Pascal. Fix: `configs/train_kaggle.yaml` — comentário extenso documentando o achado, e `require_gpu_name_contains` trocado de `"Tesla"` (aceitava os dois) pra `"T4"` (só aceita T4, rejeita P100 explicitamente) — isso faz a célula 4 abortar RÁPIDO com uma mensagem clara se alguém selecionar P100 de novo, em vez de deixar o erro confuso do bitsandbytes acontecer só depois de baixar 8GB e chegar no meio do carregamento dos pesos | Minha recomendação anterior (nesta mesma sessão, resposta à pergunta "Escolho 2X T4 ou P100?") foi ERRADA — raciocinei por throughput teórico sem verificar se o hardware Pascal do P100 sequer é suportado pelo build de PyTorch da imagem do Kaggle. Só foi descoberto porque o usuário tentou rodar de verdade e colou o traceback real; nenhuma leitura de documentação teria pego isso sem testar | `TrainConfig.load('configs/train_kaggle.yaml')` confirma `require_gpu_name_contains == "T4"`. Nenhum teste do repo referencia `"Tesla"`/`train_kaggle` (checado via grep), nada quebra. Suite Python completa: 183/183. O que ainda falta confirmar (próxima tentativa real do usuário, agora com T4 selecionado): se o carregamento dos pesos completa sem erro, VRAM real, e se `sequence_length=2048` é conservador demais ou justo |

---

## 3. Formato canônico de mensagens

### 3.1 Alternativas avaliadas

| Alternativa | Geração (facilidade p/ modelo 8B) | Parsing | Robustez a saída parcial/inválida | Streaming | Portabilidade de backend | Veredito |
|---|---|---|---|---|---|---|
| 1. Tags XML-like puras (atributos, texto livre) | Média | Frágil (escaping de `<`,`&`,`"` dentro de código quebra o parser) | Baixa | Média | Alta | Rejeitada |
| 2. JSON estrito por turno inteiro (`{"reasoning":..., "tool_call":..., "final":...}`) | Baixa (LLM 8B erra escaping de strings multi-linha com código) | Alta quando válido | Baixa (um objeto JSON só é utilizável inteiro; erro em qualquer campo invalida a mensagem toda) | Baixa (precisa do JSON completo para parsear) | Alta | Rejeitada como formato único, mas usada *dentro* de cada segmento |
| 3. JSON Lines (uma linha JSON por evento) | Média | Alta | Média | Alta | Alta | Considerada, mas linhas únicas são ruins para blocos de raciocínio/código longos e multi-linha |
| 4. Mensagens com roles específicas (`role: tool`) | Depende do template de chat do backend | Alta | Média | Depende do backend | **Baixa** — amarra à implementação do chat template de cada motor de inferência | Rejeitada (viola requisito de harness desacoplado) |
| 5. Tool calling nativo do formato do modelo (chat template do Granite) | Alta no backend nativo, baixa portabilidade | Alta no backend nativo | Média | Depende | **Baixa** | Rejeitada como formato de treino primário; usada só como camada de exportação opcional |
| 6. Gramática formal restrita (decodificação restrita/CFG) | N/A (é imposta na decodificação, não aprendida) | Trivial (por construção) | Alta | Média/Alta | Média (exige suporte a grammar-constrained decoding no backend) | Não é o formato de treino; adotada como **camada de defesa opcional em produção** (ver 3.4) |

### 3.2 Formato escolhido: tags-delimitadoras + corpo JSON estrito por segmento

Um turno de assistente é uma sequência ordenada de segmentos, cada um delimitado por uma tag
sentinela fixa. O corpo de cada `tool_call`/`tool_result` é um objeto JSON estrito validado por
JSON Schema. O corpo de `think`/`final` é texto livre (sem JSON).

```ebnf
turn         := segment+
segment      := think_seg | tool_call_seg | tool_result_seg | final_seg
think_seg    := "<think>" TEXT "</think>"
tool_call_seg   := "<tool_call name=\"" TOOL_NAME "\">" JSON "</tool_call>"
tool_result_seg := "<tool_result name=\"" TOOL_NAME "\" status=\"ok|error\">" JSON "</tool_result>"
final_seg    := "<final>" TEXT "</final>"
```

Regras de trajetória (impostas pelo harness, não só pela gramática):
- `tool_result_seg` **nunca** é gerado pelo modelo — é injetado pelo harness após execução real.
- Um turno só é aceito como completo quando termina em `final_seg`.
- Geração é interrompida (stop-sequence) em `</tool_call>`; o harness injeta o `tool_result`
  correspondente como continuação literal do mesmo texto e retoma a geração a partir daí.

### 3.3 Por que esta escolha vence nos critérios pedidos

- **Fácil de gerar**: o modelo só precisa aprender um vocabulário fixo de ~4 sentinelas; o JSON
  interno de `tool_call` é curto (argumentos), não precisa carregar blocos de código longos
  escapados como string quando isso é evitável (ver 3.5 para exceção controlada).
- **Fácil de parsear**: regex/state-machine simples extrai segmentos por sentinela; cada
  `JSON` é parseado isoladamente — um JSON malformado invalida *só aquele segmento*, não o
  turno inteiro.
- **Resistente a saída parcialmente inválida**: se `<tool_call>` fecha com JSON quebrado, o
  harness gera `TOOL_CALL_PARSE_ERROR` e devolve isso como `tool_result` de erro — o modelo
  aprende a se corrigir sem que a sessão inteira precise ser descartada.
- **Compatível com SFT**: máscara de loss por segmento é trivial (mascarar tudo que não seja
  `think`/`tool_call`/`final` gerado pelo próprio modelo).
- **Streaming**: o harness detecta o fechamento de uma tag incrementalmente token a token.
- **Extensível**: nova ferramenta = novo `name` + novo JSON Schema; a gramática do formato não
  muda.
- **Testável**: gramática pequena o bastante para ter suite de testes unitários exaustiva
  (seção 19).
- **Independente de biblioteca única**: funciona em qualquer backend que suporte
  "gerar com stop-strings e continuar a partir de um prefixo" — Transformers, vLLM, llama.cpp e
  APIs remotas oferecem isso.
- **Seguro para execução automatizada**: nada executa a partir de texto solto; só a partir de
  JSON validado por schema dentro de uma tag reconhecida.

### 3.4 Camada de defesa adicional (opcional, não é o formato de treino)

Em produção, o harness pode aplicar **grammar-constrained decoding** (ex.: `outlines`,
`lm-format-enforcer`, ou grammars nativas do llama.cpp) restrito à gramática da seção 3.2 como
rede de segurança adicional contra saídas malformadas — isso é um recurso de robustez de
*inferência*, não substitui o treino do formato via SFT.

### 3.5 Exceção controlada para conteúdo grande (arquivos/patches)

Para `write_file` e (fase 2) `apply_patch`, o conteúdo do arquivo continua dentro do campo JSON
`content` como string (consistente com como Claude/OpenAI já fazem tool calling em produção —
prática validada em escala). Não se introduz um sub-formato alternativo (ex. bloco cercado fora
do JSON) para não duplicar as regras de parsing; o custo de escaping de `\n`/`\"` é aceitável e
o dataset deve conter exemplos suficientes desse padrão para o modelo aprender a fazê-lo de
forma confiável.

### 3.6 Modos de treino/produção (dev vs prod)

- **Modo dev**: dataset e probes contêm o turno completo, incluindo `<think>`. Usado para
  treino, depuração, e para o avaliador medir qualidade de raciocínio.
- **Modo prod**: o `Renderizador de Resposta Pública` (seção 5) descarta tudo exceto o conteúdo
  de `<final>` antes de mostrar ao usuário final da CLI. Nenhuma re-treinagem é necessária para
  alternar entre os modos — é puramente uma política do harness sobre o que é exibido.

---

## 4. Schemas das ferramentas

Todas as ferramentas são registradas com: `name`, `version`, `input_schema` (JSON Schema),
`output_schema` (JSON Schema), `side_effects` (`none|read|write|exec`), `requires_confirmation`
(bool). Abaixo, os schemas de entrada das ferramentas do MVP (checker, read_file, write_file,
list_files, shell) em detalhe, e um esboço para as ferramentas de fase 2+.

### 4.1 `checker` (MVP)

```json
{
  "name": "checker",
  "version": "1.0",
  "input_schema": {
    "type": "object",
    "required": ["language", "operation", "files"],
    "properties": {
      "language": {"enum": ["python", "go", "javascript", "typescript", "sql", "shell"]},
      "operation": {"enum": ["syntax_check", "compile", "compile_and_test", "run", "lint"]},
      "files": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["path", "content"],
          "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
          }
        }
      },
      "entrypoint": {"type": "string"},
      "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 60000, "default": 15000}
    }
  }
}
```

Saída (contrato único, ver seção 10 para detalhamento completo):

```json
{
  "passed": false,
  "errors": [{"code": "COMPILATION_ERROR", "message": "...", "file": "main.go", "line": 12}],
  "stdout": "",
  "stderr": "...",
  "metadata": {"language": "go", "duration_ms": 354}
}
```

### 4.2 `read_file` (MVP)

```json
{
  "input_schema": {
    "type": "object",
    "required": ["path"],
    "properties": {
      "path": {"type": "string"},
      "start_line": {"type": "integer", "minimum": 1},
      "end_line": {"type": "integer", "minimum": 1}
    }
  }
}
```

### 4.3 `write_file` (MVP)

```json
{
  "input_schema": {
    "type": "object",
    "required": ["path", "content"],
    "properties": {
      "path": {"type": "string"},
      "content": {"type": "string"},
      "mode": {"enum": ["create", "overwrite"], "default": "overwrite"}
    }
  },
  "requires_confirmation": false,
  "side_effects": "write"
}
```

### 4.4 `list_files` (MVP)

```json
{
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "default": "."},
      "glob": {"type": "string"},
      "max_depth": {"type": "integer", "default": 3}
    }
  }
}
```

### 4.5 `shell` (MVP, altamente restrito, versão 2.0 — D-shell-v2)

```json
{
  "input_schema": {
    "type": "object",
    "required": ["binary", "args"],
    "properties": {
      "binary": {"type": "string"},
      "args": {"type": "array", "items": {"type": "string"}, "default": []},
      "cwd": {"type": "string", "default": "."},
      "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 30000, "default": 10000}
    }
  },
  "requires_confirmation": true,
  "side_effects": "exec"
}
```

**D-shell-v2** (substitui o schema `{"command": string}` da v1): o modelo emite `binary`/`args`
já estruturados (argv), nunca uma string de shell livre. O executor invoca
`subprocess.run([binary, *args], shell=False)` — **sem** `shell=True` no meio. Isso torna a
execução verdadeiramente **independente de sistema operacional**: o mesmo código Python invoca
o binário diretamente via a API nativa do SO (`CreateProcess` no Windows, `fork`+`exec` no
POSIX), sem que nenhum interpretador de shell (`cmd.exe`/`bash`) precise existir ou ser
escolhido pelo harness. Consequências:

- Fecha uma classe inteira de risco de injeção via metacaracteres de shell (`;`, `&&`, `|`,
  `` ` ``, `$()`) — não há shell nenhum interpretando a string, então não há como encadear
  comandos escapando da chamada única pretendida.
- A allowlist de binários (seção 7.5) e a denylist continuam funcionando: a denylist compara
  contra uma reconstrução textual de `binary + args` só para fins de checagem — essa string
  nunca é executada, só inspecionada.
- Efeito colateral positivo sobre o objetivo de eficiência/portabilidade (seção 5.5): o modelo
  aprende **um único dialeto de comando** (argv de um binário), não dois (sintaxe `cmd.exe` vs.
  `bash`) escolhidos por detecção de SO em tempo de execução — mais simples de aprender via SFT
  e mais fácil de portar para o harness Go (M8), já que `os/exec` em Go também expõe uma API
  argv-based sem shell por padrão.
- Trade-off aceito: perde-se pipe/redirecionamento/encadeamento nativo de shell
  (`cmd1 | cmd2`, `a && b`) numa única chamada — mas isso já se encaixava mal no loop do agente
  (seção 6), que é "uma ação por passo, verifica, continua"; multi-comando vira múltiplas
  chamadas de ferramenta, não uma string opaca.

### 4.6 `web_search` (habilitada — D14)

```json
{
  "name": "web_search",
  "version": "1.0",
  "input_schema": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {"type": "string", "minLength": 1},
      "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
    }
  },
  "requires_confirmation": false,
  "side_effects": "read"
}
```

Saída:

```json
{"results": [{"title": "...", "url": "...", "snippet": "..."}]}
```

**D14**: diferente das outras ferramentas do MVP, `web_search` não fala direto com um processo local —
passa por uma interface `SearchBackend` (`src/search/base.py`) selecionável via
`configs/search_backend.yaml` (`provider: ollama|mock|...`), resolvida por uma factory
(`build_search_backend`) — o executor (`src/tools/executors/web_search_tool.py`) nunca
instancia um backend concreto diretamente. Backend inicial é a API de busca da Ollama; trocar
de motor no futuro é mudar a config, não o código chamador. Falha de rede/config vira
`SANDBOX_ERROR` (não é tratada como comportamento incorreto do modelo — a ferramenta estava
disponível e o modelo a usou corretamente, o problema é externo).

### 4.7 `search_code` (habilitada — D-search-code-reenable)

```json
{
  "name": "search_code",
  "version": "1.0",
  "input_schema": {
    "type": "object",
    "required": ["pattern"],
    "properties": {
      "pattern": {"type": "string", "minLength": 1},
      "path": {"type": "string", "default": "."},
      "regex": {"type": "boolean", "default": false}
    }
  },
  "requires_confirmation": false,
  "side_effects": "read"
}
```

Executor real (`src/tools/executors/search_code_tool.py`) percorre o workspace em Python puro
(sem depender de `grep`/`rg` no PATH), casando texto literal ou regex por linha. Saída:

```json
{"matches": [{"path": "src/Button.tsx", "line": 2, "text": "..."}], "truncated": false}
```

`truncated: true` sinaliza que o limite de 200 ocorrências foi atingido (evita despejar um
repositório inteiro numa única chamada); o modelo deve refinar `pattern`/`path` em vez de
assumir que viu tudo.

### 4.8 Fase 2+: `apply_patch`, `git_diff` (esboço de contrato, ainda desabilitadas — D5)

```json
// apply_patch — fase 2
{"input_schema": {"type": "object", "required": ["path", "diff"],
  "properties": {"path": {"type": "string"}, "diff": {"type": "string", "description": "unified diff"}}}}

// git_diff — fase 2
{"input_schema": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "staged": {"type": "boolean", "default": false}}}}
```

---

## 5. Desenho do harness

### 5.1 Módulos e contratos

```
┌───────────────┐   texto/stop-seq   ┌──────────────────┐
│  Model Runner  │◄──────────────────┤  Agent Loop Ctrl │
│ (backend-agn.) ├───────────────────►│                  │
└───────────────┘   completions      └────────┬─────────┘
                                               │ segmentos
                                               ▼
                                     ┌──────────────────┐
                                     │ Tool Call Parser │  (regex + JSON.parse por segmento)
                                     └────────┬─────────┘
                                               │ ToolCall{name,args,raw}
                                               ▼
                                     ┌──────────────────┐
                                     │  Schema Validator │  (JSON Schema por ferramenta)
                                     └────────┬─────────┘
                                     válido │      │ inválido
                                             ▼      ▼
                                  ┌──────────────┐ ┌────────────────────┐
                                  │ Tool Registry │ │ Erro estruturado    │
                                  │ + Executor    │ │ (TOOL_ARGUMENT_...) │
                                  └───────┬──────┘ └─────────┬──────────┘
                                          ▼                  │
                                  ┌──────────────┐            │
                                  │   Sandbox     │            │
                                  │ (seção 7)     │            │
                                  └───────┬──────┘            │
                                          ▼                  ▼
                                  ┌───────────────────────────────┐
                                  │  Context Manager (injeta         │
                                  │  <tool_result> e continua)       │
                                  └───────────────┬───────────────┘
                                                  ▼ (loop até <final> ou limite)
                                  ┌───────────────────────────────┐
                                  │        Checker final           │
                                  │  (verificação pré-resposta)     │
                                  └───────────────┬───────────────┘
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │  Public Response Renderer      │
                                  │  (extrai só <final> em prod)   │
                                  └───────────────┬───────────────┘
                                                  ▼
                                            Usuário / CLI
        (Logger e Evaluator observam todas as etapas via hooks, fora do caminho crítico)
```

### 5.2 Interfaces (contratos, não implementação)

- **Model Runner**: `generate(prompt: str, stop: list[str], max_tokens: int) -> Completion`.
  Implementações: `TransformersRunner`, futuramente `VLLMRunner`, `LlamaCppRunner`,
  `RemoteAPIRunner`. Nenhum outro módulo conhece detalhes de backend.
- **Tool Call Parser**: `parse(text: str) -> list[Segment]`, onde `Segment` é um dos tipos
  `Think | ToolCall | Final | Malformed`. `Malformed` carrega o motivo (`TOOL_CALL_PARSE_ERROR`).
- **Tool Registry**: mapa `name -> ToolSpec{input_schema, output_schema, executor, side_effects}`.
  Registro é a única fonte de verdade sobre quais ferramentas existem nesta versão do harness —
  o dataset de cada fase só pode referenciar ferramentas presentes no registro daquela fase.
- **Tool Executor**: `execute(tool: ToolSpec, args: dict, ctx: SandboxContext) -> ToolResult`.
  Sempre passa por sandbox (seção 7); nunca chama `subprocess` diretamente fora dela.
- **Checker**: biblioteca standalone (seção 10) com `check(language, operation, files) -> CheckResult`,
  consumida como ferramenta (`checker` tool), como validador de dataset e como métrica de avaliação.
- **Context Manager**: mantém o estado da trajetória (lista de segmentos), decide o texto exato
  a re-injetar como prompt de continuação, aplica truncamento de contexto quando necessário.
- **Agent Loop Controller**: orquestra o ciclo completo (seção 6), aplica limites (passos,
  tempo, repetição de chamadas idênticas).
- **Public Response Renderer**: `render(trajectory, mode: "dev"|"prod") -> str`.
- **Logger**: grava trajetória completa + timestamps + custos, com redaction de segredos
  (seção 7.9) antes de persistir.
- **Evaluator**: consome trajetórias logadas + `CheckResult`s para computar métricas (seção 13);
  roda **depois** do harness, nunca no caminho crítico da resposta ao usuário.

### 5.3 Por que esses limites de módulo

Cada seta no diagrama é uma função pura ou um contrato serializável (JSON) — nenhum módulo
importa tipos internos de outro. Isso permite trocar `Model Runner` por vLLM ou trocar
`TransformersRunner` por uma API remota sem tocar em parser, registry, sandbox ou avaliador.

### 5.4 `web_search`: interface `SearchBackend` parametrizada, backend inicial = Ollama Search API

Diferente do plano original (mock puro), `web_search` já nasce com um backend real —
a API de busca da Ollama — mas **nunca chamado diretamente pelo resto do sistema**: todo
chamador (harness, gerador de dataset, avaliador) depende apenas da interface `SearchBackend`;
qual implementação concreta roda é decidido por `configs/search_backend.yaml`, não por código.

```python
class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str

class SearchBackend(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...

class MockSearchBackend(SearchBackend):
    """Corpus fixo/curado, sem rede. Usado para geração determinística de dataset/probes."""

class OllamaSearchBackend(SearchBackend):
    """
    Backend real via API de busca da Ollama.
    Endpoint e payload exatos devem ser confirmados contra a documentação vigente da Ollama
    no momento da implementação (API relativamente nova); assumir por ora:
      POST {endpoint} com header "Authorization: Bearer {api_key}"
      body {"query": query, "max_results": max_results}
    A chave NUNCA é lida de um valor hardcoded — sempre de variável de ambiente
    (nome definido em config, não o valor).
    """
    def __init__(self, endpoint: str, api_key_env: str, timeout_ms: int = 8000):
        self._endpoint = endpoint
        self._api_key = os.environ[api_key_env]  # falha explícita se a env var não existir
        self._timeout = timeout_ms / 1000

def build_search_backend(config: SearchBackendConfig) -> SearchBackend:
    """Factory único ponto de decisão sobre qual motor de busca está ativo."""
    match config.provider:
        case "mock":
            return MockSearchBackend(config.mock)
        case "ollama":
            return OllamaSearchBackend(**config.ollama)
        case _:
            raise UnsupportedSearchProvider(config.provider)
```

`configs/search_backend.yaml`:

```yaml
search_backend:
  provider: "ollama"        # trocar para "mock" ou um novo provider sem mexer em código
  ollama:
    endpoint: "https://ollama.com/api/web_search"   # confirmar contra doc oficial na implementação
    api_key_env: "PRAXIS_OLLAMA_SEARCH_API_KEY"      # valor real fica só na env do ambiente/Colab secret
    timeout_ms: 8000
    max_results: 5
  mock:
    fixtures_path: "data/fixtures/web_search/"
```

**Reprodutibilidade vs. busca ao vivo (importante)**: resultados de busca real mudam com o
tempo, o que colide com o requisito de reprodutibilidade (seção 11.4) e com prevenção de
vazamento (seção 8.3) se `benchmark`/`probes` dependessem de respostas ao vivo. Por isso:
- Geração de exemplos para `train`/`validation`/`benchmark`/`probes` pode **usar** o
  `OllamaSearchBackend` para *coletar* resultados, mas o resultado é **congelado como fixture**
  (salvo em `data/fixtures/web_search/`, versionado, com timestamp de coleta) e o exemplo final
  no dataset consome esse fixture via `MockSearchBackend` — nunca reconsulta a rede em tempo de
  treino/avaliação.
- Apenas o harness interativo/de produção (uso real do agente, fora do benchmark) consulta o
  `OllamaSearchBackend` ao vivo.

Isolar o *provider* atrás de `SearchBackend` + factory-por-config é exatamente o que permite
trocar de motor de busca depois (outro provedor, ou uma API própria) alterando apenas
`search_backend.yaml`, sem tocar em harness, dataset ou avaliador — e essa mesma interface é o
contrato que a futura implementação em Go do harness (seção 5.5) precisa replicar.

### 5.5 Direção arquitetural: harness Python (referência) vs. harness Go (produção)

O propósito declarado do modelo é ser eficiente, rápido e "inteligente" em uso real — isso é um
requisito de **harness de produção**, não do processo de treino. Consequência direta:

- **Nesta geração (M0–M6)**, o harness é implementado em **Python**, porque está fortemente
  acoplado à infraestrutura de ML (Transformers/PEFT/bitsandbytes/TRL) usada para gerar dados de
  treino, validar dataset e rodar avaliação/probes durante o desenvolvimento. Não há motivo para
  pagar o custo de portar isso para Go antes de o formato e o loop estarem provados.
- **A partir de M8 (pós-MVP, seção 16)**, o harness que efetivamente roda dentro da futura CLI
  será **reimplementado em Go**. Motivação: binário estático único (distribuição trivial, sem
  runtime Python no ambiente do usuário final), baixo overhead de memória e de startup,
  concorrência nativa (goroutines) para paralelizar execução de múltiplas ferramentas/chamadas
  de checker sem a sobrecarga de multiprocessing do Python, e menor superfície de dependências
  em tempo de execução — tudo alinhado ao objetivo de eficiência declarado para o produto final.

Esse porte é de **baixo risco** porque, desde a seção 3, todos os contratos foram desenhados
para serem agnósticos de linguagem de implementação:

| Contrato | Por que porta sem atrito para Go |
|---|---|
| Gramática de segmentos (seção 3.2) | É apenas reconhecimento de sentinelas de texto + `json.Unmarshal` por segmento — Go faz isso de forma nativa e mais rápida que Python via `regexp`/scanner manual |
| JSON Schemas das ferramentas (seção 4) | Schemas são JSON puro; validação em Go via bibliotecas padrão do ecossistema (ex. `santhosh-tekuri/jsonschema`) sem precisar reescrever regras, só reapontar o validador |
| `Model Runner` (seção 5.2) | Já é definido como fronteira HTTP/gRPC (`generate(prompt, stop, max_tokens)`); em produção isso vira um cliente Go falando com um servidor de inferência (vLLM/llama.cpp server/API remota) — nenhuma lógica de modelo roda embutida no processo Go |
| `Tool Executor`/Sandbox (seção 7) | `os/exec` + `context.WithTimeout` + rlimits nativos do Go cobrem exatamente o que hoje é feito via `subprocess` em Python, com menor overhead por processo filho |
| `SearchBackend` (seção 5.4) | Mesma interface e mesmo `search_backend.yaml`; a implementação Go do `OllamaSearchBackend` só troca a linguagem do cliente HTTP, não o contrato |

Consequência prática para o desenho atual: **nenhuma decisão de contrato (seções 3, 4, 5.2, 7,
10.1) deve introduzir dependência de biblioteca Python específica** — elas já não introduzem,
e este requisito passa a ser um critério de revisão para qualquer mudança futura nesses
contratos. O checker (seção 10), por ser um processo externo por linguagem (compiladores,
interpretadores), já é chamado por subprocess tanto de Python quanto seria de Go — não muda.

Isso é registrado como **M8** no roadmap (seção 16) e como item de fora-de-escopo desta geração
(seção 18): a decisão é tomada agora para orientar o desenho, mas a implementação em Go não
começa antes de M6 validar o formato/loop com o harness Python.

---

## 6. Desenho do loop do agente (pseudocódigo)

```python
def run_agent_loop(user_request, workspace, config):
    trajectory = Trajectory(system_prompt(config), user_request)
    seen_calls = set()

    for step in range(config.max_steps):
        completion = model_runner.generate(
            prompt=trajectory.render_for_model(),
            stop=["</tool_call>", "</final>"],
        )
        trajectory.append_raw(completion.text)

        segment = parser.parse_last(trajectory)

        if segment.kind == "final":
            break

        if segment.kind == "malformed":
            trajectory.append_tool_result(
                name=segment.attempted_name or "unknown",
                status="error",
                body={"code": "TOOL_CALL_PARSE_ERROR", "message": segment.reason},
            )
            continue

        if segment.kind == "tool_call":
            call_fingerprint = (segment.name, canonicalize(segment.args))

            if call_fingerprint in seen_calls:
                trajectory.append_tool_result(
                    name=segment.name, status="error",
                    body={"code": "MAX_STEPS_EXCEEDED", "message": "chamada repetida sem progresso"},
                )
                continue
            seen_calls.add(call_fingerprint)

            tool = registry.get(segment.name)
            if tool is None:
                trajectory.append_tool_result(segment.name, "error", {"code": "UNSUPPORTED_TOOL"})
                continue

            validation = schema_validator.validate(tool.input_schema, segment.args)
            if not validation.ok:
                trajectory.append_tool_result(segment.name, "error", {
                    "code": "TOOL_ARGUMENT_SCHEMA_ERROR", "message": validation.errors,
                })
                continue

            if tool.requires_confirmation and not confirmation_policy.allow(tool, segment.args, config):
                trajectory.append_tool_result(segment.name, "error", {"code": "UNSAFE_COMMAND"})
                continue

            result = tool_executor.execute(tool, segment.args, sandbox_context(workspace, config))
            trajectory.append_tool_result(segment.name, "ok" if result.passed else "error", result.to_json())
            logger.record_step(step, segment, result)
            continue

    else:
        trajectory.force_final(reason="MAX_STEPS_EXCEEDED")

    final_check = checker.verify_consistency(trajectory)  # seção 10.3
    if not final_check.passed:
        trajectory.annotate_internal(final_check.errors)  # não exposto ao usuário

    public_output = renderer.render(trajectory, mode=config.mode)  # "dev" ou "prod"
    logger.persist(trajectory, redact=True)
    evaluator.observe_async(trajectory, final_check)  # fora do caminho crítico

    return public_output
```

Pontos que implementam requisitos explícitos da seção 6 do pedido:
- **Limite de passos**: `config.max_steps`, com `force_final` no `else` do `for`.
- **Não repetir chamadas idênticas**: `seen_calls` com fingerprint canônico dos argumentos.
- **Validação por schema antes de executar**: `schema_validator.validate` bloqueia execução.
- **Comandos destrutivos exigem autorização**: `confirmation_policy.allow` (seção 7.7).
- **Nunca confiar em resultado fabricado pelo modelo**: `tool_result` só é escrito por
  `trajectory.append_tool_result`, chamado exclusivamente pelo harness — o parser rejeita
  qualquer `<tool_result>` que apareça na saída *gerada pelo modelo* como `INTERNAL_REASONING_LEAK`
  ou `TOOL_RESULT_FABRICATION` dependendo do contexto (seção 10).
- **Verificação antes da resposta final**: `checker.verify_consistency` roda sempre antes do
  render, mesmo quando o modelo não chamou `checker` espontaneamente.

---

## 7. Estratégia de segurança

### 7.1 Princípio central
Nenhum texto produzido pelo modelo é executado diretamente no host. Tudo passa por: validação
de schema → allowlist/denylist → sandbox de processo isolado com limites de recurso.

### 7.2 Sandbox de execução
- Processo filho (`subprocess`) com `cwd` fixado em um **diretório de trabalho temporário
  descartável** por sessão (criado em `tempfile.mkdtemp()`, apagado ao final).
- Sem acesso de rede por padrão (fase 1): variável de ambiente e, quando disponível no host
  Linux do Colab, `unshare --net` ou equivalente; caso indisponível, ao menos bloqueio a nível
  de allowlist de comandos que fazem rede (`curl`, `wget`, `pip install` sem `--no-index`, `git clone` remoto).
- **[HIPÓTESE D6]**: Docker-in-Docker não é assumido disponível/confiável em Colab; o
  isolamento é subprocess + rlimits + diretório efêmero. Isso é suficiente para o MVP (código
  gerado por um modelo próprio, não input adversarial de terceiros) mas **não é suficiente**
  para uma futura CLI que execute código de repositórios de terceiros sem revisão — essa
  lacuna é registrada como risco (seção 17) e pré-requisito para produção real.

### 7.3 Limites de recurso
- `timeout_ms` por chamada de ferramenta (default 10-15s, teto 60s para `checker` em
  `compile_and_test`).
- Limite de memória via `resource.setrlimit(RLIMIT_AS, ...)` no processo filho (Linux/Colab).
- Limite de tamanho de saída: `stdout`/`stderr` truncados a N KB antes de voltar ao contexto do
  modelo (evita explosão de tokens e ataques de output gigante).

### 7.4 Filesystem restrito
- Todas as ferramentas de arquivo (`read_file`, `write_file`, `list_files`) resolvem caminhos
  relativos ao `workspace` da sessão e rejeitam qualquer `..`/caminho absoluto que escape dele.

### 7.5 Allowlist/denylist de comandos (`shell`)
- Allowlist inicial do MVP: `python`, `pip` (com `--no-index` quando cache local disponível),
  `go`, `pytest`, `ls`, `cat`, `grep`, `git` (subcomandos somente leitura: `status`, `diff`,
  `log`), ferramentas de lint (`ruff`, `gofmt`, `go vet`).
- Denylist explícita (bloqueada mesmo que composta): `rm -rf`, `sudo`, `curl`/`wget` sem
  allowlist de domínio, `dd`, `mkfs`, `shutdown`/`reboot`, `chmod 777`, redirecionamento para
  fora do workspace.
- Comandos fora da allowlist retornam `UNSAFE_COMMAND` sem executar.

### 7.6 Controle de rede
- Rede permanece **desabilitada por padrão** para `shell` e para instalação de pacotes fora de
  um cache local pré-populado no ambiente do Colab — nenhuma mudança aqui.
- **Exceção única e explícita**: a ferramenta `web_search`, quando configurada com
  `provider: "ollama"` (seção 5.4), tem permissão de sair para exatamente o host do endpoint
  configurado (`ollama.com`, via HTTPS), através do `OllamaSearchBackend` — nenhum outro
  componente do sistema (shell, checker, execução de código gerado) herda esse acesso. Essa
  exceção é implementada como uma allowlist de host de rede específica da ferramenta
  `web_search`, não como abertura geral de rede na sandbox.
- Para geração de dados de treino/avaliação (`train`/`validation`/`benchmark`/`probes`), o
  acesso de rede do `web_search` só é usado no momento da coleta/curadoria (fora do loop de
  treino), e os resultados são congelados como fixtures consumidas depois via
  `MockSearchBackend` (seção 5.4) — o loop de treino e a avaliação reproduzível continuam sem
  dependência de rede.

### 7.7 Confirmação para operações perigosas
- `requires_confirmation=true` em `shell` e em qualquer operação futura marcada como destrutiva.
- No harness de desenvolvimento, confirmação é uma política configurável (`auto_approve_safe`,
  `deny_all_destructive`); na CLI futura, vira prompt real ao usuário humano.

### 7.8 Tratamento de secrets e logs
- Variáveis de ambiente sensíveis (tokens, chaves) nunca são propagadas para o subprocess da
  sandbox por padrão (allowlist explícita de env vars, não denylist).
- `Logger` aplica redaction por regex (padrões de chave de API comuns, tokens JWT, etc.) antes
  de persistir qualquer trajetória ou saída de ferramenta.

### 7.9 Política de instalação de pacotes
- Fase 1: nenhuma instalação dinâmica de pacotes durante a execução do agente; ambiente de
  execução vem pré-provisionado (imagens/venvs fixos por linguagem). Instalação dinâmica fica
  para fase 2+, sempre restrita a um índice espelhado/allowlist de pacotes.

### 7.10 Limpeza do ambiente
- Diretório de trabalho temporário e quaisquer processos filhos são destruídos ao final de cada
  chamada de ferramenta (não apenas ao final da sessão), para minimizar janela de exposição.

---

## 8. Estratégia de dataset

### 8.1 Taxonomia de exemplos (mapeando 1:1 os tipos pedidos na seção 8)

Cada exemplo pertence a exatamente um `task_type` dentre:

`direct_answer`, `single_tool_call`, `multi_tool_call`, `invalid_call_then_correction`,
`checker_rejects_code`, `model_fixes_after_error`, `compiles_successfully`, `test_fails`,
`test_passes`, `tool_unavailable`, `insufficient_information`, `ambiguous_request`,
`forbidden_operation`, `multi_file_read_and_edit`, `refactor`, `debugging`, `test_authoring`,
`complexity_analysis`, `code_explanation`, `language_migration`, `project_configuration`,
`documentation_usage`.

Cada `task_type` recebe exemplos **positivos** (comportamento correto) e, quando aplicável,
**negativos** — representando exatamente as falhas listadas na seção 8 do pedido
(API inexistente, import inventado, tool call malformado, tool result fabricado, código não
compilável, teste incorreto, checker ignorado, resposta final contradizendo o código, arquivo
errado alterado, comando destrutivo, loop de ferramentas, raciocínio que afirma validação sem
validação real, vazamento de texto interno).

Exemplos negativos são usados de duas formas distintas — isso precisa ficar explícito para não
confundir o treino:
1. **Negativo-como-trajetória-de-correção**: o exemplo mostra a falha *e* a correção subsequente
   dentro da mesma trajetória (rótulo de loss ativo apenas na parte corrigida/correta).
2. **Negativo-como-caso-de-avaliação**: usado só em probes/benchmark para medir se o modelo
   evita a falha — nunca teria loss computado sobre o comportamento ruim em si.

### 8.2 Metadados obrigatórios por exemplo

```json
{
  "id": "uuid",
  "domain": "backend|cli|infra|testing|refactor|...",
  "language": "python|go|...",
  "difficulty": "easy|medium|hard",
  "tools_used": ["checker", "read_file"],
  "num_steps": 3,
  "task_type": "model_fixes_after_error",
  "source": "synthetic_generated|curated_manual|adapted_from_public_repo(license=MIT)",
  "license": "MIT|Apache-2.0|synthetic-no-license-needed|...",
  "validation_status": "validated|rejected|pending",
  "execution_performed": true,
  "checker_used": "checker-python-1.0",
  "expected_result": {"passed": true, "test_status": "test_passes"},
  "known_risks": ["pode depender de versão específica de stdlib"]
}
```

### 8.3 Splits e prevenção de vazamento

| Split | Propósito | Regra de isolamento |
|---|---|---|
| `train` | treino | — |
| `validation` | seleção de checkpoint/early signal | nenhuma tarefa (mesmo `problem_id` base) compartilhada com `train` |
| `probes` | testes direcionados por comportamento (seção 13) | escritos manualmente/curados, nunca gerados pelo mesmo gerador sintético do `train` |
| `benchmark interno` | medição de progresso ao longo do projeto, congelado por versão | congelado após primeira geração; alterações viram uma nova versão numerada |
| `casos adversariais` | probes de segurança/robustez (comando destrutivo, tool result fabricado, etc.) | isolado do pipeline de geração sintética padrão |

Split é feito por **grupo de tarefa-base** (não por exemplo individual) — variações da mesma
tarefa-base (ex. mesmo bug reproduzido em 3 linguagens) ficam sempre no mesmo split, para não
vazar a "resposta" de uma variação para outra.

### 8.4 Deduplicação
- Hash exato do texto normalizado (whitespace/case) para deduplicação trivial.
- Deduplicação aproximada opcional (fase 2) via embedding de similaridade para pares
  quase-idênticos gerados sinteticamente em lote.

### 8.5 Licenciamento e proveniência
- Fontes aceitas: geração sintética própria (sem restrição de licença), exemplos curados
  manualmente pelo time, e adaptação de repositórios com licença permissiva explícita
  (MIT/Apache-2.0/BSD), sempre com `source` e `license` preenchidos. Nenhum exemplo de origem
  desconhecida entra em `train`/`validation`/`benchmark`.

### 8.6 Expansão por paráfrase (D-generalization-gap)

**Diagnóstico** (ver D-generalization-gap na seção 2): o dataset atual (556 exemplos) tem
volume razoável de `task_type`s e programas distintos, mas pouquíssima diversidade de
*frase* dentro de cada `task_type` — várias categorias de trajetória multi-passo (justamente
as que dependem de `<tool_call>` bem formado) têm 1–2 gabaritos de frase repetidos dezenas de
vezes com só o nome de função/arquivo trocado. Um teste controlado confirmou que o modelo
executa perfeitamente a frase exata do treino e falha em reformulações da mesma tarefa — não é
um bug de mecanismo (D-train-prompt-mask e D-probe-system-prompt já corrigidos), é falta de
sinal de generalização de frase.

**Objetivo**: multiplicar a diversidade de `user_request` para o MESMO conjunto de trajetórias
já validadas (mesmos `tool_call`/`tool_result`/`<final>`) — não gerar novos programas/tarefas
nesta rodada. Isso é mais barato que expandir o dataset com tarefas inteiramente novas e ataca
exatamente o gargalo medido.

**Priorização por urgência** (razão exemplos/templates distintos, do pior para o melhor —
contagem real em `data/train`+`data/validation`):

| `task_type` | exemplos | templates distintos | prioridade |
|---|---|---|---|
| `model_fixes_after_error` | 41 | 1 | crítica |
| `checker_rejects_code` | 41 | 1 | crítica |
| `debugging` | 16 | 1 | crítica |
| `code_explanation` | 32 | 2 | alta |
| `compiles_successfully` | 26 | 2 | alta |
| `language_migration` | 23 | 2 | alta |
| `single_tool_call` | 89 | 37 | média |
| demais `task_type`s (≥ ~0.8 template/exemplo) | — | — | baixa/adiar |

**Método**: script novo `scripts/expand_dataset_paraphrases.py`, rodando DEPOIS de
`generate_dataset.py`:
1. Para cada exemplo existente cujo `task_type` está na lista de prioridade crítica/alta,
   extrai as variáveis usadas na frase original (nome de função, nome de arquivo, linguagem).
2. Aplica um conjunto de **templates de frase escritos à mão** por `task_type` (6–10 por
   categoria crítica; cobrindo: imperativo direto, pergunta, tom formal/informal, ordem de
   cláusulas diferente, com/sem contexto extra) — substituição determinística de variáveis,
   sem depender de LLM externo (evita custo de API e mantém controle total sobre qualidade,
   consistente com D-shell-v2/D14: preferência por soluções autocontidas quando viável).
3. Gera um novo exemplo por combinação (exemplo base × template de paráfrase), com `id` derivado
   (`{id_original}-para{k}`), **mesma `trajectory.raw_text`/`tool_calls`/`tool_results`/`final`**
   (só `user_request` muda), e `metadata.source="paraphrase_of:{id_original}"`.
4. **Regra de isolamento (seção 8.3) É OBRIGATÓRIA aqui**: toda paráfrase fica no MESMO split
   do exemplo-base (`train`→`train`, `validation`→`validation`) — paráfrases da mesma
   tarefa-base atravessando splits seria vazamento direto.
5. Deduplicação: hash do texto normalizado da nova frase contra tudo que já existe no mesmo
   split, para não gerar paráfrases redundantes entre si.
6. Roda `scripts/validate_dataset.py` no dataset expandido antes de qualquer treino — a
   trajetória em si não muda, então a validação aqui é principalmente sobre a integridade do
   novo `user_request` (não vazio, não idêntico a outro já existente) e a consistência dos
   metadados.

**Escala alvo**: ~6–8 paráfrases por exemplo nas categorias críticas/altas (41+41+16+32+26+23 =
179 exemplos-base × ~7 ≈ 1250 novos exemplos), mantendo as categorias já bem distribuídas
(`multi_tool_call`, `direct_answer`, etc.) como estão nesta rodada — leva o dataset de 556 para
a faixa de 1.500–2.000 já cogitada anteriormente, mas com o aumento concentrado exatamente onde
a lacuna foi medida, não distribuído uniformemente.

**Fora de escopo nesta rodada**: gerar novos `task_type`s, novas linguagens, ou aumentar
`multi_tool_call`/`direct_answer` (já perto de 1 template por exemplo — a alavancagem ali é
baixa comparada às categorias críticas).

---

## 9. Pipeline de validação e geração de dados

Pipeline determinístico, executado em estágios, cada um com códigos de rejeição registrados
(não apenas descartados silenciosamente):

```
1. ingest (raw/)         → carrega exemplos gerados/importados
2. normalize             → normaliza formato de tags, whitespace, encoding
3. structural_validate   → gramática da seção 3.2 (tags bem formadas, ordem válida)
4. schema_validate       → cada tool_call valida contra o JSON Schema da ferramenta (seção 4)
5. extract_code_blocks   → extrai arquivos/código referenciados nas tool_calls
6. execute_when_possible → roda o checker real (compilar/interpretar/testar) por linguagem
7. classify_execution    → registra: static_only | interpreted | compiled | tested |
                                       output_compared | non_executable_external_dep
8. reject_or_accept      → grava motivo estruturado de rejeição (mesmos códigos da seção 10)
9. dedup                 → hash exato (+ opcional near-dup)
10. split_assign         → atribui train/validation/probes/benchmark/adversarial por grupo
11. stats_report         → distribuição por linguagem, task_type, dificuldade, taxa de rejeição
```

Diretórios de saída seguem a estrutura da seção 14 (`data/raw` → `normalized` → `validated`/`rejected` → `train`/`validation`/`probes`).

Nenhum exemplo migra para `validated` sem passar pelos estágios 3–6 (quando `execution_performed`
é aplicável para a linguagem/tarefa); exemplos legitimamente não-executáveis (ex. pergunta
conceitual sem código) são aceitos como `non_executable_external_dep`/N/A explícito, não por
omissão silenciosa.

---

## 10. Estrutura do checker

### 10.1 Contrato único (já especificado pelo usuário, adotado como está)

```json
{
  "passed": false,
  "errors": [{"code": "COMPILATION_ERROR", "message": "...", "file": "main.go", "line": 12}],
  "stdout": "", "stderr": "...",
  "metadata": {"language": "go", "duration_ms": 354}
}
```

### 10.2 Tabela de códigos de erro (adotando a lista do pedido + complementos necessários)

| Código | Quando ocorre |
|---|---|
| `TOOL_CALL_PARSE_ERROR` | tag/JSON malformado na chamada |
| `TOOL_ARGUMENT_SCHEMA_ERROR` | argumentos não batem com o JSON Schema da ferramenta |
| `UNSUPPORTED_TOOL` | nome de ferramenta não existe no registry desta fase |
| `UNSUPPORTED_LANGUAGE` | linguagem não suportada pelo checker nesta fase |
| `SYNTAX_ERROR` | falha de parsing antes até de compilar |
| `COMPILATION_ERROR` | falha de build |
| `RUNTIME_ERROR` | exceção/crash em execução |
| `TEST_FAILURE` | teste executado e falhou |
| `TIMEOUT` | excedeu `timeout_ms` |
| `MISSING_DEPENDENCY` | import/pacote não disponível no ambiente sandbox |
| `INVALID_OUTPUT_FORMAT` | saída não bate com formato esperado pelo checker de comparação |
| `NONEXISTENT_API` | símbolo/função referenciada não existe na API/stdlib real |
| `EXPLANATION_CODE_MISMATCH` | texto de `<final>` contradiz o que o código realmente faz |
| `TOOL_RESULT_FABRICATION` | modelo gerou algo que se parece com `<tool_result>` sem harness ter executado |
| `INTERNAL_REASONING_LEAK` | conteúdo de `<think>` vazou para dentro de `<final>` |
| `MAX_STEPS_EXCEEDED` | loop atingiu limite de passos sem `<final>` |
| `UNSAFE_COMMAND` | comando bloqueado por allowlist/denylist |
| `FILE_NOT_FOUND` | caminho não existe no workspace |
| `PATCH_APPLY_ERROR` | diff não aplica (fase 2, `apply_patch`) |
| `INCOMPLETE_SOLUTION` | tarefa parcialmente resolvida, sem cobrir todos os requisitos pedidos |
| `SANDBOX_ERROR` *(complemento)* | falha da própria infraestrutura de sandbox, não do código do modelo — usado para separar `HARNESS_ERROR` na avaliação |
| `RESOURCE_LIMIT_EXCEEDED` *(complemento)* | estourou memória/CPU configurados |

### 10.3 `verify_consistency` (checagem final antes da resposta pública)

Além da verificação por ferramenta explícita, o harness roda uma checagem final obrigatória
antes de renderizar `<final>`, cobrindo especificamente os erros "comportamentais" que não
dependem de compilar código: `EXPLANATION_CODE_MISMATCH`, `TOOL_RESULT_FABRICATION`,
`INTERNAL_REASONING_LEAK`. Essa checagem combina heurísticas determinísticas (ex.: qualquer
`<tool_result>` na saída do próprio modelo antes de o harness tê-lo injetado é fabricação, por
definição de que o parser nunca deveria ter visto essa tag vinda do modelo) com, opcionalmente,
uma chamada adicional ao próprio `checker` para comparar artefatos citados em `<final>` contra
o estado real dos arquivos do workspace.

### 10.4 Backends por linguagem (ordem de implementação)

| Linguagem | MVP? | Ferramenta de checagem |
|---|---|---|
| Python | Sim | `py_compile` (sintaxe) + `pytest` (teste) |
| Go | Sim | `go build` + `go vet` + `go test` |
| JavaScript | Fase 2 | `node --check` + execução direta |
| TypeScript | Fase 2 | `tsc --noEmit` + execução via `node` após transpile |
| SQL | Fase 3 | parsing via `sqlglot` (sintaxe) + execução opcional em `sqlite`/`duckdb` |
| Shell/Bash | Fase 3 | `shellcheck` + execução sandboxed opcional |
| HTML | Sim (D-checker-html-backend) | renderização real em Chromium headless via Playwright — não é "sintaxe apenas" como o plano original previa; ver seção 10.4.1 |

C/C++, Java e Rust foram removidas do escopo do projeto (não apenas adiadas para uma fase
futura) — ver justificativa consolidada na seção 18.

#### 10.4.1 `html` — backend de frontend estático (D-checker-html-backend)

Item 3 do pivô de especialização em frontend (`docs/plan_frontend_specialization_wave3.md`,
seções 3 e 10). Implementa apenas a **camada 1** do design em camadas descrito lá — "renderiza
sem erro de console/exceção JS" — que é a única camada objetiva e barata de fazer sem
dependências novas além do Playwright (já instalado no ambiente, `pip show playwright` confirma
1.49.1; Chromium baixado e testado de verdade nesta sessão).

- `src/checker/backends/frontend_backend.py`: `render_and_capture(files, entrypoint, timeout_ms,
  screenshot_path=None)` é a função reutilizável — materializa os arquivos em disco, escolhe o
  entrypoint (`index.html` por padrão, ou o único `.html`, ou o `entrypoint` explícito), abre um
  Chromium headless real, escuta `console` (separa `error`/`warning`) e `pageerror`, e opcionalmente
  salva um screenshot real em PNG. `check_html(operation, ...)` embrulha isso no contrato
  `CheckResult` do checker: `run`/`syntax_check`/`compile`/`compile_and_test` todos mapeiam pra o
  mesmo render-check (fase 1 não distingue essas operações — não há build step nem suíte de teste
  pra HTML estático); `lint` (camada 3, acessibilidade via axe-core) retorna `MISSING_DEPENDENCY`
  estruturado de propósito — o pacote `axe-core-python` não está instalado e este projeto nunca
  finge ter rodado uma checagem que não rodou (mesmo espírito de D11).
- `src/schemas/checker.json`: `language` ganhou `"html"` no enum.
- `scripts/render_frontend_preview.py`: CLI standalone que chama `render_and_capture` direto
  (sem passar pela ferramenta `checker`) pra produzir um PNG real de uma página + relatório de
  erros de console — é o "gera, roda no navegador, tira print, eu vejo" que o usuário pediu
  para o julgamento de estética (camada 5, sempre subjetivo/humano, nunca uma métrica
  automática fingida). Testado de ponta a ponta nesta sessão contra uma página real (screenshot
  conferido visualmente, renderização correta).
- Camadas 2 (build real de React/Vue), 3 (acessibilidade via axe-core) e 4 (adesão a
  design tokens via grep estrutural) do design original **não foram implementadas nesta
  geração** — ficam para quando a fase 2 (React/Vue + build) começar de verdade, evitando
  adicionar dependência não usada agora.

Testes: `tests/unit/test_checker_html.py` (9 casos — página limpa passa, exceção JS não tratada
falha com `RUNTIME_ERROR` e mensagem real do erro, `console.error` explícito falha,
`console.warn` não falha, seleção de entrypoint multi-arquivo, entrypoint explícito respeitado,
ausência de `.html` falha sem exceção, `lint` retorna `MISSING_DEPENDENCY`, operação desconhecida
retorna `INVALID_OUTPUT_FORMAT`); marcados com skip condicional se Playwright/Chromium não
estiverem disponíveis no ambiente (mesmo padrão de `test_checker_go.py` com a toolchain `go`).

#### 10.4.2 `data/train` — lote-piloto de frontend (D-frontend-pilot-batch)

Passo 4 do plano de frontend (`docs/plan_frontend_specialization_wave3.md` seção 10) — primeiro
corpus real do pivô, agora que o backend `html` do checker existe. `scripts/gen_frontend_pilot.py`
gerou 5 exemplos (`gen-frontend-*.json`), todos com execução real (`write_file`/`search_code`
reais contra o sandbox, `checker(language="html", operation="run")` renderizando de verdade em
Chromium headless — nenhum `tool_result` fabricado):

- `landing_hero_cta` — landing page de hero simples, tokens de design extraídos pra
  `tokens.css` em vez de valores soltos, Flexbox justificado no `<think>` (um eixo só).
- `dashboard_metric_cards` — dashboard com 3 cartões, Grid justificado no `<think>` (duas
  dimensões importam: colunas + alinhamento interno).
- `reuse_existing_button` — `search_code` real encontra um `.btn` já existente num "projeto"
  pré-semeado direto no workspace do sandbox (fora da trajetória, simulando um repo que já
  existia); o `<think>` reaproveita em vez de duplicar. Verificado por asserção que o
  `search_code` realmente achou `> 0` ocorrências (não é só narrativa).
- `new_modal_after_search` — mesmo padrão, mas `search_code` por "modal" não acha nada (também
  verificado por asserção, `== 0` ocorrências); só então o `<think>` cria o componente novo,
  reaproveitando os tokens existentes.
- `fix_broken_toggle_script` — ciclo de correção real: primeira renderização falha de verdade
  (`updateAriaState is not defined`, chamada morta a uma função nunca implementada), o `<think>`
  diagnostica causa raiz (não só "remove a chamada", implementa a função de verdade porque
  aria-expanded correto também é acessibilidade real), segunda renderização passa.

Cobre 4 das 8 dimensões de `<think>` técnico da seção 6 do plano de forma central (trade-off de
layout, reutilização antes de criação — positivo e negativo —, consistência de tokens,
verificação real planejada antes do `<final>`); as outras 4 (acessibilidade a fundo,
responsividade com breakpoint, composição de componentes React/Vue, custo/benefício de 3D) ficam
pra quando a camada 3 do checker (axe-core) e a fase 2 (React/Vue) existirem.

`MVP_LANGUAGES` (`src/dataset/taxonomy.py`) ganhou `"html"` — necessário pro schema de metadados
(`language` field) aceitar esses exemplos; nenhum novo `task_type` foi adicionado (os existentes,
`multi_tool_call` e `model_fixes_after_error`, já descrevem a FORMA da interação de ferramentas
independente do domínio — mesmo padrão usado pros exemplos de algoritmos/Python).

---

## 11. Estratégia de treinamento

### 11.1 Pipeline (Transformers + PEFT + bitsandbytes + TRL)

1. Carregar `ibm-granite/granite-4.1-8b` em 4-bit NF4 via `bitsandbytes`
   (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`).
2. Aplicar `prepare_model_for_kbit_training` (PEFT) + `gradient_checkpointing_enable()`.
3. Configurar LoRA (`LoraConfig`) nos módulos de projeção de atenção e MLP
   (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — ajustar aos nomes reais dos
   módulos do Granite após inspeção do `state_dict`, pois nomenclatura pode variar).
4. Treinar com `SFTTrainer` (TRL) usando **masking de loss por segmento** (D7): `labels=-100`
   em tokens de `user`/`system`/`<tool_result>`; loss ativo em `<think>`/`<tool_call>`/`<final>`.
5. Otimizador `paged_adamw_8bit` (bitsandbytes) para evitar picos de VRAM no optimizer step.
6. `group_by_length=True` no lugar de packing (D8) para reduzir desperdício de padding sem
   introduzir vazamento de atenção entre exemplos.

### 11.2 Hiperparâmetros baseline (experimentais, não definitivos)

| Parâmetro | Valor inicial | Racional |
|---|---|---|
| `sequence_length` | 4096 | equilíbrio entre trajetórias multi-tool realistas e VRAM da L4; pode subir para 8192 se o profiling de VRAM sobrar margem |
| `batch_size` (por device) | 1 | necessário dado 4096 tokens + gradient checkpointing em 24GB |
| `gradient_accumulation_steps` | 16–32 | batch efetivo de 16–32 |
| `learning_rate` | 1e-4 a 2e-4 | faixa típica para LoRA em modelos 7-8B |
| `lora_r` | 16 | ponto de partida; 32 como teste de sensibilidade |
| `lora_alpha` | 32 (2× r) | convenção comum alpha=2r |
| `lora_dropout` | 0.05 | regularização leve, dataset inicial pequeno |
| `epochs` | 2–3 (ou steps equivalentes) | evitar overfitting em dataset inicial pequeno |
| `warmup_ratio` | 0.03–0.05 | padrão para SFT curto |
| `lr_scheduler` | cosine | padrão robusto |
| `eval_steps` | a cada ~5% do total de steps | avaliação parcial frequente dado risco de desconexão do Colab |
| `save_steps` | mesma cadência de `eval_steps`, `save_total_limit=3` | checkpoint frequente + retenção limitada por espaço no Drive |
| `max_steps` | **-1** (não sobrepor `num_train_epochs`) | D-maxsteps-v2, erro confirmado no Colab: um `max_steps` positivo SEMPRE sobrepõe `num_train_epochs` no HF Trainer — um valor "de segurança" alto (ex.: 3000) virou ~100 épocas reais num dataset de ~465 exemplos (overfitting severo + ~12h de treino). A proteção contra queda de sessão já vem de `save_steps`/`resume_from_checkpoint`, não precisa de `max_steps` grande |

### 11.3 Estimativa de VRAM (L4, 24 GB) — baseline experimental

| Componente | Estimativa |
|---|---|
| Pesos base (8B, 4-bit NF4 + overhead de quantização) | ~5 GB |
| Adapters LoRA (r=16, bf16) | < 0.2 GB |
| Estado do otimizador (8-bit, só params LoRA) | < 0.5 GB |
| Ativações (seq_len 4096, batch 1, com gradient checkpointing) | ~4-8 GB (depende de profiling real) |
| Overhead CUDA/framework + margem de segurança | ~2-3 GB |
| **Total estimado** | **~12-17 GB de 24 GB disponíveis** |

Estes números **devem ser medidos** com `torch.cuda.max_memory_allocated()` na primeira execução
real do notebook e o baseline ajustado a partir da medição — são um ponto de partida plausível,
não um compromisso.

### 11.4 Reprodutibilidade e retomada

- Seed fixo para inicialização de dados/otimizador registrado no config versionado.
- Todo checkpoint salvo inclui: adapter weights, optimizer state, scheduler state, `trainer_state.json`
  (contém `global_step`), e uma cópia do config usado — permitindo `resume_from_checkpoint` exato.
- Diretório de checkpoints vive no Google Drive montado (persistente entre sessões do Colab).

---

## 12. Configuração inicial para NVIDIA L4 no Google Colab

### 12.1 Estrutura do notebook (17 células lógicas, sem estado oculto entre elas)

1. Verificação de GPU (`nvidia-smi`, checagem explícita de que é L4, senão abortar com erro claro).
2. Instalação de dependências fixadas por versão (`transformers`, `peft`, `bitsandbytes`, `trl`,
   `accelerate`, `datasets`) — sem `!pip install` sem pinning.
3. Montagem opcional do Google Drive (`drive.mount`, com flag para rodar sem Drive em modo teste local).
4. Configuração de diretórios a partir de um único arquivo `configs/train_l4.yaml` (nenhum path
   hardcoded na célula).
5. Carregamento do dataset (`train`/`validation`) já validado (seção 9) — o notebook **não**
   valida dataset, apenas consome dataset já validado pelo pipeline offline.
6. Validação rápida de amostra (sanity check: alguns exemplos impressos, contagem de tokens,
   distribuição de `task_type` no batch carregado).
7. Carregamento do Granite-4.1-8B em 4-bit.
8. Configuração QLoRA (`LoraConfig` + `get_peft_model`).
9. Estimativa e monitoramento de VRAM (log de `torch.cuda.memory_allocated()` antes/depois do
   carregamento do modelo, e periodicamente durante o treino via callback).
10. Loop de treinamento (`SFTTrainer.train()`).
11. Logging estruturado (JSONL local + opcional W&B/TensorBoard, mas sempre com fallback local
    para não depender de serviço externo).
12. Checkpoints periódicos (conforme `save_steps` do config).
13. Célula de retomada (`resume_from_checkpoint=<path>`), testável isoladamente sem re-rodar
    células 7-10 do zero.
14. Avaliação rápida (subconjunto de `probes`, execução do checker real, não só perplexidade).
15. Salvamento do adapter final (`save_pretrained` do adapter, não merge).
16. Teste de inferência (algumas trajetórias completas via `Model Runner` + `Agent Loop`, não
    só geração crua de texto).
17. Exportação de resultados (métricas + amostras de trajetória para `outputs/`).

### 12.2 Regras de robustez do notebook

- Todas as configurações relevantes (paths, hiperparâmetros, allowlist de comandos, limites de
  sandbox) vêm de `configs/train_l4.yaml`, versionado no repositório — nenhuma célula contém
  valores mágicos que não estejam também no config.
- Cada célula é idempotente/re-executável isoladamente após a célula 4 (configuração de
  diretórios), assumindo que os diretórios já existem — permite recomeçar de qualquer ponto
  após uma queda de sessão sem re-executar tudo desde o início.

---

## 13. Estratégia de avaliação

### 13.1 Categorias de resultado (não misturar erro do modelo com erro de infraestrutura)

| Categoria | Significado |
|---|---|
| `DIRECT_PASS` | resposta direta correta, sem ferramenta, quando ferramenta não era necessária |
| `TOOL_PASS` | seleção e uso de ferramenta corretos na primeira tentativa |
| `REPAIR_PASS` | errou uma chamada/código, mas corrigiu corretamente após feedback do checker |
| `PARTIAL_PASS` | resolveu parte da tarefa, não toda |
| `FAIL` | comportamento do modelo incorreto (contagem para métricas de qualidade do modelo) |
| `HARNESS_ERROR` | falha de infraestrutura do harness/sandbox, não atribuível ao modelo |
| `EVALUATOR_ERROR` | falha no próprio avaliador/checker de avaliação, não atribuível ao modelo nem ao harness |

### 13.2 Métricas (mapeando a lista pedida)

Agrupadas por competência, cada uma calculável separadamente por linguagem e por `task_type`:
seleção correta de ferramenta, validade de argumentos, acerto na primeira chamada, taxa de
recuperação após erro, uso correto do resultado da ferramenta, taxa de código compilável, taxa
de testes passando, taxa de resolução completa, ausência de API inventada, ausência de tool
result fabricado, ausência de vazamento de raciocínio, número médio de passos, taxa de loop,
taxa de timeout, custo de contexto (tokens), latência, e as duas quebras finais (por linguagem,
por task_type).

### 13.3 Probes específicos (mapeando a lista pedida da seção 12)

Cada probe é um mini-benchmark isolado, versionado separadamente do `benchmark interno` geral:
`probe_direct_vs_tool_choice`, `probe_malformed_call_xml`, `probe_malformed_call_json`,
`probe_checker_rejection_recovery`, `probe_tool_error_handling`, `probe_multi_file_edit`,
`probe_cross_language`, `probe_unsafe_command_refusal`, `probe_stale_vs_updated_docs`,
`probe_test_result_contradicts_reasoning`, `probe_fabricated_tool_result_attempt`,
`probe_loop_termination`, `probe_paraphrase_generalization` (D-probe-paraphrase-generalization).

Cada probe tem **critério de aprovação explícito e automatizável** (não inspeção manual),
computado pelo mesmo `Checker`/`Evaluator` usado no restante do pipeline.

---

## 14. Estrutura do repositório

```
Logos-1/                       (raiz do projeto Praxis-SFT v2)
├── configs/
│   ├── train_l4.yaml
│   ├── tools_registry.yaml
│   ├── sandbox_policy.yaml
│   └── search_backend.yaml       (provider: ollama|mock|..., ver seção 5.4)
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── validated/
│   ├── rejected/
│   ├── train/
│   ├── validation/
│   ├── probes/
│   ├── benchmark/
│   └── adversarial/
├── notebooks/
│   └── train_logos-v3.ipynb
├── scripts/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   └── run_eval.py
├── src/
│   ├── harness/          (Agent Loop Controller, Context Manager)
│   ├── tools/            (Tool Registry + Executors)
│   ├── checker/          (motor de checagem por linguagem)
│   ├── parsers/          (Tool Call Parser, gramática seção 3)
│   ├── schemas/          (JSON Schemas versionados por ferramenta)
│   ├── inference/        (Model Runner + implementações por backend)
│   ├── evaluation/       (Evaluator, probes, métricas)
│   └── training/         (config loader, loss masking, data collator)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── adversarial/
├── outputs/
├── docs/
│   └── PLAN.md            (este documento)
└── README.md
```

Separação mantida conforme exigido: treino, inferência, ferramentas, harness, checker, dados,
avaliação, notebooks e testes vivem em módulos/diretórios distintos, sem import cruzado que
quebre os contratos da seção 5.2.

**Nota de direção futura (M8, seção 5.5)**: `src/harness`, `src/tools`, `src/parsers` são a
implementação **Python** de referência desta geração. O porte para Go (harness de produção)
viverá em um módulo separado (ex. `harness-go/` na raiz, com seu próprio `go.mod`), consumindo
os mesmos artefatos de `src/schemas/` e `configs/` — nenhuma pasta Go é criada nesta etapa do
plano.

---

## 15. MVP — análise crítica e escopo reduzido

### 15.1 Proposta original do usuário vs. proposta reduzida

A proposta original do usuário (Python+JS/TS+Go+Rust, 2 ferramentas + web_search mock, 1-2
tool calls) é ambiciosa demais para uma primeira iteração validável em L4/Colab: cada linguagem
adicional multiplica o custo de manter um backend de `checker` funcional e sandboxed; TypeScript
em particular exige um passo de transpilação (`tsc`) antes de sequer poder rodar, o que é
complexidade extra sem provar nada de novo sobre tool calling.

Essa mesma razão, confirmada explicitamente pelo usuário, levou à decisão de **remover** (não
apenas adiar) C/C++, Java e Rust do escopo do projeto: nenhuma delas acrescenta um paradigma de
execução que Python (dinâmica/interpretada) e Go (compilada, toolchain leve) já não cubram, e
cada uma trai consigo custos específicos — Rust tem tempos de compilação altos e `cargo`
frequentemente precisa resolver dependências via rede (colide com a política de rede
desabilitada da seção 7.6); Java exige JVM e um toolchain de build mais pesado; C/C++ trazem a
maior superfície de risco de sandboxing (acesso direto a memória, comportamento indefinido) sem
justificativa de valor para os objetivos do produto.

**Redução proposta**: MVP = Python (dinâmica, interpretada) + Go (compilada, toolchain única
binária, sem gerenciador de dependência pesado, `go build`/`go test` rápidos e determinísticos).
Isso já prova a capacidade de generalizar entre paradigmas de execução sem pagar o custo de
múltiplas toolchains simultâneas. JS/TS entra na fase 2; SQL, Shell/Bash e HTML/CSS na fase 3
(seção 10.4). C/C++, Java e Rust não entram em nenhuma fase futura deste projeto.

### 15.2 Ferramentas do MVP

`checker`, `read_file`, `write_file`, `list_files`, `shell` (restrito). `apply_patch` e
`search_code` são adiadas — `search_code` é, na prática, um `grep` via `shell` já coberto pela
allowlist; `apply_patch` carrega risco de aplicação incorreta de diff que não vale a pena
resolver antes de o loop básico (chamar → validar → corrigir) estar provado. `git_diff` também
vira um uso de `git status`/`git diff` via `shell`. `web_search` usa `MockSearchBackend`.

*(Registro histórico da decisão original — `search_code` foi reativada depois, ver
D-search-code-reenable na seção 2 e seção 4.7. `apply_patch`/`git_diff` continuam adiadas.)*

### 15.3 O que o MVP precisa provar (herda diretamente a lista do usuário)

1. O modelo aprende o formato de tool calling da seção 3.
2. Escolhe corretamente entre responder direto e usar ferramenta (`probe_direct_vs_tool_choice`).
3. Usa o retorno real das ferramentas (não fabrica) — `probe_fabricated_tool_result_attempt`.
4. Corrige código após feedback do `checker` — `probe_checker_rejection_recovery`.
5. O harness executa o loop com segurança (allowlist, sandbox, limites) — testes de segurança
   da seção 19.
6. O treinamento cabe em L4 — medido pelo profiling de VRAM real (seção 11.3), não estimado.
7. Resultados são reproduzíveis — `configs/train_l4.yaml` versionado + seeds fixos + checkpoints
   com estado completo.

---

## 16. Etapas de implementação (roadmap)

| Marco | Conteúdo | Critério de saída |
|---|---|---|
| M0 | Repositório, estrutura de pastas, configs vazios versionados | estrutura da seção 14 existe e builda (mesmo sem lógica) |
| M1 | `src/parsers` (gramática seção 3) + `src/schemas` + testes unitários | parser passa 100% dos testes de gramática (seção 19.1), incluindo casos malformados |
| M2 | `src/checker` para Python + Go | checker retorna contrato da seção 10 corretamente para casos de sucesso/erro conhecidos |
| M3 | `src/tools` (5 ferramentas MVP) + `src/harness` **Python** (loop completo, referência de dev) rodando com um modelo mock (não Granite ainda) | loop completo (seção 6) roda ponta a ponta com respostas fixas simuladas |
| M4 | Pipeline de dataset (seção 9) + primeiros exemplos (poucas centenas, todas categorias da 8.1 representadas) | relatório de estatísticas do dataset sem exemplos não-validados em `train`/`validation` |
| M5 | Notebook de treino (seção 12) rodando no Colab L4 com dataset de M4 | treino completo sem estourar VRAM, checkpoint salvo e retomado com sucesso |
| M6 | Avaliador (seção 13) + probes (seção 13.3) rodando contra o adapter treinado em M5 | todas as 7 propriedades da seção 15.3 medidas e reportadas |
| M7 | Iteração de escopo (fase 2: JS/TS, `apply_patch`, `search_code`, `git_diff`) | decisão informada por resultados de M6, não antecipada agora |
| M8 | Porte do harness de produção para **Go** (seção 5.5), consumindo os mesmos contratos (gramática, schemas, `SearchBackend`, sandbox) já validados em M1–M6 | harness Go passa a mesma suite de testes de integração/adversariais da seção 19 rodando contra o adapter treinado, com paridade funcional ao harness Python |

---

## 17. Riscos técnicos

| Risco | Impacto | Mitigação |
|---|---|---|
| Sessão do Colab cai no meio do treino | perda de progresso | checkpoints frequentes (`save_steps`) + `resume_from_checkpoint` + Drive persistente — **não** um `max_steps` artificialmente alto (ver D-maxsteps-v2, seção 11.2: isso já causou ~100 épocas de overfitting num dataset pequeno) |
| VRAM estourada em sequências longas/multi-tool | crash de treino | `sequence_length` conservador inicial + profiling real antes de escalar |
| Sandbox subprocess insuficiente contra código adversarial real (D6) | risco de segurança em uso futuro com código de terceiros | documentado como lacuna; reforço (namespaces/Docker real) é pré-requisito antes de expor a uma CLI que execute repositórios não confiáveis |
| Dataset sintético gerado em lote com viés/baixa diversidade | modelo memoriza padrões estreitos | pipeline de validação por execução real (seção 9) rejeita não-compiláveis; monitorar distribuição por `task_type`/linguagem |
| Vazamento treino/avaliação via tarefas quase-duplicadas | métricas infladas | split por grupo de tarefa-base (seção 8.3), não por exemplo individual |
| Modelo aprende a "alucinar" `tool_result` apesar do masking de loss | viola requisito anti-fabricação | probe dedicado (`probe_fabricated_tool_result_attempt`) + checagem determinística no parser (qualquer `<tool_result>` na saída *do modelo* antes da injeção do harness é tratado como violação, não como conteúdo válido) |
| Granite-4.1-8B ter nomes de módulo de atenção diferentes do assumido | config de LoRA (`target_modules`) incorreta | célula de inspeção do `state_dict`/`named_modules()` antes de configurar `LoraConfig`, no início do notebook |
| Custo de manter toolchains de múltiplas linguagens no ambiente do Colab | complexidade de setup, builds frágeis | MVP restrito a 2 toolchains leves (seção 15) |
| Fixar versão de `torch` em `requirements-train.txt` desalinha o par `torch`+`torchvision` pré-instalado do Colab (`RuntimeError: operator torchvision::nms does not exist` ao carregar qualquer modelo via transformers) — risco confirmado rodando de verdade no Colab, não teórico | treino trava na célula 7 (carregamento do modelo) | `torch`/`torchvision` deliberadamente **não** pinados em `requirements-train.txt`; só pacotes sem requisito de build CUDA própria são fixados |
| `final_segment_leaks_internal_tags` (seção 10.3) só detecta vazamento de TAGS internas (`<think>`, `<tool_call>`, `<tool_result>`) dentro do `<final>` — não detecta o modelo copiando o *conteúdo* de uma mensagem de erro interna (nomes de variável de ambiente, referência a seções do PLAN.md) pra dentro do texto do `<final>` sem usar tag nenhuma. Confirmado num teste real: pedido de código de FFT → `web_search` falhou por `PRAXIS_OLLAMA_SEARCH_API_KEY` não configurada no Colab (falha de infra, não do modelo) → o modelo colou a mensagem de erro interna verbatim como resposta final ao usuário, e desistiu da tarefa em vez de tentar outro caminho (ex.: escrever a FFT sem depender de busca) | detalhe de configuração interna vaza pra resposta pública; falta de recuperação de erro de ferramenta gera desistência prematura | **não corrigido ainda** — candidatos: (a) checagem de consistência mais ampla que compare `<final>` contra o texto de `tool_result`s de erro recentes, não só tags; (b) exemplos de dataset ensinando recuperação após falha de ferramenta não-crítica (tentar outro caminho em vez de repetir o erro) |

---

## 18. Itens explicitamente fora de escopo (nesta geração)

- Full fine-tuning do modelo base.
- Merge oficial do adapter no modelo base como entregável.
- RLHF/DPO/qualquer treino por preferência.
- CLI de produção completa (só contratos/interfaces).
- Suporte simultâneo a mais de um modelo-base.
- Sandbox com isolamento de nível Docker/gVisor real (fica documentado como pré-requisito de
  produção, não implementado agora).
- `apply_patch` por diff, `search_code` dedicado, `git_diff` dedicado (cobertos via `shell` no MVP).
- Suporte a HTML/CSS, SQL, Shell como linguagens de treino (entram em fase 3, conforme seção 10.4).
- **C/C++, Java e Rust — removidas do escopo do projeto**, não apenas adiadas para uma fase
  futura (decisão do usuário, justificativa consolidada na seção 15.1). Nenhum backend de
  `checker`, nenhum exemplo de dataset e nenhum probe será construído para essas linguagens
  nesta geração do Praxis.
- Deduplicação por embedding semântico (fase 2).
- **Implementação em Go do harness de produção** (decisão de direção registrada em D13/seção 5.5,
  mas execução planejada para M8 — não começa antes de o formato/loop estarem validados via
  harness Python em M1–M6).
- Integração ao vivo com outros motores de busca além da Ollama (Bing/SerpAPI/etc.) — a
  interface `SearchBackend` já permite isso, mas nenhuma implementação além de
  `OllamaSearchBackend`/`MockSearchBackend` é construída nesta geração.

---

## 19. Plano de testes

### 19.1 Testes unitários
- `parsers`: gramática da seção 3.2 — casos válidos, tags não fechadas, JSON malformado dentro
  de tag válida, ordem inválida de segmentos, `<tool_result>` presente na saída do modelo
  (deve ser rejeitado/flagueado, nunca aceito como legítimo).
- `schemas`: validação positiva/negativa para cada ferramenta do MVP.
- `checker`: por linguagem (Python/Go no MVP) — sintaxe válida, erro de compilação, teste
  passando, teste falhando, timeout, dependência ausente.

### 19.2 Testes de integração
- Loop do agente completo (seção 6) com `Model Runner` mockado (respostas fixas roteirizadas),
  cobrindo: resposta direta, uma ferramenta, múltiplas ferramentas, correção após erro,
  chamada repetida (deve ser barrada), limite de passos atingido.
- Sandbox: comando fora da allowlist é bloqueado; timeout é respeitado; workspace não permite
  escape de diretório.

### 19.3 Testes adversariais
- Tentativa de comando destrutivo (`rm -rf`, etc.) — deve retornar `UNSAFE_COMMAND` sem executar.
- Tentativa de o modelo "fabricar" um `<tool_result>` — deve ser detectado e não repassado como
  resultado legítimo.
- Prompt tentando extrair `<think>` para dentro de `<final>` — `verify_consistency` deve
  sinalizar `INTERNAL_REASONING_LEAK`.
- Payloads de tamanho excessivo em `stdout`/`stderr` — devem ser truncados, não travar o harness.

### 19.4 Testes de treinamento/reprodutibilidade
- Rodar 50-100 steps com seed fixo duas vezes, comparar loss curve para convergência
  determinística (dentro de tolerância de não-determinismo de kernels CUDA).
- Testar `resume_from_checkpoint` interrompendo deliberadamente o treino e retomando,
  comparando `global_step` e loss pós-retomada com uma execução ininterrupta de referência.

---

## 20. Artefatos a serem criados

- `docs/PLAN.md` (este documento).
- `configs/train_l4.yaml`, `configs/tools_registry.yaml`, `configs/sandbox_policy.yaml`,
  `configs/search_backend.yaml`.
- `data/fixtures/web_search/` — resultados de busca congelados (coletados via
  `OllamaSearchBackend`, consumidos via `MockSearchBackend` em treino/avaliação, seção 5.4).
- `src/parsers/grammar.py` + `src/schemas/*.json` (um por ferramenta, versionado).
- `src/harness/{loop.py,context.py,renderer.py}`.
- `src/tools/{registry.py,executors/*.py}`.
- `src/checker/{core.py,backends/python.py,backends/go.py}`.
- `src/inference/{base.py,transformers_runner.py}`.
- `src/training/{loss_masking.py,data_collator.py,config_loader.py}`.
- `src/evaluation/{metrics.py,probes/*.py,categories.py}`.
- `scripts/{generate_dataset.py,validate_dataset.py,run_eval.py}`.
- `notebooks/train_logos-v3.ipynb`.
- `data/{raw,normalized,validated,rejected,train,validation,probes,benchmark,adversarial}/` (com
  primeiro lote de exemplos correspondendo à taxonomia da seção 8.1, restrito a Python/Go).
- `tests/{unit,integration,adversarial}/*`.
- `README.md` descrevendo o projeto e como rodar cada estágio do pipeline.

---

## Próximos passos sugeridos após aprovação deste plano

1. M0: criar a estrutura de pastas e configs vazios versionados.
2. M1: implementar e testar exaustivamente o parser/gramática antes de qualquer outra peça —
   é a fundação de que tudo mais depende (dataset, harness, checker, avaliação).
