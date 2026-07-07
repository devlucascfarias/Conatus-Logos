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
| D5 | MVP tools = `checker`, `read_file`, `write_file`, `list_files`, `shell` | `search_code`/`git_diff` são compostos triviais via `shell`; `apply_patch` adia risco de aplicação incorreta de diff | **[HIPÓTESE]** |
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

### 4.6 Fase 2+: `apply_patch`, `search_code`, `git_diff`, `web_search` (esboço de contrato)

```json
// apply_patch — fase 2
{"input_schema": {"type": "object", "required": ["path", "diff"],
  "properties": {"path": {"type": "string"}, "diff": {"type": "string", "description": "unified diff"}}}}

// search_code — fase 2
{"input_schema": {"type": "object", "required": ["pattern"],
  "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}, "regex": {"type": "boolean", "default": false}}}}

// git_diff — fase 2
{"input_schema": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "staged": {"type": "boolean", "default": false}}}}

// web_search — fase 2, backend mockável via interface abstrata (ver 5.4)
{"input_schema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}}
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
| HTML/CSS | Fase 3 | validadores de parsing (sintaxe apenas, sem execução) |

C/C++, Java e Rust foram removidas do escopo do projeto (não apenas adiadas para uma fase
futura) — ver justificativa consolidada na seção 18.

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
`probe_loop_termination`.

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
│   └── train_granite_l4.ipynb
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
- `notebooks/train_granite_l4.ipynb`.
- `data/{raw,normalized,validated,rejected,train,validation,probes,benchmark,adversarial}/` (com
  primeiro lote de exemplos correspondendo à taxonomia da seção 8.1, restrito a Python/Go).
- `tests/{unit,integration,adversarial}/*`.
- `README.md` descrevendo o projeto e como rodar cada estágio do pipeline.

---

## Próximos passos sugeridos após aprovação deste plano

1. M0: criar a estrutura de pastas e configs vazios versionados.
2. M1: implementar e testar exaustivamente o parser/gramática antes de qualquer outra peça —
   é a fundação de que tudo mais depende (dataset, harness, checker, avaliação).
