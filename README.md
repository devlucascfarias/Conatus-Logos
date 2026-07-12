# Praxis-SFT v2

Modelo especializado em engenharia de software, derivado via **QLoRA** de
`ibm-granite/granite-4.1-8b` (`configs/train_l4.yaml`, D-granite-swap — ver `docs/PLAN.md`),
treinado para raciocinar, chamar ferramentas estruturadas, interpretar retornos reais dessas
ferramentas, verificar sua própria solução e emitir uma resposta pública limpa. Treino em NVIDIA
L4 (Google Colab). Sem modo de "thinking" nativo, evitando conflito com a gramática
`<think>`/`<tool_call>`/`<final>` treinada por cima; escolhido depois que 3 treinos reais no
`Qwen/Qwen3-4B-Instruct-2507` (usado antes, `D-frontend-pivot-model-swap`) mostraram o mesmo
padrão de gramática malformada independente da quantidade de passos de treino. GPUs menores
(RTX A2000 12GB local, `configs/train_a2000.yaml`) e GPU gratuita do Kaggle (P100/T4 16GB,
`configs/train_kaggle.yaml`, `D-kaggle-training-support`) ainda apontam pro Qwen3-4B — não
foram migradas junto, ver `docs/PLAN.md` antes de rodar treino por um desses caminhos.

O plano técnico completo — decisões arquiteturais, formato canônico de trajetória, schemas de
ferramenta, desenho do harness/loop do agente, estratégia de segurança, dataset, treino,
avaliação e roadmap — está em [`docs/PLAN.md`](docs/PLAN.md). Este README documenta apenas
**como rodar** o que já foi implementado (M0-M6 do roadmap); leia o plano para entender **por
quê** cada peça existe.

## Status

| Marco | Conteúdo | Status |
|---|---|---|
| M0 | Estrutura de pastas + configs | ✅ |
| M1 | Parser/gramática (`src/parsers`) + schemas (`src/schemas`) | ✅ (35 testes) |
| M2 | Checker Python + Go (`src/checker`) | ✅ (19 testes, 7 pulados sem toolchain Go local) |
| M3 | Sandbox, ferramentas, busca, inferência mock, harness/loop (`src/security`, `src/tools`, `src/search`, `src/inference`, `src/harness`) | ✅ (32 testes) |
| M4 | Pipeline de validação de dataset (`src/dataset`) + 556 exemplos (Python + Go, todos os `task_type` da taxonomia, gerados com biblioteca de ~78 funções + injetores de bug + checker como oráculo) | ✅ **556/556 validados** (465 train / 78 validation / 13 adversarial) — degrau "generalização" (500-800), ainda abaixo do que um dataset de produção real usaria (milhares+) |
| M5 | Notebook de treino (`notebooks/train_logos-v3.ipynb`) + `TransformersModelRunner.generate()` | ✅ `generate()` testado ponta a ponta contra modelo real minúsculo (`tests/unit/test_transformers_runner.py`); notebook rodado de verdade em A2000 local e Kaggle (T4) com o Qwen3-4B-Instruct-2507 — modelo não é gated, `lora_target_modules` batem certo, treino em andamento; VRAM/tempo real na L4 ainda não medidos |
| M6 | Avaliador + 4 probes reais (`src/evaluation`) | ✅ mecanismo provado contra `ScriptedModelRunner` **e** contra Transformers real (modelo de teste); falta rodar contra o adapter Qwen3-4B treinado de verdade (treino em andamento) |
| M7/M8 | Fase 2 (JS/TS, `apply_patch`...) / porte do harness para Go | Fora de escopo desta geração (ver seção 18 do plano) |

## Requisitos

```bash
pip install -r requirements.txt          # harness, checker, dataset, avaliação, testes
pip install -r requirements-train.txt    # só dentro do notebook de treino (Colab)
python -m playwright install chromium    # backend "html" do checker (renderização real)
```

O backend `go` do checker (`src/checker/backends/go_backend.py`) precisa do binário `go` no
`PATH`; sem ele, os testes correspondentes são pulados e chamadas de checker em Go retornam
`MISSING_DEPENDENCY` de forma estruturada (não uma exceção). O mesmo vale para o backend `html`
(`src/checker/backends/frontend_backend.py`) sem o Chromium do Playwright instalado.

O backend `javascript`/`typescript` (`src/checker/backends/node_backend.py`, D-checker-node-backend)
precisa de `node`/`npm` no `PATH` **e** de `node_modules` instalado no projeto-template em
`checker_templates/react_three_fiber/` (React + TypeScript + Vite + React Three Fiber + drei) —
sem isso, as chamadas retornam `MISSING_DEPENDENCY` estruturado, os testes correspondentes são
pulados:

```bash
cd checker_templates/react_three_fiber
npm install    # uma vez só — node_modules não é versionado (ver .gitignore)
```

A operação `run` builda com `vite build` de verdade e renderiza o resultado num Chromium headless
real (servido por HTTP local, não `file://` — scripts `type="module"` são bloqueados por CORS ao
carregar via `file://`), capturando erros de console genuínos, WebGL incluso.

## Rodar os testes

```bash
python -m pytest -q
```

## Pipeline de dataset

```bash
# 1. Gera ~556 exemplos (todos os task_type da seção 8.1, biblioteca de funções + injetores de
#    bug + checker como oráculo) em data/raw/
python scripts/generate_dataset.py --seed-demo

# 2. Normaliza, valida gramática/schema, RE-EXECUTA todo tool_call de checker de verdade
#    (D11 — nunca confia em texto do modelo), classifica, deduplica, separa splits e
#    produz estatísticas (seção 9)
python scripts/validate_dataset.py
```

Importante: exemplos cujo `<tool_call name="checker">` usa `"language": "go"` só validam com
sucesso se o binário `go` estiver instalado no ambiente que roda `validate_dataset.py` — a
rejeição por `MISSING_DEPENDENCY`/divergência de resultado é o comportamento correto e
esperado (D11), não um bug, quando `go` não está disponível.

## Avaliação (probes)

```bash
python scripts/run_eval.py --demo
```

Roda os 4 probes implementados (`src/evaluation/probes/`) contra um `ScriptedModelRunner` com
respostas roteirizadas — prova o mecanismo de avaliação. `TransformersModelRunner.generate()`
(`src/inference/transformers_runner.py`) já está implementado e testado contra um modelo real
minúsculo do Hugging Face Hub; rodar os probes contra o **adapter Qwen3-4B treinado de verdade**
depende do treino real (M5) terminar.

## Rodar o notebook de treino (Colab/Kaggle)

`notebooks/train_logos-v3.ipynb` detecta o ambiente (Colab, Kaggle, ou local) e ajusta clone do
repositório, leitura de secrets e exportação de artefatos automaticamente — nenhuma edição do
notebook é necessária pra trocar de ambiente. Fora do Colab/Kaggle (ex.: `jupyter nbconvert`
local), essas etapas são puladas e o notebook assume que já está numa cópia local do repositório.

Pré-requisito (uma vez, fora do notebook): gere um Personal Access Token no GitHub (Settings →
Developer settings → Personal access tokens → Fine-grained, acesso só ao repositório
`Conatus-Logos`, permissão de leitura em "Contents"). **Nunca** cole o token direto numa célula.

- **Colab**: ícone de chave 🔑 na barra lateral esquerda → "Add new secret" → nome `GH_TOKEN`,
  valor = o token. Repita para `PRAXIS_OLLAMA_SEARCH_API_KEY` (opcional, só afeta `web_search`).
  O toggle "Notebook access" precisa estar ligado pra cada notebook — criar o secret uma vez
  não basta.
- **Kaggle**: aba "Add-ons" → "Secrets" → "Add a new secret", mesmos dois nomes. Marque a opção
  de anexar o secret ao notebook, e habilite "Internet" em "Settings" (sem isso a leitura do
  secret falha).

Qual config usar (`configs/train_l4.yaml`, `train_a2000.yaml` ou `train_kaggle.yaml`) é
selecionado via a variável de ambiente `TRAIN_CONFIG_PATH`; ausente, o notebook usa
`train_l4.yaml` por padrão.

## Estrutura

```
configs/            hiperparâmetros, allowlist/denylist, registro de ferramentas, backend de busca
data/                raw → normalized → validated/rejected → train/validation/probes/benchmark/adversarial
notebooks/           notebook de treino para Colab/L4
scripts/             CLIs que orquestram dataset e avaliação
src/
├── parsers/         gramática canônica de trajetória (seção 3.2)
├── schemas/          JSON Schemas por ferramenta + registro (D10)
├── checker/          camada de verificação independente do modelo (D11) — Python + Go
├── security/         política de sandbox (allowlist/denylist/limites) + execução isolada
├── search/            SearchBackend parametrizado (mock / Ollama, D14)
├── tools/             executores das 5 ferramentas do MVP
├── inference/         ModelRunner backend-agnóstico (mock hoje, Transformers em M5)
├── harness/           Trajectory, loop do agente, renderer dev/prod, logger
├── dataset/           taxonomia, schema de metadados, pipeline de validação (seção 9)
├── evaluation/        categorias de resultado + probes
└── training/          config loader + máscara de loss por segmento (D7)
tests/
├── unit/
├── integration/
└── adversarial/
docs/PLAN.md          plano técnico completo
```

## Segurança

Nenhum texto gerado pelo modelo é executado diretamente: toda chamada de ferramenta passa por
validação de schema → allowlist/denylist → sandbox de processo isolado com timeout, limite de
saída e workspace efêmero (`configs/sandbox_policy.yaml`, `src/security/`). Comandos
destrutivos são bloqueados por denylist independentemente de confirmação (ver
`tests/adversarial/`).

## Segredos

Nenhuma chave de API é lida de valor hardcoded em código ou config — `configs/search_backend.yaml`
referencia apenas o **nome** da variável de ambiente (`api_key_env`); o valor real fica só no
ambiente de execução (shell local ou Colab secret), nunca em um arquivo versionado.
