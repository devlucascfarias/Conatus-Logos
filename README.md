# Praxis-SFT v2

Modelo especializado em engenharia de software, derivado via **QLoRA** de
`Qwen/Qwen3-4B-Instruct-2507`, treinado para raciocinar, chamar ferramentas estruturadas,
interpretar retornos reais dessas ferramentas, verificar sua própria solução e emitir uma
resposta pública limpa. Treino em NVIDIA L4 (Google Colab) ou GPUs menores (ex.: RTX A2000
12GB, `configs/train_a2000.yaml`) — o modelo de 4B (`D-frontend-pivot-model-swap`, ver
`docs/PLAN.md`) foi escolhido por caber com folga em hardware mais modesto que o 8B generalista
anterior (`ibm-granite/granite-4.1-8b`) e por não ter modo de "thinking" nativo, evitando
conflito com a gramática `<think>`/`<tool_call>`/`<final>` treinada por cima.

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
| M5 | Notebook de treino (`notebooks/train_granite_l4.ipynb`) + `TransformersModelRunner.generate()` | ✅ `generate()` implementado e testado ponta a ponta contra modelo real minúsculo (`tests/unit/test_transformers_runner.py`); células 7-16 do notebook ainda exigem GPU real (Colab) e não foram executadas contra o Granite de verdade — riscos abertos: acesso ao modelo (gated?), nomes de `lora_target_modules`, VRAM real |
| M6 | Avaliador + 4 probes reais (`src/evaluation`) | ✅ mecanismo provado contra `ScriptedModelRunner` **e** contra Transformers real (modelo de teste); falta rodar contra o adapter Granite treinado de verdade |
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
minúsculo do Hugging Face Hub; rodar os probes contra o **adapter Granite treinado de verdade**
ainda depende de M5 (treino real no Colab).

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
