# Praxis-SFT v2

Modelo especializado em engenharia de software, fine-tuned via **QLoRA** sobre `ibm-granite/granite-4.1-8b`.

O projeto implementa um agente capaz de:

* gerar respostas estruturadas;
* executar ferramentas;
* validar código em ambientes reais;
* interpretar o resultado das ferramentas;
* corrigir a solução quando necessário;
* retornar apenas a resposta final ao usuário.

## Como funciona

O fluxo principal é:

```text
Prompt
  ↓
ModelRunner
  ↓
<think> / <tool_call>
  ↓
Parser + Schema Validation
  ↓
Security Policy + Sandbox
  ↓
Tool Execution
  ↓
Tool Result
  ↓
Modelo
  ↓
<final>
```

O modelo **não executa código diretamente**. Chamadas de ferramenta passam por validação de schema, política de segurança e execução isolada antes do resultado voltar ao modelo.

## Componentes

```text
src/
├── parsers/       parser da trajetória gerada pelo modelo
├── schemas/       schemas das ferramentas
├── checker/       validação e execução de código
├── security/      sandbox e políticas de execução
├── search/        backend de busca
├── tools/         implementação das ferramentas
├── inference/     interface de inferência dos modelos
├── harness/       loop principal do agente
├── dataset/       geração e validação do dataset
├── evaluation/    probes e métricas
└── training/      configuração e utilitários de treino
```

## Checker

O checker funciona como um verificador independente do modelo.

Backends disponíveis:

* Python
* Go
* JavaScript / TypeScript
* Node.js
* HTML
* SCSS

Dependendo do backend, são usados runtimes reais como `go`, `node`, `vite`, `vitest`, `eslint`, `sass` e Chromium via Playwright.

## Instalação

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Para os checkers JavaScript/TypeScript/Node/SCSS:

```bash
cd checker_templates/react_three_fiber
npm install
```

## Testes

```bash
python -m pytest -q
```

## Dataset

Gerar os exemplos:

```bash
python scripts/generate_dataset.py --seed-demo
```

Validar e gerar os splits:

```bash
python scripts/validate_dataset.py
```

A validação inclui parsing, schemas, deduplicação e reexecução real das chamadas ao checker.

## Avaliação

```bash
python scripts/run_eval.py --demo
```

Os probes exercitam o loop completo do agente e verificam o comportamento das respostas e chamadas de ferramenta.

## Treinamento

O notebook principal está em:

```text
notebooks/train_logos-v3.ipynb
```

A configuração de treino pode ser selecionada com:

```bash
TRAIN_CONFIG_PATH=configs/train_l4.yaml
```

As dependências específicas de treino estão em:

```bash
pip install -r requirements-train.txt
```

## Segurança

Toda execução de ferramenta passa por:

```text
schema validation
→ allowlist / denylist
→ sandbox isolado
→ timeout e limites de saída
→ resultado estruturado
```

Nenhum código produzido pelo modelo é executado diretamente fora desse pipeline.

## Documentação

Detalhes de arquitetura, decisões técnicas, dataset, treinamento e roadmap:

```text
docs/PLAN.md
```
