# Plano: pivô do Logos para especialização em frontend (wave 3)

Status: **proposta para revisão — nenhum código escrito ainda**. Este documento consolida
tudo que foi discutido em conversa na sessão de 2026-07-09, depois de a wave 2 (identidade
Logos-3/Conatus + generalização multi-turno) estar completa e depois de o treino do modelo
generalista de 8B (Granite-4.1-8B) se mostrar custoso demais no hardware disponível (RTX A2000
12GB — ver histórico de OOM em `configs/train_a2000.yaml`).

## 1. Motivação e tese

Duas pressões reais convergindo:

1. **Escassez de hardware pra treino generalista**: reproduzir um agente de engenharia de
   software GENÉRICO num modelo de 8B está sendo custoso — a A2000 12GB mal cabe o QLoRA do
   8B (precisou cair pra `sequence_length=1024` e ainda assim brigou com o desktop gráfico
   pela VRAM). Um modelo generalista compete de frente com Copilot/Cursor/Claude Code num
   espaço lotado, exigindo capacidade ampla que hardware modesto não sustenta bem.
2. **Nicho subexplorado com demanda real**: gerar UI de frontend genuinamente bonita e
   específica (landing pages, dashboards elegantes) é um domínio onde um modelo PEQUENO e
   ESPECIALIZADO pode ficar realmente bom, em vez de medíocre-em-tudo. Precedente de mercado:
   v0 (Vercel), Bolt, Lovable — todos apostam que "UI bonita e específica" vale mais que
   competir como generalista. A tese "modelo pequeno especializado bate generalista maior
   dentro do domínio" é um padrão real de fine-tuning.

**Tese central**: transformar o Logos de agente de engenharia de software GENÉRICO num modelo
ESPECIALIZADO em frontend — capaz de reproduzir landing pages, dashboards e componentes
visualmente elegantes — usando um modelo base menor (4B) que treina com folga no hardware
disponível.

### 1.1 O risco honesto que este pivô carrega

Frontend bonito NÃO troca um problema difícil por um fácil. Troca "amplo mas objetivo"
(algoritmo passa no teste ou não) por "estreito mas subjetivo" (o que torna um dashboard
"elegante"? Não tem um `assert` claro). Todo este projeto foi construído sobre a disciplina de
NUNCA fabricar `tool_result`, sempre execução REAL. Pra algoritmo isso é trivial (o `checker`
roda o teste de verdade). Pra "essa landing page ficou elegante", qual checker roda? **Resolver
essa pergunta é o pré-requisito de tudo o mais neste plano** — modelo, dataset e tooling vêm
depois dela. A resposta adotada está na seção 4 (checker de frontend por renderização real +
screenshot + console + métricas objetivas, com o julgamento estético puro feito por avaliação
externa de VLM/humano, nunca fabricado como métrica automática).

## 2. Modelo base: Qwen3-4B-Instruct-2507

Escolhido depois de comparar contra Gemma 4 E4B e Phi-4-mini. Bate os quatro critérios pedidos
pelo usuário (rápido, eficiente, inteligente pra frontend, SEM think nativo):

- **Sem think nativo (crítico)**: o model card confirma "supports only non-thinking mode and
  does not generate thinking blocks in its output". É a variante que a Qwen lançou de
  propósito separada do "Qwen3-4B-Thinking" — zero risco de conflito entre um formato de
  raciocínio nativo do modelo e a gramática `<think>`/`<tool_call>`/`<final>` que treinamos por
  cima. Isso importa: um modelo com think nativo brigaria com o nosso `<think>` (dois formatos
  de raciocínio competindo).
- **4.0B parâmetros densos** (não MoE): mais simples de quantizar e treinar em QLoRA que o
  Gemma 4 E4B (4.5B efetivo via mixture-of-experts, roteamento adiciona complexidade).
- **Base forte pra fine-tuning**: benchmark independente (distil labs, 12 modelos pequenos x 8
  tarefas) mostra que iguala/supera o professor em 7 de 8 benchmarks.
- **Família com histórico de código**: Qwen3 vence Gemma em código agentic na mesma classe de
  parâmetros ativos (referência: Qwen3.6-35B-A3B 73.4% vs Gemma 4 26B A4B 52.0% no SWE-bench
  Verified — não é a escala de 4B, mas é sinal de prioridade de código na família).
- **Prático**: licença Apache 2.0 (permissiva pra produto), disponível pronto no Ollama
  (encaixa direto na CLI `conatus` existente), suporte da Unsloth pra QLoRA eficiente
  (ajuda com a escassez de hardware).
- **Contexto nativo de 262k tokens**: folga enorme comparado aos 4096 do Granite — relevante
  porque HTML/CSS/JSX de uma página inteira é longo.

### 2.1 Impacto no que já existe

Trocar `granite-4.1-8b` por `Qwen/Qwen3-4B-Instruct-2507` toca:
- `configs/train_l4.yaml` / `configs/train_a2000.yaml`: `base_model` + revalidar VRAM (4B deve
  caber com MUITO mais folga que o 8B — provavelmente resolve todo o drama de OOM da wave 2, e
  talvez até permita voltar `sequence_length` pra 2048+).
- `lora.target_modules`: a arquitetura do Qwen3 tem nomes de módulo diferentes do Granite —
  precisa inspecionar `named_modules()` do Qwen3-4B antes (mesmo risco já registrado na seção
  17 do PLAN.md pro Granite; NÃO assumir que `q_proj`/`k_proj`/... batem).
- `PRAXIS_SYSTEM_PROMPT` (`src/harness/system_prompt.py`) e `agent.SystemPrompt`
  (`cli/internal/agent/agent.go`): a identidade continua "Logos-3, família Conatus", mas o
  texto provavelmente ganha uma linha sobre a especialização em frontend (ver seção 6).
- Notebook de treino: sem mudança estrutural (o pipeline QLoRA é agnóstico ao modelo base).

## 3. Ferramentas (tools)

### 3.1 Mantidas sem mudança (base de qualquer CLI)

`write_file`, `read_file`, `list_files`, `shell`, `web_search` — decisão explícita do usuário:
não são específicas de domínio, são o alicerce de qualquer agente de CLI. Continuam exatamente
como estão, independente do pivô.

### 3.2 `search_code` — REATIVAR (já existe como esboço)

Já está no schema (`src/schemas/search_code.json` — `pattern`, `path`, `regex`) mas está
`enabled: false` em `configs/tools_registry.yaml` desde a decisão original do MVP (D5: "é um
grep via shell já coberto"). Para o caso de uso de frontend, uma ferramenta DEDICADA é mais
ergonômica que montar argumentos de shell toda vez: em vez de `{"binary": "grep", "args":
[...]}`, o modelo escreve `{"pattern": "Button", "path": "src/components"}`.

**Uso pedagógico central**: antes de criar qualquer componente novo (`Button`, `Card`,
`Input`), o modelo procura se já existe um equivalente — evita componente duplicado e código
morto. É o mesmo padrão "nunca agir sem localizar primeiro" que já está registrado em memória
de projeto (`feedback_dataset_scenario_realism` — o modelo deve localizar antes de editar/criar,
nunca assumir), agora aplicado a reutilização de componente.

**Trabalho pra ativar**: mudar `enabled: false` → `true` em `configs/tools_registry.yaml`,
implementar o executor `src/tools/executors/search_code_tool.py` (delega a `grep`/ripgrep via
sandbox, mesmo padrão de `shell_tool.py`), registrar em `src/tools/registry.py`. O schema já
existe.

### 3.3 `checker` — NOVO backend de frontend (a peça central e mais difícil)

Ver seção 4 inteira. Hoje o `checker` tem backends Python (`py_compile`/`pytest`) e Go
(`go build`/`go test`); ganha um backend de frontend que renderiza de verdade e captura
evidência real.

## 4. Checker de frontend — como "ficou bom" vira execução REAL, não fabricação

Esta é a peça que viabiliza (ou inviabiliza) o projeto inteiro. Segue exatamente a mesma
disciplina do `checker` atual: "execução real, resultado real, nunca fabricado". A diferença é
que a "execução" de frontend é renderizar num navegador de verdade e capturar evidência.

### 4.1 Camadas de avaliação (do mais objetivo ao mais subjetivo)

| Camada | O que mede | Como (execução real) | Objetividade |
|---|---|---|---|
| 1. Renderiza sem erro | O código roda? Canvas em branco? Erro de console? | Sobe preview real, checa console limpo (sem erro/warning fatal) | Totalmente objetivo (passou/falhou) |
| 2. Build (fase React/Vue) | TypeScript/JSX compila? | `npm install` + `vite build`/`next build` REAL, captura erro real de tsc/build | Totalmente objetivo |
| 3. Acessibilidade | Contraste WCAG, semântica, ARIA, hierarquia de heading | axe-core / Lighthouse rodando de verdade contra o HTML renderizado | Objetivo (score numérico + violações listadas) |
| 4. Aderência ao design system | Usa tokens/componentes definidos, não valores soltos | Script que confere classes/tokens usados contra o sistema (grep estrutural) | Objetivo (bate ou não bate) |
| 5. Estética pura | "Ficou lindo?" | Screenshot real → julgamento de VLM/humano por amostragem | SUBJETIVO — nunca vira métrica automática fingida |

**Regra de ouro herdada do projeto**: camadas 1-4 são checáveis por máquina e entram no
`checker` como resultado real. A camada 5 (estética pura) NUNCA é fabricada como número — é
julgamento externo (VLM como juiz, ou humano por amostragem), registrado honestamente como tal.
Fingir que "elegância" virou uma métrica automática seria exatamente o tipo de fabricação que
o projeto inteiro se recusou a fazer com `tool_result`.

### 4.2 Ferramentas de renderização já disponíveis

O ambiente já tem as ferramentas de preview necessárias (mesmas que o assistente usa):
`preview_start` (sobe servidor real), `preview_screenshot` (print real do render),
`preview_console_logs` (erros de console — crítico pra Three.js, ver 4.4),
`preview_inspect`/`preview_click` (CSS computado real + interação). Um backend de frontend do
checker orquestra essas: materializa os arquivos → sobe preview → captura screenshot + console
+ métricas → devolve resultado real.

### 4.3 Dois usos do checker de frontend

1. **Geração de dataset**: gerar a página/dashboard → renderizar de verdade → screenshot real
   → só ENTÃO escrever `<think>`/`<final>`. Vira um ciclo de correção real igual ao de bug de
   código (gerar → ver que ficou ruim → ajustar → ver que melhorou), só que o "bug" agora é
   estético/estrutural. Isso mantém a disciplina de nunca escrever uma trajetória que afirma
   algo sobre um resultado que não foi de fato observado.
2. **Avaliação do modelo treinado**: depois de um checkpoint, gerar N exemplos → renderizar →
   julgar (camadas 1-4 automático, camada 5 amostragem). Vira métrica de progresso REAL entre
   retreinos, não só perplexidade.

### 4.4 Three.js / WebGL — cuidado específico

Cena 3D pode renderizar um canvas EM BRANCO (contexto WebGL perdido, erro de shader) sem isso
aparecer óbvio só no screenshot. Por isso a camada 1 (console limpo via `preview_console_logs`)
é obrigatória pra qualquer exemplo com Three.js — não basta olhar a imagem.

## 5. Stack e escopo em fases

### Fase 1 — HTML/CSS/JS estático (+ Three.js vanilla)

Começar pelo mais simples reduz a superfície de erro: "não compilou" não atrapalha a avaliação
de "ficou bonito". Renderiza direto no navegador, sem `npm run build` no meio.

- HTML semântico + CSS (provavelmente Tailwind via CDN pra ter um design system pronto e
  tokens consistentes de graça, mas decisão em aberto).
- JS vanilla pra interatividade.
- **Three.js vanilla** (via `<script type="module">` ou import de CDN) — NÃO precisa de build,
  cabe já nesta fase. 3D em hero section é o que separa "landing genérica" de "landing
  memorável".

### Fase 2 — React/Vue com build REAL (+ react-three-fiber)

Só depois que a base estética estiver sólida. Aqui a "execução real" ganha a camada 2
(`npm install` + build de verdade, erro real de TypeScript/JSX).

- Componentes React (ou Vue), com pipeline de build real.
- **react-three-fiber (R3F)** pro 3D — wrapper React do Three.js, mantém tudo no modelo de
  componente declarativo em vez de misturar imperativo (Three.js cru) com declarativo (JSX).
- Aqui o `search_code` (3.2) brilha: reutilizar componentes existentes em vez de recriar.

## 6. Estilo de `<think>` técnico para frontend

Mesma disciplina da wave 2 (profundidade proporcional à dificuldade, verbalização de intenção
de ferramenta, trade-off explícito, sem hífen/travessão de frase), adaptada ao domínio. As
dimensões técnicas que dão o mesmo rigor de "causa raiz vs sintoma" que fizemos pra algoritmos:

1. **Semântica e acessibilidade**: `<button>` em vez de `<div onClick>`, hierarquia de heading
   correta, ARIA só quando a semântica nativa não basta, contraste WCAG real. Verbalizar a
   escolha, não só aplicar.
2. **Trade-off de layout**: Flexbox vs Grid com justificativa técnica (Grid quando as duas
   dimensões importam, Flexbox quando é uma linha/coluna). Mesmo padrão de `dict.fromkeys` vs
   `set()` da wave 2.
3. **Responsividade**: verbalizar a decisão de breakpoint ("colapsa pra uma coluna em 768px
   porque a sidebar espremeria o conteúdo principal"), não declarar valor mágico.
4. **Reutilização antes de criação**: antes de criar `Button`/`Card`/`Input`, o `<think>`
   verbaliza a busca real via `search_code` e só cria se confirmar que não existe — evita
   componente duplicado e código morto.
5. **Composição de componentes (fase React/Vue)**: quando extrair subcomponente vs deixar
   inline, controlado vs não controlado, evitar prop drilling. "Causa raiz vs sintoma" aplicado
   a estrutura de componente.
6. **Consistência visual**: reaproveitar token de espaçamento/cor definido em vez de valor
   solto — checável por script (grep contra os tokens), não precisa de julgamento humano nessa
   parte.
7. **Custo/benefício do 3D**: verbalizar se a cena precisa mesmo ser WebGL ou se CSS/SVG
   resolveria com menos peso. "Compara as duas abordagens antes de escolher" aplicado a "vale
   3D aqui?".
8. **Verificação real planejada**: o `<think>` verbaliza a intenção de renderizar e conferir
   antes de considerar pronto ("vou gerar isso e olhar o resultado renderizado + console antes
   de responder") — mesma verbalização de intenção de ferramenta da wave 2, com a "ferramenta"
   sendo o preview real.

## 7. Identidade e system prompt

A identidade Logos-3/Conatus (fechada na wave 2, `D-conatus-logos3-identity`) permanece. O
system prompt provavelmente ganha uma frase sobre a especialização em frontend — decisão em
aberto se isso muda o texto-base (que exigiria regerar/revalidar o dataset de novo, com o
mesmo cuidado train/inference da wave 2) ou se fica só implícito no conteúdo dos exemplos. A
definição da família Conatus continua aparecendo só quando perguntado diretamente.

## 8. Geração de dataset especializado

O dataset atual (~2455 exemplos em `data/train`) é majoritariamente algoritmos/backend/CLI
genéricos. Um pivô pra frontend precisa de um corpus novo, gerado com o mesmo mecanismo já
validado (templates hand-authored com execução REAL — nunca self-play/LLM-em-loop sem
validação, decisão de custo da wave 2). Categorias candidatas (a detalhar num plano de execução
separado, não aqui):

- Landing pages (hero + features + CTA + footer), com e sem Three.js.
- Dashboards (cards de métrica, tabelas, gráficos, layout responsivo).
- Componentes isolados (button, card, modal, form) — treinando reutilização via `search_code`.
- Ciclos de correção estética real (gerar → renderizar → ver que ficou ruim → ajustar).
- Refatoração de UI existente (dado um HTML feio, deixar elegante).

Cada exemplo passa pelo checker de frontend (seção 4) ANTES de entrar no dataset — nada de
afirmar "ficou elegante" sobre algo que não foi renderizado de verdade.

## 9. O que muda no que já existe — resumo de impacto

| Componente | Mudança |
|---|---|
| `configs/train_*.yaml` | `base_model` → Qwen3-4B; revalidar VRAM (deve sobrar folga); revisar `lora.target_modules` pra arquitetura Qwen3 |
| `configs/tools_registry.yaml` | `search_code` → `enabled: true` |
| `src/tools/executors/` | novo `search_code_tool.py`; novo backend de frontend do checker |
| `src/checker/backends/` | novo backend de frontend (render + screenshot + console + a11y + design-system) |
| `src/schemas/checker.json` | estender pra aceitar `language: "html"`/`"react"`/etc. |
| `src/dataset/taxonomy.py` | novos task_types de frontend (landing_page, dashboard, component, ui_refactor...) |
| `PRAXIS_SYSTEM_PROMPT` + CLI | possível linha sobre especialização frontend (decisão aberta) |
| `data/train` | corpus novo de frontend (plano de execução separado) |

## 10. Ordem de execução proposta (não fazer tudo de uma vez)

1. **Trocar o modelo base pra Qwen3-4B e revalidar VRAM** — item mais barato, alívio imediato
   de hardware, destrava tudo o mais. Confirmar que o QLoRA cabe com folga na A2000 (deve
   caber com MUITA folga vs o 8B).
2. **Reativar `search_code`** — pequeno, isolado, testável, útil independente do resto.
3. **Construir o backend de frontend do checker** — a peça central e mais difícil; sem ela,
   nada de dataset de frontend confiável. Começar pela fase 1 (HTML estático: render +
   screenshot + console + axe-core).
4. **Gerar um lote-piloto pequeno de frontend** (landing pages estáticas), validar qualidade
   com o checker novo + julgamento manual, antes de escalar.
5. **Retreinar e avaliar** com o checker de frontend como métrica de progresso.
6. **Fase 2 (React/Vue + R3F)** só depois que a fase 1 estiver sólida.

## 11. Perguntas em aberto (decisões que faltam antes de executar)

- Tailwind (via CDN na fase 1) como design system, ou CSS puro com tokens próprios?
- O system prompt muda o texto-base (regera dataset) ou a especialização fica só implícita?
- React ou Vue pra fase 2? (usuário disse "React/Vue" — escolher um pra focar).
- Manter os ~2455 exemplos genéricos atuais no treino (pra não perder capacidade base de
  CLI/código) ou treinar só no corpus de frontend novo? Provavelmente MANTER uma fração dos
  genéricos (as tools base ainda importam), mas a proporção é uma decisão real.
- Escala do corpus de frontend (a wave 2 calibrou ~700-900 total; frontend com render real por
  exemplo é mais caro — número a definir).
