#!/usr/bin/env python
"""Lote-piloto de frontend (docs/plan_frontend_specialization_wave3.md, seção 10, passo 4) —
primeiro corpus real do pivô de especialização. Fase 1 apenas (HTML/CSS/JS estático, sem
build step, sem Three.js ainda — "landing pages... com e sem Three.js" na ordem mais simples
primeiro).

Mesma disciplina de sempre (D11 + seção 8 do plano de frontend): NUNCA fabricar um `tool_result`
de checker — todo `write_file`/`search_code`/`checker` roda de verdade contra o sandbox real,
`checker(language="html")` renderiza num Chromium headless de verdade (D-checker-html-backend).
`build_example` levanta `AssertionError` se qualquer passo não se comportar como o esperado —
um exemplo com asserção falha não entra no dataset (não existe fallback silencioso).

Cobre 4 das 8 dimensões de `<think>` técnico da seção 6 do plano de forma central:
  4. reutilização antes de criação (via `search_code` real contra um "projeto" pré-existente
     escrito direto no workspace do sandbox, fora da trajetória do modelo — simula um repo que
     já tinha componentes antes do pedido do usuário)
  2. trade-off de layout (Flexbox vs Grid, verbalizado no `<think>`)
  6. consistência visual (reaproveitar tokens de `tokens.css` em vez de valor solto)
  8. verificação real planejada (o `<think>` antes do `<final>` verbaliza a intenção de
     renderizar/checar console antes de considerar pronto)
As outras 4 (acessibilidade a fundo, responsividade com breakpoint, composição de componentes
React/Vue, custo/benefício de 3D) ficam pra quando a fase 2 e a camada 3 do checker existirem.

Uso:
    python scripts/gen_frontend_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.harness.system_prompt import PRAXIS_SYSTEM_PROMPT  # noqa: E402
from src.harness.trajectory import Trajectory  # noqa: E402
from src.security import SandboxContext, SandboxPolicy  # noqa: E402
from src.tools import ToolExecutorRegistry  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "train"

_POLICY = SandboxPolicy.load().with_confirmation_mode("auto_approve_safe")
_REGISTRY = ToolExecutorRegistry()

# "Design system" mínimo reaproveitado entre exemplos — tokens reais, não fingidos, checáveis
# por grep (dimensão 6 do think técnico de frontend).
TOKENS_CSS = (
    ":root {\n"
    "  --color-bg: #0f0f11;\n"
    "  --color-surface: #1a1a1e;\n"
    "  --color-text: #f5f5f7;\n"
    "  --color-muted: #a1a1aa;\n"
    "  --color-primary: #6366f1;\n"
    "  --space-sm: 0.5rem;\n"
    "  --space-md: 1rem;\n"
    "  --space-lg: 2rem;\n"
    "  --space-xl: 4rem;\n"
    "  --radius-md: 0.5rem;\n"
    "}\n"
)

COMPONENTS_CSS = (
    ".btn {\n"
    "  display: inline-block;\n"
    "  padding: var(--space-sm) var(--space-lg);\n"
    "  background: var(--color-primary);\n"
    "  color: var(--color-text);\n"
    "  border: none;\n"
    "  border-radius: var(--radius-md);\n"
    "  font-weight: 600;\n"
    "  cursor: pointer;\n"
    "}\n\n"
    ".card {\n"
    "  background: var(--color-surface);\n"
    "  border-radius: var(--radius-md);\n"
    "  padding: var(--space-lg);\n"
    "}\n"
)


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


def _seed_existing_project(sandbox, files: dict[str, str]) -> None:
    """Escreve arquivos DIRETO no workspace do sandbox, fora da trajetória — representa um
    projeto que já existia antes do pedido do usuário chegar (o que `search_code`/`read_file`
    vão encontrar), não uma ação que o modelo "fez" nesta conversa."""
    for path, content in files.items():
        dest = sandbox.workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def build_example(
    id_: str,
    difficulty: str,
    user_request: str,
    seed_files: dict[str, str],
    steps: list[tuple[str, str, dict]],
    task_type: str,
    final_text: str,
) -> dict:
    """`steps` é uma lista de (think_text, tool_name, tool_args) ou
    (think_text, tool_name, tool_args, verify_fn) aplicada em ordem — `verify_fn(result)` checa
    que o resultado real bate com o que o `<think>` afirma (ex.: search_code achou/não achou o
    que o texto diz que achou/não achou). O último passo precisa ser um `checker` que passa."""
    sandbox = SandboxContext(policy=_POLICY)
    try:
        _seed_existing_project(sandbox, seed_files)

        traj = Trajectory(system_prompt=PRAXIS_SYSTEM_PROMPT, user_request=user_request)

        tools_used: list[str] = []
        last_result = None
        last_tool_name = None
        for step in steps:
            think_text, tool_name, tool_args = step[0], step[1], step[2]
            verify_fn = step[3] if len(step) > 3 else None
            traj.append_raw(f"<think>{think_text}</think>")
            last_result = _tool_call(traj, sandbox, tool_name, tool_args)
            last_tool_name = tool_name
            tools_used.append(tool_name)
            if tool_name in ("write_file", "search_code"):
                assert last_result.passed, f"{tool_name} falhou inesperadamente em {id_}: {last_result.to_json()}"
            if verify_fn is not None:
                assert verify_fn(last_result), f"{tool_name} em {id_} não bateu com o que o <think> afirma: {last_result.to_json()}"

        assert last_tool_name == "checker", f"{id_}: último passo precisa ser 'checker'"
        assert last_result.passed, f"checker final falhou inesperadamente em {id_}: {last_result.to_json()}"

        traj.append_raw(f"<final>{final_text}</final>")
    finally:
        sandbox.cleanup()

    return {
        "metadata": {
            "id": id_,
            "domain": "frontend",
            "language": "html",
            "difficulty": difficulty,
            "tools_used": tools_used,
            "num_steps": len(steps),
            "task_type": task_type,
            "source": "curated_manual_real_execution",
            "license": "synthetic-no-license-needed",
            "validation_status": "validated",
            "execution_performed": True,
            "split": "train",
            "checker_used": "checker-html-1.0",
            "expected_result": {"passed": True},
            "execution_classification": "interpreted",
        },
        "trajectory": {
            "system_prompt": PRAXIS_SYSTEM_PROMPT,
            "user_request": user_request,
            "raw_text": traj.raw_text,
        },
    }


# --- exemplo 1: landing page simples, projeto vazio, primeira vez ---------------------------

EX_LANDING_HERO = dict(
    id_="gen-frontend-landing_hero_cta",
    difficulty="easy",
    user_request="Crie uma landing page simples com um hero, um título, uma descrição curta e um botão de call-to-action.",
    seed_files={},
    task_type="multi_tool_call",
    steps=[
        (
            "É um projeto novo, vou verificar se já existe algum arquivo de estilos antes de "
            "inventar tokens do zero.",
            "search_code",
            {"pattern": "--color-primary", "path": "."},
        ),
        (
            "Não encontrei nada — projeto vazio. Vou definir tokens de cor/espaçamento num "
            "arquivo próprio em vez de espalhar valores soltos pelo HTML, assim qualquer página "
            "futura reaproveita as mesmas variáveis.",
            "write_file",
            {"path": "tokens.css", "content": TOKENS_CSS},
        ),
        (
            "Para o hero, uma única coluna centralizada não precisa de Grid — Flexbox com "
            "direção coluna resolve com menos código e é a ferramenta certa quando só existe "
            "um eixo de alinhamento.",
            "write_file",
            {
                "path": "index.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>Landing</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                    "<style>\n"
                    "body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                    "font-family: system-ui, sans-serif; }\n"
                    ".hero { display: flex; flex-direction: column; align-items: center; "
                    "justify-content: center; min-height: 100vh; text-align: center; "
                    "gap: var(--space-md); padding: var(--space-lg); }\n"
                    ".hero h1 { font-size: 2.5rem; margin: 0; }\n"
                    ".hero p { color: var(--color-muted); max-width: 32rem; margin: 0; }\n"
                    ".btn { display: inline-block; padding: var(--space-sm) var(--space-lg); "
                    "background: var(--color-primary); color: var(--color-text); border: none; "
                    "border-radius: var(--radius-md); font-weight: 600; cursor: pointer; }\n"
                    "</style>\n</head>\n<body>\n"
                    "<section class=\"hero\">\n"
                    "<h1>Construa mais rápido</h1>\n"
                    "<p>Uma ferramenta pequena, direta ao ponto, sem enrolação.</p>\n"
                    "<button class=\"btn\">Começar agora</button>\n"
                    "</section>\n</body>\n</html>\n"
                ),
            },
        ),
        (
            "A página está pronta. Antes de responder, vou renderizar de verdade num navegador "
            "e conferir o console — não vou considerar isso pronto só porque o HTML parece "
            "correto na leitura.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS},
                    {
                        "path": "index.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>Landing</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                            "<style>\n"
                            "body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                            "font-family: system-ui, sans-serif; }\n"
                            ".hero { display: flex; flex-direction: column; align-items: center; "
                            "justify-content: center; min-height: 100vh; text-align: center; "
                            "gap: var(--space-md); padding: var(--space-lg); }\n"
                            ".hero h1 { font-size: 2.5rem; margin: 0; }\n"
                            ".hero p { color: var(--color-muted); max-width: 32rem; margin: 0; }\n"
                            ".btn { display: inline-block; padding: var(--space-sm) var(--space-lg); "
                            "background: var(--color-primary); color: var(--color-text); border: none; "
                            "border-radius: var(--radius-md); font-weight: 600; cursor: pointer; }\n"
                            "</style>\n</head>\n<body>\n"
                            "<section class=\"hero\">\n"
                            "<h1>Construa mais rápido</h1>\n"
                            "<p>Uma ferramenta pequena, direta ao ponto, sem enrolação.</p>\n"
                            "<button class=\"btn\">Começar agora</button>\n"
                            "</section>\n</body>\n</html>\n"
                        ),
                    },
                ],
                "entrypoint": "index.html",
            },
        ),
    ],
    final_text=(
        "Criei a landing page em index.html com um hero centralizado (Flexbox em coluna, já que "
        "só há um eixo de alinhamento aqui, Grid seria complexidade sem benefício), extraí cores "
        "e espaçamentos para tokens.css em vez de valores soltos, e confirmei renderizando num "
        "navegador de verdade que a página carrega sem erro de console."
    ),
)


# --- exemplo 2: dashboard com Grid, projeto vazio ------------------------------------------

EX_DASHBOARD_GRID = dict(
    id_="gen-frontend-dashboard_metric_cards",
    difficulty="medium",
    user_request="Faça um dashboard com 3 cartões de métrica lado a lado, cada um com um número grande e um rótulo.",
    seed_files={},
    task_type="multi_tool_call",
    steps=[
        (
            "Aqui as duas dimensões importam: preciso controlar tanto as colunas (3 cartões "
            "lado a lado) quanto o alinhamento das linhas dentro de cada cartão — isso é "
            "exatamente o caso em que Grid vale mais que Flexbox, que só controla bem um eixo "
            "por vez.",
            "write_file",
            {
                "path": "dashboard.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>Dashboard</title>\n<style>\n"
                    ":root { --color-bg: #0f0f11; --color-surface: #1a1a1e; --color-text: #f5f5f7; "
                    "--color-muted: #a1a1aa; --space-md: 1rem; --space-lg: 2rem; --radius-md: 0.5rem; }\n"
                    "body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                    "font-family: system-ui, sans-serif; padding: var(--space-lg); }\n"
                    ".grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-md); }\n"
                    ".card { background: var(--color-surface); border-radius: var(--radius-md); "
                    "padding: var(--space-lg); text-align: center; }\n"
                    ".card .value { font-size: 2.5rem; font-weight: 700; display: block; }\n"
                    ".card .label { color: var(--color-muted); }\n"
                    "</style>\n</head>\n<body>\n"
                    "<div class=\"grid\">\n"
                    "<div class=\"card\"><span class=\"value\">128</span><span class=\"label\">Usuários ativos</span></div>\n"
                    "<div class=\"card\"><span class=\"value\">42</span><span class=\"label\">Pedidos hoje</span></div>\n"
                    "<div class=\"card\"><span class=\"value\">97%</span><span class=\"label\">Uptime</span></div>\n"
                    "</div>\n</body>\n</html>\n"
                ),
            },
        ),
        (
            "Antes de responder, vou renderizar e checar o console — o CSS parece certo na "
            "leitura, mas isso não substitui a execução real.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {
                        "path": "dashboard.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>Dashboard</title>\n<style>\n"
                            ":root { --color-bg: #0f0f11; --color-surface: #1a1a1e; --color-text: #f5f5f7; "
                            "--color-muted: #a1a1aa; --space-md: 1rem; --space-lg: 2rem; --radius-md: 0.5rem; }\n"
                            "body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                            "font-family: system-ui, sans-serif; padding: var(--space-lg); }\n"
                            ".grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-md); }\n"
                            ".card { background: var(--color-surface); border-radius: var(--radius-md); "
                            "padding: var(--space-lg); text-align: center; }\n"
                            ".card .value { font-size: 2.5rem; font-weight: 700; display: block; }\n"
                            ".card .label { color: var(--color-muted); }\n"
                            "</style>\n</head>\n<body>\n"
                            "<div class=\"grid\">\n"
                            "<div class=\"card\"><span class=\"value\">128</span><span class=\"label\">Usuários ativos</span></div>\n"
                            "<div class=\"card\"><span class=\"value\">42</span><span class=\"label\">Pedidos hoje</span></div>\n"
                            "<div class=\"card\"><span class=\"value\">97%</span><span class=\"label\">Uptime</span></div>\n"
                            "</div>\n</body>\n</html>\n"
                        ),
                    }
                ],
                "entrypoint": "dashboard.html",
            },
        ),
    ],
    final_text=(
        "Criei o dashboard em dashboard.html usando CSS Grid com 3 colunas — escolhi Grid em vez "
        "de Flexbox porque aqui as duas dimensões importam (colunas dos cartões e alinhamento "
        "interno de cada um), e confirmei renderizando de verdade que não há erro de console."
    ),
)


# --- exemplo 3: reuso de componente já existente (search_code encontra) --------------------

EX_REUSE_BUTTON = dict(
    id_="gen-frontend-reuse_existing_button",
    difficulty="easy",
    user_request="Adicione um botão de 'Assinar' na página de preços.",
    seed_files={"tokens.css": TOKENS_CSS, "components.css": COMPONENTS_CSS},
    task_type="multi_tool_call",
    steps=[
        (
            "Antes de inventar um estilo de botão novo, vou procurar se o projeto já tem um "
            "componente de botão padronizado.",
            "search_code",
            {"pattern": ".btn", "path": "."},
            lambda r: len(r.data["matches"]) > 0,
        ),
        (
            "Encontrei .btn já definido em components.css, reaproveitando tokens de tokens.css. "
            "Vou usar exatamente essa classe em vez de criar uma nova — duplicar aqui só geraria "
            "dois estilos de botão divergentes no mesmo produto.",
            "write_file",
            {
                "path": "pricing.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>Preços</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                    "<link rel=\"stylesheet\" href=\"components.css\">\n"
                    "<style>body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                    "font-family: system-ui, sans-serif; padding: var(--space-lg); text-align: center; }"
                    "</style>\n</head>\n<body>\n"
                    "<h1>Plano Pro</h1>\n"
                    "<button class=\"btn\">Assinar</button>\n"
                    "</body>\n</html>\n"
                ),
            },
        ),
        (
            "Vou renderizar pra confirmar que o botão reaproveitado aparece corretamente e não "
            "há erro de console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS},
                    {"path": "components.css", "content": COMPONENTS_CSS},
                    {
                        "path": "pricing.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>Preços</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                            "<link rel=\"stylesheet\" href=\"components.css\">\n"
                            "<style>body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                            "font-family: system-ui, sans-serif; padding: var(--space-lg); text-align: center; }"
                            "</style>\n</head>\n<body>\n"
                            "<h1>Plano Pro</h1>\n"
                            "<button class=\"btn\">Assinar</button>\n"
                            "</body>\n</html>\n"
                        ),
                    },
                ],
                "entrypoint": "pricing.html",
            },
        ),
    ],
    final_text=(
        "Encontrei que o projeto já tem um componente .btn padronizado em components.css e "
        "reaproveitei ele em pricing.html em vez de criar um estilo de botão novo, evitando "
        "duplicação. Confirmei renderizando que a página carrega sem erro de console."
    ),
)


# --- exemplo 4: componente novo, confirmado que NÃO existe ---------------------------------

EX_NEW_COMPONENT_AFTER_CHECK = dict(
    id_="gen-frontend-new_modal_after_search",
    difficulty="medium",
    user_request="Preciso de um modal de confirmação com um título, uma mensagem e dois botões (confirmar/cancelar).",
    seed_files={"tokens.css": TOKENS_CSS, "components.css": COMPONENTS_CSS},
    task_type="multi_tool_call",
    steps=[
        (
            "Antes de criar um componente novo, vou procurar se já existe algo de modal no "
            "projeto.",
            "search_code",
            {"pattern": "modal", "path": "."},
            lambda r: len(r.data["matches"]) == 0,
        ),
        (
            "Não encontrei nenhuma classe ou componente de modal existente — components.css só "
            "tem .btn e .card. Posso criar um novo com segurança, reaproveitando os tokens de "
            "cor/espaçamento já definidos em vez de inventar valores soltos.",
            "write_file",
            {
                "path": "components.css",
                "content": (
                    COMPONENTS_CSS + "\n"
                    ".modal-backdrop {\n"
                    "  position: fixed; inset: 0; background: rgba(0,0,0,0.6);\n"
                    "  display: flex; align-items: center; justify-content: center;\n"
                    "}\n\n"
                    ".modal {\n"
                    "  background: var(--color-surface); border-radius: var(--radius-md);\n"
                    "  padding: var(--space-lg); max-width: 24rem; text-align: center;\n"
                    "}\n\n"
                    ".modal .actions {\n"
                    "  display: flex; gap: var(--space-sm); justify-content: center;\n"
                    "  margin-top: var(--space-md);\n"
                    "}\n"
                ),
            },
        ),
        (
            "Vou montar a página de exemplo usando o novo .modal e renderizar de verdade antes "
            "de considerar pronto.",
            "write_file",
            {
                "path": "index.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>Modal</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                    "<link rel=\"stylesheet\" href=\"components.css\">\n"
                    "<style>body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                    "font-family: system-ui, sans-serif; }</style>\n</head>\n<body>\n"
                    "<div class=\"modal-backdrop\">\n"
                    "<div class=\"modal\">\n"
                    "<h2>Confirmar exclusão</h2>\n"
                    "<p>Essa ação não pode ser desfeita.</p>\n"
                    "<div class=\"actions\">\n"
                    "<button class=\"btn\">Confirmar</button>\n"
                    "<button class=\"btn\">Cancelar</button>\n"
                    "</div>\n</div>\n</div>\n</body>\n</html>\n"
                ),
            },
        ),
        (
            "Renderizando agora pra confirmar que não há erro de console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS},
                    {
                        "path": "components.css",
                        "content": (
                            COMPONENTS_CSS + "\n"
                            ".modal-backdrop {\n"
                            "  position: fixed; inset: 0; background: rgba(0,0,0,0.6);\n"
                            "  display: flex; align-items: center; justify-content: center;\n"
                            "}\n\n"
                            ".modal {\n"
                            "  background: var(--color-surface); border-radius: var(--radius-md);\n"
                            "  padding: var(--space-lg); max-width: 24rem; text-align: center;\n"
                            "}\n\n"
                            ".modal .actions {\n"
                            "  display: flex; gap: var(--space-sm); justify-content: center;\n"
                            "  margin-top: var(--space-md);\n"
                            "}\n"
                        ),
                    },
                    {
                        "path": "index.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>Modal</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n"
                            "<link rel=\"stylesheet\" href=\"components.css\">\n"
                            "<style>body { margin: 0; background: var(--color-bg); color: var(--color-text); "
                            "font-family: system-ui, sans-serif; }</style>\n</head>\n<body>\n"
                            "<div class=\"modal-backdrop\">\n"
                            "<div class=\"modal\">\n"
                            "<h2>Confirmar exclusão</h2>\n"
                            "<p>Essa ação não pode ser desfeita.</p>\n"
                            "<div class=\"actions\">\n"
                            "<button class=\"btn\">Confirmar</button>\n"
                            "<button class=\"btn\">Cancelar</button>\n"
                            "</div>\n</div>\n</div>\n</body>\n</html>\n"
                        ),
                    },
                ],
                "entrypoint": "index.html",
            },
        ),
    ],
    final_text=(
        "Procurei no projeto e confirmei que não existia nenhum componente de modal ainda, "
        "então criei .modal-backdrop/.modal/.actions em components.css reaproveitando os tokens "
        "existentes (cor de superfície, raio, espaçamento) em vez de valores soltos, montei o "
        "exemplo em index.html reaproveitando também o .btn já existente, e confirmei "
        "renderizando que a página carrega sem erro de console."
    ),
)


# --- exemplo 5: ciclo de correção real (bug de JS -> checker falha -> corrige) --------------

EX_FIX_AFTER_CONSOLE_ERROR = dict(
    id_="gen-frontend-fix_broken_toggle_script",
    difficulty="medium",
    user_request="Crie uma seção de FAQ onde clicar na pergunta expande a resposta, usando JavaScript simples.",
    seed_files={},
    task_type="model_fixes_after_error",
    steps=[
        (
            "Vou implementar um toggle simples com JS puro: cada pergunta tem um botão que "
            "alterna a classe 'open' da resposta correspondente.",
            "write_file",
            {
                "path": "faq.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>FAQ</title>\n<style>\n"
                    "body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; }\n"
                    ".answer { display: none; }\n.answer.open { display: block; }\n"
                    "</style>\n</head>\n<body>\n"
                    "<div class=\"faq-item\">\n"
                    "<button onclick=\"toggleAnswer(this)\">O que é o Logos?</button>\n"
                    "<p class=\"answer\">Um assistente especializado em frontend.</p>\n"
                    "</div>\n"
                    "<script>\n"
                    "function toggleAnswer(button) {\n"
                    "  const answer = button.nextElementSibling;\n"
                    "  answer.classList.toggle('open');\n"
                    "  updateAriaState(button);\n"
                    "}\n"
                    "</script>\n</body>\n</html>\n"
                ),
            },
        ),
        (
            "Antes de responder, vou renderizar de verdade e conferir o console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {
                        "path": "faq.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>FAQ</title>\n<style>\n"
                            "body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; }\n"
                            ".answer { display: none; }\n.answer.open { display: block; }\n"
                            "</style>\n</head>\n<body>\n"
                            "<div class=\"faq-item\">\n"
                            "<button onclick=\"toggleAnswer(this)\">O que é o Logos?</button>\n"
                            "<p class=\"answer\">Um assistente especializado em frontend.</p>\n"
                            "</div>\n"
                            "<script>\n"
                            "function toggleAnswer(button) {\n"
                            "  const answer = button.nextElementSibling;\n"
                            "  answer.classList.toggle('open');\n"
                            "  updateAriaState(button);\n"
                            "}\n"
                            "</script>\n</body>\n</html>\n"
                        ),
                    }
                ],
                "entrypoint": "faq.html",
            },
        ),
        (
            "O checker apontou 'updateAriaState is not defined' — cliquei sem querer numa "
            "função auxiliar que eu tinha em mente pra sincronizar aria-expanded, mas nunca "
            "implementei. A causa raiz não é a lógica de toggle (essa está certa), é essa "
            "chamada morta. Vou implementar a função de verdade em vez de só removê-la, já que "
            "aria-expanded correto é parte real de acessibilidade num toggle.",
            "write_file",
            {
                "path": "faq.html",
                "content": (
                    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                    "<title>FAQ</title>\n<style>\n"
                    "body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; }\n"
                    ".answer { display: none; }\n.answer.open { display: block; }\n"
                    "</style>\n</head>\n<body>\n"
                    "<div class=\"faq-item\">\n"
                    "<button onclick=\"toggleAnswer(this)\" aria-expanded=\"false\">O que é o Logos?</button>\n"
                    "<p class=\"answer\">Um assistente especializado em frontend.</p>\n"
                    "</div>\n"
                    "<script>\n"
                    "function toggleAnswer(button) {\n"
                    "  const answer = button.nextElementSibling;\n"
                    "  answer.classList.toggle('open');\n"
                    "  updateAriaState(button);\n"
                    "}\n"
                    "function updateAriaState(button) {\n"
                    "  const expanded = button.getAttribute('aria-expanded') === 'true';\n"
                    "  button.setAttribute('aria-expanded', String(!expanded));\n"
                    "}\n"
                    "</script>\n</body>\n</html>\n"
                ),
            },
        ),
        (
            "Vou renderizar de novo pra confirmar que o erro sumiu.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {
                        "path": "faq.html",
                        "content": (
                            "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
                            "<title>FAQ</title>\n<style>\n"
                            "body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; }\n"
                            ".answer { display: none; }\n.answer.open { display: block; }\n"
                            "</style>\n</head>\n<body>\n"
                            "<div class=\"faq-item\">\n"
                            "<button onclick=\"toggleAnswer(this)\" aria-expanded=\"false\">O que é o Logos?</button>\n"
                            "<p class=\"answer\">Um assistente especializado em frontend.</p>\n"
                            "</div>\n"
                            "<script>\n"
                            "function toggleAnswer(button) {\n"
                            "  const answer = button.nextElementSibling;\n"
                            "  answer.classList.toggle('open');\n"
                            "  updateAriaState(button);\n"
                            "}\n"
                            "function updateAriaState(button) {\n"
                            "  const expanded = button.getAttribute('aria-expanded') === 'true';\n"
                            "  button.setAttribute('aria-expanded', String(!expanded));\n"
                            "}\n"
                            "</script>\n</body>\n</html>\n"
                        ),
                    }
                ],
                "entrypoint": "faq.html",
            },
        ),
    ],
    final_text=(
        "Criei a seção de FAQ em faq.html com toggle via JavaScript. A primeira renderização "
        "apontou um erro real de console: updateAriaState não estava definida. A causa raiz era "
        "uma chamada morta a uma função de sincronização de aria-expanded que eu pretendia "
        "escrever mas esqueci — implementei a função de verdade, já que isso também melhora a "
        "acessibilidade do toggle, e confirmei renderizando de novo que o erro sumiu."
    ),
)


EXAMPLES = [
    EX_LANDING_HERO,
    EX_DASHBOARD_GRID,
    EX_REUSE_BUTTON,
    EX_NEW_COMPONENT_AFTER_CHECK,
    EX_FIX_AFTER_CONSOLE_ERROR,
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for spec in EXAMPLES:
        example = build_example(**spec)
        out_path = OUT_DIR / f"{example['metadata']['id']}.json"
        out_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {out_path.name} (renderizado de verdade em Chromium headless, checker passou)")
        written += 1
    print(f"\n{written} exemplos de frontend gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
