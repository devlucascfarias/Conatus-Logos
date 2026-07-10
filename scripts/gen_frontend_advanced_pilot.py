#!/usr/bin/env python
"""Segunda leva do lote-piloto de frontend (docs/plan_frontend_specialization_wave3.md, seção
10, passo 4 — complementa `gen_frontend_pilot.py`) — exemplos mais complexos: Three.js vanilla
(seção 5, "Fase 1... + Three.js vanilla"), glassmorphism, e a dimensão 7 do `<think>` técnico
(custo/benefício de 3D) nos dois sentidos: um caso em que 3D SE justifica e um em que NÃO se
justifica (o pedido menciona "3D" mas a engenharia certa é recusar).

Achado real desta sessão, documentado em docs/PLAN.md (D-frontend-threejs-pilot): addons do
Three.js (ex. OrbitControls) importam via specifier nu (`import ... from 'three'`), que o
Chromium NÃO resolve sem um `<script type="importmap">` mapeando "three" pra URL do CDN — sem
isso, falha de verdade com "Failed to resolve module specifier". Confirmado rodando o checker
`html` de verdade antes de escrever qualquer exemplo (nunca assumido). Isso também é a
justificativa real por trás da seção 4.4 do plano (cuidado específico com Three.js/WebGL).

Mesma disciplina de sempre: `write_file`/`search_code`/`checker` reais, `checker(language="html")`
renderiza de verdade em Chromium headless (WebGL via SwiftShader/software rendering nesse
ambiente — confirmado sem erro de console, só warning de fallback, que não falha o checker).

Uso:
    python scripts/gen_frontend_advanced_pilot.py
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

THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
ORBIT_CONTROLS_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js"

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

TOKENS_CSS_WITH_GLASS = (
    TOKENS_CSS[:-2] + "\n"
    "  --glass-bg: rgba(255, 255, 255, 0.08);\n"
    "  --glass-border: rgba(255, 255, 255, 0.18);\n"
    "  --blur-md: 12px;\n"
    "}\n"
)


def _tool_call(traj: Trajectory, sandbox, name: str, args: dict):
    traj.append_raw(f'<tool_call name="{name}">{json.dumps(args, ensure_ascii=False)}</tool_call>')
    result = _REGISTRY.execute(name, args, sandbox)
    traj.append_tool_result(name, "ok" if result.passed else "error", result.to_json())
    return result


def _seed_existing_project(sandbox, files: dict[str, str]) -> None:
    for path, content in files.items():
        dest = sandbox.workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def build_example(
    id_: str,
    difficulty: str,
    user_request: str,
    seed_files: dict[str, str],
    steps: list[tuple],
    task_type: str,
    final_text: str,
    expect_last_checker_passed: bool = True,
) -> dict:
    """Mesmo formato de `gen_frontend_pilot.py::build_example`. `steps` é uma lista de
    (think_text, tool_name, tool_args) ou (think_text, tool_name, tool_args, verify_fn)."""
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
                assert verify_fn(last_result), f"{tool_name} em {id_} não bateu com o <think>: {last_result.to_json()}"

        assert last_tool_name == "checker", f"{id_}: último passo precisa ser 'checker'"
        assert last_result.passed == expect_last_checker_passed, (
            f"checker final em {id_} esperado passed={expect_last_checker_passed}, veio "
            f"{last_result.passed}: {last_result.to_json()}"
        )

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


# --- exemplo A: hero com Three.js (torus girando) + botão glassmorphism --------------------

_HERO_THREEJS_HTML = (
    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>Landing 3D</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n<style>\n"
    "* { box-sizing: border-box; }\n"
    "body { margin: 0; background: var(--color-bg); color: var(--color-text); "
    "font-family: system-ui, sans-serif; overflow: hidden; }\n"
    "#scene { position: fixed; inset: 0; }\n"
    ".hero-content { position: relative; z-index: 1; display: flex; flex-direction: column; "
    "align-items: center; justify-content: center; min-height: 100vh; text-align: center; "
    "gap: var(--space-md); padding: var(--space-lg); pointer-events: none; }\n"
    ".hero-content h1 { font-size: 3rem; margin: 0; }\n"
    ".hero-content p { color: var(--color-muted); max-width: 32rem; margin: 0; }\n"
    ".btn-glass { pointer-events: auto; padding: var(--space-sm) var(--space-lg); "
    "background: var(--glass-bg); border: 1px solid var(--glass-border); "
    "backdrop-filter: blur(var(--blur-md)); -webkit-backdrop-filter: blur(var(--blur-md)); "
    "color: var(--color-text); border-radius: var(--radius-md); font-weight: 600; "
    "cursor: pointer; }\n"
    "</style>\n</head>\n<body>\n"
    "<canvas id=\"scene\"></canvas>\n"
    "<div class=\"hero-content\">\n"
    "<h1>Construa em outra dimensão</h1>\n"
    "<p>Uma plataforma que sai do achatado.</p>\n"
    "<button class=\"btn-glass\">Começar agora</button>\n"
    "</div>\n"
    "<script type=\"module\">\n"
    f"import * as THREE from '{THREE_CDN}';\n"
    "const canvas = document.getElementById('scene');\n"
    "const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });\n"
    "renderer.setSize(window.innerWidth, window.innerHeight);\n"
    "renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));\n"
    "const scene = new THREE.Scene();\n"
    "const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);\n"
    "camera.position.z = 5;\n"
    "const geometry = new THREE.TorusKnotGeometry(1, 0.32, 128, 24);\n"
    "const material = new THREE.MeshStandardMaterial({ color: 0x6366f1, roughness: 0.3, metalness: 0.6 });\n"
    "const knot = new THREE.Mesh(geometry, material);\n"
    "scene.add(knot);\n"
    "scene.add(new THREE.AmbientLight(0xffffff, 0.4));\n"
    "const light = new THREE.PointLight(0xffffff, 1.2);\n"
    "light.position.set(3, 3, 3);\n"
    "scene.add(light);\n"
    "function animate() {\n"
    "  requestAnimationFrame(animate);\n"
    "  knot.rotation.x += 0.004;\n"
    "  knot.rotation.y += 0.006;\n"
    "  renderer.render(scene, camera);\n"
    "}\n"
    "animate();\n"
    "window.addEventListener('resize', () => {\n"
    "  camera.aspect = window.innerWidth / window.innerHeight;\n"
    "  camera.updateProjectionMatrix();\n"
    "  renderer.setSize(window.innerWidth, window.innerHeight);\n"
    "});\n"
    "</script>\n</body>\n</html>\n"
)

EX_HERO_THREEJS = dict(
    id_="gen-frontend-hero_threejs_torus_glass",
    difficulty="hard",
    user_request=(
        "Crie uma landing page com uma seção hero que tenha um objeto 3D girando ao fundo, "
        "usando Three.js, e um botão de call-to-action com efeito de vidro (glassmorphism) por cima."
    ),
    seed_files={},
    task_type="multi_tool_call",
    steps=[
        (
            "Um objeto 3D animado de fundo é exatamente o tipo de elemento que separa uma "
            "landing genérica de uma memorável, e é o elemento central pedido aqui, não um "
            "enfeite incidental — o custo de carregar Three.js via CDN compensa. Vou usar "
            "Three.js vanilla (import de módulo, sem build) e primeiro definir os tokens de "
            "glassmorphism (fundo translúcido, borda sutil, blur) além dos já existentes de "
            "cor/espaçamento.",
            "write_file",
            {"path": "tokens.css", "content": TOKENS_CSS_WITH_GLASS},
        ),
        (
            "Agora o HTML: um <canvas> fixo de fundo com a cena Three.js (torus knot girando "
            "com requestAnimationFrame) e o conteúdo do hero sobreposto com z-index, incluindo "
            "o botão com backdrop-filter reaproveitando os tokens de vidro que acabei de "
            "definir em vez de valores soltos.",
            "write_file",
            {"path": "index.html", "content": _HERO_THREEJS_HTML},
        ),
        (
            "Antes de considerar pronto, vou renderizar de verdade e conferir o console — uma "
            "cena 3D pode parecer correta na leitura do código e ainda assim renderizar um "
            "canvas em branco ou falhar num erro de shader/contexto WebGL, isso só se confirma "
            "executando.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS_WITH_GLASS},
                    {"path": "index.html", "content": _HERO_THREEJS_HTML},
                ],
                "entrypoint": "index.html",
            },
        ),
    ],
    final_text=(
        "Criei a landing em index.html com um hero em que o fundo é uma cena Three.js real "
        "(torus knot girando via requestAnimationFrame, iluminação ambiente + pontual) e o "
        "conteúdo sobreposto usa um botão com glassmorphism (fundo translúcido + blur), "
        "reaproveitando tokens de tokens.css em vez de valores soltos. Justifiquei o uso de 3D "
        "aqui porque é o elemento central do pedido, não decoração, e confirmei renderizando de "
        "verdade num navegador que a cena carrega sem erro de console."
    ),
)


# --- exemplo B: preços com glassmorphism (CSS puro, sem JS) ---------------------------------

_PRICING_GLASS_HTML = (
    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>Preços</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n<style>\n"
    "* { box-sizing: border-box; }\n"
    "body { margin: 0; min-height: 100vh; font-family: system-ui, sans-serif; color: var(--color-text); "
    "background: radial-gradient(circle at top left, #4338ca, #0f0f11 60%); "
    "padding: var(--space-xl) var(--space-lg); }\n"
    ".pricing-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-md); "
    "max-width: 60rem; margin: 0 auto; align-items: stretch; }\n"
    ".plan { background: var(--glass-bg); border: 1px solid var(--glass-border); "
    "backdrop-filter: blur(var(--blur-md)); -webkit-backdrop-filter: blur(var(--blur-md)); "
    "border-radius: var(--radius-md); padding: var(--space-lg); text-align: center; "
    "display: flex; flex-direction: column; gap: var(--space-sm); }\n"
    ".plan .price { font-size: 2rem; font-weight: 700; }\n"
    ".plan .name { color: var(--color-muted); text-transform: uppercase; font-size: 0.8rem; "
    "letter-spacing: 0.05em; }\n"
    "</style>\n</head>\n<body>\n"
    "<div class=\"pricing-grid\">\n"
    "<div class=\"plan\"><span class=\"name\">Básico</span><span class=\"price\">R$29/mês</span></div>\n"
    "<div class=\"plan\"><span class=\"name\">Pro</span><span class=\"price\">R$79/mês</span></div>\n"
    "<div class=\"plan\"><span class=\"name\">Empresa</span><span class=\"price\">R$199/mês</span></div>\n"
    "</div>\n</body>\n</html>\n"
)

EX_PRICING_GLASS = dict(
    id_="gen-frontend-pricing_glassmorphism_grid",
    difficulty="medium",
    user_request=(
        "Crie uma seção de preços com 3 cartões usando um efeito de vidro (glassmorphism) sobre "
        "um fundo com gradiente colorido."
    ),
    seed_files={"tokens.css": TOKENS_CSS},
    task_type="multi_tool_call",
    steps=[
        (
            "Vou verificar se o projeto já tem algum token ou classe de glassmorphism antes de "
            "inventar valores novos.",
            "search_code",
            {"pattern": "backdrop-filter", "path": "."},
            lambda r: len(r.data["matches"]) == 0,
        ),
        (
            "Não encontrei nada — o tokens.css atual só tem cor/espaçamento. Vou estender esse "
            "mesmo arquivo com tokens de vidro (fundo translúcido, borda, blur) em vez de criar "
            "um arquivo novo, mantendo um único lugar de verdade pros tokens.",
            "write_file",
            {"path": "tokens.css", "content": TOKENS_CSS_WITH_GLASS},
        ),
        (
            "Para os 3 cartões, as duas dimensões importam de novo: colunas E altura igual "
            "entre eles (align-items: stretch), então Grid continua sendo a escolha certa, "
            "igual no dashboard.",
            "write_file",
            {"path": "pricing.html", "content": _PRICING_GLASS_HTML},
        ),
        (
            "Vou renderizar pra confirmar que o blur aplica de verdade e não há erro de "
            "console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS_WITH_GLASS},
                    {"path": "pricing.html", "content": _PRICING_GLASS_HTML},
                ],
                "entrypoint": "pricing.html",
            },
        ),
    ],
    final_text=(
        "Confirmei que o projeto não tinha nenhum token de glassmorphism ainda, então estendi o "
        "mesmo tokens.css (em vez de criar outro arquivo) com fundo translúcido/borda/blur, e "
        "montei pricing.html com Grid de 3 colunas com altura igual entre os cartões. Confirmei "
        "renderizando que o efeito de vidro aplica de verdade e não há erro de console."
    ),
)


# --- exemplo C: showcase de produto com OrbitControls — ciclo de correção real -------------

_ORBIT_BROKEN_HTML = (
    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>Showcase</title>\n<style>body{margin:0;background:#0f0f11}</style>\n</head>\n<body>\n"
    "<script type=\"module\">\n"
    f"import * as THREE from '{THREE_CDN}';\n"
    f"import {{ OrbitControls }} from '{ORBIT_CONTROLS_CDN}';\n"
    "const scene = new THREE.Scene();\n"
    "const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);\n"
    "camera.position.z = 4;\n"
    "const renderer = new THREE.WebGLRenderer({ antialias: true });\n"
    "renderer.setSize(window.innerWidth, window.innerHeight);\n"
    "document.body.appendChild(renderer.domElement);\n"
    "const geometry = new THREE.IcosahedronGeometry(1.4, 0);\n"
    "const material = new THREE.MeshStandardMaterial({ color: 0x22c55e, flatShading: true });\n"
    "scene.add(new THREE.Mesh(geometry, material));\n"
    "scene.add(new THREE.AmbientLight(0xffffff, 0.6));\n"
    "const controls = new OrbitControls(camera, renderer.domElement);\n"
    "controls.enableDamping = true;\n"
    "function animate() {\n"
    "  requestAnimationFrame(animate);\n"
    "  controls.update();\n"
    "  renderer.render(scene, camera);\n"
    "}\n"
    "animate();\n"
    "</script>\n</body>\n</html>\n"
)

_ORBIT_FIXED_HTML = _ORBIT_BROKEN_HTML.replace(
    "<title>Showcase</title>\n<style>body{margin:0;background:#0f0f11}</style>\n</head>\n<body>\n",
    (
        "<title>Showcase</title>\n<style>body{margin:0;background:#0f0f11}</style>\n"
        "<script type=\"importmap\">\n"
        "{\"imports\": {\"three\": \"" + THREE_CDN + "\"}}\n"
        "</script>\n</head>\n<body>\n"
    ),
).replace(
    f"import * as THREE from '{THREE_CDN}';\n",
    "import * as THREE from 'three';\n",
)

EX_ORBIT_FIX = dict(
    id_="gen-frontend-product_showcase_orbitcontrols_fix",
    difficulty="hard",
    user_request=(
        "Crie uma cena 3D de showcase de produto, com um objeto no centro que o usuário possa "
        "rotacionar arrastando o mouse, usando Three.js e OrbitControls."
    ),
    seed_files={},
    task_type="model_fixes_after_error",
    steps=[
        (
            "Rotação interativa por arraste é interação real de mouse com um objeto 3D — CSS "
            "não faz isso, então o WebGL se justifica aqui, diferente de um enfeite decorativo. "
            "Vou usar Three.js vanilla com o addon OrbitControls, ambos via CDN.",
            "write_file",
            {"path": "index.html", "content": _ORBIT_BROKEN_HTML},
        ),
        (
            "Antes de responder, vou renderizar de verdade e conferir o console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [{"path": "index.html", "content": _ORBIT_BROKEN_HTML}],
                "entrypoint": "index.html",
            },
        ),
        (
            "O console apontou 'Failed to resolve module specifier \"three\"'. A causa raiz não "
            "é a minha importação do Three.js em si (essa usa a URL completa do CDN e funciona) "
            "— é que o próprio OrbitControls.js importa internamente via 'import ... from "
            "\"three\"', um specifier nu que o navegador não sabe resolver sozinho. Preciso de "
            "um <script type=\"importmap\"> mapeando o nome \"three\" pra URL do CDN, e aí trocar "
            "minha própria importação pra usar o mesmo nome em vez da URL direta, senão os dois "
            "caminhos carregam o Three.js duas vezes.",
            "write_file",
            {"path": "index.html", "content": _ORBIT_FIXED_HTML},
        ),
        (
            "Vou renderizar de novo pra confirmar que o erro sumiu.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [{"path": "index.html", "content": _ORBIT_FIXED_HTML}],
                "entrypoint": "index.html",
            },
        ),
    ],
    final_text=(
        "Criei o showcase em index.html com Three.js + OrbitControls (rotação por arraste "
        "justifica o uso de WebGL aqui, é interação real, não decoração). A primeira "
        "renderização falhou de verdade: o addon OrbitControls importa 'three' como specifier "
        "nu internamente, que só resolve com um import map — adicionei "
        "<script type=\"importmap\"> mapeando 'three' pra URL do CDN e ajustei minha própria "
        "importação pra usar o mesmo caminho, evitando carregar a biblioteca duas vezes. "
        "Confirmei renderizando de novo que o erro sumiu."
    ),
)


# --- exemplo D: pedido menciona "3D" mas a engenharia certa é recusar ----------------------

_SPINNER_CSS_HTML = (
    "<!doctype html>\n<html lang=\"pt-br\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>Carregando</title>\n<link rel=\"stylesheet\" href=\"tokens.css\">\n<style>\n"
    "body { margin: 0; min-height: 100vh; display: flex; align-items: center; "
    "justify-content: center; background: var(--color-bg); }\n"
    ".spinner { width: 3rem; height: 3rem; border-radius: 50%; "
    "border: 4px solid var(--color-surface); border-top-color: var(--color-primary); "
    "animation: spin 0.8s linear infinite; }\n"
    "@keyframes spin { to { transform: rotate(360deg); } }\n"
    "</style>\n</head>\n<body>\n"
    "<div class=\"spinner\" role=\"status\" aria-label=\"Carregando\"></div>\n"
    "</body>\n</html>\n"
)

EX_SPINNER_NO_3D = dict(
    id_="gen-frontend-loading_spinner_css_not_3d",
    difficulty="easy",
    user_request="Adicione um ícone de carregamento 3D bem elaborado na tela de espera do app.",
    seed_files={"tokens.css": TOKENS_CSS},
    task_type="multi_tool_call",
    steps=[
        (
            "O pedido menciona '3D', mas um spinner de carregamento é um elemento pequeno, "
            "recorrente, e crítico justo no momento em que a rede pode estar lenta — depender "
            "de Three.js aqui significa esperar uma biblioteca externa carregar e um contexto "
            "WebGL inicializar só pra girar um anel. Uma animação CSS (@keyframes rotate numa "
            "borda com um segmento colorido) produz visualmente o mesmo giro contínuo, pesa "
            "quase nada e não depende de rede nenhuma. Vou recusar o WebGL aqui e usar CSS "
            "puro, reaproveitando os tokens de cor já existentes.",
            "write_file",
            {"path": "loading.html", "content": _SPINNER_CSS_HTML},
        ),
        (
            "Vou renderizar pra confirmar que a animação de fato aplica e não há erro de "
            "console.",
            "checker",
            {
                "language": "html",
                "operation": "run",
                "files": [
                    {"path": "tokens.css", "content": TOKENS_CSS},
                    {"path": "loading.html", "content": _SPINNER_CSS_HTML},
                ],
                "entrypoint": "loading.html",
            },
        ),
    ],
    final_text=(
        "Implementei o indicador de carregamento em loading.html com uma animação CSS "
        "(@keyframes rotate numa borda com segmento colorido) em vez de Three.js/WebGL — para "
        "um elemento pequeno e recorrente que aparece justo quando a rede pode estar lenta, "
        "depender de uma biblioteca 3D externa custaria mais do que entregaria, já que o efeito "
        "visual de giro contínuo é o mesmo. Confirmei renderizando que a animação funciona sem "
        "erro de console."
    ),
)


EXAMPLES = [
    EX_HERO_THREEJS,
    EX_PRICING_GLASS,
    EX_ORBIT_FIX,
    EX_SPINNER_NO_3D,
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
    print(f"\n{written} exemplos avançados de frontend gerados com execução real em {OUT_DIR}")


if __name__ == "__main__":
    main()
