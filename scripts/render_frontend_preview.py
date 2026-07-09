"""CLI offline para o passo de julgamento visual do pivô de frontend (docs/
plan_frontend_specialization_wave3.md seção 3, camada 5): renderiza um arquivo HTML de verdade
num Chromium headless, salva um screenshot real e imprime erros de console — depois disso um
humano (ou Claude, olhando o PNG) julga se ficou bonito. Não existe métrica automática de
"beleza" neste projeto e este script não finge ter uma (D-search-code-reenable e D11 seguem o
mesmo princípio: nunca fabricar resultado de verificação).

Uso:
    python scripts/render_frontend_preview.py caminho/para/index.html
    python scripts/render_frontend_preview.py caminho/para/index.html --out preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.checker.backends.frontend_backend import render_and_capture  # noqa: E402
from src.checker.core import CheckFile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_path", type=Path, help="arquivo .html de entrada (relativo ou absoluto)")
    parser.add_argument("--out", type=Path, default=None, help="onde salvar o screenshot (default: ao lado do .html, mesmo nome + .png)")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()

    html_path = args.html_path.resolve()
    if not html_path.exists():
        print(f"arquivo não encontrado: {html_path}", file=sys.stderr)
        return 1

    out_path = args.out.resolve() if args.out else html_path.with_suffix(".png")

    project_dir = html_path.parent
    files = [
        CheckFile(path=str(p.relative_to(project_dir)).replace("\\", "/"), content=p.read_text(encoding="utf-8"))
        for p in project_dir.rglob("*")
        if p.is_file() and p.suffix in (".html", ".css", ".js")
    ]
    entrypoint = str(html_path.relative_to(project_dir)).replace("\\", "/")

    try:
        result = render_and_capture(files, entrypoint, args.timeout_ms, screenshot_path=out_path)
    except RuntimeError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    print(f"screenshot salvo em: {result.screenshot_path}")
    print(f"passou (sem erro de console/JS): {result.passed}")
    if result.console_errors:
        print("erros de console:")
        for msg in result.console_errors:
            print(f"  - {msg}")
    if result.page_errors:
        print("exceções JS não tratadas:")
        for msg in result.page_errors:
            print(f"  - {msg}")
    if result.console_warnings:
        print(f"({len(result.console_warnings)} warning(s) de console, não bloqueiam o passed)")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
