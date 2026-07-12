"""Testes do backend Node.js server-side/CLI do checker (D-checker-node-server-backend, ver
docs/PLAN.md) — distinto de "javascript"/"typescript" (`node_backend.py`, orientado a React/
Vite/browser).

Pulados (não falham) se `node`/`npm` não estiverem no PATH ou se o projeto-template não tiver
`node_modules` instalado.
"""

import shutil
from pathlib import Path

import pytest

from src.checker import check

_HAS_NODE = shutil.which("node") is not None and shutil.which("npm") is not None
_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "checker_templates" / "react_three_fiber"
_HAS_TEMPLATE = (_TEMPLATE_DIR / "node_modules").is_dir()
_requires_node = pytest.mark.skipif(
    not (_HAS_NODE and _HAS_TEMPLATE),
    reason="node/npm ausente ou node_modules do template não instalado (ver README.md)",
)

_FIZZBUZZ = (
    "function fizzbuzz(n) {\n"
    "  const out = [];\n"
    "  for (let i = 1; i <= n; i++) {\n"
    "    if (i % 15 === 0) out.push('FizzBuzz');\n"
    "    else if (i % 3 === 0) out.push('Fizz');\n"
    "    else if (i % 5 === 0) out.push('Buzz');\n"
    "    else out.push(String(i));\n"
    "  }\n"
    "  return out;\n"
    "}\n"
    "console.log(fizzbuzz(15).join(' '));\n"
)

_EXPRESS_APP = (
    "import express from 'express';\n\n"
    "export function createApp() {\n"
    "  const app = express();\n"
    "  app.use(express.json());\n"
    "  const items = [];\n\n"
    "  app.get('/health', (req, res) => res.json({ status: 'ok' }));\n\n"
    "  app.post('/items', (req, res) => {\n"
    "    const { name } = req.body;\n"
    "    if (!name) return res.status(400).json({ error: 'name é obrigatório' });\n"
    "    const item = { id: items.length + 1, name };\n"
    "    items.push(item);\n"
    "    res.status(201).json(item);\n"
    "  });\n\n"
    "  app.get('/items', (req, res) => res.json(items));\n\n"
    "  return app;\n"
    "}\n"
)

_EXPRESS_TEST = (
    "import { describe, it, expect } from 'vitest';\n"
    "import request from 'supertest';\n"
    "import { createApp } from './app.js';\n\n"
    "describe('API de itens', () => {\n"
    "  it('GET /health responde ok', async () => {\n"
    "    const app = createApp();\n"
    "    const res = await request(app).get('/health');\n"
    "    expect(res.status).toBe(200);\n"
    "  });\n\n"
    "  it('POST /items sem name retorna 400', async () => {\n"
    "    const app = createApp();\n"
    "    const res = await request(app).post('/items').send({});\n"
    "    expect(res.status).toBe(400);\n"
    "  });\n"
    "});\n"
)


@_requires_node
def test_run_executes_real_cli_script():
    result = check(
        language="node",
        operation="run",
        files=[{"path": "fizzbuzz.js", "content": _FIZZBUZZ}],
        entrypoint="fizzbuzz.js",
        timeout_ms=30000,
    )
    assert result.passed, result.stderr
    assert "Fizz" in result.stdout and "Buzz" in result.stdout


@_requires_node
def test_run_captures_real_runtime_error():
    broken = "throw new Error('falha proposital');\n"
    result = check(
        language="node", operation="run", files=[{"path": "broken.js", "content": broken}], entrypoint="broken.js", timeout_ms=30000
    )
    assert not result.passed
    assert result.errors[0].code == "RUNTIME_ERROR"


@_requires_node
def test_compile_and_test_runs_real_express_server_via_supertest():
    result = check(
        language="node",
        operation="compile_and_test",
        files=[{"path": "app.js", "content": _EXPRESS_APP}, {"path": "app.test.js", "content": _EXPRESS_TEST}],
        timeout_ms=30000,
    )
    assert result.passed, result.to_json()


@_requires_node
def test_compile_and_test_fails_when_endpoint_broken():
    broken_app = _EXPRESS_APP.replace("res.status(400)", "res.status(200)")
    result = check(
        language="node",
        operation="compile_and_test",
        files=[{"path": "app.js", "content": broken_app}, {"path": "app.test.js", "content": _EXPRESS_TEST}],
        timeout_ms=30000,
    )
    assert not result.passed
    assert result.errors[0].code == "TEST_FAILURE"


@_requires_node
def test_lint_catches_real_undefined_variable():
    result = check(
        language="node",
        operation="lint",
        files=[{"path": "greet.js", "content": "function greet(name) {\n  console.log(greeting + name);\n}\ngreet('mundo');\n"}],
        timeout_ms=30000,
    )
    assert not result.passed
    assert any("no-undef" in e.message for e in result.errors)


@_requires_node
def test_lint_passes_clean_code_without_false_positive_on_console():
    result = check(
        language="node",
        operation="lint",
        files=[{"path": "greet.js", "content": "function greet(name) {\n  console.log('Ola, ' + name);\n}\ngreet('mundo');\n"}],
        timeout_ms=30000,
    )
    assert result.passed, result.to_json()


@_requires_node
def test_lint_never_leaks_absolute_path_or_temp_dir():
    result = check(
        language="node",
        operation="lint",
        files=[{"path": "greet.js", "content": "console.log(undefinedThing);\n"}],
        timeout_ms=30000,
    )
    combined = result.stdout + result.stderr + str(result.errors)
    assert "praxis_checker_" not in combined
    assert "AppData" not in combined


def test_missing_node_toolchain_reports_structured_error(monkeypatch):
    import src.checker.backends.node_server_backend as node_server_backend

    monkeypatch.setattr(node_server_backend, "_node_available", lambda: False)
    result = check(language="node", operation="run", files=[{"path": "x.js", "content": "1;"}], entrypoint="x.js")
    assert not result.passed
    assert result.errors[0].code == "MISSING_DEPENDENCY"


def test_node_registered_as_distinct_language_from_javascript():
    from src.checker.core import registered_languages

    langs = registered_languages()
    assert "node" in langs
    assert "javascript" in langs
    assert "typescript" in langs
