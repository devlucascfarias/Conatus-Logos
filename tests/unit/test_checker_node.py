"""Testes do backend Node/React/Vite/React Three Fiber do checker (D-checker-node-backend, ver
docs/PLAN.md).

Pulados (não falham) se `node`/`npm` não estiverem no PATH, ou se o projeto-template
(`checker_templates/react_three_fiber/`) não tiver `node_modules` instalado ainda (`npm install`
manual, ver README.md) — o backend em si já trata as duas ausências como MISSING_DEPENDENCY
estruturado (ver test_missing_node_toolchain_reports_structured_error).
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

_R3F_SCENE = (
    'import { Canvas } from "@react-three/fiber";\n'
    'import { OrbitControls } from "@react-three/drei";\n\n'
    "function Scene() {\n"
    "  return (\n"
    "    <>\n"
    "      <ambientLight intensity={0.5} />\n"
    "      <directionalLight position={[3, 3, 3]} intensity={1} />\n"
    "      <mesh rotation={[0.4, 0.2, 0]}>\n"
    "        <torusKnotGeometry args={[1, 0.3, 128, 32]} />\n"
    '        <meshStandardMaterial color="#7c5cff" metalness={0.6} roughness={0.3} />\n'
    "      </mesh>\n"
    "      <OrbitControls enableZoom={false} />\n"
    "    </>\n"
    "  );\n"
    "}\n\n"
    "export default function App() {\n"
    "  return (\n"
    "    <Canvas camera={{ position: [0, 0, 5] }}>\n"
    "      <Scene />\n"
    "    </Canvas>\n"
    "  );\n"
    "}\n"
)


@_requires_node
def test_syntax_check_passes_for_valid_r3f_scene():
    result = check(
        language="typescript",
        operation="syntax_check",
        files=[{"path": "src/App.tsx", "content": _R3F_SCENE}],
        timeout_ms=60000,
    )
    assert result.passed, result.stderr
    assert result.metadata["language"] == "typescript"


@_requires_node
def test_syntax_check_fails_for_type_error():
    broken = _R3F_SCENE.replace("intensity={0.5}", 'intensity="nao-e-um-numero"')
    result = check(
        language="typescript",
        operation="syntax_check",
        files=[{"path": "src/App.tsx", "content": broken}],
        timeout_ms=60000,
    )
    assert not result.passed
    assert result.errors[0].code == "COMPILATION_ERROR"


@_requires_node
def test_run_builds_and_renders_r3f_scene_without_console_errors():
    result = check(
        language="typescript",
        operation="run",
        files=[{"path": "src/App.tsx", "content": _R3F_SCENE}],
        timeout_ms=60000,
    )
    assert result.passed, result.to_json()
    assert result.errors == []


@_requires_node
def test_run_catches_real_runtime_error_from_undefined_reference():
    broken = _R3F_SCENE + "\nundefinedFunctionCall();\n"
    result = check(
        language="typescript",
        operation="run",
        files=[{"path": "src/App.tsx", "content": broken}],
        timeout_ms=60000,
    )
    assert not result.passed
    assert any(e.code in ("RUNTIME_ERROR", "COMPILATION_ERROR") for e in result.errors)


@_requires_node
def test_run_never_leaks_absolute_temp_dir_path():
    broken = _R3F_SCENE + "\nundefinedFunctionCall();\n"
    result = check(
        language="typescript",
        operation="run",
        files=[{"path": "src/App.tsx", "content": broken}],
        timeout_ms=60000,
    )
    combined = result.stdout + result.stderr + str(result.errors)
    assert "praxis_checker_node_" not in combined


def test_missing_node_toolchain_reports_structured_error(monkeypatch):
    import src.checker.backends.node_backend as node_backend

    monkeypatch.setattr(node_backend, "_node_available", lambda: False)
    result = check(
        language="typescript",
        operation="syntax_check",
        files=[{"path": "src/App.tsx", "content": _R3F_SCENE}],
    )
    assert not result.passed
    assert result.errors[0].code == "MISSING_DEPENDENCY"


def test_javascript_and_typescript_both_resolve_to_the_node_backend():
    from src.checker.core import registered_languages

    langs = registered_languages()
    assert "javascript" in langs
    assert "typescript" in langs


@_requires_node
def test_lint_catches_real_assignment_in_condition():
    # no-undef é desligado pelo typescript-eslint em .ts/.tsx de propósito (tsc já cobre isso,
    # via a operação "compile" — não é um gap deste backend). Testamos uma regra que o eslint
    # de fato cobre em TS: atribuição dentro de condicional (bug clássico de = em vez de ==).
    result = check(
        language="typescript",
        operation="lint",
        files=[{"path": "src/bad.ts", "content": "function check(a: number) {\n  if (a = 5) { console.log(a); }\n}\n"}],
        timeout_ms=30000,
    )
    assert not result.passed
    assert any("no-cond-assign" in e.message for e in result.errors)


@_requires_node
def test_lint_no_false_positive_on_console_global():
    result = check(
        language="typescript",
        operation="lint",
        files=[{"path": "src/good.ts", "content": "function greet(name: string) {\n  console.log('Ola, ' + name);\n}\ngreet('mundo');\n"}],
        timeout_ms=30000,
    )
    assert result.passed, result.to_json()
