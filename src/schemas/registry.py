"""Registro de ferramentas (D10, seção 5.2 do PLAN.md).

Carrega `configs/tools_registry.yaml` + os JSON Schemas correspondentes deste diretório, e
expõe validação de argumentos por ferramenta. `ToolRegistry` é a única fonte de verdade sobre
quais ferramentas existem nesta fase — nenhum outro módulo deve ler tools_registry.yaml
diretamente nem instanciar `jsonschema.Validator` por conta própria.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

SCHEMAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCHEMAS_DIR.parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "tools_registry.yaml"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    input_schema: dict[str, Any]
    side_effects: str  # "none" | "read" | "write" | "exec"
    requires_confirmation: bool
    enabled: bool


class ToolRegistry:
    """Mapa name -> ToolSpec, carregado de configs/tools_registry.yaml."""

    def __init__(self, specs: dict[str, ToolSpec]):
        self._specs = specs
        self._validators: dict[str, jsonschema.Draft202012Validator] = {}

    @classmethod
    def load(cls, registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> ToolRegistry:
        registry_path = Path(registry_path)
        with registry_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        specs: dict[str, ToolSpec] = {}
        for name, entry in raw.get("tools", {}).items():
            schema_path = SCHEMAS_DIR / entry["schema_path"]
            with schema_path.open("r", encoding="utf-8") as sf:
                input_schema = json.load(sf)
            specs[name] = ToolSpec(
                name=name,
                version=entry["version"],
                input_schema=input_schema,
                side_effects=entry["side_effects"],
                requires_confirmation=entry["requires_confirmation"],
                enabled=entry["enabled"],
            )
        return cls(specs)

    def get(self, name: str) -> ToolSpec | None:
        """None tanto para nome inexistente quanto para ferramenta desabilitada nesta fase —
        do ponto de vista do loop do agente (seção 6), ambos os casos são UNSUPPORTED_TOOL."""
        spec = self._specs.get(name)
        if spec is None or not spec.enabled:
            return None
        return spec

    def all_enabled(self) -> list[ToolSpec]:
        return [s for s in self._specs.values() if s.enabled]

    def _validator_for(self, spec: ToolSpec) -> jsonschema.Draft202012Validator:
        cached = self._validators.get(spec.name)
        if cached is None:
            cached = jsonschema.Draft202012Validator(spec.input_schema)
            self._validators[spec.name] = cached
        return cached

    def validate_args(self, name: str, args: dict[str, Any]) -> list[str]:
        """Retorna lista de mensagens de erro (vazia se válido). Nunca lança exceção —
        o chamador (loop do agente) decide o que fazer, convertendo em
        TOOL_ARGUMENT_SCHEMA_ERROR ou UNSUPPORTED_TOOL conforme o caso."""
        spec = self.get(name)
        if spec is None:
            return [f"ferramenta desconhecida ou desabilitada nesta fase: {name}"]
        validator = self._validator_for(spec)
        return [error.message for error in validator.iter_errors(args)]
