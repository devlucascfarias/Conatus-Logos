"""`ModelRunner` sobre a API HTTP local do Ollama (seção 5.2) — usado para validar
comportamento e velocidade do adapter já mergeado/quantizado (GGUF) servido localmente, fora
do processo Python de treino/avaliação.

Importante (D2, seção 3.6): o harness NUNCA usa o chat template nativo do backend — o prompt
já chega pronto como texto contínuo (`Trajectory.render_for_model()`). Por isso este runner usa
`"raw": true` na API do Ollama, que ignora completamente o template de chat do `Modelfile` e
trata `prompt` como texto cru de continuação — exatamente o que `run_agent_loop` espera. Rodar
com `ollama run` normal (sem `raw`) testaria o template de chat padrão, não o formato que o
adapter foi treinado para reconhecer.

A parada por stop-sequence é feita no lado do CLIENTE (streaming, checando o sufixo do texto
acumulado a cada pedaço recebido), não pelo parâmetro `options.stop` da Ollama — a API da
Ollama trunca a stop-sequence PARA FORA do texto retornado quando usada nativamente, mas o
harness precisa da tag de fechamento (`</tool_call>`/`</final>`) incluída no texto anexado à
trajetória, mesmo contrato de `TransformersModelRunner.generate()` (verificado em
`tests/unit/test_transformers_runner.py::test_generate_stops_exactly_at_stop_sequence`)."""

from __future__ import annotations

import json
from typing import Optional

import requests

from .base import Completion


class OllamaModelRunner:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout_s: float = 300.0):
        self._model = model
        self._host = host.rstrip("/")
        self._timeout_s = timeout_s

    def generate(self, prompt: str, stop: list[str], max_tokens: int = 1024) -> Completion:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }

        text = ""
        matched_stop: Optional[str] = None
        done_reason: Optional[str] = None

        with requests.post(
            f"{self._host}/api/generate", json=payload, stream=True, timeout=self._timeout_s
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                text += chunk.get("response", "")

                for candidate in stop:
                    if candidate and text.endswith(candidate):
                        matched_stop = candidate
                        break
                if matched_stop is not None:
                    break

                if chunk.get("done"):
                    done_reason = chunk.get("done_reason")
                    break

        if matched_stop is not None:
            return Completion(text=text, stop_reason="stop_sequence", matched_stop=matched_stop)
        if done_reason == "length":
            return Completion(text=text, stop_reason="max_tokens", matched_stop=None)
        return Completion(text=text, stop_reason="eos", matched_stop=None)
