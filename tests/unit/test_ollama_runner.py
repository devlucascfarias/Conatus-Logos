"""Testes de `OllamaModelRunner` (seção 5.2) contra um `requests.post` fake — sem bater numa
Ollama de verdade. Confirma o contrato que `run_agent_loop` depende: modo `raw` (sem chat
template), parada de stop-sequence com a tag INCLUÍDA no texto retornado (mesmo contrato de
`TransformersModelRunner`, não o comportamento nativo de truncamento da Ollama), e
`stop_reason` correto nos dois outros casos (`max_tokens`, `eos`)."""

from __future__ import annotations

import json

import pytest
import requests

from src.inference.ollama_runner import OllamaModelRunner


class _FakeStreamResponse:
    def __init__(self, chunks: list[dict]):
        self._lines = [json.dumps(chunk).encode("utf-8") for chunk in chunks]

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _install_fake_post(monkeypatch, chunks, captured: dict):
    def _fake_post(url, json=None, stream=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["stream"] = stream
        captured["timeout"] = timeout
        return _FakeStreamResponse(chunks)

    monkeypatch.setattr("src.inference.ollama_runner.requests.post", _fake_post)


def test_generate_uses_raw_mode_not_chat_template(monkeypatch):
    """D2 — o harness manda texto cru; a Ollama NUNCA deve aplicar o template de chat do
    Modelfile, senão o modelo não reconhece o formato que foi treinado para seguir."""
    captured: dict = {}
    _install_fake_post(monkeypatch, [{"response": "oi", "done": True, "done_reason": "stop"}], captured)

    runner = OllamaModelRunner(model="logos-v2")
    runner.generate("PROMPT CRU EXATO", stop=[], max_tokens=10)

    assert captured["json"]["raw"] is True
    assert captured["json"]["prompt"] == "PROMPT CRU EXATO"
    assert captured["json"]["model"] == "logos-v2"
    assert captured["json"]["options"]["num_predict"] == 10
    assert captured["stream"] is True


def test_generate_defaults_to_temperature_zero(monkeypatch):
    """D-ollama-temperature-zero — sem isso, a Ollama usa amostragem com temperatura > 0 por
    padrão, tornando o mesmo prompt não-determinístico entre rodadas (confirmado num teste
    real: checker correto numa vez, shell com módulo inexistente na outra) e misturando ruído
    de amostragem com o efeito real de merge/quantização."""
    captured: dict = {}
    _install_fake_post(monkeypatch, [{"response": "ok", "done": True, "done_reason": "stop"}], captured)

    runner = OllamaModelRunner(model="logos-v2")
    runner.generate("prompt", stop=[], max_tokens=10)

    assert captured["json"]["options"]["temperature"] == 0.0


def test_generate_allows_overriding_temperature(monkeypatch):
    captured: dict = {}
    _install_fake_post(monkeypatch, [{"response": "ok", "done": True, "done_reason": "stop"}], captured)

    runner = OllamaModelRunner(model="logos-v2", temperature=0.7)
    runner.generate("prompt", stop=[], max_tokens=10)

    assert captured["json"]["options"]["temperature"] == 0.7


def test_generate_stops_with_tag_included_in_text(monkeypatch):
    """Diferente do truncamento nativo da Ollama — o texto retornado precisa TERMINAR com a
    stop-sequence, não excluí-la, para bater com o contrato de TransformersModelRunner."""
    captured: dict = {}
    chunks = [
        {"response": "<think>oi</think>", "done": False},
        {"response": "<tool_call", "done": False},
        {"response": ">", "done": False},
        {"response": "{}", "done": False},
        {"response": "</tool_call>", "done": False},
        {"response": "texto que nao deveria ser lido", "done": True, "done_reason": "stop"},
    ]
    _install_fake_post(monkeypatch, chunks, captured)

    runner = OllamaModelRunner(model="logos-v2")
    result = runner.generate("prompt", stop=["</tool_call>", "</final>"], max_tokens=100)

    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "</tool_call>"
    assert result.text.endswith("</tool_call>")
    assert "texto que nao deveria ser lido" not in result.text


def test_generate_returns_max_tokens_when_done_reason_is_length(monkeypatch):
    captured: dict = {}
    chunks = [
        {"response": "texto sem nenhuma tag de parada", "done": False},
        {"response": "", "done": True, "done_reason": "length"},
    ]
    _install_fake_post(monkeypatch, chunks, captured)

    runner = OllamaModelRunner(model="logos-v2")
    result = runner.generate("prompt", stop=["</final>"], max_tokens=5)

    assert result.stop_reason == "max_tokens"
    assert result.matched_stop is None


def test_generate_returns_eos_when_model_stops_without_matching_our_stop(monkeypatch):
    captured: dict = {}
    chunks = [
        {"response": "resposta curta", "done": True, "done_reason": "stop"},
    ]
    _install_fake_post(monkeypatch, chunks, captured)

    runner = OllamaModelRunner(model="logos-v2")
    result = runner.generate("prompt", stop=["</final>"], max_tokens=100)

    assert result.stop_reason == "eos"
    assert result.matched_stop is None
    assert result.text == "resposta curta"


def test_generate_respects_custom_host(monkeypatch):
    captured: dict = {}
    _install_fake_post(monkeypatch, [{"response": "ok", "done": True, "done_reason": "stop"}], captured)

    runner = OllamaModelRunner(model="logos-v2", host="http://192.168.0.42:11434/")
    runner.generate("prompt", stop=[], max_tokens=10)

    assert captured["url"] == "http://192.168.0.42:11434/api/generate"


def test_generate_raises_on_http_error(monkeypatch):
    class _FailingResponse(_FakeStreamResponse):
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

    def _fake_post(url, json=None, stream=None, timeout=None):
        return _FailingResponse([])

    monkeypatch.setattr("src.inference.ollama_runner.requests.post", _fake_post)

    runner = OllamaModelRunner(model="logos-v2")
    with pytest.raises(requests.HTTPError):
        runner.generate("prompt", stop=[], max_tokens=10)
