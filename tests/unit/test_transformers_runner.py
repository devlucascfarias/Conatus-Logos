"""Testes de `TransformersModelRunner` (PLAN.md seção 5.2/5.5) contra um modelo real minúsculo
do Hugging Face Hub (`hf-internal-testing/tiny-random-gpt2` — publicado pela própria HF
especificamente para testes automatizados, poucos MB, pesos aleatórios). Isso dá confiança de
que a lógica de stop-sequence/continuação funciona com transformers de verdade, não só com
`ScriptedModelRunner` — antes de apontar para o Granite-4.1-8B real no notebook de treino (M5).

Pulado automaticamente se `transformers`/`peft`/`torch` não estiverem instalados (ambientes que
só rodam o harness de referência não precisam dessas dependências pesadas, seção 5.5)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("torch")

from src.inference.transformers_runner import TransformersModelRunner  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def runner() -> TransformersModelRunner:
    return TransformersModelRunner(_TINY_MODEL, device="cpu")


def test_loads_tiny_model_without_error(runner: TransformersModelRunner):
    assert runner is not None


def test_generate_respects_max_tokens_when_no_stop_matches(runner: TransformersModelRunner):
    result = runner.generate("Hello world", stop=["<this_never_appears_in_output>"], max_tokens=10)
    assert result.stop_reason == "max_tokens"
    assert result.matched_stop is None
    assert isinstance(result.text, str) and len(result.text) > 0


def test_generate_stops_exactly_at_stop_sequence(runner: TransformersModelRunner):
    # Geração é gulosa (do_sample=False) e determinística — "ct" aparece nesse modelo de teste
    # bem antes de 50 tokens, o que prova que o StoppingCriteria realmente corta a geração no
    # ponto certo, não só decodifica tudo e checa o sufixo por acaso.
    result = runner.generate("Hello world", stop=["ct"], max_tokens=50)
    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "ct"
    assert result.text.endswith("ct")


def test_generate_returns_completion_with_correct_types(runner: TransformersModelRunner):
    result = runner.generate("Hello", stop=[], max_tokens=5)
    assert hasattr(result, "text")
    assert hasattr(result, "stop_reason")
    assert hasattr(result, "matched_stop")


def test_from_loaded_reuses_existing_model_without_reloading():
    """D-oom-eval: evita OutOfMemoryError confirmado no Colab (segunda cópia do modelo
    carregada em VRAM já quase toda ocupada pelo treino) — `from_loaded` nunca chama
    `from_pretrained`, só empacota um model/tokenizer já existentes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)

    runner = TransformersModelRunner.from_loaded(model, tokenizer)
    result = runner.generate("Hello world", stop=["ct"], max_tokens=50)

    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "ct"


def test_missing_dependencies_raises_clear_runtime_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("simulado: peft não instalado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="requirements-train.txt"):
        TransformersModelRunner(_TINY_MODEL, device="cpu")
