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

from src.inference.transformers_runner import TransformersModelRunner, _find_earliest_stop  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


# --- _find_earliest_stop (D-stop-sequence-merge) ------------------------------------------


def test_find_earliest_stop_matches_exact_suffix():
    match = _find_earliest_stop("hello world</final>", ["</final>"])
    assert match == ("</final>", 11)


def test_find_earliest_stop_matches_mid_string_not_just_suffix():
    # O caso real que quebrava com `text.endswith(...)`: a stop sequence aparece no MEIO do
    # texto (o tokenizer gerou conteúdo depois dela, fundindo o fim da tag com o token
    # seguinte) — precisa achar mesmo assim, não só quando está no fim exato.
    match = _find_earliest_stop("<tool_call>...</tool_call>\nmais texto gerado depois", ["</tool_call>"])
    assert match == ("</tool_call>", 14)


def test_find_earliest_stop_picks_earliest_among_multiple_candidates():
    text = "<think>ok</think><tool_call>...</tool_call>"
    match = _find_earliest_stop(text, ["</tool_call>", "</think>"])
    assert match is not None
    assert match[0] == "</think>"


def test_find_earliest_stop_returns_none_when_nothing_matches():
    assert _find_earliest_stop("texto qualquer", ["</final>", "</tool_call>"]) is None


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
    # Geração é gulosa (do_sample=False) e determinística — "if" aparece nesse modelo de teste
    # bem antes de 50 tokens, o que prova que o StoppingCriteria realmente corta a geração no
    # ponto certo, não só decodifica tudo e checa o sufixo por acaso. Substring escolhida
    # depois de D-repetition-loop adicionar repetition_penalty ao generate() — "ct" (usado
    # antes) parou de aparecer na saída determinística deste modelo de pesos aleatórios com a
    # penalidade ativa; "if" continua confiável.
    result = runner.generate("Hello world", stop=["if"], max_tokens=50)
    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "if"
    assert result.text.endswith("if")


def test_generate_truncates_trailing_content_when_stop_detected_late(runner: TransformersModelRunner, monkeypatch):
    """D-stop-sequence-merge, reproduzindo o achado real: simula a StoppingCriteria NÃO
    disparando a tempo (`model.generate` devolvido diretamente com conteúdo gerado depois da
    stop sequence, como aconteceria se o tokenizer fundisse o fim da tag com o token
    seguinte). Mesmo assim, o `Completion.text` final tem que vir truncado exatamente no fim
    da stop sequence — nunca vazar o que veio depois. Sem essa truncagem pós-geração, foi
    exatamente isso que aconteceu contra o adapter real: dois `<tool_call>` fabricados (com
    ferramentas que nem existem) mais um `<final>` alegando sucesso, tudo numa Completion só,
    nunca interrompida a tempo, nunca executada/rejeitada pelo harness."""
    import torch

    prompt = "Hello world"
    inputs = runner._tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    extra_text = "</tool_call>\nconteudo fabricado que nunca deveria aparecer na Completion"
    extra_ids = runner._tokenizer(extra_text, return_tensors="pt")["input_ids"][0]
    fake_output = torch.cat([inputs["input_ids"][0], extra_ids]).unsqueeze(0)

    monkeypatch.setattr(runner._model, "generate", lambda **kwargs: fake_output)

    result = runner.generate(prompt, stop=["</tool_call>"], max_tokens=50)

    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "</tool_call>"
    assert result.text.endswith("</tool_call>")
    assert "conteudo fabricado" not in result.text


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
    result = runner.generate("Hello world", stop=["if"], max_tokens=50)

    assert result.stop_reason == "stop_sequence"
    assert result.matched_stop == "if"


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
