"""Testes de src/training (PLAN.md seção 11): config loader real + máscara de loss (D7)
usando um tokenizer mock nível-caractere (determinístico, sem depender de transformers)."""

from src.training import TrainConfig, apply_loss_mask, compute_loss_mask


def _char_level_offsets(text: str) -> list[tuple[int, int]]:
    """Tokenizer mock: 1 token por caractere — suficiente para testar a lógica de máscara
    por span de caracteres sem precisar de um tokenizer real (Granite só chega em M5)."""
    return [(i, i + 1) for i in range(len(text))]


# --- config_loader -------------------------------------------------------------------


def test_train_config_loads_from_default_yaml():
    # D-frontend-pivot-model-swap: base trocada pra Qwen3-4B-Instruct-2507 (ver docs/PLAN.md).
    # target_modules NÃO inclui q_proj/k_proj — QK-norm do Qwen3 é incompatível com LoRA nessas
    # duas projeções (D-qwen3-qknorm-lora).
    config = TrainConfig.load()
    assert config.base_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.load_in_4bit is True
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert "q_proj" not in config.lora_target_modules
    assert "k_proj" not in config.lora_target_modules
    assert "v_proj" in config.lora_target_modules
    assert config.sequence_length == 4096
    assert config.optim == "paged_adamw_8bit"
    assert config.require_gpu_name_contains == "L4"


# --- loss_masking --------------------------------------------------------------------


def test_think_and_final_segments_get_loss():
    text = "<think>a</think><final>b</final>"
    mask = compute_loss_mask(text, _char_level_offsets(text))
    # todo caractere dentro de <think>...</think> ou <final>...</final> (incluindo as tags)
    # pertence a um segmento elegível — só teríamos texto fora de segmento se houvesse lixo
    # entre as tags, o que não é o caso aqui.
    assert all(mask)


def test_tool_result_segment_is_masked_out():
    text = (
        '<tool_call name="read_file">{"path": "a.py"}</tool_call>'
        '<tool_result name="read_file" status="ok">{"content": "x"}</tool_result>'
        "<final>ok</final>"
    )
    segments_text_tool_result_start = text.index('<tool_result')
    segments_text_tool_result_end = text.index("</tool_result>") + len("</tool_result>")

    mask = compute_loss_mask(text, _char_level_offsets(text))

    # dentro do <tool_result>...</tool_result>, loss deve estar sempre desligado
    assert not any(mask[segments_text_tool_result_start:segments_text_tool_result_end])
    # antes (tool_call) e depois (final), loss deve estar ligado
    assert mask[text.index('<tool_call')]
    assert mask[text.index("<final>")]


def test_apply_loss_mask_replaces_masked_tokens_with_ignore_index():
    token_ids = [10, 20, 30, 40]
    mask = [True, False, True, False]
    labels = apply_loss_mask(token_ids, mask)
    assert labels == [10, -100, 30, -100]


def test_apply_loss_mask_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        apply_loss_mask([1, 2, 3], [True, False])


def test_full_trajectory_masking_end_to_end():
    text = (
        "<think>vou ler o arquivo</think>"
        '<tool_call name="read_file">{"path": "a.py"}</tool_call>'
        '<tool_result name="read_file" status="ok">{"content": "print(1)"}</tool_result>'
        "<think>agora respondo</think>"
        "<final>o arquivo imprime 1</final>"
    )
    offsets = _char_level_offsets(text)
    mask = compute_loss_mask(text, offsets)
    token_ids = list(range(len(text)))  # ids arbitrários, só para testar o replace
    labels = apply_loss_mask(token_ids, mask)

    tool_result_start = text.index("<tool_result")
    tool_result_end = text.index("</tool_result>") + len("</tool_result>")
    assert all(label == -100 for label in labels[tool_result_start:tool_result_end])
    assert labels[text.index("<think>")] != -100
    assert labels[text.index("<final>")] != -100
