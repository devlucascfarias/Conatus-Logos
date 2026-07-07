"""Testes de `build_pretokenized_dataset`/`TrajectoryDataCollator` (PLAN.md D7, D-sfttrainer-v2)
contra um `trl.SFTTrainer` real (não mockado) — confirma que o dataset pré-tokenizado faz o
SFTTrainer pular sua própria tokenização (que exigiria uma coluna "text" incompatível com a
gramática canônica) e que o treino roda de ponta a ponta com a máscara de loss aplicada.

Pulado automaticamente se `transformers`/`trl`/`datasets`/`torch` não estiverem instalados."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("torch")

from src.training import TrajectoryDataCollator, build_pretokenized_dataset  # noqa: E402
from src.training.loss_masking import IGNORE_INDEX  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


_TRAJECTORY_WITH_TOOL_RESULT = (
    '<think>vou ler o arquivo</think>'
    '<tool_call name="read_file">{"path": "a.py"}</tool_call>'
    '<tool_result name="read_file" status="ok">{"content": "print(1)"}</tool_result>'
    "<final>o arquivo imprime 1</final>"
)
_DIRECT_TRAJECTORY = "<think>não preciso de ferramenta</think><final>2 + 2 = 4</final>"


def test_build_pretokenized_dataset_has_input_ids_and_labels(tokenizer):
    dataset = build_pretokenized_dataset([_TRAJECTORY_WITH_TOOL_RESULT, _DIRECT_TRAJECTORY], tokenizer, max_length=128)
    assert set(dataset.column_names) == {"input_ids", "labels"}
    assert len(dataset) == 2
    assert len(dataset[0]["input_ids"]) == len(dataset[0]["labels"])


def test_pretokenized_dataset_masks_tool_result_span(tokenizer):
    dataset = build_pretokenized_dataset([_TRAJECTORY_WITH_TOOL_RESULT], tokenizer, max_length=128)
    labels = dataset[0]["labels"]
    # Nem todo token pode estar mascarado (think/tool_call/final continuam com loss ativo) —
    # mas pelo menos alguns tokens devem estar mascarados (o <tool_result>, D7).
    assert any(label == IGNORE_INDEX for label in labels)
    assert any(label != IGNORE_INDEX for label in labels)


def test_collator_pads_batch_to_same_length(tokenizer):
    dataset = build_pretokenized_dataset([_TRAJECTORY_WITH_TOOL_RESULT, _DIRECT_TRAJECTORY], tokenizer, max_length=128)
    collator = TrajectoryDataCollator(tokenizer)
    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2

    # A entrada mais curta deve ter padding (attention_mask com zeros no final).
    lengths = [len(dataset[0]["input_ids"]), len(dataset[1]["input_ids"])]
    shorter_idx = lengths.index(min(lengths))
    assert 0 in batch["attention_mask"][shorter_idx].tolist()


def test_sft_trainer_skips_own_tokenization_and_trains(tokenizer, tmp_path):
    """Teste de ponta a ponta real (não mockado): confirma que o SFTTrainer reconhece o
    dataset como já processado (não busca uma coluna "text") e que trainer.train() completa
    sem erro — a regressão original era KeyError: 'text' vindo do tokenize_fn interno do trl."""
    from transformers import AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)

    train_dataset = build_pretokenized_dataset(
        [_TRAJECTORY_WITH_TOOL_RESULT, _DIRECT_TRAJECTORY], tokenizer, max_length=128
    )
    collator = TrajectoryDataCollator(tokenizer)

    args = TrainingArguments(
        output_dir=str(tmp_path / "smoketest"),
        per_device_train_batch_size=2,
        max_steps=1,
        logging_steps=1,
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, data_collator=collator)
    trainer.train()  # não deve lançar KeyError: 'text'
