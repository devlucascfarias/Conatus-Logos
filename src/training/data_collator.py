"""Data collator + pré-tokenização de treino (M5, seção 11.1) — combina tokenização real
(tokenizer do Granite) com `compute_loss_mask`/`apply_loss_mask` (loss_masking.py) para produzir
`labels` corretamente mascarados (D7).

D-sfttrainer-v2: o `trl.SFTTrainer` desta geração (>= 1.5) SEMPRE tokeniza o dataset ele mesmo
internamente (`_prepare_dataset`), procurando uma coluna `"text"` — **a menos que** o dataset
já contenha uma coluna `input_ids`, caso em que ele reconhece o dataset como "já processado" e
pula esse passo por completo. É esse o mecanismo que usamos aqui: `build_pretokenized_dataset`
tokeniza e aplica a máscara de loss por segmento ANTES de entregar ao `SFTTrainer`, e
`TrajectoryDataCollator` vira um collator de **padding puro** (não tokeniza mais nada) — isso
evita que o `SFTTrainer` re-tokenize com seu próprio pipeline (que não sabe nada sobre a
gramática canônica nem sobre mascarar `<tool_result>`).

Import deste módulo nunca falha por falta de `transformers`/`datasets` — só a construção do
collator ou a chamada de `build_pretokenized_dataset`."""

from __future__ import annotations

from typing import Any

from .loss_masking import IGNORE_INDEX, apply_loss_mask, compute_loss_mask


def build_pretokenized_dataset(raw_texts: list[str], tokenizer: Any, max_length: int) -> Any:
    """Tokeniza cada trajetória e aplica a máscara de loss por segmento (D7), retornando um
    `datasets.Dataset` com colunas `input_ids`/`labels` — a presença de `input_ids` é o que
    faz o `SFTTrainer` pular sua própria tokenização (ver D-sfttrainer-v2 acima)."""
    from datasets import Dataset

    records = []
    for raw_text in raw_texts:
        encoded = tokenizer(raw_text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        mask = compute_loss_mask(raw_text, encoded["offset_mapping"])
        labels = apply_loss_mask(encoded["input_ids"], mask)
        records.append({"input_ids": encoded["input_ids"], "labels": labels})

    return Dataset.from_list(records)


class TrajectoryDataCollator:
    """Collator de **padding** para exemplos já tokenizados por `build_pretokenized_dataset`
    (cada exemplo é um dict com `input_ids`/`labels`, ambos já com a máscara de loss aplicada).
    Não tokeniza nada aqui — só preenche (pad) até o maior comprimento do lote e monta a
    `attention_mask` correspondente.

    Uso pretendido dentro do notebook de treino (M5):

        pretokenized = build_pretokenized_dataset(raw_texts, tokenizer, config.sequence_length)
        collator = TrajectoryDataCollator(tokenizer)
        trainer = SFTTrainer(..., train_dataset=pretokenized, data_collator=collator)
    """

    def __init__(self, tokenizer: Any):
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        self._pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(ex["input_ids"]) for ex in examples)

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for ex in examples:
            ids = list(ex["input_ids"])
            labels = list(ex["labels"])
            pad_len = max_len - len(ids)

            input_ids_batch.append(ids + [self._pad_token_id] * pad_len)
            labels_batch.append(labels + [IGNORE_INDEX] * pad_len)
            attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
