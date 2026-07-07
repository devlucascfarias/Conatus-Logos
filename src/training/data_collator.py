"""Data collator de treino (M5, seção 11.1) — combina tokenização real (tokenizer do Granite)
com `compute_loss_mask`/`apply_loss_mask` (loss_masking.py) para produzir batches com `labels`
corretamente mascarados. Não executável neste ambiente de desenvolvimento (exige o tokenizer
real do modelo, instalado só via `requirements-train.txt`) — a lógica de mascaramento em si já
está implementada e testada em `loss_masking.py` sem depender de nenhum tokenizer real.

Import deste módulo nunca falha por falta de `transformers` — só a construção do collator."""

from __future__ import annotations

from typing import Any

from .loss_masking import apply_loss_mask, compute_loss_mask


class TrajectoryDataCollator:
    """Uso pretendido dentro do notebook de treino (M5):

        collator = TrajectoryDataCollator(tokenizer, max_length=config.sequence_length)
        batch = collator([{"raw_text": ex["trajectory"]["raw_text"]} for ex in examples])

    Cada exemplo já validado pelo pipeline de dataset (seção 9) contém `raw_text` na gramática
    canônica; este collator tokeniza e aplica a máscara de loss por segmento (D7)."""

    def __init__(self, tokenizer: Any, max_length: int):
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for example in examples:
            raw_text = example["raw_text"]
            encoded = self._tokenizer(
                raw_text,
                truncation=True,
                max_length=self._max_length,
                return_offsets_mapping=True,
            )
            mask = compute_loss_mask(raw_text, encoded["offset_mapping"])
            labels = apply_loss_mask(encoded["input_ids"], mask)

            input_ids_batch.append(encoded["input_ids"])
            labels_batch.append(labels)
            attention_mask_batch.append(encoded["attention_mask"])

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }
