"""Máscara de loss por segmento (D7, PLAN.md seção 11.1) — `labels=-100` em tudo que não foi
gerado pelo assistente. Aplicado sobre a gramática canônica (seção 3): loss ativo em
`<think>`/`<tool_call>`/`<final>`; loss mascarado em `<tool_result>` (injetado pelo harness,
nunca deveria ser "previsto" pelo modelo) e em qualquer texto fora de um segmento reconhecido.

Este módulo é deliberadamente independente de tokenizer/framework: opera sobre spans de
caracteres (`token_offsets`), então funciona com qualquer tokenizer que exponha um
`offset_mapping` (todos os tokenizers rápidos do Hugging Face expõem isso) — o notebook de
treino (M5) só precisa chamar `compute_loss_mask` com o `offset_mapping` real do tokenizer do
Granite."""

from __future__ import annotations

from src.parsers import parse_segments

_LOSS_ELIGIBLE_KINDS = frozenset({"think", "tool_call", "final"})

IGNORE_INDEX = -100  # convenção padrão do PyTorch/Hugging Face para "ignorar no loss"


def compute_loss_mask(raw_text: str, token_offsets: list[tuple[int, int]]) -> list[bool]:
    """Retorna, por token, True se o loss deve ser computado (token pertence a um segmento
    gerado pelo assistente) ou False (token pertence a `<tool_result>` ou está fora de
    qualquer segmento reconhecido — nunca deveria ocorrer num exemplo já validado pelo
    pipeline de dataset, mas por segurança tratamos como não-elegível, não como erro)."""
    segments = parse_segments(raw_text)
    eligible_spans = [(seg.start, seg.end) for seg in segments if seg.kind in _LOSS_ELIGIBLE_KINDS]

    mask = []
    for token_start, _token_end in token_offsets:
        eligible = any(span_start <= token_start < span_end for span_start, span_end in eligible_spans)
        mask.append(eligible)
    return mask


def apply_loss_mask(token_ids: list[int], mask: list[bool], ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Converte uma máscara booleana em `labels` no formato esperado por
    `transformers.Trainer`/`trl.SFTTrainer`: token original onde há loss, `ignore_index` onde
    não há."""
    if len(token_ids) != len(mask):
        raise ValueError(f"token_ids ({len(token_ids)}) e mask ({len(mask)}) têm tamanhos diferentes")
    return [tok if keep else ignore_index for tok, keep in zip(token_ids, mask)]
