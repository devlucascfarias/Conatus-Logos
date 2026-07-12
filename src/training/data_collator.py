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

D-train-prompt-mask: `build_pretokenized_dataset` tokeniza o texto COMPLETO
(system_prompt + user_request + `<raw_text>`), usando exatamente o mesmo template de
`Trajectory.render_for_model()` (harness/trajectory.py) — nunca só `raw_text` isolado. Um bug
anterior tokenizava apenas `raw_text`, então o modelo nunca via system_prompt/user_request
durante o treino; na inferência real (via `run_agent_loop`), o prompt SEMPRE inclui esse
prefixo — confirmado no Colab: o modelo treinado ignorava o pedido do usuário e alucinava
`<final>` sem chamar nenhuma ferramenta, e degenerava em loops de repetição after um
`<tool_result status="error">`, porque o formato de inferência era inédito para ele. O prefixo
inteiro (tudo antes de `raw_text`) fica com loss mascarado (`prefix_len` em
`compute_loss_mask`) — só o texto gerado pelo assistente (menos `<tool_result>`, D7) treina.

D-dataset-history-loss-mask (item 2 de docs/plan_dataset_expansion_wave2_identity_multiturn.md
seção 4, depois de D-dataset-history-schema ter adicionado o campo `trajectory.history` ao
schema): turnos de `history` (se houver) entram no PREFIXO, nunca no texto mascarável. Isso é
uma consequência direta de como `compute_loss_mask` já funcionava: ele só calcula spans
elegíveis dentro de `raw_text` do turno ATUAL (nunca do histórico) e trata qualquer coisa antes
de `prefix_len` como não-elegível — bastou fazer `_render_prefix` também incluir os turnos de
`history` (via `Trajectory.render_for_model()`, que já sabe concatenar `history` no formato
`[USER]/[ASSISTANT]`, ver D-dataset-history-schema) para que TODO o conteúdo de `history`
(inclusive `<think>`/`<tool_call>`/`<final>` de turnos passados, que teriam parecido
"elegíveis" se fossem reparseados) fique automaticamente com loss mascarado, sem precisar
tocar em `compute_loss_mask`/`loss_masking.py` — só o turno atual entra na loss, exatamente
como antes desta mudança para exemplos de turno único.

Import deste módulo nunca falha por falta de `transformers`/`datasets` — só a construção do
collator ou a chamada de `build_pretokenized_dataset`."""

from __future__ import annotations

from typing import Any

from .loss_masking import IGNORE_INDEX, apply_loss_mask, compute_loss_mask


def _render_prefix(
    system_prompt: str,
    user_request: str,
    history: list[dict[str, str]] | None = None,
    environment: dict | None = None,
) -> str:
    """Mesmo template de `Trajectory.render_for_model()` (harness/trajectory.py), até o ponto
    onde `raw_text` do turno ATUAL começaria — reaproveitado aqui via uma trajetória com
    `raw_text=""` para garantir que treino e inferência NUNCA divirjam nesse formato
    (D-train-prompt-mask). `history` (se houver) entra ANTES do turno atual, no prefixo — ver
    D-dataset-history-loss-mask acima para por que isso é suficiente pra mascarar o histórico
    inteiro sem lógica extra de máscara. `environment` (D-prompt-environment-block) entra logo
    após o system prompt, também no prefixo — então o bloco de SO/shell é contexto mascarado,
    nunca tokens treinados."""
    from src.harness.trajectory import HistoryTurn, Trajectory

    history_turns = [HistoryTurn(user_request=h["user_request"], raw_text=h["raw_text"]) for h in (history or [])]
    return Trajectory(
        system_prompt=system_prompt, user_request=user_request, history=history_turns, environment=environment
    ).render_for_model()


def build_pretokenized_dataset(trajectories: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Any:
    """Tokeniza cada trajetória completa (prefixo, incluindo `history` quando presente, +
    `raw_text` do turno atual) e aplica a máscara de loss por segmento (D7) só sobre `raw_text`
    do turno atual, retornando um `datasets.Dataset` com colunas `input_ids`/`labels` — a
    presença de `input_ids` é o que faz o `SFTTrainer` pular sua própria tokenização (ver
    D-sfttrainer-v2 acima).

    `trajectories` é uma lista de dicts com chaves `system_prompt`, `user_request`, `raw_text`
    e opcionalmente `history` (lista de `{user_request, raw_text}` — mesmo shape de
    `ex["trajectory"]` no dataset gerado, ver D-train-prompt-mask/D-dataset-history-schema)."""
    from datasets import Dataset

    records = []
    for traj in trajectories:
        raw_text = traj["raw_text"]
        prefix = _render_prefix(traj["system_prompt"], traj["user_request"], traj.get("history"), traj.get("environment"))
        full_text = prefix + raw_text

        encoded = tokenizer(full_text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        mask = compute_loss_mask(raw_text, encoded["offset_mapping"], prefix_len=len(prefix))
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
