#!/usr/bin/env python
"""Mede o pico REAL de VRAM (torch.cuda.max_memory_allocated()) de um passo de treino
completo — carregamento do modelo em 4-bit + LoRA + forward + backward — antes de comprometer
horas de treino numa GPU nova (ex.: RTX A2000 12GB) sem saber se cabe.

Mesma lógica das células 19/21/23 do notebook (notebooks/train_granite_l4.ipynb), mas: (a) roda
como script standalone, sem as partes específicas do Colab (montagem de Drive); (b) mede o pico
depois de um passo de treino de VERDADE (forward + backward + optimizer.step()), não só depois
de carregar o modelo — ativações durante o backward são o componente mais variável da estimativa
da seção 11.3 do PLAN.md (~4-8 GB, "depende de profiling real"), e é exatamente isso que este
script mede de fato, em vez de estimar.

Uso:
    python scripts/vram_smoketest.py --config configs/train_a2000_smoketest.yaml
    python scripts/vram_smoketest.py --config configs/train_a2000_smoketest.yaml --num-examples 5
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _detect_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_a2000_smoketest.yaml"))
    parser.add_argument("--num-examples", type=int, default=3, help="quantos exemplos reais de data/train usar no lote de teste")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.training import TrainConfig, TrajectoryDataCollator, build_pretokenized_dataset

    config = TrainConfig.load(args.config)

    if not torch.cuda.is_available():
        raise SystemExit("Nenhuma GPU CUDA detectada — este script precisa rodar na máquina com a GPU real.")

    gpu_name = _detect_gpu_name()
    print(f"GPU detectada: {gpu_name!r}")
    if config.require_gpu_name_contains not in gpu_name:
        print(
            f"AVISO: o config espera uma GPU com {config.require_gpu_name_contains!r} no nome, "
            f"encontrado {gpu_name!r}. Continuando mesmo assim (isto é um smoke test, não o "
            "treino completo com require_gpu_name_contains obrigatório)."
        )

    torch.cuda.reset_peak_memory_stats()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    print(f"Carregando {config.base_model} em 4-bit... (primeira vez baixa o modelo, pode demorar)")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, quantization_config=bnb_config, device_map="auto"
    )

    after_load_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Pico de VRAM após carregar o modelo base (4-bit): {after_load_gb:.2f} GB")

    model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    after_lora_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Pico de VRAM após aplicar LoRA: {after_lora_gb:.2f} GB")

    # Carrega alguns exemplos REAIS de data/train (não dado sintético) pra montar um lote de
    # teste com o mesmo formato/tamanho de trajetória que o treino de verdade usaria.
    example_paths = sorted(glob.glob(str(REPO_ROOT / config.train_path / "*.json")))[: args.num_examples]
    if not example_paths:
        raise SystemExit(f"Nenhum exemplo encontrado em {config.train_path}")
    trajectories = [json.loads(Path(p).read_text(encoding="utf-8"))["trajectory"] for p in example_paths]
    print(f"Usando {len(trajectories)} exemplos reais de {config.train_path} pro passo de teste.")

    dataset = build_pretokenized_dataset(trajectories, tokenizer, max_length=config.sequence_length)
    collator = TrajectoryDataCollator(tokenizer)
    batch = collator([dataset[i] for i in range(len(dataset))])
    batch = {k: v.to(model.device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    print()
    print("=" * 60)
    print(f"PICO REAL DE VRAM (modelo + LoRA + forward + backward + optimizer.step): {peak_gb:.2f} GB")
    print(f"VRAM reservada pelo allocator do PyTorch (pode ser maior que o alocado): {reserved_gb:.2f} GB")
    print(f"sequence_length usado neste teste: {config.sequence_length}")
    print("Baseline experimental do PLAN.md (seção 11.3, numa L4 24GB, seq_len=4096): ~12-17 GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
