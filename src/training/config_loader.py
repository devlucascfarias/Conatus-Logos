"""Carrega `configs/train_l4.yaml` (PLAN.md seções 11/12) num dataclass tipado — nenhum
hiperparâmetro deve ser lido diretamente do YAML fora daqui (seção 12.2: nenhuma célula do
notebook contém valores mágicos que não estejam também no config)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train_l4.yaml"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple

    train_path: str
    validation_path: str
    sequence_length: int
    group_by_length: bool

    batch_size_per_device: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    num_train_epochs: int
    max_steps: int
    optim: str
    gradient_checkpointing: bool
    seed: int

    eval_steps: int
    save_steps: int
    save_total_limit: int

    output_dir: str
    logs_dir: str
    adapter_dir: str
    drive_mount_point: str

    run_on_colab: bool
    require_gpu_name_contains: str
    mixed_precision: str

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            base_model=raw["model"]["base_model"],
            load_in_4bit=raw["model"]["load_in_4bit"],
            bnb_4bit_quant_type=raw["model"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=raw["model"]["bnb_4bit_compute_dtype"],
            bnb_4bit_use_double_quant=raw["model"]["bnb_4bit_use_double_quant"],
            lora_r=raw["lora"]["r"],
            lora_alpha=raw["lora"]["alpha"],
            lora_dropout=raw["lora"]["dropout"],
            lora_target_modules=tuple(raw["lora"]["target_modules"]),
            train_path=raw["data"]["train_path"],
            validation_path=raw["data"]["validation_path"],
            sequence_length=raw["data"]["sequence_length"],
            group_by_length=raw["data"]["group_by_length"],
            batch_size_per_device=raw["training"]["batch_size_per_device"],
            gradient_accumulation_steps=raw["training"]["gradient_accumulation_steps"],
            learning_rate=raw["training"]["learning_rate"],
            lr_scheduler_type=raw["training"]["lr_scheduler_type"],
            warmup_ratio=raw["training"]["warmup_ratio"],
            num_train_epochs=raw["training"]["num_train_epochs"],
            max_steps=raw["training"]["max_steps"],
            optim=raw["training"]["optim"],
            gradient_checkpointing=raw["training"]["gradient_checkpointing"],
            seed=raw["training"]["seed"],
            eval_steps=raw["evaluation"]["eval_steps"],
            save_steps=raw["evaluation"]["save_steps"],
            save_total_limit=raw["evaluation"]["save_total_limit"],
            output_dir=raw["paths"]["output_dir"],
            logs_dir=raw["paths"]["logs_dir"],
            adapter_dir=raw["paths"]["adapter_dir"],
            drive_mount_point=raw["paths"]["drive_mount_point"],
            run_on_colab=raw["runtime"]["run_on_colab"],
            require_gpu_name_contains=raw["runtime"]["require_gpu_name_contains"],
            mixed_precision=raw["runtime"]["mixed_precision"],
        )
