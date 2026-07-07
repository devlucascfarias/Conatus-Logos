"""`ModelRunner` sobre Hugging Face Transformers (seção 5.2) — implementação de produção do
notebook de treino (M5, seção 12), não deste harness de referência local (M3).

Importar este módulo NUNCA falha por falta de `transformers`/`peft`/`torch` — a falha só
ocorre ao tentar instanciar `TransformersModelRunner`, com uma mensagem explícita apontando
para `requirements-train.txt`. Isso permite que o resto do harness (parser, checker, tools)
seja testado neste ambiente de desenvolvimento sem exigir GPU nem as dependências pesadas
de treino."""

from __future__ import annotations

from typing import Optional

from .base import Completion


class TransformersModelRunner:
    def __init__(self, model_name_or_path: str, adapter_path: Optional[str] = None, device: str = "cuda"):
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "TransformersModelRunner requer 'transformers'/'peft'/'torch' instalados "
                "(ver requirements-train.txt) — não disponíveis neste ambiente de dev. "
                "Esta classe é destinada a rodar dentro do notebook de treino (M5)."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device)
        self._model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model

    def generate(self, prompt: str, stop: list[str], max_tokens: int = 1024) -> Completion:
        raise NotImplementedError(
            "geração real via Transformers contra o adapter treinado é implementada no "
            "notebook de treino (M5) — fora do escopo do harness de referência local (M3), "
            "que usa ScriptedModelRunner (src.inference.mock_runner) para testes."
        )
