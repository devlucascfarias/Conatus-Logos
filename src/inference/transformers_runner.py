"""`ModelRunner` sobre Hugging Face Transformers (seção 5.2) — implementação de produção do
notebook de treino (M5, seção 12).

Importar este módulo NUNCA falha por falta de `transformers`/`peft`/`torch` — a falha só
ocorre ao tentar instanciar `TransformersModelRunner`, com uma mensagem explícita apontando
para `requirements-train.txt`. Isso permite que o resto do harness (parser, checker, tools)
seja testado neste ambiente de desenvolvimento sem exigir GPU nem as dependências pesadas
de treino.

`generate()` é testado de ponta a ponta (não só com mocks) contra um modelo minúsculo de teste
do Hugging Face Hub (`hf-internal-testing/tiny-random-gpt2`) — ver
tests/unit/test_transformers_runner.py — o que dá confiança real de que a lógica de
stop-sequence/continuação (D2, seção 3.2/5.5) funciona antes de apontar para o Granite-4.1-8B
de verdade no Colab."""

from __future__ import annotations

from typing import Any, Optional

from .base import Completion


class TransformersModelRunner:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        quantization_config: Optional[Any] = None,
        repetition_penalty: float = 1.15,
    ):
        """`quantization_config` (ex.: `transformers.BitsAndBytesConfig` em 4-bit NF4) é
        OPCIONAL mas altamente recomendado ao carregar um modelo de vários bilhões de
        parâmetros direto (sem reaproveitar um `model` já em memória via `from_loaded`) — sem
        isso, `from_pretrained` carrega em precisão cheia (ex.: ~32 GB para um modelo de 8B em
        fp32), o que não cabe numa L4 de 24 GB e faz OOM já no carregamento, antes de qualquer
        geração. O chamador decide o `quantization_config` (normalmente o mesmo usado no
        treino, via `TrainConfig`) — este módulo não depende de `TrainConfig` de propósito.

        `repetition_penalty` (D-repetition-loop): 1.0 desativa. Valor moderado (1.15) por
        padrão — ver nota em `generate()`."""
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
        if self._tokenizer.pad_token_id is None:
            # Vários modelos (família GPT-2, inclusive o fixture de teste) não têm pad_token
            # definido — usar eos_token como pad é a convenção padrão para geração em lote/máscara.
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device, quantization_config=quantization_config
        )
        self._model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
        self._model.eval()
        self._repetition_penalty = repetition_penalty

    @classmethod
    def from_loaded(
        cls, model: Any, tokenizer: Any, repetition_penalty: float = 1.15
    ) -> "TransformersModelRunner":
        """Reaproveita um modelo/tokenizer JÁ carregados em memória, sem chamar
        `from_pretrained` de novo — evita duplicar a VRAM ao avaliar logo após o treino, no
        mesmo processo/notebook (`OutOfMemoryError` confirmado no Colab: carregar uma segunda
        cópia do modelo enquanto o `model` do treino ainda ocupa quase toda a VRAM da L4)."""
        instance = cls.__new__(cls)
        instance._tokenizer = tokenizer
        if instance._tokenizer.pad_token_id is None:
            instance._tokenizer.pad_token = instance._tokenizer.eos_token
        instance._model = model
        instance._model.eval()
        instance._repetition_penalty = repetition_penalty
        return instance

    def generate(self, prompt: str, stop: list[str], max_tokens: int = 1024) -> Completion:
        """Gera a continuação de `prompt` até encontrar uma das `stop` sequences ou atingir
        `max_tokens` — nunca usa o chat-template/roles nativo do backend (D2): o prompt já
        chega pronto como texto contínuo (seção 3.6, `Trajectory.render_for_model`), e esta
        função só precisa saber decodificar e parar."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        class _StopOnStrings(StoppingCriteria):
            def __init__(self, tokenizer, prompt_len: int, stop_strings: list[str]):
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.stop_strings = stop_strings
                self.matched: Optional[str] = None

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                if not self.stop_strings:
                    return False
                generated_ids = input_ids[0][self.prompt_len :]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                for candidate in self.stop_strings:
                    if text.endswith(candidate):
                        self.matched = candidate
                        return True
                return False

        stopper = _StopOnStrings(self._tokenizer, prompt_len, stop)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                stopping_criteria=StoppingCriteriaList([stopper]),
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                # D-usecache: gradient_checkpointing (treino) costuma deixar
                # model.config.use_cache=False — sem o cache de atenção, cada token gerado
                # recalcula a sequência inteira do zero, o que é bem mais lento E consome
                # cada vez mais VRAM conforme a geração cresce (confirmado no Colab: VRAM
                # subindo até quase o teto durante uma única chamada de generate()). Forçar
                # use_cache=True aqui sempre, independente do que ficou configurado no
                # model.config depois do treino.
                use_cache=True,
                # D-repetition-loop: decodificação gulosa (do_sample=False) sem nenhuma
                # penalidade de repetição não tem como escapar de um loop assim que o modelo
                # começa a repetir uma frase — repetir vira, token a token, a opção de maior
                # probabilidade. Confirmado num caso real no Colab (adapter pós-D-hypothesis-
                # revision-expansion): um <think> de diagnóstico entrou em loop de repetição
                # da mesma frase até UNTERMINATED_TAG, quatro vezes seguidas, até
                # MAX_STEPS_EXCEEDED — sem fabricação (o harness segurou a honestidade), mas
                # a tarefa nunca foi concluída. 1.15 é moderado de propósito: alto o
                # suficiente pra quebrar loops, baixo o suficiente pra não penalizar
                # repetição legítima de código (indentação, `self.`, `def `, chaves de teste).
                repetition_penalty=self._repetition_penalty,
            )

        generated_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stopper.matched is not None:
            return Completion(text=text, stop_reason="stop_sequence", matched_stop=stopper.matched)
        if generated_ids.shape[0] >= max_tokens:
            return Completion(text=text, stop_reason="max_tokens", matched_stop=None)
        return Completion(text=text, stop_reason="eos", matched_stop=None)
