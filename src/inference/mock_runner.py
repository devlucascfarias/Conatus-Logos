"""`ModelRunner` roteirizado (M3, seção 16) — respostas fixas, usado nos testes de integração
do loop do agente antes de existir um adapter treinado (esse chega em M5/M6)."""

from __future__ import annotations

from .base import Completion


class ScriptedModelRunner:
    """Devolve, em ordem, uma lista de textos pré-escritos a cada chamada de `generate`."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._calls = 0

    def generate(self, prompt: str, stop: list[str], max_tokens: int = 1024) -> Completion:
        if self._calls >= len(self._responses):
            raise AssertionError(
                f"ScriptedModelRunner esgotou as {len(self._responses)} respostas roteirizadas "
                f"(chamada número {self._calls + 1})"
            )
        text = self._responses[self._calls]
        self._calls += 1
        matched = next((s for s in stop if text.endswith(s)), None)
        return Completion(text=text, stop_reason="stop_sequence" if matched else "eos", matched_stop=matched)

    @property
    def calls_made(self) -> int:
        return self._calls
