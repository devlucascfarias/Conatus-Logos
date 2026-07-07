"""Public Response Renderer (PLAN.md seção 5.2, D9, 3.6) — extrai só `<final>` em modo prod.

Alternar entre modo dev/prod é puramente uma política de exibição do harness; nenhuma
re-treinagem do modelo é necessária para isso (seção 3.6)."""

from __future__ import annotations

from src.parsers.segments import FinalSegment

from .trajectory import Trajectory

VALID_MODES = ("dev", "prod")


def render(trajectory: Trajectory, mode: str = "dev") -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"modo de renderização desconhecido: {mode!r} (esperado um de {VALID_MODES})")

    if mode == "dev":
        return trajectory.raw_text

    final_segments = [seg for seg in trajectory.segments() if isinstance(seg, FinalSegment)]
    if not final_segments:
        return ""
    return final_segments[-1].text
