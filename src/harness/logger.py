"""Logger estruturado do harness (PLAN.md seção 5.2, 7.8) — grava trajetória completa com
redaction de segredos ANTES de persistir; nunca grava texto cru sem passar pela política de
redaction (`SandboxPolicy.redact`). Roda fora do caminho crítico da resposta ao usuário."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.security.policy import DEFAULT_POLICY_PATH, SandboxPolicy

from .trajectory import Trajectory


class TrajectoryLogger:
    def __init__(self, output_dir: Path | str, policy: Optional[SandboxPolicy] = None):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._policy = policy or SandboxPolicy.load(DEFAULT_POLICY_PATH)

    def persist(self, trajectory: Trajectory, session_id: str, extra: Optional[dict[str, Any]] = None) -> Path:
        record = {
            "session_id": session_id,
            "user_request": self._policy.redact(trajectory.user_request),
            "raw_text": self._policy.redact(trajectory.raw_text),
            "forced_final_reason": trajectory.forced_final_reason,
            **(extra or {}),
        }
        path = self._output_dir / f"trajectory_{session_id}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
