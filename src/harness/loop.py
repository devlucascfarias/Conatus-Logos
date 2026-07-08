"""Loop do agente (PLAN.md seção 6) — orquestra geração, parsing, validação por schema,
execução de ferramenta e verificação final. Referência de desenvolvimento em Python; a versão
de produção será portada para Go (M8, seção 5.5) consumindo os mesmos contratos.

Deliberadamente diferente do pseudocódigo da seção 6 num ponto: logging/persistência de
trajetória fica FORA desta função (seção 5.2 — Logger/Evaluator são hooks que rodam fora do
caminho crítico), para manter o loop puro e fácil de testar sem I/O de disco. Quem chama
`run_agent_loop` decide se/como persistir o resultado."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.checker import errors as error_codes
from src.parsers import (
    contains_fabricated_tool_result,
    final_segment_leaks_internal_tags,
    parse_last_segment,
)
from src.parsers.segments import FinalSegment, MalformedSegment, ToolCallSegment
from src.tools import ToolExecutorRegistry

from .renderer import render
from .trajectory import Trajectory


@dataclass(frozen=True)
class AgentLoopConfig:
    max_steps: int = 8
    stop_sequences: tuple = ("</tool_call>", "</final>")
    mode: str = "dev"  # "dev" | "prod" — seção 3.6
    # D-maxtokens: teto de tokens por CHAMADA de generate() (não por turno inteiro). Sem isso,
    # um modelo que não emite a stop-sequence de forma limpa gera até o default do backend
    # (1024 em TransformersModelRunner) em cada um dos até `max_steps` passos — o contexto
    # acumulado pode explodir a ponto de estourar VRAM só no prefill da próxima chamada
    # (OutOfMemoryError de ~6.67 GiB numa única atenção, confirmado no Colab).
    # D-maxtokens-oop: 256 bastava para as trajetórias curtas da geração anterior, mas a
    # expansão de dataset OOP/erro/padrões/módulos (docs/plan_dataset_expansion_oop_shell.md)
    # inclui `checker` chamado com uma classe inteira + arquivo de teste inteiro embutidos no
    # mesmo JSON — confirmado em teste real pós-retreino: a chamada de `checker` para a classe
    # Calculator truncava no meio (`UNTERMINATED_TAG`) e, após falhas repetidas, o modelo
    # chegou a tentar fabricar um `<tool_result>` (bloqueado pelo harness) antes de esgotar
    # `max_steps`. 512 dá margem para esses payloads maiores sem reabrir o risco de OOM que
    # motivou o teto original.
    max_tokens_per_step: int = 512


@dataclass
class AgentLoopResult:
    trajectory: Trajectory
    public_output: str
    steps_taken: int
    forced_final: bool
    consistency_errors: list = field(default_factory=list)


def _canonicalize(args: dict[str, Any]) -> str:
    """Fingerprint estável de argumentos para detectar chamadas repetidas idênticas (seção 6)."""
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def run_agent_loop(
    user_request: str,
    system_prompt: str,
    model_runner,
    tool_registry: ToolExecutorRegistry,
    sandbox,
    config: Optional[AgentLoopConfig] = None,
) -> AgentLoopResult:
    config = config or AgentLoopConfig()
    trajectory = Trajectory(system_prompt=system_prompt, user_request=user_request)
    seen_calls: set = set()
    forced_final = False
    steps_taken = 0

    for step in range(config.max_steps):
        steps_taken = step + 1
        completion = model_runner.generate(
            prompt=trajectory.render_for_model(),
            stop=list(config.stop_sequences),
            max_tokens=config.max_tokens_per_step,
        )

        # Checagem anti-fabricação (seção 10.3) sobre o texto RECÉM-GERADO, antes de anexar
        # qualquer <tool_result> real — um <tool_result> vindo do próprio modelo é sempre
        # fabricação, nunca um resultado legítimo.
        if contains_fabricated_tool_result(completion.text):
            trajectory.append_raw(completion.text)
            trajectory.append_tool_result(
                name="unknown",
                status="error",
                body={
                    "code": error_codes.TOOL_RESULT_FABRICATION,
                    "message": "o modelo tentou fabricar um <tool_result> em vez de aguardar o harness",
                },
            )
            continue

        trajectory.append_raw(completion.text)
        segment = parse_last_segment(trajectory.raw_text)

        if isinstance(segment, FinalSegment):
            break

        if isinstance(segment, MalformedSegment):
            trajectory.append_tool_result(
                name=segment.attempted_name or "unknown",
                status="error",
                body={"code": segment.reason, "message": segment.message},
            )
            continue

        if isinstance(segment, ToolCallSegment):
            _handle_tool_call(segment, trajectory, tool_registry, sandbox, seen_calls)
            continue

        # None ou tipo inesperado (ex.: tool_result vindo do modelo sem ser pego pela
        # checagem de fabricação por algum motivo) — nunca deveria ocorrer, mas não trava o loop.
        trajectory.append_tool_result(
            name="unknown",
            status="error",
            body={"code": error_codes.TOOL_CALL_PARSE_ERROR, "message": "segmento não reconhecido pelo loop"},
        )

    else:
        trajectory.force_final(reason=error_codes.MAX_STEPS_EXCEEDED)
        forced_final = True

    consistency_errors = _verify_consistency(trajectory)
    public_output = render(trajectory, mode=config.mode)

    return AgentLoopResult(
        trajectory=trajectory,
        public_output=public_output,
        steps_taken=steps_taken,
        forced_final=forced_final,
        consistency_errors=consistency_errors,
    )


def _handle_tool_call(
    segment: ToolCallSegment,
    trajectory: Trajectory,
    tool_registry: ToolExecutorRegistry,
    sandbox,
    seen_calls: set,
) -> None:
    fingerprint = (segment.name, _canonicalize(segment.args))
    if fingerprint in seen_calls:
        trajectory.append_tool_result(
            segment.name,
            "error",
            {"code": error_codes.MAX_STEPS_EXCEEDED, "message": "chamada repetida sem progresso"},
        )
        return
    seen_calls.add(fingerprint)

    spec = tool_registry.get_spec(segment.name)
    if spec is None:
        trajectory.append_tool_result(
            segment.name,
            "error",
            {"code": error_codes.UNSUPPORTED_TOOL, "message": f"ferramenta desconhecida/desabilitada: {segment.name}"},
        )
        return

    schema_errors = tool_registry.validate_args(segment.name, segment.args)
    if schema_errors:
        trajectory.append_tool_result(
            segment.name,
            "error",
            {"code": error_codes.TOOL_ARGUMENT_SCHEMA_ERROR, "message": "; ".join(schema_errors)},
        )
        return

    if spec.requires_confirmation and not sandbox.confirmation_allowed(spec, segment.args):
        trajectory.append_tool_result(
            segment.name,
            "error",
            {"code": error_codes.UNSAFE_COMMAND, "message": "operação requer confirmação e não foi autorizada"},
        )
        return

    exec_result = tool_registry.execute(segment.name, segment.args, sandbox)
    trajectory.append_tool_result(segment.name, "ok" if exec_result.passed else "error", exec_result.to_json())


def _verify_consistency(trajectory: Trajectory) -> list:
    """Seção 10.3 — checagem final antes da resposta pública, roda mesmo quando o modelo não
    chamou `checker` espontaneamente. Cobre aqui a parte estrutural determinística
    (vazamento de raciocínio); EXPLANATION_CODE_MISMATCH fica para o Evaluator (seção 13),
    que tem acesso ao estado real dos arquivos do workspace para comparar contra o `<final>`."""
    found = []
    for seg in trajectory.segments():
        if isinstance(seg, FinalSegment) and final_segment_leaks_internal_tags(seg.text):
            found.append(error_codes.INTERNAL_REASONING_LEAK)
    return found
