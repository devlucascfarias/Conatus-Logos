"""Utilitário de execução de subprocess compartilhado pelos backends do checker.

Isolado aqui (não em src/security) porque o checker roda compiladores/interpretadores
confiáveis do próprio ambiente de desenvolvimento (não código arbitrário do usuário final) —
o sandbox de segurança pesado (seção 7) é aplicado pela ferramenta `shell`/`Tool Executor`,
não pelo checker. Ainda assim, timeout e truncamento de saída são sempre aplicados aqui.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 65536  # mesmo teto de configs/sandbox_policy.yaml (seção 7.3)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int


def _truncate(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n...[truncado]"


def _sanitize_base_dir(text: str, cwd: Path) -> str:
    """Remove o caminho absoluto do diretório temporário de execução (`praxis_checker_py_*`/
    `praxis_checker_go_*`, sempre sob o temp dir real da máquina) de stdout/stderr antes de
    devolver ao chamador — sem isso, um traceback real (ex.: `File "C:\\Users\\<usuário>\\...\\
    praxis_checker_py_<hash>\\main.py", line 3`) vaza o nome de usuário e a estrutura de pastas
    da máquina que rodou o checker pra dentro de qualquer `<tool_result>`/`<final>` que reproduza
    essa mensagem. Achado real: 20 exemplos do dataset (`aug-debug-*`) tinham esse vazamento
    porque foram gerados rodando o checker de verdade sem essa sanitização existir ainda —
    corrigido aqui, na função compartilhada por TODOS os backends, não só no dataset já gerado,
    porque o mesmo vazamento aconteceria em qualquer uso real do checker em produção, não só na
    geração de exemplos.

    D-checker-path-leak-go-slash (achado real ao testar, não hipótese): o panic runtime do Go
    imprime o caminho com barras normais (`/`) mesmo no Windows — ex.
    `C:/Users/<usuário>/AppData/Local/Temp/praxis_checker_go_<hash>/main.go:5` — diferente do
    `py_compile`/`pytest`, que usam barra invertida nativa do SO. Sem gerar também a variante
    com `/`, o sanitizador não pegava esse caso: confirmado reproduzindo `go run` com um panic
    real (`assignment to entry in nil map`) antes de adicionar essa variante."""
    if not text:
        return text
    base_forms = {str(cwd), str(cwd.resolve())}
    for candidate in set(base_forms):
        base_forms.add(candidate.replace("\\", "/"))
    for candidate in set(base_forms):
        # também cobre a variante com barras invertidas escapadas, como aparece dentro de
        # mensagens de erro re-serializadas em JSON antes de virar texto de novo.
        base_forms.add(candidate.replace("\\", "\\\\"))
    for form in base_forms:
        text = re.sub(re.escape(form) + r"[\\/]?", "", text)
    return text


def run_command(cmd: list[str], cwd: Path, timeout_ms: int) -> CommandResult:
    import time

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_ms / 1000,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return CommandResult(
            returncode=proc.returncode,
            stdout=_truncate(_sanitize_base_dir(proc.stdout, cwd)),
            stderr=_truncate(_sanitize_base_dir(proc.stderr, cwd)),
            timed_out=False,
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            returncode=-1,
            stdout=_truncate(_sanitize_base_dir(stdout, cwd)),
            stderr=_truncate(_sanitize_base_dir(stderr, cwd)),
            timed_out=True,
            duration_ms=duration_ms,
        )


def materialize_files(base_dir: Path, files: list) -> None:
    """Escreve CheckFile(path, content) em disco sob base_dir, criando subdiretórios.

    `newline=""` desliga a tradução de \\n -> \\r\\n do Windows (universal newlines do modo
    texto padrão) — sem isso, um `content` com só `\\n` (ex.: produzido por um programa Go, que
    não traduz newline sozinho) ganha um `\\r` espúrio ao ser materializado aqui, divergindo do
    conteúdo real que outra linguagem já executou/leu antes (achado real gerando
    D-crosslayer-dataset-fase-f, docs/PLAN.md — um CSV gerado por Go, embutido como fixture
    numa chamada de checker Node, ganhava um `\\r` que não existia no arquivo real do Go)."""
    for f in files:
        dest = base_dir / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.content, encoding="utf-8", newline="")
