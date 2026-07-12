"""Importar este pacote registra todos os backends de checker disponíveis nesta fase (M2:
Python + Go, seção 10.4; `html` adicionado depois — ver docs/PLAN.md, D-checker-html-backend;
`javascript`/`typescript` — ver D-checker-node-backend). `src.checker.__init__` importa este
pacote para garantir o registro como efeito colateral de importar `src.checker`."""

from . import (  # noqa: F401
    frontend_backend,
    go_backend,
    node_backend,
    node_server_backend,
    python_backend,
    scss_backend,
)
