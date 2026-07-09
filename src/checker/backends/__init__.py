"""Importar este pacote registra todos os backends de checker disponíveis nesta fase (M2:
Python + Go, seção 10.4; `html` adicionado no pivô de frontend — docs/
plan_frontend_specialization_wave3.md seção 3). `src.checker.__init__` importa este pacote para
garantir o registro como efeito colateral de importar `src.checker`."""

from . import frontend_backend, go_backend, python_backend  # noqa: F401
