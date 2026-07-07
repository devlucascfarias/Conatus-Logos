"""Importar este pacote registra todos os backends de checker disponíveis nesta fase (M2:
Python + Go, seção 10.4). `src.checker.__init__` importa este pacote para garantir o registro
como efeito colateral de importar `src.checker`."""

from . import go_backend, python_backend  # noqa: F401
