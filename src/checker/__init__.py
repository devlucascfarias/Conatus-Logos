from . import errors
from .core import CheckError, CheckFile, CheckResult, check, register_backend, registered_languages

# Importar backends registra "python" e "go" em core._BACKENDS como efeito colateral (seção 10.4).
from . import backends  # noqa: E402,F401

__all__ = [
    "errors",
    "CheckError",
    "CheckFile",
    "CheckResult",
    "check",
    "register_backend",
    "registered_languages",
]
