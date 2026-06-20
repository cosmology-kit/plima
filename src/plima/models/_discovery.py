"""Discovery helpers for PLIMA models."""

from __future__ import annotations

import importlib
import pkgutil

_SKIP_MODULES = {
    "_discovery",
    "base",
    "model_registry",
}


def import_model_modules() -> None:
    """Import model modules so decorated models are registered."""
    import plima.models as models

    package_path = models.__path__
    package_name = models.__name__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = module_info.name

        if module_name in _SKIP_MODULES:
            continue

        importlib.import_module(f"{package_name}.{module_name}")