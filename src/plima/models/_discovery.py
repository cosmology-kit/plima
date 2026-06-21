"""Model discovery helpers for PLIMA."""

from __future__ import annotations

import importlib
import pkgutil

import plima.models as models

_SKIP_MODULES = {
    "_discovery",
    "model_registry",
}


def import_model_modules() -> None:
    """Import model modules so decorated models are registered."""
    package_path = models.__path__

    for module_info in pkgutil.iter_modules(package_path):
        if module_info.name in _SKIP_MODULES:
            continue

        if module_info.name.startswith("_"):
            continue

        importlib.import_module(f"plima.models.{module_info.name}")
