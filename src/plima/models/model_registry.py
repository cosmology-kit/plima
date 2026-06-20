"""Registry for PLIMA model functions."""

from __future__ import annotations

from collections.abc import Callable

from plima.models._discovery import import_model_modules
from plima.utils.types import ModelFunction

__all__ = [
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "register_ia_model",
    "discover_models",
    "get_model",
    "list_models",
    "list_model_aliases",
]


MODEL_REGISTRY: dict[str, ModelFunction] = {}
MODEL_ALIASES: dict[str, tuple[str, ...]] = {}
_DISCOVERED = False


def register_ia_model(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[ModelFunction], ModelFunction]:
    """Register a PLIMA model function."""

    def decorator(function: ModelFunction) -> ModelFunction:
        keys = (name, *aliases)

        for key in keys:
            if key in MODEL_REGISTRY:
                msg = f"Model {key!r} is already registered."
                raise ValueError(msg)

        for key in keys:
            MODEL_REGISTRY[key] = function

        MODEL_ALIASES[name] = aliases

        return function

    return decorator


def discover_models() -> None:
    """Import model modules once so decorated models are registered."""
    global _DISCOVERED

    if _DISCOVERED:
        return

    import_model_modules()
    _DISCOVERED = True


def get_model(name: str) -> ModelFunction:
    """Return a registered PLIMA model function."""
    discover_models()

    try:
        return MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(list_models())
        msg = f"Unknown model {name!r}. Available models: {available}."
        raise KeyError(msg) from error


def list_models(*, include_aliases: bool = True) -> tuple[str, ...]:
    """Return registered model names."""
    discover_models()

    if include_aliases:
        return tuple(sorted(MODEL_REGISTRY))

    return tuple(sorted(MODEL_ALIASES))


def list_model_aliases() -> dict[str, tuple[str, ...]]:
    """Return canonical model names and aliases."""
    discover_models()

    return dict(sorted(MODEL_ALIASES.items()))
