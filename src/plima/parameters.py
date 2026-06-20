"""Helpers for IA model parameter dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_parameter(
    params: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    """Return one scalar model parameter from a mapping."""
    return float(params.get(name, default))


def require_parameter(
    params: Mapping[str, Any],
    name: str,
) -> float:
    """Return one required scalar model parameter from a mapping."""
    if name not in params:
        msg = f"Missing required parameter {name!r}."
        raise KeyError(msg)

    return float(params[name])
