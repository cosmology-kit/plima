"""Builder for CCL intrinsic alignment backend objects.

This module provides one public entry point for building PLIMA intrinsic
alignment models for CCL.

The PLIMA model registry is the source of truth for model names and aliases.
This builder resolves models through ``plima.models.model_registry`` instead
of keeping a second alias registry.

LA and NLA models use native CCL IA bias tuples.

TATT and halo model IA use custom CCL Pk2D spectra.

The forecasting code should branch on the returned ``mode`` value instead of
checking model names directly.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any, Literal, TypedDict

from numpy.typing import ArrayLike

from plima.backends.ccl.halo_model import make_ccl_halo_model_power_spectra
from plima.backends.ccl.la import make_ccl_la_ia_bias
from plima.backends.ccl.nla import make_ccl_nla_ia_bias
from plima.backends.ccl.tatt import make_ccl_tatt_power_spectra
from plima.models.model_registry import get_model, list_model_aliases
from plima.utils.types import ModelFunction

__all__ = [
    "CCLIAMode",
    "CCLIAModel",
    "build_ccl_ia_model",
]


CCLIAMode = Literal["native_bias", "pk2d"]
CCLIABuilder = Callable[..., "CCLIAModel"]


class CCLIAModel(TypedDict):
    """Dictionary returned by the CCL IA builder."""

    model_name: str
    canonical_model_name: str
    family: str
    backend: str
    mode: CCLIAMode
    ia_bias: tuple[Any, Any] | None
    spectra: dict[str, Any]
    metadata: dict[str, Any]


def _normalize_model_name(model_name: str) -> str:
    """Return a normalized model name."""
    return model_name.strip().lower()


def _canonical_model_name(model_name: str) -> str:
    """Return the canonical registered model name for a model or alias."""
    normalized_name = _normalize_model_name(model_name)
    aliases_by_model = list_model_aliases()

    if normalized_name in aliases_by_model:
        return normalized_name

    for canonical_name, aliases in aliases_by_model.items():
        if normalized_name in aliases:
            return canonical_name

    return normalized_name


def _model_family(model_function: ModelFunction) -> str:
    """Return the PLIMA model family from the model function module."""
    return model_function.__module__.rsplit(".", maxsplit=1)[-1]


def _model_accepts_z(model_function: ModelFunction) -> bool:
    """Return whether a registered model function accepts ``z``."""
    parameters = signature(model_function).parameters

    return "z" in parameters


def _evaluate_amplitude_model(
    model_function: ModelFunction,
    z: ArrayLike,
    **kwargs: Any,
) -> Any:
    """Evaluate a registered amplitude model."""
    if _model_accepts_z(model_function):
        return model_function(z, **kwargs)

    return model_function(**kwargs)


def _native_bias_model(
    *,
    model_name: str,
    canonical_model_name: str,
    family: str,
    ia_bias: tuple[Any, Any],
    amplitude: Any,
) -> CCLIAModel:
    """Return a standard native CCL IA bias model dictionary."""
    return {
        "model_name": model_name,
        "canonical_model_name": canonical_model_name,
        "family": family,
        "backend": "ccl",
        "mode": "native_bias",
        "ia_bias": ia_bias,
        "spectra": {},
        "metadata": {
            "uses_native_ccl_ia": True,
            "uses_pk2d": False,
            "amplitude": amplitude,
        },
    }


def _pk2d_model(
    *,
    model_name: str,
    canonical_model_name: str,
    family: str,
    spectra: dict[str, Any],
    raw_backend_result: dict[str, Any],
) -> CCLIAModel:
    """Return a standard custom CCL Pk2D model dictionary."""
    return {
        "model_name": model_name,
        "canonical_model_name": canonical_model_name,
        "family": family,
        "backend": "ccl",
        "mode": "pk2d",
        "ia_bias": None,
        "spectra": spectra,
        "metadata": {
            "uses_native_ccl_ia": False,
            "uses_pk2d": True,
            "raw_backend_result": raw_backend_result,
        },
    }


def _build_la_model(
    *,
    cosmo: Any,
    model_name: str,
    canonical_model_name: str,
    model_function: ModelFunction,
    z: ArrayLike,
    **kwargs: Any,
) -> CCLIAModel:
    """Build a native CCL IA bias model for LA."""
    del cosmo

    amplitude = _evaluate_amplitude_model(model_function, z, **kwargs)
    ia_bias = make_ccl_la_ia_bias(z, amplitude=amplitude)

    return _native_bias_model(
        model_name=model_name,
        canonical_model_name=canonical_model_name,
        family="la",
        ia_bias=ia_bias,
        amplitude=amplitude,
    )


def _build_nla_model(
    *,
    cosmo: Any,
    model_name: str,
    canonical_model_name: str,
    model_function: ModelFunction,
    z: ArrayLike,
    **kwargs: Any,
) -> CCLIAModel:
    """Build a native CCL IA bias model for NLA."""
    del cosmo

    amplitude = _evaluate_amplitude_model(model_function, z, **kwargs)
    ia_bias = make_ccl_nla_ia_bias(z, amplitude=amplitude)

    return _native_bias_model(
        model_name=model_name,
        canonical_model_name=canonical_model_name,
        family="nla",
        ia_bias=ia_bias,
        amplitude=amplitude,
    )


def _build_tatt_model(
    *,
    cosmo: Any,
    model_name: str,
    canonical_model_name: str,
    model_function: ModelFunction,
    z: ArrayLike,
    **kwargs: Any,
) -> CCLIAModel:
    """Build custom CCL Pk2D spectra for TATT."""
    del model_function

    result = make_ccl_tatt_power_spectra(cosmo, z, **kwargs)

    return _pk2d_model(
        model_name=model_name,
        canonical_model_name=canonical_model_name,
        family="tatt",
        spectra=result["spectra"],
        raw_backend_result=result,
    )


def _build_halo_model(
    *,
    cosmo: Any,
    model_name: str,
    canonical_model_name: str,
    model_function: ModelFunction,
    z: ArrayLike,
    **kwargs: Any,
) -> CCLIAModel:
    """Build custom CCL Pk2D spectra for halo model IA."""
    del model_function

    result = make_ccl_halo_model_power_spectra(cosmo, z, **kwargs)

    return _pk2d_model(
        model_name=model_name,
        canonical_model_name=canonical_model_name,
        family="halo_model",
        spectra=result["spectra"],
        raw_backend_result=result,
    )


_CCL_FAMILY_BUILDERS: dict[str, CCLIABuilder] = {
    "la": _build_la_model,
    "nla": _build_nla_model,
    "tatt": _build_tatt_model,
    "halo_model": _build_halo_model,
}


def build_ccl_ia_model(
    cosmo: Any,
    model_name: str,
    z: ArrayLike,
    **kwargs: Any,
) -> CCLIAModel:
    """Build CCL backend objects for a registered PLIMA IA model.

    Args:
        cosmo: CCL cosmology object. This is used by models that build custom
            Pk2D spectra. It is accepted for all models so the public API stays
            stable.
        model_name: PLIMA model name or alias.
        z: Redshift values used by the IA model.
        **kwargs: Model keyword arguments.

    Returns:
        Standard CCL IA model dictionary.

    Raises:
        ValueError: If the registered model family has no CCL builder.
        KeyError: If ``model_name`` is not registered in PLIMA.
    """
    normalized_name = _normalize_model_name(model_name)
    canonical_name = _canonical_model_name(normalized_name)
    model_function = get_model(normalized_name)
    family = _model_family(model_function)

    try:
        builder = _CCL_FAMILY_BUILDERS[family]
    except KeyError as error:
        msg = f"Model family {family!r} has no CCL builder."
        raise ValueError(msg) from error

    return builder(
        cosmo=cosmo,
        model_name=normalized_name,
        canonical_model_name=canonical_name,
        model_function=model_function,
        z=z,
        **kwargs,
    )
