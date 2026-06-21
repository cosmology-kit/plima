"""Halo model intrinsic alignment parameter models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from plima.models.model_registry import register_ia_model
from plima.utils.constants import DEFAULT_PIVOT_REDSHIFT
from plima.utils.types import FloatArray
from plima.utils.validators import (
    as_finite_float_array,
    validate_finite,
    validate_greater_than,
    validate_positive,
)

__all__ = [
    "halo_model_ia_parameters",
]


@register_ia_model(
    "halo_model",
    aliases=(
        "hm_ia",
        "halo_model_ia",
    ),
)
def halo_model_ia_parameters(
    z: ArrayLike,
    *,
    a_ia: float = 1.0,
    eta_ia: float = 0.0,
    a1h: float = 0.001,
    eta_1h: float = 0.0,
    b: float = -2.0,
    z_pivot: float = DEFAULT_PIVOT_REDSHIFT,
    k_1h: float | None = None,
    k_2h: float | None = None,
) -> dict[str, FloatArray]:
    """Return redshift dependent halo model IA parameters.

    Args:
        z: Redshift values.
        a_ia: Large scale intrinsic alignment amplitude at the pivot redshift.
        eta_ia: Redshift evolution index for ``a_ia``.
        a1h: Satellite one halo alignment amplitude at the pivot redshift.
        eta_1h: Redshift evolution index for ``a1h``.
        b: Satellite shear radial power law slope.
        z_pivot: Pivot redshift.
        k_1h: One halo transition scale. If None, this parameter is omitted.
        k_2h: Two halo damping scale. If None, this parameter is omitted.

    Returns:
        Dictionary containing halo model IA parameter arrays.

    Raises:
        ValueError: If any numerical input is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
        ValueError: If ``z_pivot`` is less than or equal to ``-1``.
        ValueError: If ``k_1h`` or ``k_2h`` is not positive.
    """
    z = as_finite_float_array(z, name="z")

    validate_finite(a_ia, name="a_ia")
    validate_finite(eta_ia, name="eta_ia")
    validate_finite(a1h, name="a1h")
    validate_finite(eta_1h, name="eta_1h")
    validate_finite(b, name="b")
    validate_finite(z_pivot, name="z_pivot")
    validate_greater_than(z, threshold=-1.0, name="z")
    validate_greater_than(z_pivot, threshold=-1.0, name="z_pivot")

    redshift_ratio = (1.0 + z) / (1.0 + z_pivot)

    parameters = {
        "a_ia": np.asarray(a_ia * redshift_ratio**eta_ia, dtype=np.float64),
        "a1h": np.asarray(a1h * redshift_ratio**eta_1h, dtype=np.float64),
        "b": np.full_like(z, b, dtype=np.float64),
    }

    if k_1h is not None:
        validate_positive(k_1h, name="k_1h")
        parameters["k_1h"] = np.full_like(z, k_1h, dtype=np.float64)

    if k_2h is not None:
        validate_positive(k_2h, name="k_2h")
        parameters["k_2h"] = np.full_like(z, k_2h, dtype=np.float64)

    return parameters
