"""Coordinate conversion helpers.

This module provides small array safe conversion functions for common
cosmology coordinates used by PLIMA.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plima.utils.validators import (
    as_finite_float_array,
    validate_greater_than,
    validate_positive,
)

__all__ = [
    "redshift_to_scale_factor",
    "scale_factor_to_redshift",
]


def redshift_to_scale_factor(
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Return scale factor from redshift.

    Args:
        z: Redshift values. Values may be negative for future scale factors,
            but must be greater than ``-1``.

    Returns:
        Scale factor values corresponding to ``z``.

    Raises:
        ValueError: If any redshift value is not finite.
        ValueError: If any redshift value is less than or equal to ``-1``.
    """
    z = as_finite_float_array(z, name="z")
    validate_greater_than(z, threshold=-1.0, name="z")

    return 1.0 / (1.0 + z)


def scale_factor_to_redshift(
    scale_factor: ArrayLike,
) -> NDArray[np.float64]:
    """Return redshift from scale factor.

    Args:
        scale_factor: Scale factor values. Values may be greater than one,
            which corresponds to negative redshift.

    Returns:
        Redshift values corresponding to ``scale_factor``.

    Raises:
        ValueError: If any scale factor value is not finite.
        ValueError: If any scale factor value is not positive.
    """
    scale_factor = as_finite_float_array(scale_factor, name="scale_factor")
    validate_positive(scale_factor, name="scale_factor")

    return 1.0 / scale_factor - 1.0
