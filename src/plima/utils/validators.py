"""Validation helpers for numerical inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_float_array",
    "as_finite_float_array",
    "validate_finite",
    "validate_positive",
    "validate_non_negative",
    "validate_greater_than",
    "validate_fraction",
]


def as_float_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return values as a NumPy float array.

    Args:
        values: Input values to convert.
        name: Name of the input used in error messages.

    Returns:
        Input values converted to a float array.

    Raises:
        ValueError: If the input cannot be converted to floats.
    """
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to a float array."
        raise ValueError(msg) from error


def as_finite_float_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return values as a finite NumPy float array.

    Args:
        values: Input values to convert and check.
        name: Name of the input used in error messages.

    Returns:
        Input values converted to a finite float array.

    Raises:
        ValueError: If any value is not finite.
    """
    array = as_float_array(values, name=name)
    validate_finite(array, name=name)

    return array


def validate_finite(
    values: ArrayLike,
    *,
    name: str,
) -> None:
    """Check that all values are finite.

    Args:
        values: Input values to check.
        name: Name of the input used in error messages.

    Raises:
        ValueError: If any value is NaN or infinite.
    """
    array = as_float_array(values, name=name)

    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)


def validate_positive(
    values: ArrayLike,
    *,
    name: str,
) -> None:
    """Check that all values are positive.

    Args:
        values: Input values to check.
        name: Name of the input used in error messages.

    Raises:
        ValueError: If any value is less than or equal to zero.
    """
    array = as_finite_float_array(values, name=name)

    if np.any(array <= 0.0):
        msg = f"{name} must contain only positive values."
        raise ValueError(msg)


def validate_non_negative(
    values: ArrayLike,
    *,
    name: str,
) -> None:
    """Check that all values are non negative.

    Args:
        values: Input values to check.
        name: Name of the input used in error messages.

    Raises:
        ValueError: If any value is negative.
    """
    array = as_finite_float_array(values, name=name)

    if np.any(array < 0.0):
        msg = f"{name} must contain only non negative values."
        raise ValueError(msg)


def validate_greater_than(
    values: ArrayLike,
    *,
    threshold: float,
    name: str,
) -> None:
    """Check that all values are greater than a threshold.

    Args:
        values: Input values to check.
        threshold: Lower allowed bound.
        name: Name of the input used in error messages.

    Raises:
        ValueError: If any value is less than or equal to ``threshold``.
    """
    array = as_finite_float_array(values, name=name)

    if np.any(array <= threshold):
        msg = f"{name} must contain only values greater than {threshold}."
        raise ValueError(msg)


def validate_fraction(
    values: ArrayLike,
    *,
    name: str,
) -> None:
    """Check that all values are fractions between zero and one.

    Args:
        values: Input values to check.
        name: Name of the input used in error messages.

    Raises:
        ValueError: If any value is outside the closed interval from zero to one.
    """
    array = as_finite_float_array(values, name=name)

    if np.any((array < 0.0) | (array > 1.0)):
        msg = f"{name} must contain only values between 0 and 1."
        raise ValueError(msg)
