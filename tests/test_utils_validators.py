"""Unit tests for ``plima.utils.validators``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.utils.validators import (
    as_finite_float_array,
    as_float_array,
    validate_finite,
    validate_fraction,
    validate_greater_than,
    validate_non_negative,
    validate_positive,
)


def test_as_float_array_list() -> None:
    """Tests that lists convert to float arrays."""
    result = as_float_array([1, 2, 3], name="values")

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_as_float_array_scalar() -> None:
    """Tests that scalar inputs convert to zero dimensional arrays."""
    result = as_float_array(1.5, name="value")

    np.testing.assert_allclose(result, 1.5)
    assert result.shape == ()


def test_as_float_array_tuple() -> None:
    """Tests that tuples convert to float arrays."""
    result = as_float_array((1, 2), name="values")

    expected = np.array([1.0, 2.0])
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_as_float_array_bad_input() -> None:
    """Tests that non numeric inputs are rejected."""
    with pytest.raises(ValueError, match="values"):
        as_float_array(["bad"], name="values")


def test_as_finite_float_array_valid() -> None:
    """Tests that finite values convert to float arrays."""
    result = as_finite_float_array([1, 2, 3], name="values")

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "values",
    [
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, -np.inf],
    ],
)
def test_as_finite_float_array_bad(values: list[float]) -> None:
    """Tests that non finite values are rejected during conversion."""
    with pytest.raises(ValueError, match="values"):
        as_finite_float_array(values, name="values")


def test_validate_finite_valid() -> None:
    """Tests that finite values pass validation."""
    validate_finite([0.0, 1.0, -1.0], name="values")


@pytest.mark.parametrize(
    "values",
    [
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, -np.inf],
    ],
)
def test_validate_finite_bad(values: list[float]) -> None:
    """Tests that non finite values fail validation."""
    with pytest.raises(ValueError, match="values"):
        validate_finite(values, name="values")


def test_validate_positive_valid() -> None:
    """Tests that positive values pass validation."""
    validate_positive([0.1, 1.0, 2.0], name="values")


@pytest.mark.parametrize(
    "values",
    [
        [0.0],
        [-0.1],
        [1.0, np.nan],
        [1.0, np.inf],
    ],
)
def test_validate_positive_bad(values: list[float]) -> None:
    """Tests that non positive or non finite values fail validation."""
    with pytest.raises(ValueError, match="values"):
        validate_positive(values, name="values")


def test_validate_non_negative_valid() -> None:
    """Tests that non negative values pass validation."""
    validate_non_negative([0.0, 1.0, 2.0], name="values")


@pytest.mark.parametrize(
    "values",
    [
        [-0.1],
        [1.0, np.nan],
        [1.0, np.inf],
    ],
)
def test_validate_non_negative_bad(values: list[float]) -> None:
    """Tests that negative or non finite values fail validation."""
    with pytest.raises(ValueError, match="values"):
        validate_non_negative(values, name="values")


def test_validate_greater_than_valid() -> None:
    """Tests that values greater than the threshold pass validation."""
    validate_greater_than([0.1, 1.0, 2.0], threshold=0.0, name="values")


@pytest.mark.parametrize(
    "values",
    [
        [0.0],
        [-0.1],
        [1.0, np.nan],
        [1.0, np.inf],
    ],
)
def test_validate_greater_than_bad(values: list[float]) -> None:
    """Tests that values below the threshold fail validation."""
    with pytest.raises(ValueError, match="values"):
        validate_greater_than(values, threshold=0.0, name="values")


def test_validate_greater_than_message() -> None:
    """Tests that threshold appears in greater than errors."""
    with pytest.raises(ValueError, match="greater than -1.0"):
        validate_greater_than([-1.0], threshold=-1.0, name="z")


def test_validate_fraction_valid() -> None:
    """Tests that fractions in the closed unit interval pass validation."""
    validate_fraction([0.0, 0.5, 1.0], name="fraction")


@pytest.mark.parametrize(
    "values",
    [
        [-0.1],
        [1.1],
        [0.5, np.nan],
        [0.5, np.inf],
    ],
)
def test_validate_fraction_bad(values: list[float]) -> None:
    """Tests that invalid fractions fail validation."""
    with pytest.raises(ValueError, match="fraction"):
        validate_fraction(values, name="fraction")


def test_validate_fraction_bounds() -> None:
    """Tests that fraction validation includes both boundaries."""
    validate_fraction(0.0, name="fraction")
    validate_fraction(1.0, name="fraction")


def test_validators_accept_arrays() -> None:
    """Tests that validators accept NumPy arrays."""
    values = np.array([1.0, 2.0, 3.0])

    validate_finite(values, name="values")
    validate_positive(values, name="values")
    validate_non_negative(values, name="values")
    validate_greater_than(values, threshold=0.0, name="values")
    validate_fraction(np.array([0.0, 0.5, 1.0]), name="fraction")


def test_validator_names_in_errors() -> None:
    """Tests that custom names appear in validator errors."""
    with pytest.raises(ValueError, match="custom_name"):
        validate_positive([0.0], name="custom_name")
