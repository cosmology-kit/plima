"""Unit tests for ``plima.utils.converters``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.utils.converters import (
    redshift_to_scale_factor,
    scale_factor_to_redshift,
)


def test_redshift_to_scale_factor_values() -> None:
    """Tests that redshift converts to scale factor."""
    z = np.array([0.0, 1.0, 3.0])

    result = redshift_to_scale_factor(z)

    expected = np.array([1.0, 0.5, 0.25])
    np.testing.assert_allclose(result, expected)


def test_redshift_to_scale_factor_future() -> None:
    """Tests that negative redshift maps to future scale factor."""
    z = np.array([-0.5, -0.2])

    result = redshift_to_scale_factor(z)

    expected = np.array([2.0, 1.25])
    np.testing.assert_allclose(result, expected)


def test_redshift_to_scale_factor_scalar() -> None:
    """Tests that scalar redshift input is accepted."""
    result = redshift_to_scale_factor(1.0)

    np.testing.assert_allclose(result, 0.5)
    assert result.shape == ()


@pytest.mark.parametrize(
    "bad_z",
    [
        [-1.0],
        [-1.1],
        [0.0, np.nan],
        [0.0, np.inf],
        [0.0, -np.inf],
    ],
)
def test_redshift_to_scale_factor_bad_input(bad_z: list[float]) -> None:
    """Tests that invalid redshift values are rejected."""
    with pytest.raises(ValueError, match="z"):
        redshift_to_scale_factor(bad_z)


def test_scale_factor_to_redshift_values() -> None:
    """Tests that scale factor converts to redshift."""
    scale_factor = np.array([1.0, 0.5, 0.25])

    result = scale_factor_to_redshift(scale_factor)

    expected = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_scale_factor_to_redshift_future() -> None:
    """Tests that future scale factors map to negative redshift."""
    scale_factor = np.array([2.0, 1.25])

    result = scale_factor_to_redshift(scale_factor)

    expected = np.array([-0.5, -0.2])
    np.testing.assert_allclose(result, expected)


def test_scale_factor_to_redshift_scalar() -> None:
    """Tests that scalar scale factor input is accepted."""
    result = scale_factor_to_redshift(0.5)

    np.testing.assert_allclose(result, 1.0)
    assert result.shape == ()


@pytest.mark.parametrize(
    "bad_scale_factor",
    [
        [0.0],
        [-0.1],
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, -np.inf],
    ],
)
def test_scale_factor_to_redshift_bad_input(
    bad_scale_factor: list[float],
) -> None:
    """Tests that invalid scale factor values are rejected."""
    with pytest.raises(ValueError, match="scale_factor"):
        scale_factor_to_redshift(bad_scale_factor)


def test_roundtrip_redshift() -> None:
    """Tests that redshift round trip conversion is stable."""
    z = np.array([-0.5, 0.0, 0.3, 1.0, 3.0])

    result = scale_factor_to_redshift(redshift_to_scale_factor(z))

    np.testing.assert_allclose(result, z)


def test_roundtrip_scale_factor() -> None:
    """Tests that scale factor round trip conversion is stable."""
    scale_factor = np.array([0.2, 0.5, 1.0, 1.5, 2.0])

    result = redshift_to_scale_factor(scale_factor_to_redshift(scale_factor))

    np.testing.assert_allclose(result, scale_factor)


def test_list_input() -> None:
    """Tests that list inputs are converted correctly."""
    result = redshift_to_scale_factor([0.0, 1.0])

    expected = np.array([1.0, 0.5])
    np.testing.assert_allclose(result, expected)


def test_tuple_input() -> None:
    """Tests that tuple inputs are converted correctly."""
    result = scale_factor_to_redshift((1.0, 0.5))

    expected = np.array([0.0, 1.0])
    np.testing.assert_allclose(result, expected)
