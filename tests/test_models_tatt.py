"""Unit tests for ``plima.models.tatt``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.models.tatt import (
    tatt_amplitudes,
    tatt_normalized_coefficients,
    tatt_pt_biases,
    unity_ia_bias,
)
from plima.utils.constants import C1_RHO_CRITICAL, DEFAULT_PIVOT_REDSHIFT


def test_tatt_amplitudes_defaults() -> None:
    """Tests that TATT amplitudes use the expected defaults."""
    z = np.array([0.0, DEFAULT_PIVOT_REDSHIFT, 1.0])

    amplitudes = tatt_amplitudes(z)

    assert set(amplitudes) == {"a1", "a2", "a1delta"}
    np.testing.assert_allclose(amplitudes["a1"], np.ones_like(z))
    np.testing.assert_allclose(amplitudes["a2"], np.zeros_like(z))
    np.testing.assert_allclose(amplitudes["a1delta"], np.zeros_like(z))

    for value in amplitudes.values():
        assert value.dtype == np.float64
        assert value.shape == z.shape


def test_tatt_amplitudes_applies_redshift_evolution() -> None:
    """Tests that TATT amplitudes apply redshift evolution."""
    z = np.array([0.0, 0.5, 1.0])
    z_pivot = 0.5

    amplitudes = tatt_amplitudes(
        z,
        a1=2.0,
        a2=3.0,
        a1delta=4.0,
        eta1=1.0,
        eta2=2.0,
        eta1delta=-1.0,
        z_pivot=z_pivot,
    )

    ratio = (1.0 + z) / (1.0 + z_pivot)

    np.testing.assert_allclose(amplitudes["a1"], 2.0 * ratio)
    np.testing.assert_allclose(amplitudes["a2"], 3.0 * ratio**2)
    np.testing.assert_allclose(amplitudes["a1delta"], 4.0 * ratio**-1.0)


@pytest.mark.parametrize(
    ("bad_z", "match"),
    [
        ([0.0, np.nan], "z must contain only finite values."),
        ([0.0, -1.0], "z must contain only values greater than -1.0."),
        ([0.0, -2.0], "z must contain only values greater than -1.0."),
    ],
)
def test_tatt_amplitudes_rejects_invalid_redshift(
    bad_z,
    match: str,
) -> None:
    """Tests that TATT amplitudes reject invalid redshift values."""
    with pytest.raises(ValueError, match=match):
        tatt_amplitudes(bad_z)


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("a1", np.nan, "a1 must contain only finite values."),
        ("a2", np.nan, "a2 must contain only finite values."),
        ("a1delta", np.nan, "a1delta must contain only finite values."),
        ("eta1", np.nan, "eta1 must contain only finite values."),
        ("eta2", np.nan, "eta2 must contain only finite values."),
        ("eta1delta", np.nan, "eta1delta must contain only finite values."),
        ("z_pivot", np.nan, "z_pivot must contain only finite values."),
        (
            "z_pivot",
            -1.0,
            "z_pivot must contain only values greater than -1.0.",
        ),
    ],
)
def test_tatt_amplitudes_rejects_invalid_parameters(
    keyword: str,
    value: float,
    match: str,
) -> None:
    """Tests that TATT amplitudes reject invalid parameters."""
    with pytest.raises(ValueError, match=match):
        tatt_amplitudes(np.array([0.0, 0.5]), **{keyword: value})


def test_tatt_normalized_coefficients_defaults() -> None:
    """Tests that normalized TATT coefficients use the expected defaults."""
    z = np.array([0.0, 0.5, 1.0])
    growth_factor = np.array([1.0, 0.8, 0.6])
    omega_m = 0.3

    coefficients = tatt_normalized_coefficients(
        z,
        growth_factor=growth_factor,
        omega_m=omega_m,
    )

    expected_c1 = -C1_RHO_CRITICAL * omega_m / growth_factor

    assert set(coefficients) == {"c1", "c2", "cdelta"}
    np.testing.assert_allclose(coefficients["c1"], expected_c1)
    np.testing.assert_allclose(coefficients["c2"], np.zeros_like(z))
    np.testing.assert_allclose(coefficients["cdelta"], np.zeros_like(z))

    for value in coefficients.values():
        assert value.dtype == np.float64
        assert value.shape == z.shape


def test_tatt_normalized_coefficients_applies_amplitudes() -> None:
    """Tests that normalized TATT coefficients apply TATT amplitudes."""
    z = np.array([0.0, 0.5, 1.0])
    growth_factor = np.array([1.0, 0.8, 0.5])
    omega_m = 0.31
    c1_rho_critical = 0.02
    z_pivot = 0.5

    coefficients = tatt_normalized_coefficients(
        z,
        growth_factor=growth_factor,
        omega_m=omega_m,
        a1=2.0,
        a2=3.0,
        a1delta=4.0,
        eta1=1.0,
        eta2=2.0,
        eta1delta=-1.0,
        z_pivot=z_pivot,
        c1_rho_critical=c1_rho_critical,
    )

    ratio = (1.0 + z) / (1.0 + z_pivot)
    a1_z = 2.0 * ratio
    a2_z = 3.0 * ratio**2
    a1delta_z = 4.0 * ratio**-1.0

    expected_c1 = -a1_z * c1_rho_critical * omega_m / growth_factor
    expected_c2 = a2_z * 5.0 * c1_rho_critical * omega_m / growth_factor**2
    expected_cdelta = -a1delta_z * c1_rho_critical * omega_m / growth_factor

    np.testing.assert_allclose(coefficients["c1"], expected_c1)
    np.testing.assert_allclose(coefficients["c2"], expected_c2)
    np.testing.assert_allclose(coefficients["cdelta"], expected_cdelta)


def test_tatt_normalized_coefficients_supports_omega_m_squared_c2() -> None:
    """Tests that normalized TATT coefficients support the squared c2 convention."""
    z = np.array([0.0, 0.5])
    growth_factor = np.array([1.0, 0.5])
    omega_m = 0.24
    omega_m_fid = 0.3
    c1_rho_critical = 0.02

    coefficients = tatt_normalized_coefficients(
        z,
        growth_factor=growth_factor,
        omega_m=omega_m,
        a2=2.0,
        c1_rho_critical=c1_rho_critical,
        omega_m_fid=omega_m_fid,
        use_omega_m_squared_for_c2=True,
    )

    expected_c2 = (
        2.0
        * 5.0
        * c1_rho_critical
        * omega_m**2
        / omega_m_fid
        / growth_factor**2
    )

    np.testing.assert_allclose(coefficients["c2"], expected_c2)


def test_tatt_normalized_coefficients_rejects_growth_factor_shape_mismatch() -> (
    None
):
    """Tests that normalized TATT coefficients reject growth factor shape mismatch."""
    z = np.array([0.0, 0.5])
    growth_factor = np.array([1.0, 0.8, 0.6])

    with pytest.raises(
        ValueError,
        match="growth_factor must have the same shape as z.",
    ):
        tatt_normalized_coefficients(
            z,
            growth_factor=growth_factor,
            omega_m=0.3,
        )


@pytest.mark.parametrize(
    ("growth_factor", "match"),
    [
        ([1.0, np.nan], "growth_factor must contain only finite values."),
        ([1.0, 0.0], "growth_factor must contain only positive values."),
        ([1.0, -0.1], "growth_factor must contain only positive values."),
    ],
)
def test_tatt_normalized_coefficients_rejects_invalid_growth_factor(
    growth_factor,
    match: str,
) -> None:
    """Tests that normalized TATT coefficients reject invalid growth factors."""
    with pytest.raises(ValueError, match=match):
        tatt_normalized_coefficients(
            np.array([0.0, 0.5]),
            growth_factor=growth_factor,
            omega_m=0.3,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"omega_m": 0.0},
            "omega_m must contain only positive values.",
        ),
        (
            {"omega_m": -0.1},
            "omega_m must contain only positive values.",
        ),
        (
            {"omega_m": 0.3, "omega_m_fid": 0.0},
            "omega_m_fid must contain only positive values.",
        ),
        (
            {"omega_m": 0.3, "omega_m_fid": -0.1},
            "omega_m_fid must contain only positive values.",
        ),
        (
            {"omega_m": 0.3, "c1_rho_critical": np.nan},
            "c1_rho_critical must contain only finite values.",
        ),
    ],
)
def test_tatt_normalized_coefficients_rejects_invalid_normalization_parameters(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Tests that normalized TATT coefficients reject invalid normalizations."""
    with pytest.raises(ValueError, match=match):
        tatt_normalized_coefficients(
            np.array([0.0, 0.5]),
            growth_factor=np.array([1.0, 0.8]),
            **kwargs,
        )


def test_tatt_pt_biases_returns_coefficient_tuples() -> None:
    """Tests that TATT PT biases return redshift coefficient tuples."""
    z = np.array([0.0, 0.5, 1.0])
    growth_factor = np.array([1.0, 0.8, 0.6])

    biases = tatt_pt_biases(
        z,
        growth_factor=growth_factor,
        omega_m=0.3,
        a1=2.0,
        a2=3.0,
        a1delta=4.0,
    )

    coefficients = tatt_normalized_coefficients(
        z,
        growth_factor=growth_factor,
        omega_m=0.3,
        a1=2.0,
        a2=3.0,
        a1delta=4.0,
    )

    assert set(biases) == {"c1", "c2", "cdelta"}

    for key, coefficient in coefficients.items():
        bias_z, bias_value = biases[key]
        np.testing.assert_allclose(bias_z, z)
        np.testing.assert_allclose(bias_value, coefficient)


def test_tatt_pt_biases_rejects_invalid_redshift() -> None:
    """Tests that TATT PT biases reject invalid redshift values."""
    with pytest.raises(
        ValueError,
        match="z must contain only values greater than -1.0.",
    ):
        tatt_pt_biases(
            np.array([0.0, -1.0]),
            growth_factor=np.array([1.0, 0.8]),
            omega_m=0.3,
        )


@pytest.mark.parametrize(
    ("bad_z", "match"),
    [
        ([0.0, np.nan], "z must contain only finite values."),
        ([0.0, -1.0], "z must contain only values greater than -1.0."),
    ],
)
def test_unity_ia_bias_rejects_invalid_redshift(bad_z, match: str) -> None:
    """Tests that unity IA bias rejects invalid redshift values."""
    with pytest.raises(ValueError, match=match):
        unity_ia_bias(bad_z)
