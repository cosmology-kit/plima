"""Unit tests for ``plima.models.la``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plima.models.la import (
    la_amplitude,
    la_mass_amplitude,
    la_response,
    la_z_amplitude,
    lf_la_amplitude,
    p_delta_i_la,
    p_ii_la,
    red_fraction_from_luminosity_functions,
)
from plima.models.model_registry import (
    get_model,
    list_model_aliases,
    list_models,
)
from plima.utils.constants import (
    C1_RHO_CRITICAL,
    DEFAULT_HIGH_Z_PIVOT_REDSHIFT,
    DEFAULT_LOW_Z_PIVOT_REDSHIFT,
    DEFAULT_PIVOT_HALO_MASS,
    DEFAULT_PIVOT_REDSHIFT,
)


def test_la_amplitude_returns_constant_array() -> None:
    """Tests that LA amplitude returns a constant array."""
    z = np.array([0.0, 0.5, 1.0])

    amplitude = la_amplitude(z, a_ia=2.5)

    np.testing.assert_allclose(amplitude, np.full_like(z, 2.5))
    assert amplitude.dtype == np.float64
    assert amplitude.shape == z.shape


def test_la_amplitude_rejects_non_finite_redshift() -> None:
    """Tests that LA amplitude rejects non finite redshifts."""
    with pytest.raises(ValueError):
        la_amplitude([0.0, np.nan], a_ia=1.0)


def test_la_z_amplitude_matches_scale_factor_model() -> None:
    """Tests that redshift dependent LA amplitude matches the scale factor model."""
    z = np.array([0.0, DEFAULT_PIVOT_REDSHIFT, 1.0])
    a_ia = 1.2
    b_ia = -0.4

    amplitude = la_z_amplitude(z, a_ia=a_ia, b_ia=b_ia)

    scale_factor = 1.0 / (1.0 + z)
    pivot_scale_factor = 1.0 / (1.0 + DEFAULT_PIVOT_REDSHIFT)
    expected = a_ia + b_ia * (scale_factor / pivot_scale_factor - 1.0)

    np.testing.assert_allclose(amplitude, expected)
    assert amplitude.dtype == np.float64
    assert amplitude.shape == z.shape


def test_la_z_amplitude_rejects_invalid_redshift() -> None:
    """Tests that redshift dependent LA amplitude rejects invalid redshifts."""
    with pytest.raises(ValueError):
        la_z_amplitude([-1.0, 0.0], a_ia=1.0, b_ia=0.0)


def test_la_z_amplitude_rejects_invalid_pivot_redshift() -> None:
    """Tests that redshift dependent LA amplitude rejects invalid pivot redshift."""
    with pytest.raises(ValueError):
        la_z_amplitude([0.0, 1.0], a_ia=1.0, b_ia=0.0, pivot_redshift=-1.0)


def test_la_mass_amplitude_matches_mass_scaling() -> None:
    """Tests that mass dependent LA amplitude matches the expected scaling."""
    red_fraction = np.array([0.2, 0.5, 1.0])
    halo_mass = np.array([10**13.0, DEFAULT_PIVOT_HALO_MASS, 10**14.0])
    a_ia = 1.7
    beta = 0.8

    amplitude = la_mass_amplitude(
        a_ia=a_ia,
        red_fraction=red_fraction,
        halo_mass=halo_mass,
        beta=beta,
    )

    expected = a_ia * red_fraction * (halo_mass / DEFAULT_PIVOT_HALO_MASS) ** beta

    np.testing.assert_allclose(amplitude, expected)
    assert amplitude.dtype == np.float64
    assert amplitude.shape == red_fraction.shape


def test_la_mass_amplitude_accepts_broadcastable_inputs() -> None:
    """Tests that mass dependent LA amplitude accepts broadcastable inputs."""
    halo_mass = np.array([DEFAULT_PIVOT_HALO_MASS, 2.0 * DEFAULT_PIVOT_HALO_MASS])

    amplitude = la_mass_amplitude(
        a_ia=2.0,
        red_fraction=0.5,
        halo_mass=halo_mass,
        beta=1.0,
    )

    np.testing.assert_allclose(amplitude, np.array([1.0, 2.0]))
    assert amplitude.dtype == np.float64
    assert amplitude.shape == halo_mass.shape


def test_la_mass_amplitude_rejects_invalid_red_fraction() -> None:
    """Tests that mass dependent LA amplitude rejects invalid red fractions."""
    with pytest.raises(ValueError):
        la_mass_amplitude(
            a_ia=1.0,
            red_fraction=[0.5, 1.2],
            halo_mass=[DEFAULT_PIVOT_HALO_MASS, DEFAULT_PIVOT_HALO_MASS],
            beta=1.0,
        )


def test_la_mass_amplitude_rejects_non_positive_halo_mass() -> None:
    """Tests that mass dependent LA amplitude rejects non positive halo masses."""
    with pytest.raises(ValueError):
        la_mass_amplitude(
            a_ia=1.0,
            red_fraction=[0.5, 0.8],
            halo_mass=[DEFAULT_PIVOT_HALO_MASS, 0.0],
            beta=1.0,
        )


def test_la_mass_amplitude_rejects_non_positive_pivot_halo_mass() -> None:
    """Tests that mass dependent LA amplitude rejects non positive pivot halo mass."""
    with pytest.raises(ValueError):
        la_mass_amplitude(
            a_ia=1.0,
            red_fraction=[0.5, 0.8],
            halo_mass=[DEFAULT_PIVOT_HALO_MASS, DEFAULT_PIVOT_HALO_MASS],
            beta=1.0,
            pivot_halo_mass=0.0,
        )


def test_la_response_matches_expected_prefactor() -> None:
    """Tests that LA response matches the expected prefactor."""
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 2.0, -1.0])
    omega_m = 0.3

    response = la_response(growth_factor, omega_m, amplitude=amplitude)

    expected = -amplitude * C1_RHO_CRITICAL * omega_m / growth_factor

    np.testing.assert_allclose(response, expected)
    assert response.dtype == np.float64
    assert response.shape == growth_factor.shape


def test_la_response_accepts_scalar_amplitude() -> None:
    """Tests that LA response accepts scalar amplitude."""
    growth_factor = np.array([1.0, 0.5])
    omega_m = 0.25

    response = la_response(growth_factor, omega_m, amplitude=2.0)

    expected = -2.0 * C1_RHO_CRITICAL * omega_m / growth_factor

    np.testing.assert_allclose(response, expected)
    assert response.dtype == np.float64
    assert response.shape == growth_factor.shape


def test_la_response_rejects_non_positive_growth_factor() -> None:
    """Tests that LA response rejects non positive growth factors."""
    with pytest.raises(ValueError):
        la_response([1.0, 0.0], 0.3, amplitude=1.0)


def test_la_response_rejects_non_positive_omega_m() -> None:
    """Tests that LA response rejects non positive omega matter."""
    with pytest.raises(ValueError):
        la_response([1.0, 0.8], 0.0, amplitude=1.0)


def test_la_response_rejects_non_positive_c1_rho_critical() -> None:
    """Tests that LA response rejects non positive C1 rho critical."""
    with pytest.raises(ValueError):
        la_response([1.0, 0.8], 0.3, amplitude=1.0, c1_rho_critical=0.0)


def test_p_delta_i_la_matches_response_times_linear_power() -> None:
    """Tests that LA matter intrinsic power matches response times linear power."""
    linear_matter_power = np.array([10.0, 20.0, 40.0])
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 0.5, 2.0])
    omega_m = 0.3

    power = p_delta_i_la(
        linear_matter_power,
        growth_factor,
        omega_m,
        amplitude=amplitude,
    )

    response = la_response(growth_factor, omega_m, amplitude=amplitude)
    expected = response * linear_matter_power

    np.testing.assert_allclose(power, expected)
    assert power.dtype == np.float64
    assert power.shape == linear_matter_power.shape


def test_p_ii_la_matches_response_squared_times_linear_power() -> None:
    """Tests that LA intrinsic intrinsic power matches response squared times linear power."""
    linear_matter_power = np.array([10.0, 20.0, 40.0])
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 0.5, 2.0])
    omega_m = 0.3

    power = p_ii_la(
        linear_matter_power,
        growth_factor,
        omega_m,
        amplitude=amplitude,
    )

    response = la_response(growth_factor, omega_m, amplitude=amplitude)
    expected = response**2 * linear_matter_power

    np.testing.assert_allclose(power, expected)
    assert power.dtype == np.float64
    assert power.shape == linear_matter_power.shape


def test_p_delta_i_la_rejects_non_finite_linear_power() -> None:
    """Tests that LA matter intrinsic power rejects non finite linear power."""
    with pytest.raises(ValueError):
        p_delta_i_la([1.0, np.inf], [1.0, 0.8], 0.3, amplitude=1.0)


def test_p_ii_la_rejects_non_finite_linear_power() -> None:
    """Tests that LA intrinsic intrinsic power rejects non finite linear power."""
    with pytest.raises(ValueError):
        p_ii_la([1.0, np.inf], [1.0, 0.8], 0.3, amplitude=1.0)


class FakeFractions:
    """Fake LFKit fractions namespace."""

    def red_fraction(
        self,
        redshift: float,
        all_lf,
        *,
        m_bright: float,
        m_faint: float,
        n_m: int,
    ) -> float:
        """Return a deterministic fake red fraction."""
        _ = all_lf
        return 0.1 + 0.2 * redshift + 0.01 * (m_faint - m_bright) + 0.001 * n_m


def test_red_fraction_from_luminosity_functions_calls_lfkit_fraction() -> None:
    """Tests that red fraction helper calls the LFKit red fraction API."""
    z = np.array([0.0, 0.5, 1.0])
    red_lf = SimpleNamespace(fractions=FakeFractions())
    all_lf = object()

    fraction = red_fraction_from_luminosity_functions(
        z,
        red_lf=red_lf,
        all_lf=all_lf,
        m_bright=-22.0,
        m_faint=-18.0,
        n_m=10,
    )

    expected = np.array([0.15, 0.25, 0.35])

    np.testing.assert_allclose(fraction, expected)
    assert fraction.dtype == np.float64
    assert fraction.shape == z.shape


class InvalidFakeFractions:
    """Fake LFKit fractions namespace returning invalid values."""

    def red_fraction(
        self,
        redshift: float,
        all_lf,
        *,
        m_bright: float,
        m_faint: float,
        n_m: int,
    ) -> float:
        """Return an invalid fake red fraction."""
        _ = redshift, all_lf, m_bright, m_faint, n_m
        return 1.2


def test_red_fraction_from_luminosity_functions_rejects_invalid_fraction() -> None:
    """Tests that red fraction helper rejects invalid returned fractions."""
    red_lf = SimpleNamespace(fractions=InvalidFakeFractions())

    with pytest.raises(ValueError):
        red_fraction_from_luminosity_functions(
            [0.0, 0.5],
            red_lf=red_lf,
            all_lf=object(),
            m_bright=-22.0,
            m_faint=-18.0,
            n_m=10,
        )


def test_red_fraction_from_luminosity_functions_rejects_invalid_limits() -> None:
    """Tests that red fraction helper rejects invalid magnitude limits."""
    red_lf = SimpleNamespace(fractions=FakeFractions())

    with pytest.raises(ValueError):
        red_fraction_from_luminosity_functions(
            [0.0, 0.5],
            red_lf=red_lf,
            all_lf=object(),
            m_bright=-18.0,
            m_faint=-22.0,
        )


def test_red_fraction_from_luminosity_functions_rejects_invalid_n_m() -> None:
    """Tests that red fraction helper rejects invalid magnitude sample counts."""
    red_lf = SimpleNamespace(fractions=FakeFractions())

    with pytest.raises(ValueError):
        red_fraction_from_luminosity_functions(
            [0.0, 0.5],
            red_lf=red_lf,
            all_lf=object(),
            m_bright=-22.0,
            m_faint=-18.0,
            n_m=1,
        )


def test_lf_la_amplitude_matches_expected_scaling() -> None:
    """Tests that LF weighted LA amplitude matches the expected scaling."""
    z = np.array([0.1, DEFAULT_HIGH_Z_PIVOT_REDSHIFT, 1.0])
    luminosity_weighted_average = np.array([1.0, 1.5, 2.0])
    red_fraction = np.array([0.2, 0.5, 0.8])
    a_ia = 1.3
    eta_low_z = 0.7
    eta_high_z = -0.2

    amplitude = lf_la_amplitude(
        z,
        luminosity_weighted_average,
        a_ia=a_ia,
        red_fraction=red_fraction,
        eta_low_z=eta_low_z,
        eta_high_z=eta_high_z,
    )

    expected = a_ia * np.ones_like(z)
    expected *= ((1.0 + z) / (1.0 + DEFAULT_LOW_Z_PIVOT_REDSHIFT)) ** eta_low_z
    expected *= np.where(
        z > DEFAULT_HIGH_Z_PIVOT_REDSHIFT,
        ((1.0 + z) / (1.0 + DEFAULT_HIGH_Z_PIVOT_REDSHIFT)) ** eta_high_z,
        1.0,
    )
    expected *= red_fraction * luminosity_weighted_average

    np.testing.assert_allclose(amplitude, expected)
    assert amplitude.dtype == np.float64
    assert amplitude.shape == z.shape


def test_lf_la_amplitude_accepts_scalar_red_fraction() -> None:
    """Tests that LF weighted LA amplitude accepts scalar red fraction."""
    z = np.array([0.0, 0.5])
    luminosity_weighted_average = np.array([1.0, 2.0])

    amplitude = lf_la_amplitude(
        z,
        luminosity_weighted_average,
        a_ia=2.0,
        red_fraction=0.5,
    )

    expected = np.array([1.0, 2.0])

    np.testing.assert_allclose(amplitude, expected)
    assert amplitude.dtype == np.float64
    assert amplitude.shape == z.shape


def test_lf_la_amplitude_rejects_invalid_redshift() -> None:
    """Tests that LF weighted LA amplitude rejects invalid redshifts."""
    with pytest.raises(ValueError):
        lf_la_amplitude([-1.0, 0.0], [1.0, 1.0], a_ia=1.0)


def test_lf_la_amplitude_rejects_invalid_red_fraction() -> None:
    """Tests that LF weighted LA amplitude rejects invalid red fractions."""
    with pytest.raises(ValueError):
        lf_la_amplitude([0.0, 0.5], [1.0, 1.0], a_ia=1.0, red_fraction=[0.5, -0.1])


def test_lf_la_amplitude_rejects_invalid_low_z_pivot() -> None:
    """Tests that LF weighted LA amplitude rejects invalid low redshift pivot."""
    with pytest.raises(ValueError):
        lf_la_amplitude([0.0, 0.5], [1.0, 1.0], a_ia=1.0, low_z_pivot=-1.0)


def test_lf_la_amplitude_rejects_invalid_high_z_pivot() -> None:
    """Tests that LF weighted LA amplitude rejects invalid high redshift pivot."""
    with pytest.raises(ValueError):
        lf_la_amplitude([0.0, 0.5], [1.0, 1.0], a_ia=1.0, high_z_pivot=-1.0)


def test_la_models_are_registered() -> None:
    """Tests that LA models are registered."""
    models = list_models()

    assert "la" in models
    assert "la_z" in models
    assert "la_m" in models
    assert "la_lf" in models

    assert get_model("la") is la_amplitude
    assert get_model("la_z") is la_z_amplitude
    assert get_model("la_m") is la_mass_amplitude
    assert get_model("la_lf") is lf_la_amplitude


def test_la_model_aliases_are_registered() -> None:
    """Tests that LA model aliases are registered."""
    aliases = list_model_aliases()

    assert aliases["la"] == (
        "constant_la",
        "la_constant",
        "linear_alignment",
    )
    assert aliases["la_z"] == (
        "z_la",
        "redshift_la",
        "la_redshift",
        "linear_alignment_redshift",
    )
    assert aliases["la_m"] == (
        "m_la",
        "mass_la",
        "la_mass",
        "halo_mass_la",
        "linear_alignment_mass",
    )
    assert aliases["la_lf"] == (
        "lf_la",
        "luminosity_la",
        "luminosity_function_la",
        "la_luminosity_function",
        "linear_alignment_luminosity_function",
    )

    assert get_model("linear_alignment") is la_amplitude
    assert get_model("linear_alignment_redshift") is la_z_amplitude
    assert get_model("linear_alignment_mass") is la_mass_amplitude
    assert get_model("linear_alignment_luminosity_function") is lf_la_amplitude
