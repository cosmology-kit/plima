"""Unit tests for ``plima.models.nla``."""

from __future__ import annotations

import numpy as np
import pytest

from plima.models.model_registry import (
    get_model,
    list_model_aliases,
    list_models,
)
from plima.models.nla import (
    lf_nla_amplitude,
    nla_amplitude,
    nla_mass_amplitude,
    nla_response,
    nla_z_amplitude,
    p_delta_i_nla,
    p_ii_nla,
    red_fraction_from_luminosity_functions,
)
from plima.utils.constants import C1_RHO_CRITICAL, DEFAULT_PIVOT_HALO_MASS
from plima.utils.converters import redshift_to_scale_factor


def test_nla_const_shape() -> None:
    """Tests that constant NLA returns the input shape."""
    z = np.array([0.0, 0.5, 1.0])

    result = nla_amplitude(z, a_ia=1.7)

    expected = np.array([1.7, 1.7, 1.7])
    np.testing.assert_allclose(result, expected)
    assert result.shape == z.shape


def test_nla_const_bad_z() -> None:
    """Tests that constant NLA rejects bad redshift values."""
    with pytest.raises(ValueError, match="z"):
        nla_amplitude([0.0, np.nan], a_ia=1.0)


def test_nla_z_formula() -> None:
    """Tests that redshift NLA matches the scale factor formula."""
    z = np.array([0.0, 0.5, 1.0])
    a_ia = 1.2
    b_ia = -0.4
    pivot_redshift = 0.3

    result = nla_z_amplitude(
        z,
        a_ia=a_ia,
        b_ia=b_ia,
        pivot_redshift=pivot_redshift,
    )

    scale_factor = redshift_to_scale_factor(z)
    pivot_scale_factor = redshift_to_scale_factor(pivot_redshift)
    expected = a_ia + b_ia * (scale_factor / pivot_scale_factor - 1.0)

    np.testing.assert_allclose(result, expected)


def test_nla_z_pivot() -> None:
    """Tests that redshift NLA equals a_ia at the pivot."""
    result = nla_z_amplitude(
        [0.62],
        a_ia=2.3,
        b_ia=10.0,
        pivot_redshift=0.62,
    )

    np.testing.assert_allclose(result, [2.3])


@pytest.mark.parametrize("bad_z", [[-1.0], [-1.1]])
def test_nla_z_bad_z(bad_z: list[float]) -> None:
    """Tests that redshift NLA rejects redshift below the limit."""
    with pytest.raises(ValueError, match="z"):
        nla_z_amplitude(bad_z, a_ia=1.0, b_ia=0.0)


def test_nla_z_bad_pivot() -> None:
    """Tests that redshift NLA rejects a bad pivot redshift."""
    with pytest.raises(ValueError, match="pivot_redshift"):
        nla_z_amplitude(
            [0.0, 0.5],
            a_ia=1.0,
            b_ia=0.0,
            pivot_redshift=-1.0,
        )


def test_nla_mass_formula() -> None:
    """Tests that mass NLA matches fraction and mass scaling."""
    red_fraction = np.array([0.2, 0.5, 1.0])
    halo_mass = np.array(
        [
            DEFAULT_PIVOT_HALO_MASS,
            2.0 * DEFAULT_PIVOT_HALO_MASS,
            4.0 * DEFAULT_PIVOT_HALO_MASS,
        ],
    )

    result = nla_mass_amplitude(
        a_ia=1.5,
        red_fraction=red_fraction,
        halo_mass=halo_mass,
        beta=0.5,
    )

    expected = 1.5 * red_fraction
    expected *= (halo_mass / DEFAULT_PIVOT_HALO_MASS) ** 0.5
    np.testing.assert_allclose(result, expected)


def test_nla_mass_scalar() -> None:
    """Tests that mass NLA accepts scalar like inputs."""
    result = nla_mass_amplitude(
        a_ia=2.0,
        red_fraction=0.25,
        halo_mass=2.0 * DEFAULT_PIVOT_HALO_MASS,
        beta=1.0,
    )

    np.testing.assert_allclose(result, 1.0)


@pytest.mark.parametrize(
    ("red_fraction", "halo_mass", "pivot_halo_mass", "match"),
    [
        (
            [-0.1, 0.5],
            [1.0e13, 2.0e13],
            DEFAULT_PIVOT_HALO_MASS,
            "red_fraction",
        ),
        (
            [0.1, 1.2],
            [1.0e13, 2.0e13],
            DEFAULT_PIVOT_HALO_MASS,
            "red_fraction",
        ),
        ([0.1, 0.5], [0.0, 2.0e13], DEFAULT_PIVOT_HALO_MASS, "halo_mass"),
        ([0.1, 0.5], [1.0e13, 2.0e13], 0.0, "pivot_halo_mass"),
    ],
)
def test_nla_mass_bad_inputs(
    red_fraction: list[float],
    halo_mass: list[float],
    pivot_halo_mass: float,
    match: str,
) -> None:
    """Tests that mass NLA rejects invalid inputs."""
    with pytest.raises(ValueError, match=match):
        nla_mass_amplitude(
            a_ia=1.0,
            red_fraction=red_fraction,
            halo_mass=halo_mass,
            beta=1.0,
            pivot_halo_mass=pivot_halo_mass,
        )


def test_response_formula() -> None:
    """Tests that NLA response matches the standard prefactor."""
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 2.0, 3.0])
    omega_m = 0.3

    result = nla_response(growth_factor, omega_m, amplitude=amplitude)

    expected = -amplitude * C1_RHO_CRITICAL * omega_m / growth_factor
    np.testing.assert_allclose(result, expected)


def test_response_scalar_amp() -> None:
    """Tests that NLA response broadcasts scalar amplitude."""
    growth_factor = np.array([1.0, 0.5])

    result = nla_response(growth_factor, 0.3, amplitude=2.0)

    expected = -2.0 * C1_RHO_CRITICAL * 0.3 / growth_factor
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("growth_factor", "omega_m", "amplitude", "c1_rho_critical", "match"),
    [
        ([1.0, 0.0], 0.3, 1.0, C1_RHO_CRITICAL, "growth_factor"),
        ([1.0, 0.5], 0.0, 1.0, C1_RHO_CRITICAL, "omega_m"),
        ([1.0, 0.5], 0.3, 1.0, 0.0, "c1_rho_critical"),
        ([1.0, np.nan], 0.3, 1.0, C1_RHO_CRITICAL, "growth_factor"),
    ],
)
def test_response_bad_inputs(
    growth_factor: list[float],
    omega_m: float,
    amplitude: float,
    c1_rho_critical: float,
    match: str,
) -> None:
    """Tests that NLA response rejects invalid inputs."""
    with pytest.raises(ValueError, match=match):
        nla_response(
            growth_factor,
            omega_m,
            amplitude=amplitude,
            c1_rho_critical=c1_rho_critical,
        )


def test_p_delta_i_formula() -> None:
    """Tests that delta I power is response times matter power."""
    matter_power = np.array([10.0, 20.0, 30.0])
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 2.0, 3.0])
    omega_m = 0.3

    result = p_delta_i_nla(
        matter_power,
        growth_factor,
        omega_m,
        amplitude=amplitude,
    )

    expected = nla_response(growth_factor, omega_m, amplitude=amplitude)
    expected *= matter_power
    np.testing.assert_allclose(result, expected)


def test_p_ii_formula() -> None:
    """Tests that II power is response squared times matter power."""
    matter_power = np.array([10.0, 20.0, 30.0])
    growth_factor = np.array([1.0, 0.8, 0.5])
    amplitude = np.array([1.0, 2.0, 3.0])
    omega_m = 0.3

    result = p_ii_nla(
        matter_power,
        growth_factor,
        omega_m,
        amplitude=amplitude,
    )

    response = nla_response(growth_factor, omega_m, amplitude=amplitude)
    expected = response**2 * matter_power
    np.testing.assert_allclose(result, expected)


def test_spectra_bad_power() -> None:
    """Tests that NLA spectra reject bad matter power."""
    with pytest.raises(ValueError, match="matter_power"):
        p_delta_i_nla(
            [1.0, np.nan],
            [1.0, 0.8],
            0.3,
            amplitude=1.0,
        )

    with pytest.raises(ValueError, match="matter_power"):
        p_ii_nla(
            [1.0, np.nan],
            [1.0, 0.8],
            0.3,
            amplitude=1.0,
        )


def test_lf_nla_formula() -> None:
    """Tests that LF NLA matches luminosity and redshift scaling."""
    z = np.array([0.2, 0.5, 1.2])
    luminosity_weighted_average = np.array([1.0, 1.5, 2.0])
    red_fraction = np.array([0.4, 0.6, 0.8])

    result = lf_nla_amplitude(
        z,
        luminosity_weighted_average,
        a_ia=2.0,
        red_fraction=red_fraction,
        eta_low_z=1.0,
        eta_high_z=2.0,
        low_z_pivot=0.5,
        high_z_pivot=1.0,
    )

    low_z_term = (1.0 + z) / (1.0 + 0.5)
    high_z_term = np.where(z > 1.0, ((1.0 + z) / (1.0 + 1.0)) ** 2.0, 1.0)
    expected = 2.0 * low_z_term
    expected *= high_z_term * red_fraction * luminosity_weighted_average

    np.testing.assert_allclose(result, expected)


def test_lf_nla_default() -> None:
    """Tests that LF NLA defaults to luminosity weighted amplitude."""
    z = np.array([0.1, 0.5, 1.0])
    luminosity_weighted_average = np.array([0.8, 1.0, 1.2])

    result = lf_nla_amplitude(
        z,
        luminosity_weighted_average,
        a_ia=1.5,
    )

    expected = 1.5 * luminosity_weighted_average
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("z", "red_fraction", "low_z_pivot", "high_z_pivot", "match"),
    [
        ([-1.0], 0.5, 0.3, 1.0, "z"),
        ([0.5], -0.1, 0.3, 1.0, "red_fraction"),
        ([0.5], 1.1, 0.3, 1.0, "red_fraction"),
        ([0.5], 0.5, -1.0, 1.0, "low_z_pivot"),
        ([0.5], 0.5, 0.3, -1.0, "high_z_pivot"),
    ],
)
def test_lf_nla_bad_inputs(
    z: list[float],
    red_fraction: float,
    low_z_pivot: float,
    high_z_pivot: float,
    match: str,
) -> None:
    """Tests that LF NLA rejects invalid inputs."""
    with pytest.raises(ValueError, match=match):
        lf_nla_amplitude(
            z,
            [1.0],
            a_ia=1.0,
            red_fraction=red_fraction,
            low_z_pivot=low_z_pivot,
            high_z_pivot=high_z_pivot,
        )


class FakeFractions:
    """Small fake LFKit fractions API."""

    def __init__(self) -> None:
        """Tests that fake fraction calls are recorded."""
        self.calls: list[tuple[float, object, float, float, int]] = []

    def red_fraction(
        self,
        z: float,
        all_lf: object,
        *,
        m_bright: float,
        m_faint: float,
        n_m: int,
    ) -> float:
        """Return a deterministic red fraction."""
        self.calls.append((z, all_lf, m_bright, m_faint, n_m))
        return 0.2 + 0.1 * z


class FakeRedLuminosityFunction:
    """Small fake LFKit red luminosity function."""

    def __init__(self) -> None:
        """Tests that fake LFKit fractions are available."""
        self.fractions = FakeFractions()


def test_red_fraction_lfkit_api() -> None:
    """Tests that red fraction calls the LFKit style API."""
    red_lf = FakeRedLuminosityFunction()
    all_lf = object()
    z = np.array([0.0, 0.5, 1.0])

    result = red_fraction_from_luminosity_functions(
        z,
        red_lf=red_lf,
        all_lf=all_lf,
        m_bright=-24.0,
        m_faint=-18.0,
        n_m=128,
    )

    expected = np.array([0.2, 0.25, 0.3])
    np.testing.assert_allclose(result, expected)

    assert red_lf.fractions.calls == [
        (0.0, all_lf, -24.0, -18.0, 128),
        (0.5, all_lf, -24.0, -18.0, 128),
        (1.0, all_lf, -24.0, -18.0, 128),
    ]


@pytest.mark.parametrize(
    ("z", "m_bright", "m_faint", "n_m", "match"),
    [
        ([-1.0], -24.0, -18.0, 512, "z"),
        ([0.5], -18.0, -24.0, 512, "m_faint"),
        ([0.5], -24.0, -18.0, 1, "n_m"),
    ],
)
def test_red_fraction_bad_inputs(
    z: list[float],
    m_bright: float,
    m_faint: float,
    n_m: int,
    match: str,
) -> None:
    """Tests that red fraction rejects invalid inputs."""
    red_lf = FakeRedLuminosityFunction()

    with pytest.raises(ValueError, match=match):
        red_fraction_from_luminosity_functions(
            z,
            red_lf=red_lf,
            all_lf=object(),
            m_bright=m_bright,
            m_faint=m_faint,
            n_m=n_m,
        )


def test_red_fraction_bad_value() -> None:
    """Tests that red fraction rejects values outside the unit interval."""

    class BadFractions:
        """Small fake fractions API with an invalid value."""

        def red_fraction(
            self,
            z: float,
            all_lf: object,
            *,
            m_bright: float,
            m_faint: float,
            n_m: int,
        ) -> float:
            """Return an invalid red fraction."""
            return 1.2

    class BadRedLuminosityFunction:
        """Small fake red luminosity function with bad fractions."""

        fractions = BadFractions()

    with pytest.raises(ValueError, match="red_fraction"):
        red_fraction_from_luminosity_functions(
            [0.5],
            red_lf=BadRedLuminosityFunction(),
            all_lf=object(),
            m_bright=-24.0,
            m_faint=-18.0,
        )


def test_registry_discovers() -> None:
    """Tests that the registry discovers NLA models and aliases."""
    models = list_models()

    assert "nla" in models
    assert "constant_nla" in models
    assert "nla_z" in models
    assert "redshift_nla" in models
    assert "nla_m" in models
    assert "halo_mass_nla" in models
    assert "nla_lf" in models
    assert "luminosity_function_nla" in models


def test_registry_functions() -> None:
    """Tests that the registry returns registered model functions."""
    assert get_model("nla") is nla_amplitude
    assert get_model("constant_nla") is nla_amplitude
    assert get_model("nla_z") is nla_z_amplitude
    assert get_model("redshift_nla") is nla_z_amplitude
    assert get_model("nla_m") is nla_mass_amplitude
    assert get_model("halo_mass_nla") is nla_mass_amplitude
    assert get_model("nla_lf") is lf_nla_amplitude
    assert get_model("luminosity_function_nla") is lf_nla_amplitude


def test_registry_canonical() -> None:
    """Tests that the registry lists canonical names without aliases."""
    models = list_models(include_aliases=False)

    assert "nla" in models
    assert "nla_z" in models
    assert "nla_m" in models
    assert "nla_lf" in models

    assert "constant_nla" not in models
    assert "redshift_nla" not in models
    assert "halo_mass_nla" not in models
    assert "luminosity_function_nla" not in models


def test_registry_aliases() -> None:
    """Tests that the registry lists aliases by canonical name."""
    aliases = list_model_aliases()

    assert aliases["nla"] == ("constant_nla", "nla_constant")
    assert aliases["nla_z"] == ("z_nla", "redshift_nla", "nla_redshift")
    assert aliases["nla_m"] == (
        "m_nla",
        "mass_nla",
        "nla_mass",
        "halo_mass_nla",
    )
    assert aliases["nla_lf"] == (
        "lf_nla",
        "luminosity_nla",
        "luminosity_function_nla",
        "nla_luminosity_function",
    )


def test_registry_unknown() -> None:
    """Tests that unknown registry keys raise a helpful error."""
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("not_a_real_model")
