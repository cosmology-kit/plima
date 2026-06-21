"""Unit tests for ``plima.backends.ccl.halo_model``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
import pytest

from plima.backends.ccl.halo_model import (
    _evaluate_pk2d_grid,
    _pk2d_from_array,
    _power_grid,
    halo_model_nla_prefactor,
    make_ccl_halo_model_ia_power_spectra,
    make_ccl_halo_model_power_spectra,
    make_ccl_halo_model_windowed_power_spectra,
)
from plima.utils.constants import C1_RHO_CRITICAL


@pytest.fixture
def cosmo() -> ccl.Cosmology:
    """Tests that a simple CCL cosmology can be reused by backend tests."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.96,
        sigma8=0.8,
        transfer_function="bbks",
        matter_power_spectrum="linear",
    )


@pytest.fixture
def k() -> np.ndarray:
    """Tests that a positive wavenumber grid is available."""
    return np.logspace(-3.0, 1.0, 16)


@pytest.fixture
def a() -> np.ndarray:
    """Tests that a positive scale factor grid is available."""
    return np.linspace(0.25, 1.0, 8)


@pytest.fixture
def dummy_ia_power_spectra(
    k: np.ndarray,
    a: np.ndarray,
) -> dict[str, ccl.Pk2D]:
    """Tests that dummy IA power spectra are available for windowing tests."""
    one = np.ones((a.size, k.size))
    two = 2.0 * one
    three = 3.0 * one

    return {
        "ii_1h_ss": _pk2d_from_array(k=k, a=a, pk=one),
        "ii_2h_ss": _pk2d_from_array(k=k, a=a, pk=two),
        "ii_2h_cs": _pk2d_from_array(k=k, a=a, pk=two),
        "ii_2h_cc": _pk2d_from_array(k=k, a=a, pk=two),
        "ii_total": _pk2d_from_array(k=k, a=a, pk=three),
        "gi_1h_s": _pk2d_from_array(k=k, a=a, pk=one),
        "gi_2h_s": _pk2d_from_array(k=k, a=a, pk=two),
        "gi_2h_c": _pk2d_from_array(k=k, a=a, pk=two),
        "gi_total": _pk2d_from_array(k=k, a=a, pk=three),
    }


def test_halo_model_nla_prefactor_matches_expected_formula(
    cosmo: ccl.Cosmology,
    a: np.ndarray,
) -> None:
    """Tests that the NLA prefactor matches the expected CCL formula."""
    a_ia = 2.5
    c1_rho_critical = 0.02

    prefactor = halo_model_nla_prefactor(
        cosmo,
        a,
        a_ia=a_ia,
        c1_rho_critical=c1_rho_critical,
    )

    expected = (
        a_ia
        * c1_rho_critical
        * cosmo["Omega_m"]
        / cosmo.growth_factor(a)
    )

    np.testing.assert_allclose(prefactor, expected)
    assert prefactor.dtype == np.float64
    assert prefactor.shape == a.shape


def test_halo_model_nla_prefactor_uses_default_normalization(
    cosmo: ccl.Cosmology,
    a: np.ndarray,
) -> None:
    """Tests that the NLA prefactor uses the shared default normalization."""
    prefactor = halo_model_nla_prefactor(cosmo, a)

    expected = C1_RHO_CRITICAL * cosmo["Omega_m"] / cosmo.growth_factor(a)

    np.testing.assert_allclose(prefactor, expected)


def test_pk2d_from_array_evaluates_input_grid(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that a Pk2D object can be built from an input grid."""
    pk = np.outer(a, k**2)

    pk2d = _pk2d_from_array(k=k, a=a, pk=pk)

    values = _evaluate_pk2d_grid(pk2d, k=k, a=a, name="pk")

    np.testing.assert_allclose(values, pk, rtol=1.0e-6, atol=1.0e-12)


def test_pk2d_from_array_rejects_wrong_shape(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that Pk2D construction rejects an incompatible grid shape."""
    pk = np.ones((k.size, a.size + 1))

    with pytest.raises(ValueError, match="pk"):
        _pk2d_from_array(k=k, a=a, pk=pk)


def test_power_grid_accepts_expected_shape(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that power grid validation accepts the expected orientation."""
    values = np.ones((a.size, k.size))

    grid = _power_grid(values, k=k, a=a, name="values")

    np.testing.assert_allclose(grid, values)


def test_power_grid_accepts_transposed_shape(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that power grid validation transposes CCL style grids."""
    values = np.arange(k.size * a.size, dtype=np.float64).reshape(k.size, a.size)

    grid = _power_grid(values, k=k, a=a, name="values")

    np.testing.assert_allclose(grid, values.T)


def test_power_grid_rejects_wrong_shape(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that power grid validation rejects incompatible shapes."""
    values = np.ones((a.size + 1, k.size + 1))

    with pytest.raises(ValueError, match="values"):
        _power_grid(values, k=k, a=a, name="values")


def test_make_ccl_halo_model_ia_power_spectra_returns_expected_keys(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that raw halo model IA spectra return the expected components."""
    spectra = make_ccl_halo_model_ia_power_spectra(
        cosmo,
        k=k,
        a=a,
        a_ia=1.2,
        a1h=0.001,
        b=-2.0,
    )

    assert set(spectra) == {
        "ii_1h_ss",
        "ii_2h_ss",
        "ii_2h_cs",
        "ii_2h_cc",
        "ii_total",
        "gi_1h_s",
        "gi_2h_s",
        "gi_2h_c",
        "gi_total",
    }

    for pk2d in spectra.values():
        values = _evaluate_pk2d_grid(pk2d, k=k, a=a, name="spectrum")
        assert values.shape == (a.size, k.size)
        assert np.all(np.isfinite(values))


def test_make_ccl_halo_model_ia_power_spectra_accepts_nonlinear_2h(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that raw halo model IA spectra can use nonlinear matter power."""
    spectra = make_ccl_halo_model_ia_power_spectra(
        cosmo,
        k=k,
        a=a,
        use_linear_2h=False,
    )

    assert "ii_total" in spectra
    assert "gi_total" in spectra


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": [0.0, 1.0], "a": [0.5, 1.0]}, "k"),
        ({"k": [0.1, 1.0], "a": [0.0, 1.0]}, "a"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "a_ia": np.nan}, "a_ia"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "a1h": np.inf}, "a1h"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "b": np.nan}, "b"),
    ],
)
def test_make_ccl_halo_model_ia_power_spectra_rejects_invalid_inputs(
    cosmo: ccl.Cosmology,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Tests that raw halo model IA spectra reject invalid inputs."""
    with pytest.raises(ValueError, match=match):
        make_ccl_halo_model_ia_power_spectra(cosmo, **kwargs)


def test_make_ccl_halo_model_windowed_power_spectra_returns_expected_keys(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
    dummy_ia_power_spectra: dict[str, ccl.Pk2D],
) -> None:
    """Tests that windowed halo model IA spectra return expected components."""
    spectra = make_ccl_halo_model_windowed_power_spectra(
        cosmo,
        k=k,
        a=a,
        ia_power_spectra=dummy_ia_power_spectra,
        k_1h=2.0,
        k_2h=3.0,
    )

    assert set(spectra) == {
        "ii",
        "gi",
        "ii_nla",
        "gi_nla",
        "ii_1h",
        "gi_1h",
    }

    for pk2d in spectra.values():
        values = _evaluate_pk2d_grid(pk2d, k=k, a=a, name="spectrum")
        assert values.shape == (a.size, k.size)
        assert np.all(np.isfinite(values))


def test_make_ccl_halo_model_windowed_power_spectra_uses_default_scales(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
    dummy_ia_power_spectra: dict[str, ccl.Pk2D],
) -> None:
    """Tests that windowed halo model IA spectra use default transition scales."""
    spectra = make_ccl_halo_model_windowed_power_spectra(
        cosmo,
        k=k,
        a=a,
        ia_power_spectra=dummy_ia_power_spectra,
    )

    assert "ii" in spectra
    assert "gi" in spectra


def test_make_ccl_halo_model_windowed_power_spectra_requires_one_halo_terms(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that windowed halo model IA spectra require one halo components."""
    with pytest.raises(KeyError, match="ii_1h_ss"):
        make_ccl_halo_model_windowed_power_spectra(
            cosmo,
            k=k,
            a=a,
            ia_power_spectra={},
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": [0.0, 1.0], "a": [0.5, 1.0]}, "k"),
        ({"k": [0.1, 1.0], "a": [0.0, 1.0]}, "a"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "a_ia": np.nan}, "a_ia"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "k_1h": 0.0}, "k_1h"),
        ({"k": [0.1, 1.0], "a": [0.5, 1.0], "k_2h": -1.0}, "k_2h"),
    ],
)
def test_make_ccl_halo_model_windowed_power_spectra_rejects_invalid_inputs(
    cosmo: ccl.Cosmology,
    dummy_ia_power_spectra: dict[str, ccl.Pk2D],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Tests that windowed halo model IA spectra reject invalid inputs."""
    with pytest.raises(ValueError, match=match):
        make_ccl_halo_model_windowed_power_spectra(
            cosmo,
            ia_power_spectra=dummy_ia_power_spectra,
            **kwargs,
        )


def test_make_ccl_halo_model_power_spectra_returns_components_and_spectra(
    cosmo: ccl.Cosmology,
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that the convenience helper returns components and spectra."""
    output = make_ccl_halo_model_power_spectra(
        cosmo,
        k=k,
        a=a,
        a_ia=1.1,
        a1h=0.002,
        b=-1.8,
        k_1h=2.0,
        k_2h=3.0,
    )

    assert set(output) == {"components", "spectra"}
    assert "ii_total" in output["components"]
    assert "gi_total" in output["components"]
    assert "ii" in output["spectra"]
    assert "gi" in output["spectra"]


def test_evaluate_pk2d_grid_rejects_nonfinite_values(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that Pk2D grid evaluation rejects nonfinite values."""

    class BadPk2D:
        def __call__(self, k_eval: np.ndarray, a_eval: np.ndarray) -> np.ndarray:
            values = np.ones((a_eval.size, k_eval.size))
            values[0, 0] = np.nan
            return values

    with pytest.raises(ValueError, match="bad_pk"):
        _evaluate_pk2d_grid(BadPk2D(), k=k, a=a, name="bad_pk")


def test_pk2d_from_array_rejects_nonfinite_values(
    k: np.ndarray,
    a: np.ndarray,
) -> None:
    """Tests that Pk2D construction rejects nonfinite values."""
    pk = np.ones((a.size, k.size))
    pk[0, 0] = np.nan

    with pytest.raises(ValueError, match="pk"):
        _pk2d_from_array(k=k, a=a, pk=pk)


def test_halo_model_nla_prefactor_rejects_scalar_scale_factor(
    cosmo: ccl.Cosmology,
) -> None:
    """Tests that the NLA prefactor rejects scalar scale factors."""
    with pytest.raises(ValueError, match="a"):
        halo_model_nla_prefactor(cosmo, 0.5)


def test_halo_model_nla_prefactor_rejects_multidimensional_scale_factor(
    cosmo: ccl.Cosmology,
) -> None:
    """Tests that the NLA prefactor rejects multidimensional scale factors."""
    with pytest.raises(ValueError, match="a must be one dimensional."):
        halo_model_nla_prefactor(cosmo, np.array([[0.5, 1.0]]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0.1, "a": [0.5, 1.0]}, "k must be one dimensional."),
        ({"k": [0.1, 1.0], "a": 0.5}, "a must be one dimensional."),
        ({"k": [[0.1, 1.0]], "a": [0.5, 1.0]}, "k must be one dimensional."),
        ({"k": [0.1, 1.0], "a": [[0.5, 1.0]]}, "a must be one dimensional."),
    ],
)
def test_make_ccl_halo_model_ia_power_spectra_rejects_invalid_grid_dimensions(
    cosmo: ccl.Cosmology,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Tests that raw halo model IA spectra reject invalid grid dimensions."""
    with pytest.raises(ValueError, match=match):
        make_ccl_halo_model_ia_power_spectra(cosmo, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0.1, "a": [0.5, 1.0]}, "k must be one dimensional."),
        ({"k": [0.1, 1.0], "a": 0.5}, "a must be one dimensional."),
        ({"k": [[0.1, 1.0]], "a": [0.5, 1.0]}, "k must be one dimensional."),
        ({"k": [0.1, 1.0], "a": [[0.5, 1.0]]}, "a must be one dimensional."),
    ],
)
def test_make_ccl_halo_model_windowed_power_spectra_rejects_invalid_grid_dimensions(
    cosmo: ccl.Cosmology,
    dummy_ia_power_spectra: dict[str, ccl.Pk2D],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Tests that windowed halo model IA spectra reject invalid grid dimensions."""
    with pytest.raises(ValueError, match=match):
        make_ccl_halo_model_windowed_power_spectra(
            cosmo,
            ia_power_spectra=dummy_ia_power_spectra,
            **kwargs,
        )
