"""Unit tests for ``plima.models.halo_model``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from plima.models.halo_model import halo_model_ia_parameters
from plima.models.model_registry import (
    get_model,
    list_model_aliases,
    list_models,
)
from plima.utils.constants import DEFAULT_PIVOT_REDSHIFT


def test_halo_model_ia_parameters_defaults() -> None:
    """Tests that halo model IA parameters use the expected defaults."""
    z = np.array([0.0, DEFAULT_PIVOT_REDSHIFT, 1.0])

    parameters = halo_model_ia_parameters(z)

    assert set(parameters) == {"a_ia", "a1h", "b"}
    np.testing.assert_allclose(parameters["a_ia"], np.ones_like(z))
    np.testing.assert_allclose(parameters["a1h"], np.full_like(z, 0.001))
    np.testing.assert_allclose(parameters["b"], np.full_like(z, -2.0))

    for value in parameters.values():
        assert value.dtype == np.float64
        assert value.shape == z.shape


def test_halo_model_ia_parameters_evolve_with_redshift() -> None:
    """Tests that halo model IA amplitudes evolve around the pivot redshift."""
    z = np.array([0.0, 0.5, 1.0])
    z_pivot = 0.5

    parameters = halo_model_ia_parameters(
        z,
        a_ia=2.0,
        eta_ia=1.5,
        a1h=0.2,
        eta_1h=-0.5,
        b=-1.2,
        z_pivot=z_pivot,
    )

    redshift_ratio = (1.0 + z) / (1.0 + z_pivot)

    np.testing.assert_allclose(
        parameters["a_ia"],
        2.0 * redshift_ratio**1.5,
    )
    np.testing.assert_allclose(
        parameters["a1h"],
        0.2 * redshift_ratio**-0.5,
    )
    np.testing.assert_allclose(parameters["b"], np.full_like(z, -1.2))


def test_halo_model_ia_parameters_equal_amplitudes_at_pivot() -> None:
    """Tests that halo model IA amplitudes equal inputs at the pivot redshift."""
    z_pivot = 0.7

    parameters = halo_model_ia_parameters(
        z_pivot,
        a_ia=3.0,
        eta_ia=2.0,
        a1h=0.4,
        eta_1h=-1.0,
        b=-1.5,
        z_pivot=z_pivot,
    )

    np.testing.assert_allclose(parameters["a_ia"], 3.0)
    np.testing.assert_allclose(parameters["a1h"], 0.4)
    np.testing.assert_allclose(parameters["b"], -1.5)


def test_halo_model_ia_parameters_uses_default_pivot_redshift() -> None:
    """Tests that halo model IA amplitudes use the shared default pivot redshift."""
    z = np.array([DEFAULT_PIVOT_REDSHIFT])

    parameters = halo_model_ia_parameters(
        z,
        a_ia=2.4,
        eta_ia=7.0,
        a1h=0.8,
        eta_1h=-3.0,
    )

    np.testing.assert_allclose(parameters["a_ia"], np.array([2.4]))
    np.testing.assert_allclose(parameters["a1h"], np.array([0.8]))


def test_halo_model_ia_parameters_includes_transition_scales() -> None:
    """Tests that halo model IA parameters include optional transition scales."""
    z = np.array([0.1, 0.4, 0.9])

    parameters = halo_model_ia_parameters(
        z,
        k_1h=2.5,
        k_2h=4.0,
    )

    assert set(parameters) == {"a_ia", "a1h", "b", "k_1h", "k_2h"}
    np.testing.assert_allclose(parameters["k_1h"], np.full_like(z, 2.5))
    np.testing.assert_allclose(parameters["k_2h"], np.full_like(z, 4.0))


def test_halo_model_ia_parameters_omits_transition_scales_by_default() -> None:
    """Tests that halo model IA parameters omit transition scales by default."""
    parameters = halo_model_ia_parameters([0.1, 0.4, 0.9])

    assert "k_1h" not in parameters
    assert "k_2h" not in parameters


@pytest.mark.parametrize(
    "z",
    [
        0.5,
        np.float64(0.5),
    ],
)
def test_halo_model_ia_parameters_accepts_scalar_redshift(z: float) -> None:
    """Tests that halo model IA parameters accept scalar redshift input."""
    parameters = halo_model_ia_parameters(z)

    assert parameters["a_ia"].shape == ()
    assert parameters["a1h"].shape == ()
    assert parameters["b"].shape == ()
    np.testing.assert_allclose(parameters["a_ia"], 1.0)
    np.testing.assert_allclose(parameters["a1h"], 0.001)
    np.testing.assert_allclose(parameters["b"], -2.0)


@pytest.mark.parametrize(
    "z",
    [
        [0.0, 0.5, 1.0],
        (0.0, 0.5, 1.0),
        np.array([0, 1, 2]),
    ],
)
def test_halo_model_ia_parameters_accepts_array_like_redshifts(
    z: Any,
) -> None:
    """Tests that halo model IA parameters accept array like redshifts."""
    parameters = halo_model_ia_parameters(z)

    expected_shape = np.asarray(z, dtype=np.float64).shape

    assert parameters["a_ia"].shape == expected_shape
    assert parameters["a1h"].shape == expected_shape
    assert parameters["b"].shape == expected_shape

    for value in parameters.values():
        assert value.dtype == np.float64


def test_halo_model_ia_parameters_preserves_multidimensional_shape() -> None:
    """Tests that halo model IA parameters preserve multidimensional shapes."""
    z = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
        ],
    )

    parameters = halo_model_ia_parameters(z)

    for value in parameters.values():
        assert value.shape == z.shape


def test_halo_model_ia_parameters_accepts_empty_redshift_array() -> None:
    """Tests that halo model IA parameters accept empty redshift arrays."""
    z = np.array([], dtype=np.float64)

    parameters = halo_model_ia_parameters(z)

    assert parameters["a_ia"].shape == (0,)
    assert parameters["a1h"].shape == (0,)
    assert parameters["b"].shape == (0,)


def test_halo_model_ia_parameters_accepts_zero_amplitudes() -> None:
    """Tests that halo model IA parameters accept zero amplitude values."""
    z = np.array([0.0, 0.5, 1.0])

    parameters = halo_model_ia_parameters(
        z,
        a_ia=0.0,
        a1h=0.0,
        b=0.0,
    )

    np.testing.assert_allclose(parameters["a_ia"], np.zeros_like(z))
    np.testing.assert_allclose(parameters["a1h"], np.zeros_like(z))
    np.testing.assert_allclose(parameters["b"], np.zeros_like(z))


def test_halo_model_ia_parameters_accepts_negative_amplitudes() -> None:
    """Tests that halo model IA parameters accept negative amplitude values."""
    z = np.array([0.0, 0.5, 1.0])

    parameters = halo_model_ia_parameters(
        z,
        a_ia=-2.0,
        a1h=-0.3,
        b=-1.7,
    )

    np.testing.assert_allclose(parameters["a_ia"], np.full_like(z, -2.0))
    np.testing.assert_allclose(parameters["a1h"], np.full_like(z, -0.3))
    np.testing.assert_allclose(parameters["b"], np.full_like(z, -1.7))


def test_halo_model_ia_parameters_transition_scales_follow_input_shape() -> (
    None
):
    """Tests that halo model IA transition scales follow the redshift shape."""
    z = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    parameters = halo_model_ia_parameters(
        z,
        k_1h=1.5,
        k_2h=2.5,
    )

    np.testing.assert_allclose(parameters["k_1h"], np.full_like(z, 1.5))
    np.testing.assert_allclose(parameters["k_2h"], np.full_like(z, 2.5))
    assert parameters["k_1h"].shape == z.shape
    assert parameters["k_2h"].shape == z.shape


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"a_ia": np.inf}, "a_ia"),
        ({"a_ia": np.nan}, "a_ia"),
        ({"eta_ia": np.inf}, "eta_ia"),
        ({"eta_ia": np.nan}, "eta_ia"),
        ({"a1h": np.inf}, "a1h"),
        ({"a1h": np.nan}, "a1h"),
        ({"eta_1h": np.inf}, "eta_1h"),
        ({"eta_1h": np.nan}, "eta_1h"),
        ({"b": np.inf}, "b"),
        ({"b": np.nan}, "b"),
        ({"z_pivot": np.inf}, "z_pivot"),
        ({"z_pivot": np.nan}, "z_pivot"),
        ({"z_pivot": -1.0}, "z_pivot"),
        ({"z_pivot": -1.1}, "z_pivot"),
        ({"k_1h": 0.0}, "k_1h"),
        ({"k_1h": -1.0}, "k_1h"),
        ({"k_1h": np.inf}, "k_1h"),
        ({"k_1h": np.nan}, "k_1h"),
        ({"k_2h": 0.0}, "k_2h"),
        ({"k_2h": -1.0}, "k_2h"),
        ({"k_2h": np.inf}, "k_2h"),
        ({"k_2h": np.nan}, "k_2h"),
    ],
)
def test_halo_model_ia_parameters_rejects_invalid_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Tests that halo model IA parameters reject invalid scalar inputs."""
    with pytest.raises(ValueError, match=match):
        halo_model_ia_parameters([0.1, 0.5], **kwargs)


@pytest.mark.parametrize(
    "z",
    [
        [-1.0, 0.0],
        [-1.1, 0.0],
        [0.0, np.inf],
        [0.0, -np.inf],
        [0.0, np.nan],
        [[0.0, 0.1], [0.2, np.nan]],
    ],
)
def test_halo_model_ia_parameters_rejects_invalid_redshifts(
    z: Any,
) -> None:
    """Tests that halo model IA parameters reject invalid redshifts."""
    with pytest.raises(ValueError, match="z"):
        halo_model_ia_parameters(z)


def test_halo_model_is_registered() -> None:
    """Tests that the halo model is registered under its canonical name."""
    assert "halo_model" in list_models()
    assert get_model("halo_model") is halo_model_ia_parameters


@pytest.mark.parametrize(
    "alias",
    [
        "hm_ia",
        "halo_model_ia",
    ],
)
def test_halo_model_aliases_are_registered(alias: str) -> None:
    """Tests that halo model aliases resolve to the registered model."""
    assert get_model(alias) is halo_model_ia_parameters


def test_halo_model_aliases_are_listed() -> None:
    """Tests that halo model aliases are listed in the registry."""
    aliases = list_model_aliases()

    assert aliases["halo_model"] == (
        "hm_ia",
        "halo_model_ia",
    )


def test_halo_model_registry_entries_are_consistent() -> None:
    """Tests that halo model registry entries are internally consistent."""
    aliases = list_model_aliases()

    assert "halo_model" in aliases
    assert "halo_model" in list_models()

    for alias in aliases["halo_model"]:
        assert get_model(alias) is get_model("halo_model")
