"""Unit tests for ``plima.models._discovery``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from plima.models import _discovery
from plima.models.model_registry import (
    get_model,
    list_model_aliases,
    list_models,
)


def test_import_model_modules_skips_non_model_modules(monkeypatch) -> None:
    """Tests that registry and helper modules are not imported."""
    module_names = [
        "_discovery",
        "model_registry",
    ]
    imported_modules: list[str] = []

    def fake_iter_modules(package_path):
        assert package_path is not None
        return [SimpleNamespace(name=name) for name in module_names]

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return SimpleNamespace()

    monkeypatch.setattr(_discovery.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(
        _discovery.importlib, "import_module", fake_import_module
    )

    _discovery.import_model_modules()

    assert imported_modules == []


def test_import_model_modules_skips_registry_and_helper_modules(
    monkeypatch,
) -> None:
    """Tests that registry and helper modules are not imported."""
    module_names = [
        "_discovery",
        "model_registry",
    ]
    imported_modules: list[str] = []

    def fake_iter_modules(package_path):
        assert package_path is not None
        return [SimpleNamespace(name=name) for name in module_names]

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return SimpleNamespace()

    monkeypatch.setattr(_discovery.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(
        _discovery.importlib, "import_module", fake_import_module
    )

    _discovery.import_model_modules()

    assert imported_modules == []


def test_skip_modules_contains_expected_modules() -> None:
    """Tests that discovery skips known non model modules."""
    assert _discovery._SKIP_MODULES == {
        "_discovery",
        "model_registry",
    }


def test_real_discovery_registers_expected_models() -> None:
    """Tests that real discovery registers expected model names."""
    models = list_models(include_aliases=False)

    assert "nla" in models
    assert "nla_z" in models
    assert "nla_m" in models
    assert "nla_lf" in models
    assert "tatt" in models
    assert "halo_model" in models


def test_real_discovery_registers_expected_aliases() -> None:
    """Tests that real discovery registers expected model aliases."""
    aliases = list_model_aliases()

    assert aliases["nla"] == ("constant_nla", "nla_constant")
    assert aliases["nla_z"] == (
        "z_nla",
        "redshift_nla",
        "nla_redshift",
    )
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
    assert aliases["tatt"] == (
        "tatt_amplitudes",
        "tatt_a",
    )
    assert aliases["halo_model"] == (
        "hm_ia",
        "halo_model_ia",
    )


def test_real_discovery_retrieves_nla_model() -> None:
    """Tests that real discovery retrieves and evaluates the NLA model."""
    model = get_model("nla")

    z = np.array([0.0, 0.5, 1.0])
    amplitude = model(z, a_ia=2.5)

    np.testing.assert_allclose(amplitude, [2.5, 2.5, 2.5])


def test_real_discovery_retrieves_tatt_model() -> None:
    """Tests that real discovery retrieves and evaluates the TATT model."""
    model = get_model("tatt")

    z = np.array([0.0, 0.5, 1.0])
    amplitudes = model(
        z,
        a1=1.0,
        a2=2.0,
        a1delta=3.0,
        eta1=0.0,
        eta2=0.0,
        eta1delta=0.0,
    )

    assert set(amplitudes) == {"a1", "a2", "a1delta"}
    np.testing.assert_allclose(amplitudes["a1"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(amplitudes["a2"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(amplitudes["a1delta"], [3.0, 3.0, 3.0])


def test_real_discovery_retrieves_halo_model() -> None:
    """Tests that real discovery retrieves and evaluates the halo model."""
    model = get_model("halo_model")

    z = np.array([0.0, 0.5, 1.0])
    parameters = model(
        z,
        a_ia=2.0,
        eta_ia=0.0,
        a1h=0.003,
        eta_1h=0.0,
        b=-2.0,
        k_1h=1.5,
        k_2h=0.2,
    )

    assert set(parameters) == {"a_ia", "a1h", "b", "k_1h", "k_2h"}
    np.testing.assert_allclose(parameters["a_ia"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(parameters["a1h"], [0.003, 0.003, 0.003])
    np.testing.assert_allclose(parameters["b"], [-2.0, -2.0, -2.0])
    np.testing.assert_allclose(parameters["k_1h"], [1.5, 1.5, 1.5])
    np.testing.assert_allclose(parameters["k_2h"], [0.2, 0.2, 0.2])
