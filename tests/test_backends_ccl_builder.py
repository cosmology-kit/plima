"""Unit tests for ``plima.backends.ccl.builder``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from plima.backends.ccl import builder


def _amplitude_with_z(z, *, a_ia: float = 1.0):
    return a_ia * np.ones_like(np.asarray(z, dtype=float))


def _amplitude_without_z(*, value: float = 2.0):
    return value


def _set_module(function, module_name: str):
    function.__module__ = module_name
    return function


def test_normalize_model_name_strips_and_lowers() -> None:
    """Tests that model names are stripped and lowercased."""
    assert builder._normalize_model_name("  NLA  ") == "nla"


def test_canonical_model_name_returns_registered_name(monkeypatch) -> None:
    """Tests that canonical model names are returned unchanged."""
    monkeypatch.setattr(
        builder,
        "list_model_aliases",
        lambda: {"nla": ("nla_lf",), "tatt": ("ta",)},
    )

    assert builder._canonical_model_name("nla") == "nla"


def test_canonical_model_name_resolves_alias(monkeypatch) -> None:
    """Tests that aliases resolve to their canonical model name."""
    monkeypatch.setattr(
        builder,
        "list_model_aliases",
        lambda: {"nla": ("nla_lf",), "tatt": ("ta",)},
    )

    assert builder._canonical_model_name("nla_lf") == "nla"


def test_canonical_model_name_returns_unknown_name(monkeypatch) -> None:
    """Tests that unknown names are returned normalized."""
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"nla": ("nla_lf",)})

    assert builder._canonical_model_name("missing") == "missing"


def test_model_family_uses_function_module_name() -> None:
    """Tests that the model family is inferred from the function module."""
    model_function = _set_module(_amplitude_with_z, "plima.models.nla")

    assert builder._model_family(model_function) == "nla"


def test_model_accepts_z_detects_z_parameter() -> None:
    """Tests that models with a z argument are detected."""
    assert builder._model_accepts_z(_amplitude_with_z)


def test_model_accepts_z_rejects_models_without_z_parameter() -> None:
    """Tests that models without a z argument are detected."""
    assert not builder._model_accepts_z(_amplitude_without_z)


def test_evaluate_amplitude_model_passes_z_when_supported() -> None:
    """Tests that amplitude models with z receive redshift values."""
    z = np.array([0.0, 0.5, 1.0])

    amplitude = builder._evaluate_amplitude_model(_amplitude_with_z, z, a_ia=3.0)

    np.testing.assert_allclose(amplitude, np.full_like(z, 3.0))


def test_evaluate_amplitude_model_omits_z_when_unsupported() -> None:
    """Tests that amplitude models without z are called without redshift values."""
    z = np.array([0.0, 0.5, 1.0])

    amplitude = builder._evaluate_amplitude_model(
        _amplitude_without_z,
        z,
        value=4.0,
    )

    assert amplitude == 4.0


def test_native_bias_model_has_standard_contract() -> None:
    """Tests that native bias model dictionaries have the expected contract."""
    ia_bias = ("z", "bias")
    amplitude = np.array([1.0, 2.0])

    result = builder._native_bias_model(
        model_name="nla",
        canonical_model_name="nla",
        family="nla",
        ia_bias=ia_bias,
        amplitude=amplitude,
    )

    assert result["model_name"] == "nla"
    assert result["canonical_model_name"] == "nla"
    assert result["family"] == "nla"
    assert result["backend"] == "ccl"
    assert result["mode"] == "native_bias"
    assert result["ia_bias"] == ia_bias
    assert result["spectra"] == {}
    assert result["metadata"]["uses_native_ccl_ia"]
    assert not result["metadata"]["uses_pk2d"]
    np.testing.assert_allclose(result["metadata"]["amplitude"], amplitude)


def test_pk2d_model_has_standard_contract() -> None:
    """Tests that Pk2D model dictionaries have the expected contract."""
    spectra = {"ii": object(), "gi": object()}
    raw_backend_result = {"spectra": spectra, "metadata": {"source": "test"}}

    result = builder._pk2d_model(
        model_name="tatt",
        canonical_model_name="tatt",
        family="tatt",
        spectra=spectra,
        raw_backend_result=raw_backend_result,
    )

    assert result["model_name"] == "tatt"
    assert result["canonical_model_name"] == "tatt"
    assert result["family"] == "tatt"
    assert result["backend"] == "ccl"
    assert result["mode"] == "pk2d"
    assert result["ia_bias"] is None
    assert result["spectra"] == spectra
    assert not result["metadata"]["uses_native_ccl_ia"]
    assert result["metadata"]["uses_pk2d"]
    assert result["metadata"]["raw_backend_result"] == raw_backend_result


def test_build_la_model_delegates_to_la_bias_builder(monkeypatch) -> None:
    """Tests that LA models delegate to the LA native bias builder."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.la")

    def fake_make_ccl_la_ia_bias(z_arg, *, amplitude):
        np.testing.assert_allclose(z_arg, z)
        np.testing.assert_allclose(amplitude, np.full_like(z, 5.0))
        return ("la_z", "la_bias")

    monkeypatch.setattr(builder, "make_ccl_la_ia_bias", fake_make_ccl_la_ia_bias)

    result = builder._build_la_model(
        cosmo=object(),
        model_name="la",
        canonical_model_name="la",
        model_function=model_function,
        z=z,
        a_ia=5.0,
    )

    assert result["family"] == "la"
    assert result["mode"] == "native_bias"
    assert result["ia_bias"] == ("la_z", "la_bias")
    np.testing.assert_allclose(result["metadata"]["amplitude"], np.full_like(z, 5.0))


def test_build_nla_model_delegates_to_nla_bias_builder(monkeypatch) -> None:
    """Tests that NLA models delegate to the NLA native bias builder."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.nla")

    def fake_make_ccl_nla_ia_bias(z_arg, *, amplitude):
        np.testing.assert_allclose(z_arg, z)
        np.testing.assert_allclose(amplitude, np.full_like(z, 6.0))
        return ("nla_z", "nla_bias")

    monkeypatch.setattr(builder, "make_ccl_nla_ia_bias", fake_make_ccl_nla_ia_bias)

    result = builder._build_nla_model(
        cosmo=object(),
        model_name="nla",
        canonical_model_name="nla",
        model_function=model_function,
        z=z,
        a_ia=6.0,
    )

    assert result["family"] == "nla"
    assert result["mode"] == "native_bias"
    assert result["ia_bias"] == ("nla_z", "nla_bias")
    np.testing.assert_allclose(result["metadata"]["amplitude"], np.full_like(z, 6.0))


def test_build_tatt_model_delegates_to_tatt_spectra_builder(monkeypatch) -> None:
    """Tests that TATT models delegate to the TATT spectra builder."""
    cosmo = object()
    z = np.array([0.0, 0.5])
    spectra = {"ii": object(), "gi": object()}
    raw_result = {"spectra": spectra, "metadata": {"source": "tatt"}}

    def fake_make_ccl_tatt_power_spectra(cosmo_arg, z_arg, **kwargs):
        assert cosmo_arg is cosmo
        np.testing.assert_allclose(z_arg, z)
        assert kwargs == {"a1": 2.0}
        return raw_result

    monkeypatch.setattr(
        builder,
        "make_ccl_tatt_power_spectra",
        fake_make_ccl_tatt_power_spectra,
    )

    result = builder._build_tatt_model(
        cosmo=cosmo,
        model_name="tatt",
        canonical_model_name="tatt",
        model_function=_set_module(_amplitude_with_z, "plima.models.tatt"),
        z=z,
        a1=2.0,
    )

    assert result["family"] == "tatt"
    assert result["mode"] == "pk2d"
    assert result["ia_bias"] is None
    assert result["spectra"] == spectra
    assert result["metadata"]["raw_backend_result"] == raw_result


def test_build_halo_model_delegates_to_halo_spectra_builder(monkeypatch) -> None:
    """Tests that halo model IA delegates to the halo spectra builder."""
    cosmo = object()
    z = np.array([0.0, 0.5])
    spectra = {"ii": object(), "gi": object()}
    raw_result = {"spectra": spectra, "metadata": {"source": "halo_model"}}

    def fake_make_ccl_halo_model_power_spectra(cosmo_arg, z_arg, **kwargs):
        assert cosmo_arg is cosmo
        np.testing.assert_allclose(z_arg, z)
        assert kwargs == {"a1h": 0.01}
        return raw_result

    monkeypatch.setattr(
        builder,
        "make_ccl_halo_model_power_spectra",
        fake_make_ccl_halo_model_power_spectra,
    )

    result = builder._build_halo_model(
        cosmo=cosmo,
        model_name="halo_model",
        canonical_model_name="halo_model",
        model_function=_set_module(_amplitude_with_z, "plima.models.halo_model"),
        z=z,
        a1h=0.01,
    )

    assert result["family"] == "halo_model"
    assert result["mode"] == "pk2d"
    assert result["ia_bias"] is None
    assert result["spectra"] == spectra
    assert result["metadata"]["raw_backend_result"] == raw_result


@pytest.mark.parametrize(
    ("model_name", "canonical_model_name", "module_name", "expected_family"),
    [
        ("la", "la", "plima.models.la", "la"),
        ("nla", "nla", "plima.models.nla", "nla"),
        ("tatt", "tatt", "plima.models.tatt", "tatt"),
        ("halo_model", "halo_model", "plima.models.halo_model", "halo_model"),
    ],
)
def test_build_ccl_ia_model_dispatches_by_model_family(
    monkeypatch,
    model_name: str,
    canonical_model_name: str,
    module_name: str,
    expected_family: str,
) -> None:
    """Tests that the public builder dispatches using the model function family."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, module_name)
    calls: list[dict[str, Any]] = []

    def fake_get_model(name: str):
        assert name == model_name
        return model_function

    def fake_family_builder(**kwargs):
        calls.append(kwargs)
        return {
            "model_name": kwargs["model_name"],
            "canonical_model_name": kwargs["canonical_model_name"],
            "family": expected_family,
            "backend": "ccl",
            "mode": "native_bias",
            "ia_bias": ("z", "bias"),
            "spectra": {},
            "metadata": {},
        }

    monkeypatch.setattr(builder, "get_model", fake_get_model)
    monkeypatch.setattr(
        builder,
        "list_model_aliases",
        lambda: {canonical_model_name: ()},
    )
    monkeypatch.setitem(
        builder._CCL_FAMILY_BUILDERS,
        expected_family,
        fake_family_builder,
    )

    result = builder.build_ccl_ia_model(
        object(),
        f"  {model_name.upper()}  ",
        z,
        amplitude=1.0,
    )

    assert result["model_name"] == model_name
    assert result["canonical_model_name"] == canonical_model_name
    assert result["family"] == expected_family
    assert len(calls) == 1
    assert calls[0]["model_name"] == model_name
    assert calls[0]["canonical_model_name"] == canonical_model_name
    assert calls[0]["model_function"] is model_function
    np.testing.assert_allclose(calls[0]["z"], z)
    assert calls[0]["amplitude"] == 1.0


def test_build_ccl_ia_model_resolves_alias_to_canonical_name(monkeypatch) -> None:
    """Tests that aliases are recorded with their canonical model name."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.nla")

    def fake_family_builder(**kwargs):
        return {
            "model_name": kwargs["model_name"],
            "canonical_model_name": kwargs["canonical_model_name"],
            "family": "nla",
            "backend": "ccl",
            "mode": "native_bias",
            "ia_bias": ("z", "bias"),
            "spectra": {},
            "metadata": {},
        }

    monkeypatch.setattr(builder, "get_model", lambda name: model_function)
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"nla": ("nla_lf",)})
    monkeypatch.setitem(builder._CCL_FAMILY_BUILDERS, "nla", fake_family_builder)

    result = builder.build_ccl_ia_model(object(), "NLA_LF", z)

    assert result["model_name"] == "nla_lf"
    assert result["canonical_model_name"] == "nla"


def test_build_ccl_ia_model_raises_for_unknown_family(monkeypatch) -> None:
    """Tests that unknown model families raise a clear error."""
    model_function = _set_module(_amplitude_with_z, "plima.models.unknown")
    monkeypatch.setattr(builder, "get_model", lambda name: model_function)
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"unknown": ()})

    with pytest.raises(ValueError, match="Model family 'unknown' has no CCL builder."):
        builder.build_ccl_ia_model(object(), "unknown", np.array([0.0, 0.5]))


def test_build_ccl_ia_model_propagates_missing_model_error(monkeypatch) -> None:
    """Tests that missing registry entries keep the registry KeyError."""

    def fake_get_model(name: str):
        raise KeyError(name)

    monkeypatch.setattr(builder, "get_model", fake_get_model)
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {})

    with pytest.raises(KeyError):
        builder.build_ccl_ia_model(object(), "missing", np.array([0.0, 0.5]))


def test_normalize_model_name_rejects_non_string() -> None:
    """Tests that model names must be strings."""
    with pytest.raises(AttributeError):
        builder._normalize_model_name(123)


def test_canonical_model_name_strips_and_lowers_alias(monkeypatch) -> None:
    """Tests that canonical model lookup normalizes aliases."""
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"nla": ("nla_lf",)})

    assert builder._canonical_model_name("  NLA_LF  ") == "nla"


def test_model_family_handles_nested_module_family() -> None:
    """Tests that model family uses the final module component."""
    model_function = _set_module(_amplitude_with_z, "some.package.plima.models.nla")

    assert builder._model_family(model_function) == "nla"


def test_model_accepts_z_detects_keyword_only_z() -> None:
    """Tests that keyword only z arguments are detected."""

    def model(*, z):
        return z

    assert builder._model_accepts_z(model)


def test_model_accepts_z_detects_positional_only_z() -> None:
    """Tests that positional only z arguments are detected."""

    def model(z, /):
        return z

    assert builder._model_accepts_z(model)


def test_evaluate_amplitude_model_forwards_kwargs_when_z_supported() -> None:
    """Tests that amplitude model keyword arguments are forwarded with z."""
    z = np.array([0.0, 0.5])

    def model(z, *, scale: float, offset: float):
        return scale * np.asarray(z) + offset

    amplitude = builder._evaluate_amplitude_model(model, z, scale=2.0, offset=1.0)

    np.testing.assert_allclose(amplitude, np.array([1.0, 2.0]))


def test_evaluate_amplitude_model_forwards_kwargs_when_z_unsupported() -> None:
    """Tests that amplitude model keyword arguments are forwarded without z."""

    def model(*, scale: float, offset: float):
        return scale + offset

    amplitude = builder._evaluate_amplitude_model(model, np.array([0.0]), scale=2.0, offset=1.0)

    assert amplitude == 3.0


def test_native_bias_model_metadata_preserves_amplitude_identity() -> None:
    """Tests that native bias metadata preserves the amplitude array."""
    amplitude = np.array([1.0, 2.0])

    result = builder._native_bias_model(
        model_name="nla",
        canonical_model_name="nla",
        family="nla",
        ia_bias=("z", "bias"),
        amplitude=amplitude,
    )

    assert result["metadata"]["amplitude"] is amplitude


def test_pk2d_model_preserves_raw_backend_result_identity() -> None:
    """Tests that Pk2D model metadata preserves the raw backend result."""
    spectra = {"ii": object()}
    raw_backend_result = {"spectra": spectra, "metadata": {"source": "test"}}

    result = builder._pk2d_model(
        model_name="halo_model",
        canonical_model_name="halo_model",
        family="halo_model",
        spectra=spectra,
        raw_backend_result=raw_backend_result,
    )

    assert result["metadata"]["raw_backend_result"] is raw_backend_result


def test_build_la_model_omits_cosmo_for_native_bias_builder(monkeypatch) -> None:
    """Tests that LA native bias construction does not pass cosmology."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.la")

    def fake_make_ccl_la_ia_bias(*args, **kwargs):
        assert len(args) == 1
        np.testing.assert_allclose(args[0], z)
        assert set(kwargs) == {"amplitude"}
        return ("z", "bias")

    monkeypatch.setattr(builder, "make_ccl_la_ia_bias", fake_make_ccl_la_ia_bias)

    result = builder._build_la_model(
        cosmo=object(),
        model_name="la",
        canonical_model_name="la",
        model_function=model_function,
        z=z,
    )

    assert result["mode"] == "native_bias"


def test_build_nla_model_omits_cosmo_for_native_bias_builder(monkeypatch) -> None:
    """Tests that NLA native bias construction does not pass cosmology."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.nla")

    def fake_make_ccl_nla_ia_bias(*args, **kwargs):
        assert len(args) == 1
        np.testing.assert_allclose(args[0], z)
        assert set(kwargs) == {"amplitude"}
        return ("z", "bias")

    monkeypatch.setattr(builder, "make_ccl_nla_ia_bias", fake_make_ccl_nla_ia_bias)

    result = builder._build_nla_model(
        cosmo=object(),
        model_name="nla",
        canonical_model_name="nla",
        model_function=model_function,
        z=z,
    )

    assert result["mode"] == "native_bias"


def test_build_tatt_model_ignores_model_function(monkeypatch) -> None:
    """Tests that TATT backend construction uses backend spectra directly."""
    cosmo = object()
    z = np.array([0.0, 0.5])
    spectra = {"ii": object(), "gi": object()}
    raw_result = {"spectra": spectra, "metadata": {}}

    monkeypatch.setattr(
        builder,
        "make_ccl_tatt_power_spectra",
        lambda cosmo_arg, z_arg, **kwargs: raw_result,
    )

    result = builder._build_tatt_model(
        cosmo=cosmo,
        model_name="tatt",
        canonical_model_name="tatt",
        model_function=_set_module(_amplitude_without_z, "plima.models.tatt"),
        z=z,
    )

    assert result["spectra"] == spectra


def test_build_halo_model_ignores_model_function(monkeypatch) -> None:
    """Tests that halo backend construction uses backend spectra directly."""
    cosmo = object()
    z = np.array([0.0, 0.5])
    spectra = {"ii": object(), "gi": object()}
    raw_result = {"spectra": spectra, "metadata": {}}

    monkeypatch.setattr(
        builder,
        "make_ccl_halo_model_power_spectra",
        lambda cosmo_arg, z_arg, **kwargs: raw_result,
    )

    result = builder._build_halo_model(
        cosmo=cosmo,
        model_name="halo_model",
        canonical_model_name="halo_model",
        model_function=_set_module(_amplitude_without_z, "plima.models.halo_model"),
        z=z,
    )

    assert result["spectra"] == spectra


def test_build_ccl_ia_model_uses_family_not_canonical_name(monkeypatch) -> None:
    """Tests that dispatch follows the model function family."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.nla")

    def fake_family_builder(**kwargs):
        return {
            "model_name": kwargs["model_name"],
            "canonical_model_name": kwargs["canonical_model_name"],
            "family": "nla",
            "backend": "ccl",
            "mode": "native_bias",
            "ia_bias": ("z", "bias"),
            "spectra": {},
            "metadata": {},
        }

    monkeypatch.setattr(builder, "get_model", lambda name: model_function)
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"some_alias": ()})
    monkeypatch.setitem(builder._CCL_FAMILY_BUILDERS, "nla", fake_family_builder)

    result = builder.build_ccl_ia_model(object(), "some_alias", z)

    assert result["family"] == "nla"


def test_build_ccl_ia_model_propagates_builder_kwargs(monkeypatch) -> None:
    """Tests that public builder keyword arguments reach the family builder."""
    z = np.array([0.0, 0.5])
    model_function = _set_module(_amplitude_with_z, "plima.models.la")
    calls: list[dict[str, Any]] = []

    def fake_family_builder(**kwargs):
        calls.append(kwargs)
        return {
            "model_name": kwargs["model_name"],
            "canonical_model_name": kwargs["canonical_model_name"],
            "family": "la",
            "backend": "ccl",
            "mode": "native_bias",
            "ia_bias": ("z", "bias"),
            "spectra": {},
            "metadata": {},
        }

    monkeypatch.setattr(builder, "get_model", lambda name: model_function)
    monkeypatch.setattr(builder, "list_model_aliases", lambda: {"la": ()})
    monkeypatch.setitem(builder._CCL_FAMILY_BUILDERS, "la", fake_family_builder)

    builder.build_ccl_ia_model(
        object(),
        "la",
        z,
        a_ia=2.0,
        eta=1.5,
    )

    assert calls[0]["a_ia"] == 2.0
    assert calls[0]["eta"] == 1.5
