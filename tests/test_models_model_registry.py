"""Unit tests for ``plima.models.model_registry``."""

from __future__ import annotations

import pytest

from plima.models import model_registry


@pytest.fixture(autouse=True)
def reset_model_registry(monkeypatch) -> None:
    """Resets the model registry state for each test."""
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {})
    monkeypatch.setattr(model_registry, "MODEL_ALIASES", {})
    monkeypatch.setattr(model_registry, "_DISCOVERED", False)


def test_register_ia_model_registers_canonical_name() -> None:
    """Tests that a model function is registered by its canonical name."""

    def model_function(z):
        return z

    returned_function = model_registry.register_ia_model("nla")(model_function)

    assert returned_function is model_function
    assert model_registry.MODEL_REGISTRY == {
        "nla": model_function,
    }
    assert model_registry.MODEL_ALIASES == {
        "nla": (),
    }


def test_register_ia_model_registers_aliases() -> None:
    """Tests that aliases point to the same registered model function."""

    def model_function(z):
        return z

    model_registry.register_ia_model(
        "nla",
        aliases=("linear_alignment", "la"),
    )(model_function)

    assert model_registry.MODEL_REGISTRY == {
        "nla": model_function,
        "linear_alignment": model_function,
        "la": model_function,
    }
    assert model_registry.MODEL_ALIASES == {
        "nla": ("linear_alignment", "la"),
    }


def test_register_ia_model_rejects_duplicate_canonical_name() -> None:
    """Tests that duplicate canonical names are rejected."""

    def first_model(z):
        return z

    def second_model(z):
        return z

    model_registry.register_ia_model("nla")(first_model)

    with pytest.raises(ValueError, match="Model 'nla' is already registered."):
        model_registry.register_ia_model("nla")(second_model)


def test_register_ia_model_rejects_duplicate_alias() -> None:
    """Tests that duplicate aliases are rejected."""

    def first_model(z):
        return z

    def second_model(z):
        return z

    model_registry.register_ia_model(
        "nla",
        aliases=("la",),
    )(first_model)

    with pytest.raises(ValueError, match="Model 'la' is already registered."):
        model_registry.register_ia_model(
            "tatt",
            aliases=("la",),
        )(second_model)


def test_register_ia_model_rejects_alias_matching_existing_name() -> None:
    """Tests that aliases cannot reuse existing canonical names."""

    def first_model(z):
        return z

    def second_model(z):
        return z

    model_registry.register_ia_model("nla")(first_model)

    with pytest.raises(ValueError, match="Model 'nla' is already registered."):
        model_registry.register_ia_model(
            "tatt",
            aliases=("nla",),
        )(second_model)


def test_discover_models_imports_model_modules_once(monkeypatch) -> None:
    """Tests that discovery imports model modules only once."""
    calls = []

    def fake_import_model_modules() -> None:
        calls.append("called")

    monkeypatch.setattr(
        model_registry,
        "import_model_modules",
        fake_import_model_modules,
    )

    model_registry.discover_models()
    model_registry.discover_models()

    assert calls == ["called"]
    assert model_registry._DISCOVERED is True


def test_get_model_returns_registered_model(monkeypatch) -> None:
    """Tests that a registered model can be retrieved by name."""

    def model_function(z):
        return z

    monkeypatch.setattr(model_registry, "_DISCOVERED", True)
    model_registry.MODEL_REGISTRY["nla"] = model_function

    assert model_registry.get_model("nla") is model_function


def test_get_model_triggers_discovery(monkeypatch) -> None:
    """Tests that getting a model triggers discovery before lookup."""

    def model_function(z):
        return z

    def fake_import_model_modules() -> None:
        model_registry.MODEL_REGISTRY["nla"] = model_function
        model_registry.MODEL_ALIASES["nla"] = ()

    monkeypatch.setattr(
        model_registry,
        "import_model_modules",
        fake_import_model_modules,
    )

    assert model_registry.get_model("nla") is model_function
    assert model_registry._DISCOVERED is True


def test_get_model_raises_helpful_error_for_unknown_model(monkeypatch) -> None:
    """Tests that unknown models raise an error listing available models."""

    def model_function(z):
        return z

    monkeypatch.setattr(model_registry, "_DISCOVERED", True)
    model_registry.MODEL_REGISTRY["nla"] = model_function
    model_registry.MODEL_REGISTRY["tatt"] = model_function
    model_registry.MODEL_ALIASES["nla"] = ()
    model_registry.MODEL_ALIASES["tatt"] = ()

    with pytest.raises(
        KeyError,
        match="Unknown model 'missing'. Available models: nla, tatt.",
    ):
        model_registry.get_model("missing")


def test_list_models_includes_aliases_by_default(monkeypatch) -> None:
    """Tests that list models returns names and aliases by default."""

    def model_function(z):
        return z

    monkeypatch.setattr(model_registry, "_DISCOVERED", True)
    model_registry.MODEL_REGISTRY.update(
        {
            "tatt": model_function,
            "nla": model_function,
            "la": model_function,
        },
    )
    model_registry.MODEL_ALIASES.update(
        {
            "nla": ("la",),
            "tatt": (),
        },
    )

    assert model_registry.list_models() == ("la", "nla", "tatt")


def test_list_models_can_exclude_aliases(monkeypatch) -> None:
    """Tests that list models can return only canonical model names."""

    def model_function(z):
        return z

    monkeypatch.setattr(model_registry, "_DISCOVERED", True)
    model_registry.MODEL_REGISTRY.update(
        {
            "tatt": model_function,
            "nla": model_function,
            "la": model_function,
        },
    )
    model_registry.MODEL_ALIASES.update(
        {
            "nla": ("la",),
            "tatt": (),
        },
    )

    assert model_registry.list_models(include_aliases=False) == ("nla", "tatt")


def test_list_model_aliases_returns_sorted_copy(monkeypatch) -> None:
    """Tests that model aliases are returned as a sorted copy."""
    monkeypatch.setattr(model_registry, "_DISCOVERED", True)
    model_registry.MODEL_ALIASES.update(
        {
            "tatt": ("tidal_alignment_tidal_torquing",),
            "nla": ("la",),
        },
    )

    aliases = model_registry.list_model_aliases()

    assert aliases == {
        "nla": ("la",),
        "tatt": ("tidal_alignment_tidal_torquing",),
    }

    aliases["nla"] = ("changed",)

    assert model_registry.MODEL_ALIASES == {
        "tatt": ("tidal_alignment_tidal_torquing",),
        "nla": ("la",),
    }
