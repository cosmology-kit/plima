"""Unit tests for ``plima.utils.types``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plima.utils.types import FloatArray, InputArray, ModelFunction, ParameterDict


def test_float_array_alias() -> None:
    """Tests that FloatArray aliases a NumPy float array."""
    assert FloatArray == NDArray[np.float64]


def test_input_array_alias() -> None:
    """Tests that InputArray aliases ArrayLike."""
    assert InputArray == ArrayLike


def test_parameter_dict_alias() -> None:
    """Tests that ParameterDict aliases a parameter mapping."""
    assert ParameterDict == Mapping[str, Any]


def test_model_function_alias() -> None:
    """Tests that ModelFunction aliases a model callable."""
    assert ModelFunction == Callable[..., FloatArray]