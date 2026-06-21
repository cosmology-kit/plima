"""Shared typing aliases for PLIMA."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
InputArray = ArrayLike
ParameterDict = Mapping[str, Any]

ModelFunction = Callable[..., FloatArray]
