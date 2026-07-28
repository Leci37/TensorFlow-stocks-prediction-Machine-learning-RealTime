"""Transformaciones de estacionariedad. Puro pandas/numpy.

Un modelo ML no debe entrenarse sobre niveles de precio no estacionarios. Aquí
se convierten en retornos, diferencias o diferenciación fraccionaria.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods)


def log_returns(s: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(s).diff(periods)


def diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.diff(periods)


def _fracdiff_weights(d: float, size: int, tau: float = 1e-4) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < tau:
            break
        w.append(wk)
    return np.array(w[::-1])


def fracdiff(s: pd.Series, d: float = 0.4, tau: float = 1e-4) -> pd.Series:
    """Diferenciación fraccionaria (López de Prado): estacionariza conservando memoria."""
    w = _fracdiff_weights(d, len(s), tau)
    width = len(w)
    out = pd.Series(index=s.index, dtype="float64")
    vals = s.values
    for i in range(width - 1, len(s)):
        out.iloc[i] = np.dot(w, vals[i - width + 1 : i + 1])
    return out
