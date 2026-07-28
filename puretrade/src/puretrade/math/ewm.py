"""Medias exponenciales, idénticas a TA-Lib. Puro numpy/pandas.

TA-Lib siembra la EMA con la SMA de los primeros ``n`` valores (no con el primer
valor, como hace ``pandas.ewm``). Se replica esa inicialización para paridad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    x = s.to_numpy(dtype="float64")
    out = np.full(x.shape, np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0]
    seed = start + n - 1
    out[seed] = np.mean(x[start:start + n])
    k = 2.0 / (n + 1.0)
    for i in range(seed + 1, len(x)):
        out[i] = k * x[i] + (1.0 - k) * out[i - 1]
    return pd.Series(out, index=s.index)


def ema_seed_at(s: pd.Series, n: int, seed_idx: int) -> pd.Series:
    """EMA cuya semilla (SMA de ``n`` valores) se coloca en ``seed_idx``.

    Necesario para replicar el MACD de TA-Lib, que alinea la EMA rápida al índice
    de arranque de la lenta en vez de a su propio periodo.
    """
    x = s.to_numpy(dtype="float64")
    out = np.full(x.shape, np.nan)
    if seed_idx - n + 1 < 0 or seed_idx >= len(x):
        return pd.Series(out, index=s.index)
    out[seed_idx] = np.mean(x[seed_idx - n + 1:seed_idx + 1])
    k = 2.0 / (n + 1.0)
    for i in range(seed_idx + 1, len(x)):
        out[i] = k * x[i] + (1.0 - k) * out[i - 1]
    return pd.Series(out, index=s.index)


def dema(s: pd.Series, n: int) -> pd.Series:
    e = ema(s, n)
    return 2 * e - ema(e, n)


def tema(s: pd.Series, n: int) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def t3(s: pd.Series, n: int, vfactor: float = 0.7) -> pd.Series:
    """T3 de Tillson, idéntica a TA-Lib (vfactor por defecto 0.7)."""
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    v = vfactor
    c1 = -v ** 3
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
