"""Primitivas de ventana trasera (rolling). Puro pandas — todas point-in-time."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def rma(s: pd.Series, n: int) -> pd.Series:
    """Media móvil de Wilder, idéntica a TA-Lib.

    Semilla = media simple de los primeros ``n`` valores válidos (no el primer
    valor, como haría ``ewm``), y a partir de ahí recursión ``(prev*(n-1)+x)/n``.
    Esta inicialización es la que usa TA-Lib en RSI/ATR/ADX; ``ewm`` no coincide.
    """
    x = s.to_numpy(dtype="float64")
    out = np.full(x.shape, np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0]
    seed = start + n - 1
    out[seed] = np.mean(x[start:start + n])
    for i in range(seed + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return pd.Series(out, index=s.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = np.nan   # sin cierre previo el rango verdadero no está definido (como TA-Lib)
    return tr


def dmi_smooth(s: pd.Series, n: int) -> pd.Series:
    """Suavizado de Wilder de TA-Lib para DMI/ADX (+DM, -DM, TR).

    A diferencia de ``rma``, siembra con la SUMA de los primeros ``n-1`` valores
    y arranca la recursión ``s = s - s/n + x`` incluyendo el actual. Devuelve la
    suma suavizada (no la media): así el cociente +DM/TR reproduce TA-Lib.
    """
    a = s.to_numpy(dtype="float64")
    out = np.full(a.shape, np.nan)
    if len(a) <= n:
        return pd.Series(out, index=s.index)
    acc = np.nansum(a[1:n])
    for i in range(n, len(a)):
        acc = acc - acc / n + a[i]
        out[i] = acc
    return pd.Series(out, index=s.index)


def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()
