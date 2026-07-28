"""Señales de cruce entre series. Puro pandas — point-in-time."""
from __future__ import annotations

import pandas as pd


def crossover_signal(a: pd.Series, b: pd.Series, carry: int = 1) -> pd.Series:
    """Detecta cruces de ``a`` sobre/bajo ``b`` (los 'crash points' del proyecto).

    +1 cuando ``a`` cruza por encima de ``b``, -1 cuando cruza por debajo.
    Con ``carry>0`` la señal se propaga ``carry`` filas hacia delante (mira solo
    al pasado, sigue siendo point-in-time).
    """
    diff = a - b
    out = pd.Series(0, index=a.index, dtype="int16")
    out[(diff >= 0) & (diff.shift(1) < 0)] = 1
    out[(diff <= 0) & (diff.shift(1) > 0)] = -1
    if carry > 0:
        out[out.shift(carry) == 1] = 1
        out[out.shift(carry) == -1] = -1
    return out
