"""Familia Custom (``ti_*``): indicadores propios / no-TA-Lib.

Incluye el KONCORDE (Blai5), reimplementado fielmente al Pine Script v2 original
(no a la versión heredada del proyecto, que tenía dos errores: usaba NVI para
calcular el PVI, y un multiplicador distinto en PVI/NVI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.ewm import ema
from ..math.rolling import highest, lowest, rma, rolling_std, sma
from ..math.signal import crossover_signal


def _pine_vi(close: pd.Series, volume: pd.Series, positive: bool) -> pd.Series:
    """PVI/NVI al estilo Blai5/Pine: incremento = ROC * volumen (no * VI previo)."""
    cl = close.to_numpy(dtype="float64")
    vo = volume.to_numpy(dtype="float64")
    out = np.zeros(len(cl))
    for i in range(1, len(cl)):
        cond = vo[i] > vo[i - 1] if positive else vo[i] < vo[i - 1]
        if cond and cl[i - 1] != 0:
            out[i] = out[i - 1] + ((cl[i] - cl[i - 1]) / cl[i - 1]) * vo[i]
        else:
            out[i] = out[i - 1]
    return pd.Series(out, index=close.index)


def _rsi(src: pd.Series, n: int = 14) -> pd.Series:
    delta = src.diff()
    rs = rma(delta.clip(lower=0), n) / rma(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + rs)


@register(
    key="ti_konkorde",
    outputs=(
        "ti_konk_blue", "ti_konk_brown", "ti_konk_green", "ti_konk_avg", "ti_konk_rest",
        "ti_konk_bl_avg_crash", "ti_konk_gre_avg_crash", "ti_konk_gre_bl_crash",
    ),
    inputs=("open", "high", "low", "close", "volume"),
    family=Family.CUSTOM, nature=Nature.CUMULATIVE, warmup=90,
)
def _konkorde(df: pd.DataFrame, m: int = 15) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    tprice = (o + h + l + c) / 4

    # --- oscilador PVI (verde) ---
    pvi = _pine_vi(c, v, positive=True)
    pvim = ema(pvi, m)
    oscp = (pvi - pvim) * 100 / (highest(pvim, 90) - lowest(pvim, 90))

    # --- oscilador NVI (azul) ---
    nvi = _pine_vi(c, v, positive=False)
    nvim = ema(nvi, m)
    azul = (nvi - nvim) * 100 / (highest(nvim, 90) - lowest(nvim, 90))

    # --- Money Flow, Bollinger oscillator, RSI, estocástico ---
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    xmf = 100 * pos / (pos + neg)

    basis = sma(tprice, 25)
    dev = 2.0 * rolling_std(tprice, 25)
    boll_osc = ((tprice - basis) / (2 * dev)) * 100

    xrsi = _rsi(tprice, 14)

    ll, hh = lowest(l, 21), highest(h, 21)
    stoc = sma(100 * (tprice - ll) / (hh - ll), 3)

    marron = (xrsi + xmf + boll_osc + (stoc / 3)) / 2
    verde = marron + oscp
    media = ema(marron, m)

    out = pd.DataFrame({
        "ti_konk_blue": azul,
        "ti_konk_brown": marron,
        "ti_konk_green": verde,
        "ti_konk_avg": media,
        "ti_konk_rest": azul - verde,
    })
    out["ti_konk_bl_avg_crash"] = crossover_signal(azul, media)
    out["ti_konk_gre_avg_crash"] = crossover_signal(verde, media)
    out["ti_konk_gre_bl_crash"] = crossover_signal(verde, azul)
    return out

# TODO: resto de ti_* (donchian, keltner, supertrend, vortex, choppiness...).
