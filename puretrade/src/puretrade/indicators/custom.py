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
from ..math.ewm import ema, pyti_ema
from ..math.rolling import highest, lowest, rma, rolling_std, sma, true_range, wma
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

# ------------------------- resto de ti_* (paridad con py_ti) -------------------------

@register(key="ti_donchian", outputs=("ti_donchian_lower_20", "ti_donchian_center_20", "ti_donchian_upper_20"),
          inputs=("high", "low"), family=Family.CUSTOM, nature=Nature.ROLLING, warmup=20)
def _donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    up, lo = highest(df["high"], n), lowest(df["low"], n)
    return pd.DataFrame({f"ti_donchian_lower_{n}": lo, f"ti_donchian_center_{n}": (up + lo) / 2,
                         f"ti_donchian_upper_{n}": up})


def _vortex(df, n):
    tr = true_range(df["high"], df["low"], df["close"]).rolling(n).sum()
    vmp = (df["high"] - df["low"].shift(1)).abs().rolling(n).sum()
    vmm = (df["low"] - df["high"].shift(1)).abs().rolling(n).sum()
    return vmp / tr, vmm / tr


@register(key="ti_vortex_5", outputs=("ti_vortex_pos_5", "ti_vortex_neg_5"),
          inputs=("high", "low", "close"), family=Family.CUSTOM, nature=Nature.ROLLING, warmup=5)
def _vortex5(df: pd.DataFrame) -> pd.DataFrame:
    p, m = _vortex(df, 5)
    return pd.DataFrame({"ti_vortex_pos_5": p, "ti_vortex_neg_5": m})


@register(key="ti_vortex_14", outputs=("ti_vortex_pos_14", "ti_vortex_neg_14"),
          inputs=("high", "low", "close"), family=Family.CUSTOM, nature=Nature.ROLLING, warmup=14)
def _vortex14(df: pd.DataFrame) -> pd.DataFrame:
    p, m = _vortex(df, 14)
    return pd.DataFrame({"ti_vortex_pos_14": p, "ti_vortex_neg_14": m})


@register(key="ti_choppiness_14", outputs=("ti_choppiness_14",), inputs=("high", "low", "close"),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=14)
def _choppiness(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tr = true_range(df["high"], df["low"], df["close"]).rolling(n).sum()
    rng = highest(df["high"], n) - lowest(df["low"], n)
    return pd.DataFrame({"ti_choppiness_14": 100 * np.log10(tr / rng) / np.log10(n)})


@register(key="ti_coppock_14_11_10", outputs=("ti_coppock_14_11_10",), inputs=("close",),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=24)
def _coppock(df: pd.DataFrame) -> pd.DataFrame:
    roc = (df["close"].pct_change(14) + df["close"].pct_change(11)) * 100
    return pd.DataFrame({"ti_coppock_14_11_10": wma(roc, 10)})


@register(key="ti_mass_index_9_25", outputs=("ti_mass_index_9_25",), inputs=("high", "low"),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=34)
def _mass_index(df: pd.DataFrame) -> pd.DataFrame:
    e1 = pyti_ema(df["high"] - df["low"], 9)
    mass = e1 / pyti_ema(e1, 9)
    return pd.DataFrame({"ti_mass_index_9_25": mass.rolling(25).sum()})


@register(key="ti_chaikin_10_3", outputs=("ti_chaikin_10_3",),
          inputs=("high", "low", "close", "volume"), family=Family.CUSTOM,
          nature=Nature.CUMULATIVE, warmup=10)
def _chaikin(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    adl = ((((c - l) - (h - c)) / (h - l)) * df["volume"]).cumsum()
    return pd.DataFrame({"ti_chaikin_10_3": pyti_ema(adl, 3) - pyti_ema(adl, 10)})


@register(key="ti_force_index_13", outputs=("ti_force_index_13",), inputs=("close", "volume"),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=13)
def _force_index(df: pd.DataFrame, n: int = 13) -> pd.DataFrame:
    return pd.DataFrame({"ti_force_index_13": pyti_ema(df["close"].diff().fillna(0) * df["volume"], n)})


@register(key="ti_hma_20", outputs=("ti_hma_20",), inputs=("close",),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=20)
def _hma(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"ti_hma_20": wma(2 * wma(df["close"], n // 2) - wma(df["close"], n), int(n ** 0.5))})


@register(key="ti_keltner", outputs=("ti_kelt_20_lower", "ti_kelt_20_upper"),
          inputs=("high", "low", "close"), family=Family.CUSTOM, nature=Nature.ROLLING, warmup=20)
def _keltner(df: pd.DataFrame, n: int = 20, factor: float = 2.0) -> pd.DataFrame:
    base = sma(df["close"], n)
    atr = sma(true_range(df["high"], df["low"], df["close"]), n)
    return pd.DataFrame({"ti_kelt_20_lower": base - factor * atr, "ti_kelt_20_upper": base + factor * atr})


@register(key="ti_supertrend_20", outputs=("ti_supertrend_20",), inputs=("high", "low", "close"),
          family=Family.CUSTOM, nature=Nature.ROLLING, warmup=20)
def _supertrend(df: pd.DataFrame, n: int = 20, factor: float = 2.0) -> pd.DataFrame:
    atr = sma(true_range(df["high"], df["low"], df["close"]), n)
    hl_avg = (df["high"] + df["low"]) / 2
    ub = (hl_avg + factor * atr).to_numpy()
    lb = (hl_avg - factor * atr).to_numpy()
    close = df["close"].to_numpy()
    length = len(close)
    fub = np.zeros(length)
    flb = np.zeros(length)
    st = np.full(length, np.nan)
    for i in range(n, length):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if st[i - 1] == fub[i - 1] and close[i] <= fub[i]:
            st[i] = fub[i]
        elif st[i - 1] == fub[i - 1] and close[i] > fub[i]:
            st[i] = flb[i]
        elif st[i - 1] == flb[i - 1] and close[i] >= flb[i]:
            st[i] = flb[i]
        elif st[i - 1] == flb[i - 1] and close[i] < flb[i]:
            st[i] = fub[i]
        else:
            st[i] = 0.0
    return pd.DataFrame({"ti_supertrend_20": st}, index=df.index)

# Deferidos a propósito: ti_acc_dist (py_ti genera None) y ti_ease_of_movement_14
# (py_ti lo normaliza con Volume.max() global -> NO es point-in-time; se dejará una
# versión causal cuando se decida cómo tratar esa fuga).
