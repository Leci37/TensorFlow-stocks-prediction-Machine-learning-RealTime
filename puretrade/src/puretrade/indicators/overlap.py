"""Familia Overlap (``olap_*``). ~24 columnas. Núcleo puro.

Incluye el ejemplo canónico de dependencia padre->hijo: las bandas de Bollinger
y sus 'crash points' derivados, que el motor calcula en orden topológico.
"""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.rolling import highest, lowest, rolling_std, sma


@register(key="olap_BBAND",
          outputs=("olap_BBAND_UPPER", "olap_BBAND_MIDDLE", "olap_BBAND_LOWER", "olap_BBAND_dif"),
          inputs=("close",), family=Family.OVERLAP, nature=Nature.ROLLING, warmup=20)
def _bbands(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(df["close"], n)
    dev = rolling_std(df["close"], n) * k
    upper, lower = mid + dev, mid - dev
    return pd.DataFrame({
        "olap_BBAND_UPPER": upper,
        "olap_BBAND_MIDDLE": mid,
        "olap_BBAND_LOWER": lower,
        "olap_BBAND_dif": upper - lower,
    })


@register(key="olap_BBAND_UPPER_crash", outputs=("olap_BBAND_UPPER_crash",),
          inputs=("close",), depends_on=("olap_BBAND",),
          family=Family.OVERLAP, nature=Nature.DERIVED, warmup=20)
def _bband_upper_crash(df: pd.DataFrame) -> pd.DataFrame:
    # cruce del precio por encima de la banda superior (señal 0/1)
    crossed = (df["close"] > df["olap_BBAND_UPPER"]).astype("int8")
    return pd.DataFrame({"olap_BBAND_UPPER_crash": crossed})


@register(key="olap_MIDPOINT", outputs=("olap_MIDPOINT",), inputs=("close",),
          family=Family.OVERLAP, nature=Nature.ROLLING, warmup=14)
def _midpoint(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"olap_MIDPOINT": (highest(df["close"], n) + lowest(df["close"], n)) / 2})


@register(key="olap_MIDPRICE", outputs=("olap_MIDPRICE",), inputs=("high", "low"),
          family=Family.OVERLAP, nature=Nature.ROLLING, warmup=14)
def _midprice(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"olap_MIDPRICE": (highest(df["high"], n) + lowest(df["low"], n)) / 2})

def _session_key(index):
    """Clave de sesión (día natural) si el índice es datetime; si no, sesión única."""
    if isinstance(index, pd.DatetimeIndex):
        return index.normalize()
    return pd.Index(["all"] * len(index))


@register(key="olap_VWAP", outputs=("olap_VWAP",), inputs=("high", "low", "close", "volume"),
          family=Family.OVERLAP, nature=Nature.CUMULATIVE, warmup=1)
def _vwap(df: pd.DataFrame) -> pd.DataFrame:
    # VWAP anclado a la sesión (se reinicia cada día), como el VWAP de TradingView.
    tp = (df["high"] + df["low"] + df["close"]) / 3
    key = _session_key(df.index)
    cum_pv = (tp * df["volume"]).groupby(key).cumsum()
    cum_v = df["volume"].groupby(key).cumsum()
    return pd.DataFrame({"olap_VWAP": cum_pv / cum_v})


@register(key="pivots",
          outputs=("piv_P", "piv_R1", "piv_S1", "piv_R2", "piv_S2", "piv_R3", "piv_S3"),
          inputs=("high", "low", "close"), family=Family.OVERLAP, nature=Nature.ROLLING, warmup=1)
def _pivots(df: pd.DataFrame) -> pd.DataFrame:
    # Pivot Points clásicos (traditional): usan H/L/C de la SESIÓN ANTERIOR.
    key = _session_key(df.index)
    daily = pd.DataFrame({"H": df["high"], "L": df["low"], "C": df["close"]}).groupby(key).agg(
        H=("H", "max"), L=("L", "min"), C=("C", "last"))
    prev = daily.shift(1)
    P = (prev["H"] + prev["L"] + prev["C"]) / 3
    piv = pd.DataFrame({
        "piv_P": P,
        "piv_R1": 2 * P - prev["L"], "piv_S1": 2 * P - prev["H"],
        "piv_R2": P + (prev["H"] - prev["L"]), "piv_S2": P - (prev["H"] - prev["L"]),
        "piv_R3": prev["H"] + 2 * (P - prev["L"]), "piv_S3": prev["L"] - 2 * (prev["H"] - P),
    })
    # difunde el valor de la sesión a cada barra intradía
    out = piv.reindex(pd.Index(key)).set_axis(df.index)
    return out

# TODO (Hilbert/DSP, van a _optional/talib_bridge): SAR, SAREXT, MAMA/FAMA, HT_TRENDLINE.
