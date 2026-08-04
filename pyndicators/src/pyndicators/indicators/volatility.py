"""Familia Volatility (``vola_*``). ~16 columnas. Núcleo puro."""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.rolling import rma, true_range


@register(key="vola_TRANGE", outputs=("vola_TRANGE",), inputs=("high", "low", "close"),
          family=Family.VOLATILITY, nature=Nature.ROLLING, warmup=1)
def _trange(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"vola_TRANGE": true_range(df["high"], df["low"], df["close"])})


@register(key="vola_ATR", outputs=("vola_ATR",), inputs=("high", "low", "close"),
          depends_on=("vola_TRANGE",),
          family=Family.VOLATILITY, nature=Nature.DERIVED, warmup=14)
def _atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"vola_ATR": rma(df["vola_TRANGE"], n)})


@register(key="vola_NATR", outputs=("vola_NATR",), inputs=("high", "low", "close"),
          depends_on=("vola_ATR",),
          family=Family.VOLATILITY, nature=Nature.DERIVED, warmup=14)
def _natr(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"vola_NATR": 100 * df["vola_ATR"] / df["close"]})
