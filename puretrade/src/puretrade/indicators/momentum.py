"""Familia Momentum (``mtum_*``). ~88 columnas. Núcleo puro (migración en curso)."""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.ewm import ema
from ..math.rolling import rma


@register(key="mtum_RSI", outputs=("mtum_RSI",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _rsi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = rma(up, n) / rma(down, n)
    return pd.DataFrame({"mtum_RSI": 100 - 100 / (1 + rs)})


@register(key="mtum_MOM", outputs=("mtum_MOM",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _mom(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_MOM": df["close"].diff(n)})


@register(key="mtum_ROC", outputs=("mtum_ROC",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _roc(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ROC": df["close"].pct_change(n) * 100})


@register(key="mtum_MACD",
          outputs=("mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list"),
          inputs=("close",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=26)
def _macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    macd = ema(df["close"], fast) - ema(df["close"], slow)
    signal = ema(macd, sig)
    return pd.DataFrame({
        "mtum_MACD": macd,
        "mtum_MACD_signal": signal,
        "mtum_MACD_list": macd - signal,  # histograma
    })

# TODO: ADX/STOCH/CCI/AO/... (CCI/APO/PPO están marcados no-PIT en el catálogo).
