"""Familia Candle (``cdl_*``). ~61 patrones. Geometría pura de OHLC — 100% PIT safe.

Los patrones de vela NO necesitan TA-Lib: son comparaciones geométricas del
cuerpo/mechas de una o pocas velas. Salida categórica al estilo TA-Lib
(+100 alcista, -100 bajista, 0 sin patrón).
"""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature, OutputType
from ..core.registry import register


@register(key="cdl_DOJI", outputs=("cdl_DOJI",), inputs=("open", "high", "low", "close"),
          family=Family.CANDLE, nature=Nature.PATTERN, output=OutputType.CATEGORICAL, warmup=1)
def _doji(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, pd.NA)
    is_doji = (body <= 0.1 * rng).fillna(False)
    return pd.DataFrame({"cdl_DOJI": is_doji.astype("int16") * 100})


@register(key="cdl_ENGULFING", outputs=("cdl_ENGULFING",), inputs=("open", "high", "low", "close"),
          family=Family.CANDLE, nature=Nature.PATTERN, output=OutputType.CATEGORICAL, warmup=2)
def _engulfing(df: pd.DataFrame) -> pd.DataFrame:
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    prev_bear, prev_bull = pc < po, pc > po
    bull = prev_bear & (c > o) & (c >= po) & (o <= pc)
    bear = prev_bull & (c < o) & (c <= po) & (o >= pc)
    out = pd.Series(0, index=df.index, dtype="int16")
    out[bull] = 100
    out[bear] = -100
    return pd.DataFrame({"cdl_ENGULFING": out})

# TODO: migrar los ~59 patrones restantes (todos geometría OHLC pura).
