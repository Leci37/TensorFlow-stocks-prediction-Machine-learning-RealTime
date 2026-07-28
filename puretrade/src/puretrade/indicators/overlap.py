"""Familia Overlap (``olap_*``). ~24 columnas. Núcleo puro.

Incluye el ejemplo canónico de dependencia padre->hijo: las bandas de Bollinger
y sus 'crash points' derivados, que el motor calcula en orden topológico.
"""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.rolling import rolling_std, sma


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
