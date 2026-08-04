"""Familia Trend (``tend_*``) + indicadores de tendencia populares.

Incluye ICHIMOKU (Ichimoku Kinko Hyo), uno de los indicadores más usados en
TradingView. Se exponen las líneas como features point-in-time: cada valor en t
se calcula solo con datos disponibles en t (no se desplazan las Senkou al futuro,
y se omite Chikou por ser look-ahead).
"""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.rolling import highest, lowest


def _midchannel(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    return (highest(high, n) + lowest(low, n)) / 2


@register(key="ichimoku",
          outputs=("ichi_tenkan_9", "ichi_kijun_26", "ichi_senkou_a", "ichi_senkou_b_52"),
          inputs=("high", "low"), family=Family.TREND, nature=Nature.ROLLING, warmup=52)
def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    h, l = df["high"], df["low"]
    tenkan = _midchannel(h, l, 9)
    kijun = _midchannel(h, l, 26)
    return pd.DataFrame({
        "ichi_tenkan_9": tenkan,
        "ichi_kijun_26": kijun,
        "ichi_senkou_a": (tenkan + kijun) / 2,      # sin desplazar (valor conocido en t)
        "ichi_senkou_b_52": _midchannel(h, l, 52),
    })
    # Chikou span (close desplazado 26 al pasado en el gráfico) se omite: como
    # feature en t equivale a close[t+26], es decir mira al futuro (no PIT-safe).

# TODO: tend_PSAR*, tend_VHF, tend_renko*, tend_hh/hl/lh/ll... (familia tend_).
