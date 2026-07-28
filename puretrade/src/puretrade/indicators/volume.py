"""Familia Volume (``volu_*``). ~9 columnas. Núcleo puro.

Ejemplo de naturaleza CUMULATIVE: point-in-time seguro (no mira al futuro) pero
su valor depende de toda la historia previa, no solo de una ventana.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register


@register(key="volu_OBV", outputs=("volu_OBV",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=1)
def _obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return pd.DataFrame({"volu_OBV": (direction * df["volume"]).cumsum()})


@register(key="volu_PVT", outputs=("volu_PVT",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=1)
def _pvt(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"volu_PVT": (df["close"].pct_change() * df["volume"]).fillna(0.0).cumsum()})
