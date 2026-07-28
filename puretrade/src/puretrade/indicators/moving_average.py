"""Familia MovingAverage (``ma_*``). ~40 columnas. Núcleo puro."""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, FactorStyle, Nature
from ..core.registry import register
from ..math.ewm import dema, ema, tema
from ..math.rolling import sma

_PERIODS = (5, 10, 20, 50, 100)
_FUNCS = {"SMA": sma, "EMA": ema, "DEMA": dema, "TEMA": tema}


def _make(kind: str, fn, n: int):
    col = f"ma_{kind}_{n}"

    @register(
        key=col,
        outputs=(col,),
        inputs=("close",),
        family=Family.MOVING_AVERAGE,
        nature=Nature.ROLLING,
        warmup=n,
    )
    def _f(df: pd.DataFrame, _fn=fn, _n=n, _col=col) -> pd.DataFrame:
        return pd.DataFrame({_col: _fn(df["close"], _n)})


for _kind, _fn in _FUNCS.items():
    for _n in _PERIODS:
        _make(_kind, _fn, _n)

# TODO: KAMA, WMA, T3, TRIMA (TRIMA/WMA algunos marcados no-PIT en el catálogo).
