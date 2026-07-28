"""Familia MovingAverage (``ma_*``). ~40 columnas. Núcleo puro."""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, FactorStyle, Nature
from ..core.registry import register
from ..math.ewm import dema, ema, t3, tema
from ..math.rolling import kama, sma, trima, wma

_PERIODS = (5, 10, 20, 50, 100)
_FUNCS = {"SMA": sma, "EMA": ema, "DEMA": dema, "TEMA": tema,
          "WMA": wma, "TRIMA": trima, "KAMA": kama, "T3": t3}


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

# Nota: SMA/EMA/DEMA/TEMA/WMA/TRIMA/KAMA/T3, todas con paridad exacta vs TA-Lib.
# TRIMA/WMA estaban en la lista negra del proyecto viejo, pero son rolling y
# point-in-time seguras; se registran como Rolling.
