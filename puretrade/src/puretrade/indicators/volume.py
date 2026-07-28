"""Familia Volume (``volu_*``). Núcleo puro.

- OBV, Chaikin AD y ADOSC: paridad exacta con TA-Lib.
- PVT, EFI, PVOL: fórmula estándar (TA-Lib no los tiene, pandas-ta no disponible
  para referencia byte-a-byte aquí); verificados point-in-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.ewm import ema


@register(key="volu_OBV", outputs=("volu_OBV",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=1)
def _obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    obv = (direction * df["volume"]).cumsum() + df["volume"].iloc[0]  # TA-Lib arranca en volume[0]
    return pd.DataFrame({"volu_OBV": obv})


@register(key="volu_Chaikin_AD", outputs=("volu_Chaikin_AD",),
          inputs=("high", "low", "close", "volume"),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=1)
def _ad(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    clv = (((c - l) - (h - c)) / (h - l)).where(h != l, 0.0)
    return pd.DataFrame({"volu_Chaikin_AD": (clv * df["volume"]).cumsum()})


@register(key="volu_Chaikin_ADOSC", outputs=("volu_Chaikin_ADOSC",),
          inputs=("high", "low", "close", "volume"), depends_on=("volu_Chaikin_AD",),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=10)
def _adosc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
    ad = df["volu_Chaikin_AD"]
    # TA-Lib siembra estas EMA en el primer valor (no en la SMA).
    fast_e = ad.ewm(span=fast, adjust=False, min_periods=1).mean()
    slow_e = ad.ewm(span=slow, adjust=False, min_periods=1).mean()
    return pd.DataFrame({"volu_Chaikin_ADOSC": fast_e - slow_e})


@register(key="volu_PVT", outputs=("volu_PVT",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.CUMULATIVE, warmup=1)
def _pvt(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"volu_PVT": (df["close"].pct_change() * df["volume"]).fillna(0.0).cumsum()})


@register(key="volu_EFI_13", outputs=("volu_EFI_13",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.ROLLING, warmup=13)
def _efi(df: pd.DataFrame, n: int = 13) -> pd.DataFrame:
    return pd.DataFrame({"volu_EFI_13": ema(df["close"].diff() * df["volume"], n)})


@register(key="volu_PVOL", outputs=("volu_PVOL",), inputs=("close", "volume"),
          family=Family.VOLUME, nature=Nature.ROLLING, warmup=1)
def _pvol(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"volu_PVOL": df["close"] * df["volume"]})

# TODO (requieren referencia pandas-ta): volu_NVI_1, volu_PVI_1, volu_PVR.
