"""Familia Momentum (``mtum_*``). Núcleo puro, con paridad exacta contra TA-Lib."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.ewm import ema, ema_seed_at
from ..math.rolling import highest, lowest, rma, sma


@register(key="mtum_RSI", outputs=("mtum_RSI",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _rsi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    rs = rma(delta.clip(lower=0), n) / rma(-delta.clip(upper=0), n)
    return pd.DataFrame({"mtum_RSI": 100 - 100 / (1 + rs)})


@register(key="mtum_CMO", outputs=("mtum_CMO",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _cmo(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    su, sd = rma(delta.clip(lower=0), n), rma(-delta.clip(upper=0), n)
    return pd.DataFrame({"mtum_CMO": 100 * (su - sd) / (su + sd)})


@register(key="mtum_MOM", outputs=("mtum_MOM",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _mom(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_MOM": df["close"].diff(n)})


@register(key="mtum_ROC", outputs=("mtum_ROC",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _roc(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ROC": df["close"].pct_change(n) * 100})


@register(key="mtum_ROCP", outputs=("mtum_ROCP",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _rocp(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ROCP": df["close"].pct_change(n)})


@register(key="mtum_ROCR", outputs=("mtum_ROCR",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _rocr(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ROCR": df["close"] / df["close"].shift(n)})


@register(key="mtum_ROCR100", outputs=("mtum_ROCR100",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=10)
def _rocr100(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ROCR100": df["close"] / df["close"].shift(n) * 100})


@register(key="mtum_BOP", outputs=("mtum_BOP",), inputs=("open", "high", "low", "close"),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=1)
def _bop(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"mtum_BOP": (df["close"] - df["open"]) / (df["high"] - df["low"])})


@register(key="mtum_WILLIAMS_R", outputs=("mtum_WILLIAMS_R",), inputs=("high", "low", "close"),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _willr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    hh, ll = highest(df["high"], n), lowest(df["low"], n)
    return pd.DataFrame({"mtum_WILLIAMS_R": -100 * (hh - df["close"]) / (hh - ll)})


@register(key="mtum_CCI", outputs=("mtum_CCI",), inputs=("high", "low", "close"),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _cci(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean_dev = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return pd.DataFrame({"mtum_CCI": (tp - sma(tp, n)) / (0.015 * mean_dev)})


@register(key="mtum_APO", outputs=("mtum_APO",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=26)
def _apo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    return pd.DataFrame({"mtum_APO": sma(df["close"], fast) - sma(df["close"], slow)})


@register(key="mtum_PPO", outputs=("mtum_PPO",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=26)
def _ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    sf, ss = sma(df["close"], fast), sma(df["close"], slow)
    return pd.DataFrame({"mtum_PPO": 100 * (sf - ss) / ss})


@register(key="mtum_MACD",
          outputs=("mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list"),
          inputs=("close",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=33)
def _macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    # TA-Lib alinea la EMA rápida al arranque de la lenta (semilla en slow-1).
    macd = ema_seed_at(df["close"], fast, slow - 1) - ema(df["close"], slow)
    signal = ema_seed_at(macd, sig, (slow - 1) + (sig - 1))
    return pd.DataFrame({
        "mtum_MACD": macd,
        "mtum_MACD_signal": signal,
        "mtum_MACD_list": macd - signal,
    })


def _stoch_fastk(df: pd.DataFrame, n: int) -> pd.Series:
    ll = lowest(df["low"], n)
    return 100 * (df["close"] - ll) / (highest(df["high"], n) - ll)


@register(key="mtum_STOCH", outputs=("mtum_STOCH_k", "mtum_STOCH_d"),
          inputs=("high", "low", "close"), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=9)
def _stoch(df: pd.DataFrame, k: int = 5, d1: int = 3, d2: int = 3) -> pd.DataFrame:
    slow_k = sma(_stoch_fastk(df, k), d1)
    return pd.DataFrame({"mtum_STOCH_k": slow_k, "mtum_STOCH_d": sma(slow_k, d2)})


@register(key="mtum_STOCH_Fa", outputs=("mtum_STOCH_Fa_k", "mtum_STOCH_Fa_d"),
          inputs=("high", "low", "close"), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=7)
def _stochf(df: pd.DataFrame, k: int = 5, d: int = 3) -> pd.DataFrame:
    fast_k = _stoch_fastk(df, k)
    return pd.DataFrame({"mtum_STOCH_Fa_k": fast_k, "mtum_STOCH_Fa_d": sma(fast_k, d)})

# TODO: ADX/DMI (Wilder), AROON, ULTOSC, TRIX, MFI, STOCH_RSI...
