"""Familia Momentum (``mtum_*``). Núcleo puro, con paridad exacta contra TA-Lib."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.ewm import ema, ema_seed_at
from ..math.rolling import dmi_smooth, highest, lowest, rma, sma, true_range


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

@register(key="mtum_MFI", outputs=("mtum_MFI",), inputs=("high", "low", "close", "volume"),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _mfi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return pd.DataFrame({"mtum_MFI": 100 * pos / (pos + neg)})

# ------------------------- DMI / ADX (suavizado Wilder de TA-Lib) -------------------------

@register(key="mtum_DMI",
          outputs=("mtum_PLUS_DM", "mtum_MINUS_DM", "mtum_PLUS_DI", "mtum_MINUS_DI", "mtum_DX"),
          inputs=("high", "low", "close"), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    s_tr = dmi_smooth(true_range(df["high"], df["low"], df["close"]), n)
    s_p, s_m = dmi_smooth(plus_dm, n), dmi_smooth(minus_dm, n)
    plus_di, minus_di = 100 * s_p / s_tr, 100 * s_m / s_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return pd.DataFrame({
        "mtum_PLUS_DM": s_p, "mtum_MINUS_DM": s_m,
        "mtum_PLUS_DI": plus_di, "mtum_MINUS_DI": minus_di, "mtum_DX": dx,
    })


@register(key="mtum_ADX", outputs=("mtum_ADX",), inputs=("high", "low", "close"),
          depends_on=("mtum_DMI",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=27)
def _adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"mtum_ADX": rma(df["mtum_DX"], n)})


@register(key="mtum_ADXR", outputs=("mtum_ADXR",), inputs=("high", "low", "close"),
          depends_on=("mtum_ADX",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=40)
def _adxr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    adx = df["mtum_ADX"]
    return pd.DataFrame({"mtum_ADXR": (adx + adx.shift(n - 1)) / 2})


# ------------------------- AROON / ULTOSC / TRIX / STOCH-RSI -------------------------

@register(key="mtum_AROON", outputs=("mtum_AROON_up", "mtum_AROON_down", "mtum_AROONOSC"),
          inputs=("high", "low"), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=15)
def _aroon(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].rolling(n + 1, min_periods=n + 1).apply(lambda x: 100 * np.argmax(x) / n, raw=True)
    down = df["low"].rolling(n + 1, min_periods=n + 1).apply(lambda x: 100 * np.argmin(x) / n, raw=True)
    return pd.DataFrame({"mtum_AROON_up": up, "mtum_AROON_down": down, "mtum_AROONOSC": up - down})


@register(key="mtum_ULTOSC", outputs=("mtum_ULTOSC",), inputs=("high", "low", "close"),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=28)
def _ultosc(df: pd.DataFrame, s: int = 7, m: int = 14, ln: int = 28) -> pd.DataFrame:
    prev = df["close"].shift(1)
    bp = df["close"] - np.minimum(df["low"], prev)
    tr = np.maximum(df["high"], prev) - np.minimum(df["low"], prev)
    a = [bp.rolling(p).sum() / tr.rolling(p).sum() for p in (s, m, ln)]
    return pd.DataFrame({"mtum_ULTOSC": 100 * (4 * a[0] + 2 * a[1] + a[2]) / 7})


@register(key="mtum_TRIX", outputs=("mtum_TRIX",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=90)
def _trix(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    e3 = ema(ema(ema(df["close"], n), n), n)
    return pd.DataFrame({"mtum_TRIX": 100 * (e3 - e3.shift(1)) / e3.shift(1)})


@register(key="mtum_STOCH_RSI", outputs=("mtum_STOCH_RSI_k", "mtum_STOCH_RSI_d"),
          inputs=("close",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=19)
def _stochrsi(df: pd.DataFrame, n: int = 14, k: int = 5, d: int = 3) -> pd.DataFrame:
    delta = df["close"].diff()
    rsi = 100 - 100 / (1 + rma(delta.clip(lower=0), n) / rma(-delta.clip(upper=0), n))
    fast_k = 100 * (rsi - lowest(rsi, k)) / (highest(rsi, k) - lowest(rsi, k))
    return pd.DataFrame({"mtum_STOCH_RSI_k": fast_k, "mtum_STOCH_RSI_d": sma(fast_k, d)})
