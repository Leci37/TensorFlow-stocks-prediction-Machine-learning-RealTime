"""Familia Candle (``cdl_*``). Patrones de vela en geometría pura de OHLC.

100% point-in-time (cada valor depende solo de la vela actual y las previas).
Salida categórica estilo TA-Lib: +100 alcista, -100 bajista, 0 sin patrón.
Las implementaciones se validan contra TA-Lib en tests/test_candles.py.
"""
from __future__ import annotations

import pandas as pd

from ..core.enums import Family, Nature, OutputType
from ..core.registry import register
from ..math.candle import candle_avg, parts


def _register(name: str, fn):
    key = f"cdl_{name}"

    @register(key=key, outputs=(key,), inputs=("open", "high", "low", "close"),
              family=Family.CANDLE, nature=Nature.PATTERN,
              output=OutputType.CATEGORICAL, warmup=12)
    def _wrap(df: pd.DataFrame, _fn=fn, _key=key) -> pd.DataFrame:
        return pd.DataFrame({_key: _fn(df).fillna(0).astype("int16")})


def _sig(cond_white, cond_black=None):
    """Construye la serie -100/0/100 a partir de máscaras booleanas."""
    out = pd.Series(0, index=cond_white.index, dtype="int16")
    out[cond_white.fillna(False)] = 100
    if cond_black is not None:
        out[cond_black.fillna(False)] = -100
    return out


# ------------------------- patrones de una vela -------------------------

def _doji(df):
    body, hl, us, ls, white = parts(df)
    return _sig(body <= candle_avg(df, "BodyDoji"))


def _dragonfly(df):
    body, hl, us, ls, white = parts(df)
    vs = candle_avg(df, "ShadowVeryShort")
    c = (body <= candle_avg(df, "BodyDoji")) & (us < vs) & (ls > vs)
    return _sig(c)


def _gravestone(df):
    body, hl, us, ls, white = parts(df)
    vs = candle_avg(df, "ShadowVeryShort")
    c = (body <= candle_avg(df, "BodyDoji")) & (ls < vs) & (us > vs)
    return _sig(c)


def _longlegged(df):
    body, hl, us, ls, white = parts(df)
    c = (body <= candle_avg(df, "BodyDoji")) & (
        (us > candle_avg(df, "ShadowLong")) | (ls > candle_avg(df, "ShadowLong"))
    )
    return _sig(c)


def _marubozu(df):
    body, hl, us, ls, white = parts(df)
    base = (body > candle_avg(df, "BodyLong")) & (us < candle_avg(df, "ShadowVeryShort")) & (ls < candle_avg(df, "ShadowVeryShort"))
    return _sig(base & white, base & ~white)


def _closingmarubozu(df):
    body, hl, us, ls, white = parts(df)
    long = body > candle_avg(df, "BodyLong")
    vs = candle_avg(df, "ShadowVeryShort")
    return _sig(long & white & (us < vs), long & ~white & (ls < vs))


def _spinningtop(df):
    body, hl, us, ls, white = parts(df)
    base = (us > body) & (ls > body) & (body < candle_avg(df, "BodyShort"))
    return _sig(base & white, base & ~white)


def _highwave(df):
    body, hl, us, ls, white = parts(df)
    vl = candle_avg(df, "ShadowVeryLong")
    base = (us > vl) & (ls > vl) & (body < candle_avg(df, "BodyShort"))
    return _sig(base & white, base & ~white)


def _longline(df):
    body, hl, us, ls, white = parts(df)
    ss = candle_avg(df, "ShadowShort")
    base = (body > candle_avg(df, "BodyLong")) & (us < ss) & (ls < ss)
    return _sig(base & white, base & ~white)


def _shortline(df):
    body, hl, us, ls, white = parts(df)
    ss = candle_avg(df, "ShadowShort")
    base = (body < candle_avg(df, "BodyShort")) & (us < ss) & (ls < ss)
    return _sig(base & white, base & ~white)


def _belthold(df):
    body, hl, us, ls, white = parts(df)
    long = body > candle_avg(df, "BodyLong")
    vs = candle_avg(df, "ShadowVeryShort")
    return _sig(long & white & (ls < vs), long & ~white & (us < vs))


_PATTERNS = {
    "DOJI": _doji,
    "DRAGONFLYDOJI": _dragonfly,
    "GRAVESTONEDOJI": _gravestone,
    "LONGLEGGEDDOJI": _longlegged,
    "MARUBOZU": _marubozu,
    "CLOSINGMARUBOZU": _closingmarubozu,
    "SPINNINGTOP": _spinningtop,
    "HIGHWAVE": _highwave,
    "LONGLINE": _longline,
    "SHORTLINE": _shortline,
    "BELTHOLD": _belthold,
}

for _name, _fn in _PATTERNS.items():
    _register(_name, _fn)

# TODO: patrones de 2 y 3 velas (ENGULFING, HARAMI, MORNINGSTAR, 3WHITESOLDIERS...).
