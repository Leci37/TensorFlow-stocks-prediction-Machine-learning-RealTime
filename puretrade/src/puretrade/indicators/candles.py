"""Familia Candle (``cdl_*``). Patrones de vela en geometría pura de OHLC.

100% point-in-time (cada valor depende solo de la vela actual y las previas).
Salida categórica estilo TA-Lib: +100 alcista, -100 bajista, 0 sin patrón.
Las implementaciones se validan contra TA-Lib en tests/test_candles.py.
"""
from __future__ import annotations

import numpy as np
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
    """Construye la serie -100/0/100 a partir de máscaras booleanas.

    ``cond_white`` puede ser None para patrones solo bajistas.
    """
    idx = cond_white.index if cond_white is not None else cond_black.index
    out = pd.Series(0, index=idx, dtype="int16")
    if cond_white is not None:
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


# ------------------------- familia HAMMER (1 vela + contexto) -------------------------

def _hammer(df):
    body, hl, us, ls, white = parts(df)
    o, c, l = df["open"], df["close"], df["low"]
    shape = (body < candle_avg(df, "BodyShort")) & (ls > candle_avg(df, "ShadowLong")) & (us < candle_avg(df, "ShadowVeryShort"))
    ctx = np.minimum(o, c) <= l.shift(1) + candle_avg(df, "Near").shift(1)
    return _sig(shape & ctx)


def _hangingman(df):
    body, hl, us, ls, white = parts(df)
    o, c, h = df["open"], df["close"], df["high"]
    shape = (body < candle_avg(df, "BodyShort")) & (ls > candle_avg(df, "ShadowLong")) & (us < candle_avg(df, "ShadowVeryShort"))
    ctx = np.minimum(o, c) >= h.shift(1) - candle_avg(df, "Near").shift(1)
    return _sig(None, shape & ctx)


def _invertedhammer(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    shape = (body < candle_avg(df, "BodyShort")) & (us > candle_avg(df, "ShadowLong")) & (ls < candle_avg(df, "ShadowVeryShort"))
    gap_down = np.maximum(o, c) < np.minimum(o.shift(1), c.shift(1))
    return _sig(shape & gap_down)


def _shootingstar(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    shape = (body < candle_avg(df, "BodyShort")) & (us > candle_avg(df, "ShadowLong")) & (ls < candle_avg(df, "ShadowVeryShort"))
    gap_up = np.minimum(o, c) > np.maximum(o.shift(1), c.shift(1))
    return _sig(None, shape & gap_up)


def _takuri(df):
    body, hl, us, ls, white = parts(df)
    c = (body <= candle_avg(df, "BodyDoji")) & (us < candle_avg(df, "ShadowVeryShort")) & (ls > candle_avg(df, "ShadowVeryLong"))
    return _sig(c)


# ------------------------- patrones de dos velas -------------------------

def _mask(index, bull=None, bear=None):
    out = pd.Series(0, index=index, dtype="int16")
    if bull is not None:
        out[bull.fillna(False)] = 100
    if bear is not None:
        out[bear.fillna(False)] = -100
    return out


def _engulfing(df):
    o, c = df["open"], df["close"]
    white, white1 = c >= o, c.shift(1) >= o.shift(1)
    o1, c1 = o.shift(1), c.shift(1)
    bull = white & ~white1 & (c >= o1) & (o <= c1) & ((c > o1) | (o < c1))
    bear = ~white & white1 & (o >= c1) & (c <= o1) & ((o > c1) | (c < o1))
    return _mask(df.index, bull, bear)


def _harami(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    hi1, lo1 = np.maximum(o.shift(1), c.shift(1)), np.minimum(o.shift(1), c.shift(1))
    cond = (body.shift(1) > candle_avg(df, "BodyLong").shift(1)) & (body <= candle_avg(df, "BodyShort")) & (hi < hi1) & (lo > lo1)
    prev_white = c.shift(1) >= o.shift(1)
    return _mask(df.index, cond & ~prev_white, cond & prev_white)


def _haramicross(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    hi1, lo1 = np.maximum(o.shift(1), c.shift(1)), np.minimum(o.shift(1), c.shift(1))
    cond = (body.shift(1) > candle_avg(df, "BodyLong").shift(1)) & (body <= candle_avg(df, "BodyDoji")) & (hi < hi1) & (lo > lo1)
    prev_white = c.shift(1) >= o.shift(1)
    return _mask(df.index, cond & ~prev_white, cond & prev_white)


def _piercing(df):
    body, hl, us, ls, white = parts(df)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bl = candle_avg(df, "BodyLong")
    cond = ((c.shift(1) < o.shift(1)) & (body.shift(1) > bl.shift(1))
            & (c >= o) & (body > bl)
            & (o < l.shift(1)) & (c > c.shift(1) + body.shift(1) * 0.5) & (c < o.shift(1)))
    return _mask(df.index, cond)


def _darkcloud(df):
    body, hl, us, ls, white = parts(df)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bl = candle_avg(df, "BodyLong")
    cond = ((c.shift(1) >= o.shift(1)) & (body.shift(1) > bl.shift(1))
            & (c < o)
            & (o > h.shift(1)) & (c < c.shift(1) - body.shift(1) * 0.5) & (c > o.shift(1)))
    return _mask(df.index, bear=cond)


def _matchinglow(df):
    o, c = df["open"], df["close"]
    eq = candle_avg(df, "Equal").shift(1)
    cond = (c.shift(1) < o.shift(1)) & (c < o) & (c <= c.shift(1) + eq) & (c >= c.shift(1) - eq)
    return _mask(df.index, cond)


def _homingpigeon(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    cond = ((c.shift(1) < o.shift(1)) & (c < o)
            & (body.shift(1) > candle_avg(df, "BodyLong").shift(1)) & (body < candle_avg(df, "BodyShort"))
            & (o < o.shift(1)) & (c > c.shift(1)))
    return _mask(df.index, cond)


def _dojistar(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    hi1, lo1 = np.maximum(o.shift(1), c.shift(1)), np.minimum(o.shift(1), c.shift(1))
    long1 = body.shift(1) > candle_avg(df, "BodyLong").shift(1)
    doji = body <= candle_avg(df, "BodyDoji")
    prev_white = c.shift(1) >= o.shift(1)
    bear = long1 & doji & prev_white & (lo > hi1)      # gap up tras vela blanca
    bull = long1 & doji & ~prev_white & (hi < lo1)     # gap down tras vela negra
    return _mask(df.index, bull, bear)


# ------------------------- patrones de tres velas -------------------------

def _hilo_bodies(df):
    o, c = df["open"], df["close"]
    return np.maximum(o, c), np.minimum(o, c)


def _morningstar(df, pen=0.3):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = _hilo_bodies(df)
    bl, bs = candle_avg(df, "BodyLong"), candle_avg(df, "BodyShort")
    cond = ((body.shift(2) > bl.shift(2)) & (c.shift(2) < o.shift(2))
            & (body.shift(1) <= bs.shift(1))
            & (hi.shift(1) < lo.shift(2))                       # gap down 2->1
            & (c >= o) & (body > bs)                            # 3ª vela blanca de cuerpo largo
            & (c > c.shift(2) + body.shift(2) * pen))
    return _mask(df.index, bull=cond)


def _eveningstar(df, pen=0.3):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = _hilo_bodies(df)
    bl, bs = candle_avg(df, "BodyLong"), candle_avg(df, "BodyShort")
    cond = ((body.shift(2) > bl.shift(2)) & (c.shift(2) >= o.shift(2))
            & (body.shift(1) <= bs.shift(1))
            & (lo.shift(1) > hi.shift(2))                       # gap up 2->1
            & (c < o) & (body > bs)                             # 3ª vela negra de cuerpo largo
            & (c < c.shift(2) - body.shift(2) * pen))
    return _mask(df.index, bear=cond)


def _3whitesoldiers(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    vs, bs = candle_avg(df, "ShadowVeryShort"), candle_avg(df, "BodyShort")
    near, far = candle_avg(df, "Near"), candle_avg(df, "Far")
    allwhite = (c >= o) & (c.shift(1) >= o.shift(1)) & (c.shift(2) >= o.shift(2))
    shortshadows = (us < vs) & (us.shift(1) < vs.shift(1)) & (us.shift(2) < vs.shift(2))
    rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    # cada vela abre dentro del cuerpo anterior (tolerancia Near)
    within = ((o.shift(1) > o.shift(2)) & (o.shift(1) <= c.shift(2) + near.shift(2))
              & (o > o.shift(1)) & (o <= c.shift(1) + near.shift(1)))
    # sin gran desaceleración de cuerpo (tolerancia Far) y la vela actual con cuerpo largo
    no_decel = (body.shift(1) > body.shift(2) - far.shift(2)) & (body > body.shift(1) - far.shift(1))
    return _mask(df.index, bull=allwhite & shortshadows & rising & within & no_decel & (body > bs))


def _3blackcrows(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    vs, bs = candle_avg(df, "ShadowVeryShort"), candle_avg(df, "BodyShort")
    near, far = candle_avg(df, "Near"), candle_avg(df, "Far")
    allblack = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2))
    shortshadows = (ls < vs) & (ls.shift(1) < vs.shift(1)) & (ls.shift(2) < vs.shift(2))
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    within = ((o.shift(1) < o.shift(2)) & (o.shift(1) >= c.shift(2) - near.shift(2))
              & (o < o.shift(1)) & (o >= c.shift(1) - near.shift(1)))
    no_decel = (body.shift(1) > body.shift(2) - far.shift(2)) & (body > body.shift(1) - far.shift(1))
    return _mask(df.index, bear=allblack & shortshadows & falling & within & no_decel & (body > bs))


def _3inside(df):
    body, hl, us, ls, white = parts(df)
    o, c = df["open"], df["close"]
    hi, lo = _hilo_bodies(df)
    bl, bs = candle_avg(df, "BodyLong"), candle_avg(df, "BodyShort")
    harami = (body.shift(2) > bl.shift(2)) & (body.shift(1) <= bs.shift(1)) & (hi.shift(1) < hi.shift(2)) & (lo.shift(1) > lo.shift(2))
    up = harami & (c.shift(2) < o.shift(2)) & (c >= o) & (c > o.shift(2))
    down = harami & (c.shift(2) >= o.shift(2)) & (c < o) & (c < o.shift(2))
    return _mask(df.index, up, down)


def _3outside(df):
    o, c = df["open"], df["close"]
    o1, c1, o2, c2 = o.shift(1), c.shift(1), o.shift(2), c.shift(2)
    eng_bull = (c1 >= o1) & (c2 < o2) & (c1 > o2) & (o1 < c2)
    eng_bear = (c1 < o1) & (c2 >= o2) & (o1 > c2) & (c1 < o2)
    up = eng_bull & (c > c1)
    down = eng_bear & (c < c1)
    return _mask(df.index, up, down)


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
    "HAMMER": _hammer,
    "HANGINGMAN": _hangingman,
    "INVERTEDHAMMER": _invertedhammer,
    "SHOOTINGSTAR": _shootingstar,
    "TAKURI": _takuri,
    "ENGULFING": _engulfing,
    "HARAMI": _harami,
    "HARAMICROSS": _haramicross,
    "PIERCING": _piercing,
    "DARKCLOUDCOVER": _darkcloud,
    "MATCHINGLOW": _matchinglow,
    "HOMINGPIGEON": _homingpigeon,
    "DOJISTAR": _dojistar,
    "MORNINGSTAR": _morningstar,
    "EVENINGSTAR": _eveningstar,
    "3INSIDE": _3inside,
    "3OUTSIDE": _3outside,
    "3WHITESOLDIERS": _3whitesoldiers,
}

# TODO: _3blackcrows implementado pero AÚN NO registrado: no es el espejo exacto
# de 3WHITESOLDIERS (sobre-dispara frente a TA-Lib). No se expone hasta lograr
# paridad exacta (contrato: todo cdl_ registrado == paridad exacta con TA-Lib).

for _name, _fn in _PATTERNS.items():
    _register(_name, _fn)

# TODO: patrones de 2 y 3 velas (ENGULFING, HARAMI, MORNINGSTAR, 3WHITESOLDIERS...).
