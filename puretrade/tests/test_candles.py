"""Paridad exacta de los patrones de vela nativos contra TA-Lib.

Se salta automáticamente si TA-Lib no está instalado (no es dependencia del
núcleo). En CI se instala solo en el job de validación.
"""
import numpy as np
import pandas as pd
import pytest

import puretrade as pt

talib = pytest.importorskip("talib")

PATTERNS = [
    # una vela
    "DOJI", "DRAGONFLYDOJI", "GRAVESTONEDOJI", "LONGLEGGEDDOJI", "MARUBOZU",
    "CLOSINGMARUBOZU", "SPINNINGTOP", "HIGHWAVE", "LONGLINE", "SHORTLINE", "BELTHOLD",
    # dos velas
    "ENGULFING", "HARAMI", "HARAMICROSS", "PIERCING", "DARKCLOUDCOVER",
    "MATCHINGLOW", "HOMINGPIGEON", "DOJISTAR",
    # tres velas
    "MORNINGSTAR", "EVENINGSTAR", "3INSIDE", "3OUTSIDE", "3WHITESOLDIERS",
]


@pytest.fixture(scope="module")
def big_ohlcv():
    # series con tramos de tendencia + ruido, para disparar también patrones de 3 velas
    rng = np.random.default_rng(3)
    n = 12000
    drift = np.zeros(n)
    for _ in range(60):
        s = rng.integers(0, n - 40)
        drift[s:s + 30] += rng.choice([-1, 1]) * 0.006
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n) + drift))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.0012, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0018, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0018, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.mark.parametrize("pattern", PATTERNS)
def test_candle_matches_talib(big_ohlcv, pattern):
    mine = pt.compute(big_ohlcv, [f"cdl_{pattern}"])[f"cdl_{pattern}"].to_numpy()
    ref = getattr(talib, f"CDL{pattern}")(
        big_ohlcv.open, big_ohlcv.high, big_ohlcv.low, big_ohlcv.close
    ).to_numpy().astype("int16")
    mismatches = int((mine != ref).sum())
    assert mismatches == 0, f"cdl_{pattern}: {mismatches} discrepancias con TA-Lib"
