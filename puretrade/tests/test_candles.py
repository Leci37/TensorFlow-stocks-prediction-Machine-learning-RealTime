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
    "DOJI", "DRAGONFLYDOJI", "GRAVESTONEDOJI", "LONGLEGGEDDOJI", "MARUBOZU",
    "CLOSINGMARUBOZU", "SPINNINGTOP", "HIGHWAVE", "LONGLINE", "SHORTLINE", "BELTHOLD",
]


@pytest.fixture(scope="module")
def big_ohlcv():
    rng = np.random.default_rng(7)
    n = 4000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.001, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.mark.parametrize("pattern", PATTERNS)
def test_candle_matches_talib(big_ohlcv, pattern):
    mine = pt.compute(big_ohlcv, [f"cdl_{pattern}"])[f"cdl_{pattern}"].to_numpy()
    ref = getattr(talib, f"CDL{pattern}")(
        big_ohlcv.open, big_ohlcv.high, big_ohlcv.low, big_ohlcv.close
    ).to_numpy().astype("int16")
    mismatches = int((mine != ref).sum())
    assert mismatches == 0, f"cdl_{pattern}: {mismatches} discrepancias con TA-Lib"
