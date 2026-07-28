"""KONCORDE (Blai5): no tiene referencia TA-Lib, así que se valida por
point-in-time y por consistencia estructural (8 columnas, crash en {-1,0,1})."""
import numpy as np
import pandas as pd
import pytest

import puretrade as pt
from puretrade.core.validate import pit_check

KONK = [
    "ti_konk_blue", "ti_konk_brown", "ti_konk_green", "ti_konk_avg", "ti_konk_rest",
    "ti_konk_bl_avg_crash", "ti_konk_gre_avg_crash", "ti_konk_gre_bl_crash",
]


@pytest.fixture(scope="module")
def ohlcv():
    rng = np.random.default_rng(9)
    n = 1000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.0015, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1000, 9000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_koncorde_produces_all_columns(ohlcv):
    out = pt.compute(ohlcv, KONK)
    assert list(out.columns) == KONK
    assert out[["ti_konk_blue", "ti_konk_green"]].dropna().shape[0] > 100


def test_koncorde_crash_are_signals(ohlcv):
    out = pt.compute(ohlcv, KONK)
    for col in ("ti_konk_bl_avg_crash", "ti_konk_gre_avg_crash", "ti_konk_gre_bl_crash"):
        assert set(out[col].dropna().unique()) <= {-1, 0, 1}


def test_koncorde_is_point_in_time(ohlcv):
    report = pit_check(ohlcv, ["ti_konk_blue", "ti_konk_green", "ti_konk_avg"])
    assert report["pit_safe"].all(), report[~report["pit_safe"]]
