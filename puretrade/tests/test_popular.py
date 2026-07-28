"""Indicadores populares de TradingView añadidos (AO, Ichimoku, VWAP, Pivots).

No son de TA-Lib; se validan por fórmula estándar + point-in-time. VWAP y Pivots
se anclan a la sesión (día natural) y requieren índice datetime.
"""
import numpy as np
import pandas as pd
import pytest

import puretrade as pt
from puretrade.core.validate import pit_check

POPULAR = [
    "mtum_AO_5_34",
    "ichi_tenkan_9", "ichi_kijun_26", "ichi_senkou_a", "ichi_senkou_b_52",
    "olap_VWAP",
    "piv_P", "piv_R1", "piv_S1", "piv_R2", "piv_S2", "piv_R3", "piv_S3",
]


@pytest.fixture(scope="module")
def intraday():
    n = 900
    idx = pd.date_range("2020-01-01 09:30", periods=n, freq="5min")
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.0015, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = rng.integers(1000, 9000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.mark.parametrize("name", POPULAR)
def test_popular_pit_safe(intraday, name):
    report = pit_check(intraday, [name])
    assert report["pit_safe"].all(), report
    assert report["n_compared"].iloc[0] > 50


def test_vwap_resets_each_session(intraday):
    # el primer bar de cada día: VWAP == typical price de ese bar
    out = pt.compute(intraday, ["olap_VWAP"])
    tp = (intraday["high"] + intraday["low"] + intraday["close"]) / 3
    first_of_day = ~intraday.index.normalize().duplicated()
    assert np.allclose(out["olap_VWAP"][first_of_day], tp[first_of_day])


def test_pivot_relations(intraday):
    out = pt.compute(intraday, ["piv_P", "piv_R1", "piv_S1"]).dropna()
    assert (out["piv_R1"] > out["piv_P"]).all()
    assert (out["piv_S1"] < out["piv_P"]).all()
