"""KONCORDE (Blai5): no tiene referencia TA-Lib, así que se valida por
point-in-time y por consistencia estructural (8 columnas, crash en {-1,0,1})."""
import numpy as np
import pandas as pd
import pytest

import pyndicators as pt
from pyndicators.core.validate import pit_check

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


# ti_* validados byte-a-byte contra py_ti durante el desarrollo; aquí se garantiza
# que computan y que son point-in-time (la suite de la librería es autónoma).
TI = [
    "ti_donchian_upper_20", "ti_donchian_lower_20", "ti_donchian_center_20",
    "ti_vortex_pos_5", "ti_vortex_neg_5", "ti_vortex_pos_14", "ti_vortex_neg_14",
    "ti_choppiness_14", "ti_coppock_14_11_10", "ti_mass_index_9_25", "ti_chaikin_10_3",
    "ti_force_index_13", "ti_hma_20", "ti_kelt_20_lower", "ti_kelt_20_upper", "ti_supertrend_20",
]


@pytest.mark.parametrize("name", TI)
def test_ti_computes_and_is_pit_safe(ohlcv, name):
    report = pit_check(ohlcv, [name])
    assert report["pit_safe"].all(), report
    assert report["n_compared"].iloc[0] > 50
