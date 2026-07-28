"""Paridad exacta de indicadores numéricos (momentum, medias, volatilidad) vs TA-Lib."""
import numpy as np
import pandas as pd
import pytest

import puretrade as pt

talib = pytest.importorskip("talib")


@pytest.fixture(scope="module")
def ohlcv():
    rng = np.random.default_rng(5)
    n = 3000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.0015, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1000, 9000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _ref(name, df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    R = {
        "mtum_RSI": lambda: talib.RSI(c, 14),
        "mtum_CMO": lambda: talib.CMO(c, 14),
        "mtum_MOM": lambda: talib.MOM(c, 10),
        "mtum_ROC": lambda: talib.ROC(c, 10),
        "mtum_ROCP": lambda: talib.ROCP(c, 10),
        "mtum_ROCR": lambda: talib.ROCR(c, 10),
        "mtum_ROCR100": lambda: talib.ROCR100(c, 10),
        "mtum_BOP": lambda: talib.BOP(o, h, l, c),
        "mtum_WILLIAMS_R": lambda: talib.WILLR(h, l, c, 14),
        "mtum_CCI": lambda: talib.CCI(h, l, c, 14),
        "mtum_APO": lambda: talib.APO(c, 12, 26, 0),
        "mtum_PPO": lambda: talib.PPO(c, 12, 26, 0),
        "mtum_MACD": lambda: talib.MACD(c, 12, 26, 9)[0],
        "mtum_MACD_signal": lambda: talib.MACD(c, 12, 26, 9)[1],
        "mtum_MACD_list": lambda: talib.MACD(c, 12, 26, 9)[2],
        "mtum_STOCH_k": lambda: talib.STOCH(h, l, c, 5, 3, 0, 3, 0)[0],
        "mtum_STOCH_d": lambda: talib.STOCH(h, l, c, 5, 3, 0, 3, 0)[1],
        "mtum_STOCH_Fa_k": lambda: talib.STOCHF(h, l, c, 5, 3, 0)[0],
        "mtum_STOCH_Fa_d": lambda: talib.STOCHF(h, l, c, 5, 3, 0)[1],
        "mtum_MFI": lambda: talib.MFI(h, l, c, df["volume"], 14),
        "mtum_PLUS_DM": lambda: talib.PLUS_DM(h, l, 14),
        "mtum_MINUS_DM": lambda: talib.MINUS_DM(h, l, 14),
        "mtum_PLUS_DI": lambda: talib.PLUS_DI(h, l, c, 14),
        "mtum_MINUS_DI": lambda: talib.MINUS_DI(h, l, c, 14),
        "mtum_DX": lambda: talib.DX(h, l, c, 14),
        "mtum_ADX": lambda: talib.ADX(h, l, c, 14),
        "mtum_ADXR": lambda: talib.ADXR(h, l, c, 14),
        "mtum_AROON_up": lambda: talib.AROON(h, l, 14)[1],
        "mtum_AROON_down": lambda: talib.AROON(h, l, 14)[0],
        "mtum_AROONOSC": lambda: talib.AROONOSC(h, l, 14),
        "mtum_ULTOSC": lambda: talib.ULTOSC(h, l, c, 7, 14, 28),
        "mtum_TRIX": lambda: talib.TRIX(c, 30),
        "mtum_STOCH_RSI_k": lambda: talib.STOCHRSI(c, 14, 5, 3, 0)[0],
        "mtum_STOCH_RSI_d": lambda: talib.STOCHRSI(c, 14, 5, 3, 0)[1],
        "volu_OBV": lambda: talib.OBV(c, df["volume"]),
        "volu_Chaikin_AD": lambda: talib.AD(h, l, c, df["volume"]),
        "volu_Chaikin_ADOSC": lambda: talib.ADOSC(h, l, c, df["volume"], 3, 10),
        "vola_ATR": lambda: talib.ATR(h, l, c, 14),
        "vola_NATR": lambda: talib.NATR(h, l, c, 14),
        "vola_TRANGE": lambda: talib.TRANGE(h, l, c),
        "ma_EMA_20": lambda: talib.EMA(c, 20),
        "ma_DEMA_20": lambda: talib.DEMA(c, 20),
        "ma_TEMA_20": lambda: talib.TEMA(c, 20),
        "ma_SMA_20": lambda: talib.SMA(c, 20),
        "ma_WMA_20": lambda: talib.WMA(c, 20),
        "ma_TRIMA_20": lambda: talib.TRIMA(c, 20),
        "ma_KAMA_20": lambda: talib.KAMA(c, 20),
        "ma_T3_5": lambda: talib.T3(c, 5, 0.7),
        "olap_MIDPOINT": lambda: talib.MIDPOINT(c, 14),
        "olap_MIDPRICE": lambda: talib.MIDPRICE(h, l, 14),
    }
    return R[name]().to_numpy(dtype="float64")


NUMERIC = [
    "mtum_RSI", "mtum_CMO", "mtum_MOM", "mtum_ROC", "mtum_ROCP", "mtum_ROCR",
    "mtum_ROCR100", "mtum_BOP", "mtum_WILLIAMS_R", "mtum_CCI", "mtum_APO", "mtum_PPO",
    "mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list",
    "mtum_STOCH_k", "mtum_STOCH_d", "mtum_STOCH_Fa_k", "mtum_STOCH_Fa_d", "mtum_MFI",
    "mtum_PLUS_DM", "mtum_MINUS_DM", "mtum_PLUS_DI", "mtum_MINUS_DI", "mtum_DX",
    "mtum_ADX", "mtum_ADXR", "mtum_AROON_up", "mtum_AROON_down", "mtum_AROONOSC",
    "mtum_ULTOSC", "mtum_TRIX", "mtum_STOCH_RSI_k", "mtum_STOCH_RSI_d",
    "volu_OBV", "volu_Chaikin_AD", "volu_Chaikin_ADOSC",
    "vola_ATR", "vola_NATR", "vola_TRANGE",
    "ma_EMA_20", "ma_DEMA_20", "ma_TEMA_20", "ma_SMA_20",
    "ma_WMA_20", "ma_TRIMA_20", "ma_KAMA_20", "ma_T3_5",
    "olap_MIDPOINT", "olap_MIDPRICE",
]


@pytest.mark.parametrize("name", NUMERIC)
def test_numeric_matches_talib(ohlcv, name):
    mine = pt.compute(ohlcv, [name])[name].to_numpy(dtype="float64")
    ref = _ref(name, ohlcv)
    mask = ~np.isnan(mine) & ~np.isnan(ref) & (ref != 0)
    assert mask.sum() > 100
    max_diff = np.abs(mine[mask] - ref[mask]).max()
    assert max_diff < 1e-6, f"{name}: maxdiff {max_diff:.2e} vs TA-Lib"
