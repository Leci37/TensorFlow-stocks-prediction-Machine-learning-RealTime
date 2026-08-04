import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    """OHLCV sintético reproducible para tests."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = close.copy()
    open_[1:] = close[:-1]
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
