"""El test estrella: los indicadores del núcleo NO deben mirar al futuro."""
import pyndicators as pt
from pyndicators.core.validate import pit_check


def test_core_indicators_are_pit_safe(ohlcv):
    report = pit_check(ohlcv, ["mtum_RSI", "vola_ATR", "ma_SMA_20", "olap_BBAND_UPPER"])
    assert report["pit_safe"].all(), report[~report["pit_safe"]]


def test_pit_check_returns_row_per_feature(ohlcv):
    report = pit_check(ohlcv, ["mtum_RSI", "mtum_MOM"])
    assert len(report) == 2
    assert "max_abs_diff" in report.columns
