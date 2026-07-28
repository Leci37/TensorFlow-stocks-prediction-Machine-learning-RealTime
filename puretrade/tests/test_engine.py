import pandas as pd

import puretrade as pt


def test_compute_all_runs(ohlcv):
    out = pt.compute(ohlcv)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(ohlcv)
    assert out.shape[1] > 10  # hay varias familias migradas


def test_compute_subset(ohlcv):
    out = pt.compute(ohlcv, ["mtum_RSI", "vola_ATR"])
    assert list(out.columns) == ["mtum_RSI", "vola_ATR"]


def test_dependency_graph_resolves(ohlcv):
    # NATR depende de ATR, que depende de TRANGE: pedir solo NATR debe funcionar.
    out = pt.compute(ohlcv, ["vola_NATR"])
    assert "vola_NATR" in out.columns
    assert out["vola_NATR"].notna().any()


def test_derived_bband_crash(ohlcv):
    out = pt.compute(ohlcv, ["olap_BBAND_UPPER_crash"])
    assert set(out["olap_BBAND_UPPER_crash"].dropna().unique()) <= {0, 1}


def test_rt_safe_filter(ohlcv):
    full = pt.compute(ohlcv)
    safe = pt.compute(ohlcv, rt_safe_only=True)
    assert safe.shape[1] <= full.shape[1]


def test_catalog_loads():
    cat = pt.catalog()
    assert len(cat) > 250
    assert {"name", "family", "nature", "pit_safe"}.issubset(cat.columns)


def test_unknown_feature_raises(ohlcv):
    try:
        pt.compute(ohlcv, ["does_not_exist"])
    except KeyError:
        return
    raise AssertionError("debería lanzar KeyError")
