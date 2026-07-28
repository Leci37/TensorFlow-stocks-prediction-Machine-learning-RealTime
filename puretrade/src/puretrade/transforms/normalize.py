"""Normalizaciones para ML. Puro pandas."""
from __future__ import annotations

import pandas as pd


def zscore(s: pd.Series, n: int) -> pd.Series:
    """Z-score con ventana trasera (point-in-time seguro)."""
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / std


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking [0,1] entre activos en cada instante (para carteras multi-activo)."""
    return df.rank(axis=1, pct=True)
