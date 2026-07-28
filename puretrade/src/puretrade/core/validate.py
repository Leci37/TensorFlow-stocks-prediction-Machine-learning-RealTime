"""Validación point-in-time: el sello de calidad de la librería.

``pit_check`` recalcula un indicador sobre una serie truncada y verifica que los
valores del pasado NO cambian. Si cambian, el indicador filtra información del
futuro (look-ahead bias) y no es apto para backtest ni para tiempo real.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .engine import compute


def pit_check(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    *,
    split: float = 0.7,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Comprueba estabilidad point-in-time de cada feature.

    Devuelve un DataFrame con una fila por feature y las columnas
    ``[n_compared, n_mismatch, max_abs_diff, pit_safe]``.
    """
    full = compute(df, features)
    n = int(len(df) * split)
    part = compute(df.iloc[:n], features)

    rows = []
    for col in full.columns:
        a = full[col].iloc[:n]
        b = part[col] if col in part.columns else pd.Series(index=a.index, dtype=float)
        mask = a.notna() & b.notna()
        diff = (a[mask] - b[mask]).abs()
        n_cmp = int(mask.sum())
        n_bad = int((diff > atol).sum())
        rows.append(
            {
                "feature": col,
                "n_compared": n_cmp,
                "n_mismatch": n_bad,
                "max_abs_diff": float(diff.max()) if n_cmp else 0.0,
                "pit_safe": n_bad == 0,
            }
        )
    return pd.DataFrame(rows).set_index("feature")
