"""Motor de cálculo: resuelve dependencias en orden topológico y computa features.

``compute`` es el único verbo que necesita el 90% de los usuarios.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .enums import PIT_UNSAFE
from .registry import all_producers, output_index

OHLCV = ("open", "high", "low", "close", "volume")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLC obligatorias: {missing}")
    return df


def _toposort(keys: Iterable[str], producers) -> List[str]:
    ordered: List[str] = []
    seen = set()

    def visit(k: str, stack: tuple = ()):
        if k in seen:
            return
        if k in stack:
            raise ValueError(f"Ciclo de dependencias en: {' -> '.join(stack + (k,))}")
        for dep in producers[k].depends_on:
            if dep not in producers:
                raise KeyError(f"{k!r} depende de {dep!r}, que no está registrado")
            visit(dep, stack + (k,))
        seen.add(k)
        ordered.append(k)

    for k in keys:
        visit(k)
    return ordered


def compute(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    *,
    rt_safe_only: bool = False,
) -> pd.DataFrame:
    """Calcula indicadores técnicos sobre un DataFrame OHLCV.

    Parameters
    ----------
    df :
        DataFrame con columnas open/high/low/close[/volume] (mayúsc. o minúsc.).
    features :
        Nombres de indicadores o de columnas de salida. ``None`` = todos.
    rt_safe_only :
        Si ``True``, excluye los indicadores que no son point-in-time seguros.
    """
    work = _normalise(df)
    producers = all_producers()
    oidx = output_index()

    if features is None:
        keys = list(producers)
    else:
        keys = []
        for f in features:
            if f in producers:
                keys.append(f)
            elif f in oidx:
                keys.append(oidx[f])
            else:
                raise KeyError(f"Indicador/columna desconocida: {f!r}")
        keys = list(dict.fromkeys(keys))  # dedup preservando orden

    if rt_safe_only:
        keys = [k for k in keys if producers[k].nature not in PIT_UNSAFE]

    order = _toposort(keys, producers)

    parts = []
    for key in order:
        res = producers[key].func(work)
        parts.append(res)
        new_cols = [c for c in res.columns if c not in work.columns]
        if new_cols:  # disponible para productores derivados (concat, no inserción 1 a 1)
            work = pd.concat([work, res[new_cols]], axis=1)

    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=work.index)
    out = out.loc[:, ~out.columns.duplicated()]

    if features is not None:
        wanted = [f for f in features if f in out.columns]
        if wanted:
            out = out[wanted]
    return out
