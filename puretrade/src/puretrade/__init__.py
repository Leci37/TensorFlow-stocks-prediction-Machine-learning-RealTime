"""puretrade — indicadores técnicos para ML financiero, en puro numpy + pandas.

Instalación limpia (sin compilar nada):  ``pip install puretrade``

Uso básico::

    import puretrade as pt
    features = pt.compute(ohlcv_df)                 # todos los indicadores
    features = pt.compute(ohlcv_df, ["mtum_RSI"])   # solo algunos
    features = pt.compute(ohlcv_df, rt_safe_only=True)  # solo point-in-time
    cat = pt.catalog()                              # ficha de cada feature
"""
from __future__ import annotations

from .core import (
    Family,
    FactorStyle,
    Nature,
    OutputType,
    all_producers,
    compute,
    load_catalog as catalog,
    pit_check,
    register,
)
from . import indicators as _indicators  # noqa: F401  (registra las familias)

__version__ = "0.1.0"

__all__ = [
    "compute",
    "catalog",
    "pit_check",
    "register",
    "all_producers",
    "Family",
    "FactorStyle",
    "Nature",
    "OutputType",
    "__version__",
]
