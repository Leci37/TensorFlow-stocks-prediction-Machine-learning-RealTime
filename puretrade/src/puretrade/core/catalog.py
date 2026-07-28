"""Carga del catálogo de features que viaja empaquetado con la librería.

Se envía como CSV (no parquet) a propósito: leer parquet exigiría ``pyarrow``,
lo que rompería la promesa de "solo numpy + pandas".
"""
from __future__ import annotations

from importlib import resources

import pandas as pd


def load_catalog() -> pd.DataFrame:
    """Devuelve el catálogo de las ~300 features clasificadas."""
    ref = resources.files("puretrade.data").joinpath("catalog.csv")
    with resources.as_file(ref) as path:
        return pd.read_csv(path)
