"""Registro central de indicadores (el catálogo vivo) y el decorador ``@register``.

Cada indicador es un *productor*: una función que recibe el DataFrame OHLCV de
trabajo y devuelve un DataFrame con una o más columnas de salida. La metadata
(familia, naturaleza, dependencias...) viaja junto a la función, de modo que
añadir un indicador nuevo es registrar una función, no tocar cuatro archivos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple

import pandas as pd

from .enums import Family, Nature, OutputType, PIT_UNSAFE

# key del productor -> Producer
_PRODUCERS: "Dict[str, Producer]" = {}


@dataclass(frozen=True)
class Producer:
    key: str
    outputs: Tuple[str, ...]
    func: Callable[[pd.DataFrame], pd.DataFrame]
    inputs: Tuple[str, ...] = ("close",)         # padres OHLCV
    depends_on: Tuple[str, ...] = ()             # madres (otros productores)
    family: Family = Family.CUSTOM
    nature: Nature = Nature.ROLLING
    output: OutputType = OutputType.CONTINUOUS
    engine: str = "native"
    warmup: int = 0

    @property
    def pit_safe(self) -> bool:
        return self.nature not in PIT_UNSAFE


def register(
    *,
    key: str,
    outputs: Sequence[str],
    inputs: Sequence[str] = ("close",),
    depends_on: Sequence[str] = (),
    family: Family = Family.CUSTOM,
    nature: Nature = Nature.ROLLING,
    output: OutputType = OutputType.CONTINUOUS,
    engine: str = "native",
    warmup: int = 0,
):
    """Registra la función decorada como productor de indicador."""

    def deco(fn: Callable[[pd.DataFrame], pd.DataFrame]):
        if key in _PRODUCERS:
            raise ValueError(f"Indicador ya registrado: {key!r}")
        _PRODUCERS[key] = Producer(
            key=key,
            outputs=tuple(outputs),
            func=fn,
            inputs=tuple(inputs),
            depends_on=tuple(depends_on),
            family=family,
            nature=nature,
            output=output,
            engine=engine,
            warmup=warmup,
        )
        return fn

    return deco


def all_producers() -> "Dict[str, Producer]":
    return dict(_PRODUCERS)


def output_index() -> Dict[str, str]:
    """Mapa columna_de_salida -> key del productor que la genera."""
    idx: Dict[str, str] = {}
    for key, prod in _PRODUCERS.items():
        for out in prod.outputs:
            idx[out] = key
    return idx
