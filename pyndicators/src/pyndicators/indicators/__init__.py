"""Importar este paquete registra todas las familias de indicadores.

El orden no importa: cada módulo se auto-registra en el registro central vía
el decorador ``@register`` al importarse.
"""
from . import (  # noqa: F401
    candles,
    custom,
    cycles,
    momentum,
    moving_average,
    overlap,
    performance,
    statistics,
    trend,
    volatility,
    volume,
)
