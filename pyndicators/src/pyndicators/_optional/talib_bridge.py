"""Puente OPCIONAL a TA-Lib. La única grieta a la pureza, aislada aquí.

Reservado para el puñado de indicadores DSP/Hilbert (HT_TRENDLINE, MAMA, SINE,
DCPERIOD...) que no son razonablemente reimplementables en numpy puro. El núcleo
NO importa este módulo; solo se activa con ``pip install pyndicators[talib]``.
"""
from __future__ import annotations


def require_talib():
    try:
        import talib  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Este indicador requiere TA-Lib. Instálalo con:  pip install pyndicators[talib]"
        ) from exc
    return talib

# TODO: registrar aquí HT_*, MAMA, SINE... marcados engine='talib' en el catálogo.
