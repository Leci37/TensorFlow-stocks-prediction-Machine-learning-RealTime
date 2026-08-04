# Instalación

## Núcleo (recomendado)

Solo necesita `numpy` y `pandas`. No compila nada:

```bash
pip install pyndicators
```

Cubre ~95% de los indicadores en implementación nativa pura.

## Extras opcionales

```bash
pip install pyndicators[talib]   # +TA-Lib, solo para indicadores DSP/Hilbert
pip install pyndicators[dev]     # pytest, ruff, mypy (desarrollo)
pip install pyndicators[all]     # todo
```

El extra `talib` requiere la librería C de TA-Lib. **No hace falta** para el núcleo:
si no lo instalas, solo quedan sin cubrir un puñado de indicadores Hilbert
(`MAMA/FAMA`, `HT_TRENDLINE`, `SAR`…), que avisan con un error claro al pedirlos.

## Requisitos

- Python ≥ 3.9
- `numpy` ≥ 1.21, `pandas` ≥ 1.3

## Datos de entrada

`compute()` espera un `DataFrame` con columnas OHLCV. Acepta mayúsculas o
minúsculas (`Open`/`open`). `volume` solo es necesario para indicadores de volumen.
Para VWAP y Pivot Points (anclados a sesión) el índice debe ser `DatetimeIndex`.
