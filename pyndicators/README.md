# pyndicators

**Indicadores técnicos para machine learning financiero — en puro `numpy` + `pandas`.**

Sin TA-Lib obligatorio, sin nada que compilar. `pip install pyndicators` y funciona.
Cada indicador registrado está verificado **byte a byte contra TA-Lib** (o py_ti) y
es **point-in-time** (no mira al futuro).

```bash
pip install pyndicators
```

## Uso en 30 segundos

```python
import pandas as pd
import pyndicators as pt

df = pd.read_csv("ohlcv.csv", parse_dates=["Date"], index_col="Date")
# df tiene columnas open/high/low/close[/volume] (mayúsculas o minúsculas)

features = pt.compute(df)          # TODOS los indicadores disponibles
features.tail()
```

`compute()` devuelve un `DataFrame` con el mismo índice que `df` y una columna por
indicador. Es el único verbo que necesitas el 90% del tiempo.

## Cómo elegir qué generar

No siempre quieres los ~160 indicadores. Puedes pedir exactamente lo que necesitas:

```python
# 1) Todos
pt.compute(df)

# 2) Solo una familia  (Momentum, Volatility, Volume, Candle, Overlap, Trend,
#    MovingAverage, Custom)
pt.compute(df, family="Volatility")        # ATR, NATR, TRANGE...
pt.compute(df, family="Candle")            # los 33 patrones de vela
pt.compute(df, family=["Momentum", "Volume"])   # varias familias

# 3) Solo indicadores concretos (por nombre de indicador o de columna)
pt.compute(df, ["mtum_RSI", "vola_ATR", "olap_BBAND_UPPER"])

# 4) Solo por naturaleza de cálculo
pt.compute(df, nature="Pattern")           # solo patrones categóricos (velas)
pt.compute(df, nature="Cumulative")        # OBV, VWAP, Koncorde...

# 5) Solo lo que es point-in-time seguro (sin look-ahead) — para backtest/ML
pt.compute(df, rt_safe_only=True)

# Se combinan:
pt.compute(df, family="Momentum", rt_safe_only=True)
```

## Descubrir qué hay disponible

```python
pt.families()                      # ['Candle', 'Custom', 'Momentum', ...]
pt.list_features(family="Volume")  # nombres de columnas de esa familia

cat = pt.catalog()                 # DataFrame-ficha de cada indicador
cat.columns
# name, family, factor_style, nature, pit_safe, engine, warmup, output, migrated

# El catálogo se filtra con pandas normal:
cat[(cat.family == "Momentum") & cat.pit_safe & cat.migrated]
cat[cat.factor_style == "volatility/risk"]
```

## Transformaciones para ML (estacionariedad y normalización)

```python
from pyndicators import transforms

transforms.returns(df["close"])          # retornos
transforms.log_returns(df["close"])
transforms.fracdiff(df["close"], d=0.4)  # diferenciación fraccionaria
transforms.zscore(features["mtum_RSI"], 100)
```

## Tiempo real (al cerrar cada vela)

Como todos los indicadores son point-in-time, la última fila calculada es el valor
"en vivo" correcto. Mantén una ventana trasera y recalcula al cerrar la vela:

```python
buf = df.tail(500)                       # ventana trasera (1m, 5m, 15m…)
live = pt.compute(buf).iloc[-1]          # features de la vela recién cerrada
```

Un buffer de ~500 barras deja medias y velas exactas y los recursivos (EMA/RSI/
ATR) a ruido de float. Detalles, multi-timeframe y salvedades (OBV, VWAP) en
[docs/realtime.md](docs/realtime.md).

## Validar que un indicador no filtra el futuro

```python
from pyndicators.core.validate import pit_check
pit_check(df, ["mtum_RSI", "vola_ATR"])   # tabla con pit_safe por feature
```

## Por qué otra librería de indicadores

Hay decenas de envoltorios de TA-Lib. `pyndicators` es distinta en cuatro cosas:

1. **Point-in-Time garantizado.** Cada feature marcada `pit_safe`, con un test que
   verifica que recalcular con datos futuros no cambia el pasado.
2. **Catálogo con estilo de factor.** Filtra por familia, naturaleza o estilo de
   factor antes de calcular nada.
3. **Transforms de estacionariedad incluidas** (returns, z-score, fracdiff).
4. **Instalación limpia.** Núcleo solo `numpy` + `pandas`. TA-Lib es opcional
   (`pip install pyndicators[talib]`) y solo para un puñado de indicadores DSP.

## Estado

Alpha. ~160 indicadores migrados y verificados (todas las medias, el grueso de
momentum, 33 patrones de vela, VWAP/Ichimoku/Pivots, y custom como el Koncorde).
Inventario completo en `pt.catalog()`.

## Documentación

Ver la carpeta [`docs/`](docs/): instalación, guía de uso, catálogo, diseño interno
y cómo añadir indicadores.

## Licencia

MIT — Luis Leciñana.
