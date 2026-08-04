# El catálogo

Cada indicador se clasifica por **tres ejes ortogonales**. Entenderlos es la clave
para elegir bien qué generar.

## Eje 1 — Familia (qué mide)

La familia financiera clásica. Define el módulo donde vive el indicador.

| Familia | Ejemplos |
|---|---|
| `Momentum` | RSI, MACD, ADX, STOCH, CCI, AO |
| `MovingAverage` | SMA, EMA, DEMA, TEMA, WMA, KAMA, T3 |
| `Overlap` | Bollinger, MIDPOINT, VWAP, Pivots |
| `Trend` | Ichimoku, PSAR |
| `Volatility` | ATR, NATR, TRANGE |
| `Volume` | OBV, Chaikin AD/ADOSC, EFI |
| `Candle` | los patrones de vela |
| `Custom` | Koncorde, supertrend, vortex, donchian, keltner |

## Eje 2 — Naturaleza (cómo se calcula en el tiempo)

Es el eje que decide si un indicador es **point-in-time seguro**.

| Naturaleza | Cómo | PIT-safe |
|---|---|---|
| `Rolling` | ventana trasera fija | ✅ |
| `Cumulative` | acumula desde el inicio (OBV, VWAP) | ✅ |
| `Pattern` | geometría de la vela | ✅ |
| `Derived` | depende de otros indicadores | hereda del padre |
| `GlobalStat` | usa estadística global de toda la serie | ❌ filtra futuro |

## Eje 3 — Estilo de factor (qué anomalía explota)

El eje que interesa a un quant para **diversificar** señales: `trend/momentum`,
`mean-reversion`, `volatility/risk`, `flow/liquidity`, `microstructure/shape`,
`cycle/seasonality`, `statistical`, `return`.

!!! note
    De momento el `factor_style` se asigna de forma **gruesa** por familia (p. ej.
    todo `Momentum` → `trend/momentum`). Refinarlo por indicador (marcar RSI como
    `mean-reversion`, etc.) es trabajo pendiente.

## Explorar y filtrar

```python
cat = pt.catalog()

# Cuántos hay por familia
cat.groupby("family").size()

# Momentum disponible y point-in-time
cat[(cat.family == "Momentum") & cat.migrated & cat.pit_safe]

# Por estilo de factor (p. ej. riesgo/volatilidad)
cat[cat.factor_style == "volatility/risk"]

# Los que NO son point-in-time (a vigilar en backtest)
cat[~cat.pit_safe]
```

## Campos de la ficha

| Campo | Significado |
|---|---|
| `name` | nombre de la columna de salida |
| `family` / `factor_style` / `nature` | los tres ejes |
| `pit_safe` | ¿es point-in-time? |
| `engine` | `native` (puro) o `talib` (extra opcional) |
| `warmup` | filas hasta valor válido |
| `output` | `continuous` o `categorical` |
| `migrated` | ¿ya implementado y verificado? |

El catálogo se envía **empaquetado** con la librería (`data/catalog.csv`, en CSV
para no arrastrar `pyarrow` y mantener la instalación limpia).
