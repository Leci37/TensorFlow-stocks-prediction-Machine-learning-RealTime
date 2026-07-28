# Guía de uso

## El flujo básico

```python
import pandas as pd
import puretrade as pt

df = pd.read_csv("ohlcv.csv", parse_dates=["Date"], index_col="Date")
features = pt.compute(df)
```

`compute(df)` devuelve un `DataFrame` con el **mismo índice** que `df` y una
columna por indicador. Las primeras filas llevan `NaN` hasta que cada indicador
tiene suficiente histórico (su *warmup*).

## Cómo elegir qué generar

Este es el punto clave: rara vez quieres los ~160 a la vez. Hay cinco maneras de
acotar, y **se combinan** entre sí.

### 1. Todos

```python
pt.compute(df)
```

### 2. Por familia

Las familias son: `Momentum`, `MovingAverage`, `Overlap`, `Trend`, `Volatility`,
`Volume`, `Statistics`, `Cycle`, `Performance`, `Candle`, `Custom`.

```python
pt.compute(df, family="Volatility")            # ATR, NATR, TRANGE...
pt.compute(df, family="Candle")                # los 33 patrones de vela
pt.compute(df, family="Momentum")              # RSI, MACD, ADX, STOCH, CCI...
pt.compute(df, family=["Momentum", "Volume"])  # varias a la vez
```

### 3. Por nombre concreto

Acepta el nombre del indicador o de cualquiera de sus columnas de salida:

```python
pt.compute(df, ["mtum_RSI", "vola_ATR", "olap_BBAND_UPPER"])
```

Si pides una columna derivada (p. ej. `vola_NATR`), el motor calcula
automáticamente sus dependencias (`ATR` → `TRANGE`) en el orden correcto.

### 4. Por naturaleza de cálculo

`Rolling` (ventana trasera), `Cumulative` (acumulado), `Pattern` (velas
categóricas), `Derived`, `GlobalStat`:

```python
pt.compute(df, nature="Pattern")      # solo patrones de vela (salida -100/0/100)
pt.compute(df, nature="Cumulative")   # OBV, VWAP, Koncorde...
```

### 5. Solo point-in-time (para backtest / ML honesto)

```python
pt.compute(df, rt_safe_only=True)     # excluye cualquier feature con look-ahead
```

### Combinaciones

```python
pt.compute(df, family="Momentum", rt_safe_only=True)
pt.compute(df, family="Candle")       # equivalente a nature="Pattern"
```

## Descubrir qué hay disponible

```python
pt.families()                        # lista de familias
pt.list_features(family="Volume")    # nombres de columnas de una familia
pt.list_features(rt_safe_only=True)  # todo lo point-in-time
```

## El catálogo

`pt.catalog()` devuelve un `DataFrame`-ficha (una fila por indicador) con toda la
metadata. Es la forma de explorar y filtrar con pandas normal:

```python
cat = pt.catalog()
cat[(cat.family == "Momentum") & cat.migrated]      # momentum disponible
cat[cat.factor_style == "volatility/risk"]          # por estilo de factor
cat[~cat.pit_safe]                                   # los que NO son point-in-time
```

Columnas: `name`, `family`, `factor_style`, `nature`, `pit_safe`, `engine`,
`warmup`, `output`, `migrated`.

## Patrones de vela

Salen como columnas categóricas al estilo TA-Lib: **+100** (alcista), **-100**
(bajista), **0** (sin patrón).

```python
velas = pt.compute(df, family="Candle")
velas[velas["cdl_ENGULFING"] != 0]     # barras con envolvente
```

## Transformaciones para ML

Los indicadores en crudo suelen ser no estacionarios. El submódulo `transforms`
los prepara para el modelo:

```python
from puretrade import transforms
transforms.returns(df["close"])
transforms.log_returns(df["close"])
transforms.fracdiff(df["close"], d=0.4)          # estacionariza conservando memoria
transforms.zscore(features["mtum_RSI"], 100)     # z-score con ventana trasera
transforms.cross_sectional_rank(panel_df)        # ranking entre activos
```

## Verificar point-in-time

```python
from puretrade.core.validate import pit_check
pit_check(df, ["mtum_RSI", "vola_ATR"])
# -> tabla con n_compared, n_mismatch, max_abs_diff, pit_safe por feature
```

## Rendimiento

`compute()` calcula cada productor una vez y comparte resultados intermedios por
el grafo de dependencias. Pedir solo lo que necesitas (por familia o nombre) es
más rápido que calcular todo y descartar.
