# Diseño interno

Para entender cómo funciona por dentro y poder extenderlo.

## Componentes

```
src/puretrade/
├── core/
│   ├── spec.py / registry.py   registro de indicadores + metadata
│   ├── engine.py               resuelve dependencias y computa (compute)
│   ├── enums.py                Family, FactorStyle, Nature, OutputType
│   ├── catalog.py              carga data/catalog.csv
│   └── validate.py             pit_check
├── math/                       primitivas puras (rolling, ewm, candle, signal)
├── indicators/                 una familia por fichero
├── transforms/                 estacionariedad y normalización
└── data/catalog.csv            ficha de cada indicador (se envía con la lib)
```

## El registro

Cada indicador es un **productor**: una función que recibe el DataFrame de trabajo
y devuelve un DataFrame con una o más columnas. La metadata viaja con la función
vía el decorador `@register`:

```python
@register(key="mtum_RSI", outputs=("mtum_RSI",), inputs=("close",),
          family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=14)
def _rsi(df, n=14):
    ...
    return pd.DataFrame({"mtum_RSI": ...})
```

Añadir un indicador = registrar una función. No se toca el motor ni el catálogo
global.

## El grafo de dependencias

Un productor puede declarar `depends_on` (otros productores). El motor los ordena
topológicamente y **calcula cada uno una sola vez**, compartiendo resultados:

```python
@register(key="vola_ATR", depends_on=("vola_TRANGE",), ...)
def _atr(df, n=14):
    return pd.DataFrame({"vola_ATR": rma(df["vola_TRANGE"], n)})
```

Pedir `vola_NATR` levanta `TRANGE → ATR → NATR` en orden, sin recomputar nada.

## Point-in-time por diseño

`Nature` decide si un indicador filtra el futuro: los `GlobalStat` (que normalizan
con estadística de toda la serie) no son PIT-safe; el resto sí. `rt_safe_only=True`
usa esta propiedad para excluirlos. `pit_check` lo verifica empíricamente
recomputando sobre una serie truncada.

## Paridad verificada

Cada indicador nativo se valida contra TA-Lib (o py_ti para los `ti_*`) hasta
`< 1e-6` de diferencia. Los tests (`tests/test_momentum.py`, `test_candles.py`…)
se saltan solos si TA-Lib no está instalado, así el núcleo se testea sin él.

Detalles finos replicados de TA-Lib: el sembrado de Wilder (`rma`) y de la EMA
(semilla = SMA de los primeros n), la alineación de la EMA rápida del MACD, y el
suavizado específico del DMI/ADX (semilla = suma de los primeros n-1).
