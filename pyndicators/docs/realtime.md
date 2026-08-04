# Tiempo real (streaming)

Caso típico: **cuando cierra la vela de 5 minutos, generar todos los indicadores;
cuando cierra la de 1 minuto, generar los de ese marco; etc.**

Sí es posible, y de forma directa. La clave es que **todos los indicadores son
point-in-time**: el valor de la última fila calculada es el valor "en vivo"
correcto, sin mirar al futuro.

## El patrón: recalcular al cerrar vela

Mantén un *buffer* con las últimas N barras. Cuando una vela cierra, la añades,
llamas a `compute()` sobre el buffer y te quedas con **la última fila**.

```python
import collections
import pandas as pd
import pyndicators as pt


class LiveIndicators:
    """Genera features al cerrar cada vela, sobre una ventana trasera."""

    def __init__(self, maxlen=500, **compute_kwargs):
        self.buf = collections.deque(maxlen=maxlen)
        self.kwargs = compute_kwargs            # p. ej. family="Momentum"

    def on_bar_close(self, ts, o, h, l, c, v=0.0):
        self.buf.append((ts, o, h, l, c, v))
        df = (pd.DataFrame(self.buf, columns=["ts", "open", "high", "low", "close", "volume"])
                .set_index("ts"))
        return pt.compute(df, **self.kwargs).iloc[-1]   # fila de features "en vivo"


live = LiveIndicators(maxlen=500)

# en tu callback de "vela cerrada":
row = live.on_bar_close(ts, o, h, l, c, v)
rsi = row["mtum_RSI"]
```

`compute()` sobre unos cientos de barras tarda milisegundos, de sobra para 1 min /
5 min. Filtra lo que calculas con `family=` / `features=` para ir aún más rápido.

## Tamaño del buffer: importa

No todos los indicadores se comportan igual al recortar la historia. Medido sobre
el valor "en vivo" (ventana trasera) frente a recalcular toda la historia:

| Tipo de indicador | buffer 100 | buffer 300 | Recomendación |
|---|---|---|---|
| Rolling (SMA, WMA, STOCH, Donchian) y **velas** | exacto | exacto | `warmup + 1` basta |
| Recursivos (EMA, RSI, ATR, MACD, ADX) | ~1e-2 (deriva) | ~1e-9 (ruido) | **≈ 300**, o ~10× el periodo mayor |
| Acumulados sin ancla (OBV, Chaikin AD) | nivel distinto | nivel distinto | **histórico completo** |
| Anclados a sesión (VWAP, Pivots) | ok si cubre la sesión | ok | buffer ≥ 1–2 sesiones |

Reglas prácticas:

- **Buffer por defecto de ~500 barras** cubre bien el caso general: las medias y
  velas salen exactas y los recursivos convergen a ruido de float.
- **OBV / Chaikin AD** dependen del punto de partida absoluto: usa histórico
  completo, o quédate con su *pendiente/cambios* (el nivel es arbitrario).
- **VWAP / Pivot Points** se anclan al día: el buffer debe cubrir la sesión actual
  (y la anterior para pivots). Requieren índice `DatetimeIndex`.

!!! tip
    Los recursivos "olvidan" el pasado lejano de forma geométrica, por eso un
    buffer de ~10× el periodo más largo los deja indistinguibles de la historia
    completa. Si necesitas exactitud total en OBV/ATR desde el minuto uno, arranca
    el buffer con el histórico ya acumulado del día.

## Varios marcos temporales a la vez (1m, 5m, 15m…)

Un buffer (y una instancia) **por marco**. Agrega el stream de 1 min en velas de
5 min / 15 min y dispara `compute()` cuando cada una cierra:

```python
engines = {
    "1min":  LiveIndicators(maxlen=500, family="Momentum"),
    "5min":  LiveIndicators(maxlen=500),            # todos
    "15min": LiveIndicators(maxlen=500, family="Volatility"),
}

def on_bar_close(timeframe, ts, o, h, l, c, v):
    return engines[timeframe].on_bar_close(ts, o, h, l, c, v)
```

Para construir las velas de 5 min desde ticks o velas de 1 min, `pandas` resamplea:

```python
bars_5m = (ticks.resample("5min")
                 .agg(open=("price", "first"), high=("price", "max"),
                      low=("price", "min"),  close=("price", "last"),
                      volume=("size", "sum")))
```

Genera solo cuando la vela está **cerrada** (no en formación): el valor de una vela
a medio formar cambia hasta el cierre.

## ¿Y streaming incremental O(1)?

El patrón de arriba **recalcula** la ventana en cada cierre (rápido y siempre
correcto). Un modo incremental real —mantener estado por indicador y actualizar
solo con la barra nueva, sin recomputar— es más eficiente para miles de símbolos a
la vez, pero exige estado por indicador. Está en el *roadmap*; para uno o pocos
símbolos en 1m/5m, recalcular la ventana va sobrado.
