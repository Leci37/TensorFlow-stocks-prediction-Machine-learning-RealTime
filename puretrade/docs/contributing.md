# Cómo añadir un indicador

Migrar o añadir un indicador son ~10 líneas y un test.

## 1. Escribe el productor

En el fichero de su familia (`indicators/momentum.py`, `volatility.py`, …):

```python
from ..core.enums import Family, Nature
from ..core.registry import register
from ..math.rolling import sma

@register(key="mtum_MIINDICADOR", outputs=("mtum_MIINDICADOR",),
          inputs=("close",), family=Family.MOMENTUM, nature=Nature.ROLLING, warmup=20)
def _miindicador(df, n=20):
    return pd.DataFrame({"mtum_MIINDICADOR": sma(df["close"], n)})
```

- `key`: identificador único del productor.
- `outputs`: columnas que genera (una o varias).
- `inputs`: columnas OHLCV que consume.
- `depends_on`: (opcional) otros productores de los que deriva.
- `nature`: decide si es point-in-time (evita `GlobalStat` salvo que de verdad lo sea).

Reutiliza las primitivas de `math/` (`rma`, `ema`, `wma`, `true_range`,
`candle_avg`…) para que la paridad con TA-Lib sea automática.

## 2. Verifica paridad

Compara contra la referencia (TA-Lib o py_ti) sobre datos sintéticos hasta
`< 1e-6`. Regla de oro del proyecto:

> **Solo se registra lo que se puede verificar.** Si un patrón no dispara en los
> datos de prueba o no alcanza paridad exacta, se deja implementado pero **sin
> registrar**, con un TODO — nunca se expone algo sin verificar.

## 3. Añade el test

En `tests/test_*.py`, añade el nombre a la lista parametrizada de paridad
(numéricos) o de PIT-safety (custom sin referencia TA-Lib).

## 4. Actualiza el catálogo

El flag `migrated` y la naturaleza se re-sincronizan desde el registro; no hay que
editar `catalog.csv` a mano para los que ya están en el inventario. Para
indicadores nuevos (no listados), añade su fila.

## Estilo

- Puro `numpy`/`pandas` en el núcleo. TA-Lib solo tras el extra `[talib]`.
- Formato: `ruff`. Tipado: `mypy` (hay `py.typed`).
