# puretrade

**Indicadores técnicos para machine learning financiero — en puro `numpy` + `pandas`.**

Sin TA-Lib obligatorio, sin nada que compilar. `pip install puretrade` y funciona.

```bash
pip install puretrade
```

```python
import puretrade as pt

features = pt.compute(ohlcv_df)                    # todos los indicadores
features = pt.compute(ohlcv_df, ["mtum_RSI"])      # solo algunos
features = pt.compute(ohlcv_df, rt_safe_only=True) # solo point-in-time (sin leakage)
catalog  = pt.catalog()                            # ficha de cada feature
```

## Por qué otra librería de indicadores

Hay decenas de envoltorios de TA-Lib. `puretrade` es distinta en cuatro cosas:

1. **Point-in-Time garantizado.** Cada feature está marcada `pit_safe` y hay un
   test (`pit_check`) que verifica que recalcular con datos futuros no cambia el
   pasado. Sin look-ahead bias.
2. **Catálogo con estilo de factor.** Filtra por familia, naturaleza, estilo de
   factor o seguridad point-in-time antes de calcular nada.
3. **Transforms de estacionariedad incluidas** (returns, z-score, fracdiff).
4. **Instalación limpia.** Núcleo solo `numpy` + `pandas`. TA-Lib es opcional
   (`pip install puretrade[talib]`) y solo para un puñado de indicadores DSP.

## Estado

Alpha. El andamiaje (motor, registro, catálogo, validación PIT) funciona y hay
varias familias migradas. El catálogo lista ~300 indicadores; la migración desde
el código legado está en curso — ver `src/puretrade/data/catalog.csv`.

## Licencia

MIT — Luis Leciñana.
