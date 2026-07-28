# Changelog

## [0.1.0] - unreleased
### Added
- Andamiaje del núcleo: registro de indicadores, motor con topo-sort de
  dependencias, catálogo empaquetado y validación point-in-time (`pit_check`).
- Familias migradas (parcial): moving_average, momentum, volatility, overlap,
  volume, candles.
- Transforms de estacionariedad: returns, log_returns, diff, fracdiff, zscore.
- Catálogo de ~300 indicadores clasificados (family/nature/engine/pit_safe).

### Candles
- 11 patrones de una vela migrados a numpy/pandas puro con paridad exacta
  verificada contra TA-Lib (DOJI, DRAGONFLYDOJI, GRAVESTONEDOJI, LONGLEGGEDDOJI,
  MARUBOZU, CLOSINGMARUBOZU, SPINNINGTOP, HIGHWAVE, LONGLINE, SHORTLINE, BELTHOLD).
- Primitivas de vela (TA_CandleSettings) replicadas en math/candle.py.
- Catálogo anotado con columna `migrated` (47/318 ya en núcleo puro).

### Candles (cont.)
- +12 patrones de 2 y 3 velas con paridad exacta contra TA-Lib: ENGULFING,
  HARAMI, HARAMICROSS, PIERCING, DARKCLOUDCOVER, MATCHINGLOW, HOMINGPIGEON,
  DOJISTAR, MORNINGSTAR, EVENINGSTAR, 3INSIDE, 3OUTSIDE (23 velas en total).
- Test de paridad ampliado y validado sobre series con tendencia.

### Candles (cont. 2)
- 3WHITESOLDIERS migrado con paridad exacta (lógica real TA-Lib: tolerancias
  Near en aperturas y Far en desaceleración de cuerpos; verificado sobre 500+
  disparos en múltiples semillas). 24 velas nativas en total.
- _3blackcrows y la familia HAMMER (HAMMER/INVERTEDHAMMER/HANGINGMAN/SHOOTINGSTAR)
  implementados parcialmente pero NO registrados: su contexto de tendencia con
  tolerancia Near aún no da paridad exacta. Pendientes de una pasada dedicada.

### Momentum + correctness
- FIX importante: rma (Wilder) y ema ahora replican la inicialización de TA-Lib
  (semilla = SMA de los primeros n). Antes usaban pandas.ewm y NO coincidían;
  esto corrige RSI/ATR/NATR y las 20 medias EMA/DEMA/TEMA ya registradas.
- MACD alineado como TA-Lib (EMA rápida sembrada en slow-1, signal en +sig-1).
- Nuevos momentum con paridad exacta: CMO, ROCP, ROCR, ROCR100, BOP, WILLR (R),
  CCI, APO, PPO, STOCH (k/d), STOCHF (k/d).
- test_momentum.py: 26 comprobaciones de paridad numérica contra TA-Lib.

### Custom: KONCORDE (Blai5)
- KONCORDE reimplementado fiel al Pine Script v2 original: azul (NVI osc),
  marrón, verde (PVI osc), media, rest y 3 crash points. 8 columnas ti_konk_*.
- Corrige 2 bugs de la versión heredada del proyecto: (1) el PVI se calculaba
  con NVI; (2) PVI/NVI usaban el multiplicador de pandas_ta en vez del volumen
  del Pine de Blai5.
- Subcomponentes verificados vs TA-Lib (RSI/MFI/SMA/STDDEV/stoch); PVI/NVI
  replicados exactos del Pine. KONCORDE completo verificado point-in-time.
- Nuevo mtum_MFI con paridad exacta vs TA-Lib. 82/318 indicadores migrados.

### Momentum pesado (DMI/ADX + AROON + ULTOSC + TRIX + STOCH-RSI)
- Familia direccional completa con paridad exacta: PLUS_DM, MINUS_DM, PLUS_DI,
  MINUS_DI, DX, ADX, ADXR. Requirió el suavizado Wilder específico de TA-Lib
  (semilla = suma de los primeros n-1), añadido como math.rolling.dmi_smooth.
- ADX depende de DMI y ADXR de ADX vía el grafo topológico del motor.
- AROON (up/down/osc), ULTOSC, TRIX, STOCH-RSI (k/d) con paridad exacta.
