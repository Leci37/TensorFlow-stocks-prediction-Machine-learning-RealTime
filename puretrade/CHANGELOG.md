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
