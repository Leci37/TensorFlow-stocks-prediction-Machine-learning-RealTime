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
