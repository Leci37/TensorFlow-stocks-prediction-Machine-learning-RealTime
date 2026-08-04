"""Vocabulario cerrado de la librería: los ejes por los que se clasifica cada indicador."""
from enum import Enum


class Family(str, Enum):
    """Eje 1 — qué mide el indicador (familia financiera clásica)."""
    MOMENTUM = "Momentum"
    MOVING_AVERAGE = "MovingAverage"
    OVERLAP = "Overlap"
    TREND = "Trend"
    VOLATILITY = "Volatility"
    VOLUME = "Volume"
    STATISTICS = "Statistics"
    CYCLE = "Cycle"
    PERFORMANCE = "Performance"
    CANDLE = "Candle"
    CUSTOM = "Custom"


class FactorStyle(str, Enum):
    """Eje quant — qué anomalía de mercado explota (para diversificar señales)."""
    TREND_MOMENTUM = "trend/momentum"
    MEAN_REVERSION = "mean-reversion"
    VOLATILITY_RISK = "volatility/risk"
    FLOW_LIQUIDITY = "flow/liquidity"
    MICROSTRUCTURE = "microstructure/shape"
    CYCLE_SEASONALITY = "cycle/seasonality"
    STATISTICAL = "statistical"
    RETURN = "return"
    MIXED = "mixed"


class Nature(str, Enum):
    """Eje 2 — cómo se calcula en el tiempo. Decide si es point-in-time seguro."""
    ROLLING = "Rolling"          # ventana trasera fija -> PIT safe
    CUMULATIVE = "Cumulative"    # acumula desde el inicio -> PIT safe (no shift-invariante)
    PATTERN = "Pattern"          # geometría de la vela -> PIT safe
    DERIVED = "Derived"          # depende de otros indicadores -> hereda del padre
    GLOBAL_STAT = "GlobalStat"   # normaliza con estadística global -> NO PIT safe (leakage)


class OutputType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


# Naturalezas que NO son point-in-time seguras (filtran información del futuro).
PIT_UNSAFE = {Nature.GLOBAL_STAT}
