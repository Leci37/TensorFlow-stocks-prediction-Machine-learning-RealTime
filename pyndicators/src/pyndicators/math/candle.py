"""Primitivas de vela replicando el sistema de umbrales de TA-Lib. Puro pandas.

TA-Lib no usa umbrales fijos: compara cada medida (cuerpo, sombra, rango) contra
una media móvil de esa medida sobre las velas previas. Aquí se reproduce ese
mecanismo (``TA_CandleSettings``) para lograr paridad exacta sin depender de C.
"""
from __future__ import annotations

import pandas as pd

# Configuración por defecto de TA-Lib: (avg_period, factor, range_type)
#   range_type: "body" = |close-open| · "hl" = high-low · "shadows" = us+ls
SETTINGS = {
    "BodyLong":       (10, 1.0, "body"),
    "BodyVeryLong":   (10, 3.0, "body"),
    "BodyShort":      (10, 1.0, "body"),
    "BodyDoji":       (10, 0.1, "hl"),
    "ShadowLong":     (0,  1.0, "body"),
    "ShadowVeryLong": (0,  2.0, "body"),
    "ShadowShort":    (10, 1.0, "shadows"),
    "ShadowVeryShort": (10, 0.1, "hl"),
    "Near":           (5,  0.2, "hl"),
    "Far":            (5,  0.6, "hl"),
    "Equal":          (5,  0.05, "hl"),
}


def parts(df: pd.DataFrame):
    """Devuelve (body, hl, upper_shadow, lower_shadow, is_white)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    hl = h - l
    upper = h - c.where(c >= o, o)   # high - max(open, close)
    lower = c.where(c <= o, o) - l   # min(open, close) - low
    return body, hl, upper, lower, (c >= o)


def _range(df: pd.DataFrame, range_type: str) -> pd.Series:
    body, hl, upper, lower, _ = parts(df)
    if range_type == "body":
        return body
    if range_type == "hl":
        return hl
    return upper + lower  # shadows


def candle_avg(df: pd.DataFrame, setting: str) -> pd.Series:
    """Umbral de TA-Lib para ``setting`` en cada vela.

    Con avg_period>0 usa la media del rango sobre las ``period`` velas ANTERIORES
    (no incluye la actual, igual que TA-Lib). Con period==0 usa el rango de la
    propia vela. El divisor 2 aplica a los tipos 'shadows'.
    """
    period, factor, rtype = SETTINGS[setting]
    rng = _range(df, rtype)
    div = 2.0 if rtype == "shadows" else 1.0
    if period == 0:
        base = rng
    else:
        base = rng.rolling(period, min_periods=period).mean().shift(1)
    return factor * base / div
