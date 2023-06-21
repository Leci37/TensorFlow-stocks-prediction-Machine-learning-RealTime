import talib
import pandas as pd

def extract_features(df: pd.DataFrame, field = 'close'):
    df = df.copy()
    df['avg'] = (df['high'] + df['low']) / 2
    df['sma'] = talib.SMA(df[field], timeperiod=20) - df[field]
    df['ema'] = talib.EMA(df[field], timeperiod=20) - df[field]
    df['rsi'] = talib.RSI(df[field], timeperiod=14)

    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df[field], fastperiod=12, slowperiod=26, signalperiod=9)
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df[field], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df[field], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    df['upperband'] = df['upperband'] - df[field]
    df['middleband'] = df['middleband'] - df[field]
    df['lowerband'] = df['lowerband'] - df[field]

    df['avg'] = df['avg'] - df['avg'].shift(1)
    df['open'] = df['open'] - df['open'].shift(1)
    df['high'] = df['high'] - df['high'].shift(1)
    df['low'] = df['low'] - df['low'].shift(1)
    df['close'] = df['close'] - df['close'].shift(1)
    return df.dropna()