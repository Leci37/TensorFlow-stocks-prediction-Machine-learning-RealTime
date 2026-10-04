import pandas as pd
from ta.volatility import *
from ta.trend import *
from ta.momentum import *
import warnings
warnings.filterwarnings('ignore')

def extract_features(df: pd.DataFrame, debug=False):
    df = df.copy()
    if debug: print('Computing Bollinger Band...')
    bband = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_hi'] = bband.bollinger_hband()
    df['bb_lo'] = bband.bollinger_lband()

    # Calculate the difference between bollinger band and the price for better math expression
    df['bb_hi'] = df['bb_hi'] - df['close']
    df['bb_lo'] = df['close'] - df['bb_lo']

    if debug: print('Computing RSI...')
    for p in [14, 30]: df[f'rsi_{p}'] = RSIIndicator(df['close'], window=p).rsi()

    if debug: print('Computing ADX...')
    for p in [14, 30, 60]:
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=p)
        df[f'adx_{p}'] = adx_indicator.adx()
        df[f'+di_{p}'] = adx_indicator.adx_pos()
        df[f'-di_{p}'] = adx_indicator.adx_neg()

    if debug: print('Computing CCI...')
    for p in [14, 30, 60]: df[f'cci_{p}'] = CCIIndicator(df['high'], df['low'], df['close'], window=p).cci()

    if debug: print('Computing MACD...')
    macd_indicator = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_diff'] = macd_indicator.macd_diff()
    df['macd_signal'] = macd_indicator.macd_signal()

    if debug: print('Computing StochRSI...')
    for p in [14, 21]:
        srsi = StochRSIIndicator(df['close'], window=p, smooth1=3, smooth2=3)
        df[f'srsi_{p}'] = srsi.stochrsi()
        df[f'srsi_{p}_k'] = srsi.stochrsi_k()
        df[f'srsi_{p}_d'] = srsi.stochrsi_d()

    if debug: print('Computing Moving Average...')
    for ma in [3, 7, 14, 21, 60]:
        df[f'MA_{ma}'] = SMAIndicator(df['close'], window=ma).sma_indicator()
        # Calculate the difference between moving average for better math expression
        df[f'MA_{ma}'] = df[f'MA_{ma}'] - df['close']

    # Stationising original data
    df['open'] = df['open'] - df['open'].shift(1)
    df['high'] = df['high'] - df['high'].shift(1)
    df['low'] = df['low'] - df['low'].shift(1)
    df['close'] = df['close'] - df['close'].shift(1)

    return df.dropna()