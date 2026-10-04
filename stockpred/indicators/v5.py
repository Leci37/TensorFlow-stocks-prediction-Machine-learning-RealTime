import pandas as pd
import numpy as np

def SMA(df: pd.DataFrame, period: int):
    df[f'MA{period}'] = df['close'].rolling(window=period).mean() - df['close']
    return df

def MACD(df: pd.DataFrame, k=12, d=26, s=9):
    # Create MACD (26, 12)
    macd_k = df['close'].ewm(span=k, adjust=False, min_periods=k).mean()
    # Get the 12-day EMA of the closing price
    macd_d = df['close'].ewm(span=d, adjust=False, min_periods=d).mean()
    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    df['MACD'] = macd_k - macd_d
    # Get the 9-Day EMA of the MACD for the Trigger line
    df['MACD_S'] = df['MACD'].ewm(span=s, adjust=False, min_periods=s).mean()
    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    df['MACD_H'] = df['MACD'] - df['MACD_S']
    return df

# RSI is using the EMA
def RSI(df: pd.DataFrame, period: int):
    delta = df['close'].diff()[1:]
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

    roll_up = up.ewm(com=period-1, adjust=True, min_periods=period).mean()
    roll_down = down.ewm(com=period-1, adjust=True, min_periods=period).mean()
    rsi = roll_up / roll_down
    df[f'RSI{period}'] = 100 - (100/(1 + rsi))
    return df

def RSIStoch(df: pd.DataFrame, period: int, smooth_k=3, smooth_d=3):
    rsi = RSI(df.copy(), period)[f'RSI{period}']

    lowest_low_rsi = rsi.rolling(period).min()
    df[f'RSIStoch{period}'] = (rsi - lowest_low_rsi) / (rsi.rolling(period).max() - lowest_low_rsi)
    df[f'RSIStoch{period}_k'] = df[f'RSIStoch{period}'].rolling(smooth_k).mean()
    df[f'RSIStoch{period}_d'] = df[f'RSIStoch{period}_k'].rolling(smooth_d).mean()
    return df

def CCI(df: pd.DataFrame, period: int):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cci_sma = typical_price.rolling(period).mean()
    cci_mad = typical_price.rolling(period).apply(lambda x: (x - x.mean()).abs().mean())
    # cci_mad = typical_price.rolling(cci_period).apply(lambda x: pd.Series(x).mad())
    df[f'CCI{period}'] = (typical_price - cci_sma) / (0.015 * cci_mad)
    return df

def Bollinger(df: pd.DataFrame, period: int, dev: float):
    mavg = df['close'].rolling(period, min_periods=period).mean()
    mstd = df['close'].rolling(period, min_periods=period).std(ddof=0)
    df[f'BBH{period}_{dev}'] = (mavg + (dev * mstd)) - df['close']
    df[f'BBL{period}_{dev}'] = df['close'] - (mavg - (dev * mstd))
    return df

def Stochastic(df: pd.DataFrame, k: int, d: int):
    n_high = df['high'].rolling(k).max()
    n_low = df['low'].rolling(k).min()
    df[f'Stoch_k_{k}_{d}'] = (df['close'] - n_low) * 100 / (n_high - n_low)
    df[f'Stoch_d_{k}_{d}'] = df[f'Stoch_k_{k}_{d}'].rolling(d).mean()
    return df

# ATR + TR + +-DX + ADX + +-DI
# https://stackoverflow.com/questions/63020750/how-to-find-average-directional-movement-for-stocks-using-pandas
def ADX(df: pd.DataFrame, period: int):
    alpha = 1/period

    # TR
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = np.abs(df['high'] - df['close'].shift(1))
    df['L-C'] = np.abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    # Warning: The EWM has unstable period
    #          It will only be the same value after a long period such as 20X bars from the period
    df[f'ATR{period}'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['high'] - df['high'].shift(1)
    df['pL-L'] = df['low'].shift(1) - df['low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df[f'+DMI{period}'] = (df['+DM']/df[f'ATR{period}'])*100
    df[f'-DMI{period}'] = (df['-DM']/df[f'ATR{period}'])*100
    del df['+DM'], df['-DM']

    # ADX
    df[f'DX{period}'] = (np.abs(df[f'+DMI{period}'] - df[f'-DMI{period}'])/(df[f'+DMI{period}'] + df[f'-DMI{period}']))*100
    df[f'ADX{period}'] = df[f'DX{period}'].ewm(alpha=alpha, adjust=False).mean()
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Generate technical indicator features with OHLC df.

    - The df must contains the columns of : 'open', 'high', 'low', 'close'
    '''

    # Drop volume & spread field
    df = df.copy()
    if 'volume' in df.columns: df = df.drop(columns=['volume'])
    if 'spread' in df.columns: df = df.drop(columns=['spread'])

    # Moving Averages
    for period in [3, 7, 14, 21]:
        df = SMA(df, period)

    # MACD
    df = MACD(df, k=12, d=26, s=9)

    # RSI & CCI & Bollinger & ADX
    for period in [7, 14]:
        df = RSI(df, period)
        df = CCI(df, period)
        df = Bollinger(df, period, dev=2)
        df = ADX(df, period)
        df = RSIStoch(df, period, smooth_k=3, smooth_d=3)

    # Stoch
    df = Stochastic(df, k=14, d=3)
    # df = Stochastic(df, k=21, d=7)

    df['open'] = df['open'] - df['open'].shift(1)
    df['high'] = df['high'] - df['high'].shift(1)
    df['low'] = df['low'] - df['low'].shift(1)
    df['close'] = df['close'] - df['close'].shift(1)

    # automatically drop the first 32 rows without indicator feature
    return df.dropna()