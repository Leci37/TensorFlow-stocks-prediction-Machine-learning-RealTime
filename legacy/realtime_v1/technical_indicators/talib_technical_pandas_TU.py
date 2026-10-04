#https://pypi.org/project/TAcharts/
'''py_ti
Indicators With Chart Functionality
Ichimoku(df=None, filename=None, interval=None): Ichimoku Cloud  KK
Renko(df=None, filename=None, interval=None): Renko Chart  KK
double_smooth(src, n_slow, n_fast): The smoothed value of two EMAs KK TODO
mmo(src, n=2): Murrey Math oscillator of src  KK
td_sequential(src, n=2): TD sequential of src across n periods  _KK
'''
#import py_ti
import numpy as np
import pandas as pd
#import TAcharts
#from TAcharts.utils.ohlcv import OHLCV
from stocktrends import indicators

from Utils import UtilsL, Utils_Yfinance

stockId = "MSFT"
stockId = "MELI"


#https://github.com/cartercarlson/TAcharts/blob/2ff89fb37e8b6996d401015a2c6f69dca036ca06/TAcharts/utils/area_between.py
#@pd_series_to_np_array
def crossover(x1, x2):
    """ Find all instances of intersections between two lines """

    x1_gt_x2 = x1 > x2
    cross = np.diff(x1_gt_x2)
    cross = np.insert(cross, 0, False)
    cross_indices = np.flatnonzero(cross)
    return cross_indices

#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1
       #raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y



#https://github.com/cartercarlson/TAcharts/blob/2ff89fb37e8b6996d401015a2c6f69dca036ca06/TAcharts/utils/area_between.py
#@pd_series_to_np_array
def area_between(line1, line2):
    """ Return the area between line1 and line2 """

    diff = line1 - line2
    x1 = diff[:-1]
    x2 = diff[1:]

    triangle_area = sum(abs(x2 - x1) * 0.5)
    square_area = sum(np.min(zip(x1, x2), axis=1))
    _area_between = triangle_area + square_area

    return _area_between


#https://stackoverflow.com/questions/28477222/python-pandas-calculate-ichimoku-chart-components
def get_clould_Ichimoku(df):
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    df2 = df.copy()
    period9_high = df2['High'].rolling(window=9).max() #pd.rolling(high_prices, window=9)
    period9_low = df2['Low'].rolling(window=9).min()
    df2['ichi_tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = df2['High'].rolling( window=26).max()
    period26_low = df2['Low'].rolling(window=26).min()
    df2['ichi_kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    df2['ichi_senkou_a'] = ((df2['ichi_kijun_sen'] + df2['ichi_tenkan_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = df2['High'].rolling( window=52).max()
    period52_low = df2['Low'].rolling( window=52).min()
    df2['ichi_senkou_b'] = ((period52_high + period52_low) / 2).shift(26)

    df2["ichi_isin_cloud"] = 0
    #df = df[df['closing_price'].between(99, 101)]
    #AVOID_NOISE = 1.00  #2%, para estar en la nube tiene que estar bien metido

    df2.loc[ ( (df2['Close'] > df2['ichi_senkou_b']) & (df2['Close'] < df2['ichi_senkou_a']) ), "ichi_isin_cloud"] = 1
    df2.loc[ ( (df2['Close'] < df2['ichi_senkou_b']) & (df2['Close']> df2['ichi_senkou_a']) ), "ichi_isin_cloud"] = -1

    df2 = Utils_Yfinance.get_crash_points(df2, 'ichi_senkou_a', 'ichi_senkou_b', col_result ="ichi_crash", highlight_result_in_next_cell =2)
    #df2.groupby('ichi_isin_cloud')['Close'].count()

    # The most current closing price plotted 22 time periods behind (optional)
    df2['ichi_chikou_span'] = df2['Close'].shift(-22)  # 22 according to investopedia

    return df2


def get_Renko(df, brick_size = 4):
    df.columns = [i.lower() for i in df.columns]
    renko = indicators.Renko(df)
    # print('\n\nRenko box calcuation based on periodic close')
    renko.brick_size = brick_size
    renko.chart_type = indicators.Renko.PERIOD_CLOSE
    data = renko.get_ohlc_data()
    print(data.head())
    print('\n\nRenko box calcuation based on price movement')
    renko.chart_type = indicators.Renko.PRICE_MOVEMENT
    data_2 = renko.get_ohlc_data()

    data_2['tend_renko_change'] = data_2['uptrend'] != data_2['uptrend'].shift(1)
    data_2.rename(columns={'date': 'Date', 'open': 'Open','high': 'High', 'low': 'Low','close': 'Close','uptrend': 'tend_renko_uptrend' }, inplace=True)
    return data_2


def murrey_Math_Oscillator( df, n=14):
    """ Returns the Murrey Math Oscillator of the close
    No se encuentra la formula se ponen df['Close']  para sustituir un vacio de la libreria from TAcharts.indicators.mmo """
    # Donchian channel
    highest = df['High'].rolling(window=n).max()
    lowest = df['Low'].rolling(window=n).min()

    rng = highest - lowest

    # Oscillator
    rng_multiplier = rng * 0.125
    midline = lowest + rng_multiplier * 4

    _mmo = (df['Close'] - midline) / rng
    _mmo[:n] = 0

    return _mmo


def td_sequential_pure(df_close, n=14):
    """ Returns the TD sequential of the close
    Candles 8 and 9 exceed the low of candles 6 and 7 during a downtrend market
Candle 1 appears as a bearish price candle"""
    old_gt_new = df_close[:-n].reset_index(drop=True) > df_close[n:].reset_index(drop=True)  #df_close['Close'].reset_index(drop=True)
    diff_lst = np.diff(old_gt_new)
    diff_lst = np.insert(diff_lst, 0, False)

    _td_sequential = [0 for _ in range(n)]

    for diff in diff_lst:
        if not diff:
            _td_sequential.append(_td_sequential[-1] + 1)
        else:
            _td_sequential.append(1)

    return _td_sequential

def td_sequential_signo(df_close, n=14):
    """ Returns the TD sequential of the close
    Candles 8 and 9 exceed the low of candles 6 and 7 during a downtrend market
Candle 1 appears as a bearish price candle"""
    old_gt_new = df_close['Close'][:-n].reset_index(drop=True) > df_close['Close'][n:].reset_index(drop=True)  #df_close['Close'].reset_index(drop=True)
    diff_lst = np.diff(old_gt_new)
    diff_lst = np.insert(diff_lst, 0, False)

    #df_close.insert(loc=len(df_close.columns), column='change_is_pos', value=False)
    pd.options.mode.chained_assignment = None
    df_close['change_is_pos'] = (df_close['Close'] - df_close['Close'].shift(1)) > 0
    pd.options.mode.chained_assignment = 'warn'

    _td_sequential = [0 for _ in range(n)]
    len(df_close[n:]) #sin los n primeros
    #for diff in diff_lst:
    for i in range(0, len(diff_lst)):
        restar_o_sumar = 0
        if df_close['change_is_pos'][i+n]:
            restar_o_sumar = 1
        else:
            restar_o_sumar = -1
        if not diff_lst[i]:
            _td_sequential.append(_td_sequential[-1] + restar_o_sumar)
        else:
            _td_sequential.append(restar_o_sumar)

    return _td_sequential




# Function to calculate average true range
def ATR(DF, n):
    df = DF.copy() # making copy of the original dataframe
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))# high -previous close
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1)) #low - previous close
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis =1, skipna = False) # True range
    df['ATR'] = df['TR'].rolling(n).mean() # average –true range
    df = df.drop(['H-L','H-PC','L-PC'], axis =1) # dropping the unneccesary columns
    df.dropna(inplace = True) # droping null items
    return df

def get_Renko_2(df_r, days_back = 26):
    # RENKO
    df_aux = ATR(df_r[['High','Low' ,'Close']], days_back)
    df_r['TR'] = df_aux['TR']
    df_r['ATR'] = df_aux['ATR']
    df_r['tend_renko_brick'] = round(df_r["ATR"], 0)
    # df_r['tend_renko_change'] = df_r['tend_renko_brick'] != df_r['tend_renko_brick'].shift(1)
    df_r['tend_renko_change'] = 0
    df_r.loc[(df_r['tend_renko_brick'] > df_r['tend_renko_brick'].shift(1)), 'tend_renko_change'] = 1
    df_r.loc[(df_r['tend_renko_brick'] < df_r['tend_renko_brick'].shift(1)), 'tend_renko_change'] = -1

    df_r.rename(columns={'TR': 'tend_renko_TR', 'ATR': 'tend_renko_ATR'}, inplace=True)
    return df_r



def get_all_pandas_TU_tecnical(df_TU, cos_cols = None):
    df_renko = None
    df_Ichi = None

    if cos_cols is None or "mtum_murrey_math" in cos_cols:
        df_TU['mtum_murrey_math'] = murrey_Math_Oscillator(df_TU)
    if cos_cols is None or "mtum_td_seq" in cos_cols:
        df_TU['mtum_td_seq'] = td_sequential_pure(df_TU['Close'])
    if cos_cols is None or "mtum_td_seq_sig" in cos_cols:
        df_TU['mtum_td_seq_sig'] = td_sequential_signo(df_TU[['Date', 'Close']])

    if cos_cols is None or "tend_hh" in cos_cols or "tend_hl" in cos_cols or "tend_ll" in cos_cols or "tend_lh" in cos_cols  or "tend_hh_crash" in cos_cols or "tend_hl_crash" in cos_cols or "tend_ll_crash" in cos_cols or "tend_lh_crash" in cos_cols:
        df_TU = get_LowerHighs_LowerHighs(df_TU)

    if cos_cols is None or "ichi_tenkan_sen" in cos_cols or "ichi_kijun_sen" in cos_cols or "ichi_senkou_a" in cos_cols or "ichi_senkou_b" in cos_cols or "ichi_isin_cloud" in cos_cols or "ichi_crash" in cos_cols or "ichi_chikou_span" in cos_cols:
        df_Ichi = get_clould_Ichimoku(df_TU)

    if cos_cols is None or "tend_renko_TR" in cos_cols or "tend_renko_ATR" in cos_cols or "tend_renko_brick" in cos_cols or "tend_renko_change" in cos_cols:
        df_renko = get_Renko_2(df_TU)

    if (df_renko is not None) and (df_Ichi is not None):
        df_TU = pd.merge(df_Ichi, df_renko)  # df_Ichi.append(df_renko)
    elif df_Ichi is not None:
        df_TU = df_Ichi
    elif df_renko is not None:
        df_TU = df_renko

    df_TU = UtilsL.replace_bat_chars_in_columns_name(df_TU, "")

    return df_TU

'''
Bull & Bear Power Index
https://stackoverflow.com/questions/64830383/calculating-bull-bear-markets-in-pandas
'''


def deleter(Data, index, times):
    for i in range(1, times + 1):
        Data = np.delete(Data, index, axis=1)
    return Data


def jump(Data, jump):
    Data = Data[jump:, ]

    return Data

def ma(Data, lookback, what, where):
    for i in range(len(Data)):
        try:
            Data[i, where] = (Data[i - lookback + 1:i + 1, what].mean())

        except IndexError:
            pass


    return Data


def ema(Data, alpha, lookback, what, where):
    # alpha is the smoothing factor
    # window is the lookback period
    # what is the column that needs to have its average calculated
    # where is where to put the exponential moving average

    alpha = alpha / (lookback + 1.0)
    beta = 1 - alpha

    # First value is a simple SMA
    Data = ma(Data, lookback, what, where)

    # Calculating first EMA
    Data[lookback + 1, where] = (Data[lookback + 1, what] * alpha) + (Data[lookback, where] * beta)
    # Calculating the rest of EMA
    for i in range(lookback + 2, len(Data)):
        try:
            Data[i, where] = (Data[i, what] * alpha) + (Data[i - 1, where] * beta)

        except IndexError:
                pass
    return Data




#TODO DARVAS BOX https://www.tradingview.com/script/uiwRnGlw/
#esta en matlab https://github.com/vakilp/darvasBox

'''
LowerHighs_LowerHighs
https://raposa.trade/blog/higher-highs-lower-lows-and-calculating-price-trends-in-python/ 
'''
def get_LowerHighs_LowerHighs(df, order=5, k=2, get_crash_point = True):
    hh = getHigherHighs(df['Close'].values, order, K=k) #- close_values
    if not hh:
        hh = getHigherHighs(df['Close'].values, order-1, K=k)
    hl = getHigherLows(df['Close'].values, order, K=k) #- close_values
    if not hl:
        hl = getHigherLows(df['Close'].values, order-1, K=k)
    ll = getLowerLows(df['Close'].values, order, K=k) #- close_values
    if not ll:
        ll = getLowerLows(df['Close'].values, order-1, K=k)
    lh = getLowerHighs(df['Close'].values, order, K=k) #- close_values
    if not lh:
        lh = getLowerHighs(df['Close'].values, order-1, K=k)

    clean_LowerHighs_LowerHighs(df, hh, prefix_name="hh")
    clean_LowerHighs_LowerHighs(df, hl, prefix_name="hl")
    clean_LowerHighs_LowerHighs(df, ll, prefix_name="ll")
    clean_LowerHighs_LowerHighs(df, lh, prefix_name="lh")

    if get_crash_point:
        df = Utils_Yfinance.get_crash_points(df, 'tend_hh', 'Close', col_result="tend_hh_crash", highlight_result_in_next_cell=1)
        df = Utils_Yfinance.get_crash_points(df, 'tend_hl', 'Close', col_result="tend_hl_crash", highlight_result_in_next_cell=1)
        df = Utils_Yfinance.get_crash_points(df, 'tend_ll', 'Close', col_result="tend_ll_crash", highlight_result_in_next_cell=1)
        df = Utils_Yfinance.get_crash_points(df, 'tend_lh', 'Close', col_result="tend_lh_crash", highlight_result_in_next_cell=1)

    return df


def clean_LowerHighs_LowerHighs(df, hh, prefix_name):
    df_aux = pd.concat([df['Close'][i] for i in hh])  # .fillna(method='ffill')
    df['tend_' + prefix_name] = df_aux[~df_aux.index.duplicated(keep='first')]
    df['tend_' + prefix_name] = df['tend_' + prefix_name].fillna(method='ffill')


'''https://raposa.trade/blog/higher-highs-lower-lows-and-calculating-price-trends-in-python/ '''
from collections import deque
from scipy.signal import argrelextrema
def getHigherLows(data: np.array, order=5, K=2):
  '''
  Finds consecutive higher lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be higher.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are higher than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] < lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerHighs(data: np.array, order=5, K=2):
  '''
  Finds consecutive lower highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be lower.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are lower than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] > highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getHigherHighs(data: np.array, order=5, K=2):
  '''
  Finds consecutive higher highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be higher.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] < highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerLows(data: np.array, order=5, K=2):
  '''
  Finds consecutive lower lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be lower.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are lower than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] > lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema
# df = yhoo_history_stock.get_historial_data_3_month(stockId, prepos=False)
#
# ##WARN avoid TAcharts.indicators.ichimoku import Ichimoku
# df = get_all_pandas_TU_tecnical(df)


