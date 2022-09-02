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

import UtilsL
import Utils_Yfinance
import yhoo_history_stock

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

    df2 = Utils_Yfinance.get_crash_points(df2, 'ichi_senkou_a','ichi_senkou_b', col_result = "ichi_crash"  )
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

    df_close['change_is_pos'] = False
    df_close['change_is_pos'] = (df_close['Close'] - df_close['Close'].shift(1)) > 0

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



def get_all_pandas_TU_tecnical(df_TU):
    df_TU['mtum_murrey_math'] = murrey_Math_Oscillator(df_TU)
    df_TU['mtum_td_seq'] = td_sequential_pure(df_TU['Close'])
    df_TU['mtum_td_seq_sig'] = td_sequential_signo(df_TU[['Date', 'Close']])
    df_Ichi = get_clould_Ichimoku(df_TU)
    df_renko = get_Renko_2(df_TU)
    # print(df_renco.head())
    # df.ta.td_seq(asint=True, show_all=True, append=True)
    df_TU = pd.merge(df_Ichi, df_renko)  # df_Ichi.append(df_renko)

    df_TU = UtilsL.replace_bat_chars_in_columns_name(df_TU, "")

    return df_TU

# df = yhoo_history_stock.get_historial_data_3_month(stockId, prepos=False)
#
# ##WARN avoid TAcharts.indicators.ichimoku import Ichimoku
# df = get_all_pandas_TU_tecnical(df)


