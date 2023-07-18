#https://pypi.org/project/TAcharts/
'''py_ti
Indicators With Chart Functionality
Ichimoku(df=None, filename=None, interval=None): Ichimoku Cloud  KK
Renko(df=None, filename=None, interval=None): Renko Chart  KK
double_smooth(src, n_slow, n_fast): The smoothed value of two EMAs KK TODO
mmo(src, n=2): Murrey Math oscillator of src  KK
td_sequential(src, n=2): TD sequential of src across n periods  _KK
'''
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from features_W3_old.ta.utils import *

#https://stackoverflow.com/questions/28477222/python-pandas-calculate-ichimoku-chart-components
def get_clould_Ichimoku(df):
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    df2 = df.copy()
    period9_high = df2['high'].rolling(window=9).max() #pd.rolling(high_prices, window=9)
    period9_low = df2['low'].rolling(window=9).min()
    df2['ichi_tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = df2['high'].rolling( window=26).max()
    period26_low = df2['low'].rolling(window=26).min()
    df2['ichi_kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    df2['ichi_senkou_a'] = ((df2['ichi_kijun_sen'] + df2['ichi_tenkan_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = df2['high'].rolling( window=52).max()
    period52_low = df2['low'].rolling( window=52).min()
    df2['ichi_senkou_b'] = ((period52_high + period52_low) / 2).shift(26)

    df2["ichi_isin_cloud"] = 0
    #df = df[df['closing_price'].between(99, 101)]
    #AVOID_NOISE = 1.00  #2%, para estar en la nube tiene que estar bien metido

    df2.loc[ ( (df2['close'] > df2['ichi_senkou_b']) & (df2['close'] < df2['ichi_senkou_a']) ), "ichi_isin_cloud"] = 1
    df2.loc[ ( (df2['close'] < df2['ichi_senkou_b']) & (df2['close']> df2['ichi_senkou_a']) ), "ichi_isin_cloud"] = -1

    df2 = get_crash_points(df2, 'ichi_senkou_a', 'ichi_senkou_b', col_result ="ichi_crash", highlight_result_in_next_cell =2)
    #df2.groupby('ichi_isin_cloud')['close'].count()

    # DONT USE is look some future , not valid real time
    # The most current closing price plotted 22 time periods behind (optional)
    # df2['ichi_chikou_span'] = df2['close'].shift(-22)  # 22 according to investopedia

    return df2



def td_sequential_pure(df_close, n=14):
    """ Returns the TD sequential of the close
    Candles 8 and 9 exceed the low of candles 6 and 7 during a downtrend market
Candle 1 appears as a bearish price candle"""
    old_gt_new = df_close[:-n].reset_index(drop=True) > df_close[n:].reset_index(drop=True)  #df_close['close'].reset_index(drop=True)
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
    old_gt_new = df_close['close'][:-n].reset_index(drop=True) > df_close['close'][n:].reset_index(drop=True)  #df_close['close'].reset_index(drop=True)
    diff_lst = np.diff(old_gt_new)
    diff_lst = np.insert(diff_lst, 0, False)

    #df_close.insert(loc=len(df_close.columns), column='change_is_pos', value=False)
    pd.options.mode.chained_assignment = None
    df_close['change_is_pos'] = (df_close['close'] - df_close['close'].shift(1)) > 0
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
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))# high -previous close
    df['L-PC'] = abs(df['low'] - df['close'].shift(1)) #low - previous close
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis =1, skipna = False) # True range
    df['ATR'] = df['TR'].rolling(n).mean() # average –true range
    df = df.drop(['H-L','H-PC','L-PC'], axis =1) # dropping the unneccesary columns
    df.dropna(inplace = True) # droping null items
    return df

def get_Renko_2(df_r, days_back = 26):
    # RENKO
    df_aux = ATR(df_r[['high','low' ,'close']], days_back)
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
    # print("get_all_pandas_TU_tecnical")
    df_renko = None
    df_Ichi = None
    #DONT USE is repeted "mtum_murrey_math"  is the same 'tend_PSARaf_002_02'
    # if cos_cols is None or "mtum_murrey_math" in cos_cols:
    #     df_TU['mtum_murrey_math'] = murrey_Math_Oscillator(df_TU)
    if cos_cols is None or "mtum_td_seq" in cos_cols:
        df_TU['mtum_td_seq'] = td_sequential_pure(df_TU['close'])
    if cos_cols is None or "mtum_td_seq_sig" in cos_cols:
        df_TU['mtum_td_seq_sig'] = td_sequential_signo(df_TU[['close']])
    #DONT USE is look some future , not valid real time
    # if cos_cols is None or "tend_hh" in cos_cols or "tend_hl" in cos_cols or "tend_ll" in cos_cols or "tend_lh" in cos_cols  or "tend_hh_crash" in cos_cols or "tend_hl_crash" in cos_cols or "tend_ll_crash" in cos_cols or "tend_lh_crash" in cos_cols:
    #     df_TU = get_lowerhighs_lowerhighs(df_TU)

    if cos_cols is None or "ichi_tenkan_sen" in cos_cols or "ichi_kijun_sen" in cos_cols or "ichi_senkou_a" in cos_cols or "ichi_senkou_b" in cos_cols or "ichi_isin_cloud" in cos_cols or "ichi_crash" in cos_cols :
        df_TU = get_clould_Ichimoku(df_TU)
    #DONT USE is repeted tend_renko_ATR is the same than tend_renko_brick
    if cos_cols is None or "tend_renko_TR" in cos_cols or "tend_renko_brick" in cos_cols or "tend_renko_change" in cos_cols:
        df_TU = get_Renko_2(df_TU)
        df_TU = df_TU.drop(columns=['tend_renko_ATR'] )


    # if (df_renko is not None) and (df_Ichi is not None):
    #     df_TU = pd.merge(df_Ichi, df_renko)  #df_Ichi.append(df_renko) #
    # elif df_Ichi is not None:
    #     df_TU = df_Ichi
    # elif df_renko is not None:
    #     df_TU = df_renko

    df_TU = replace_bad_chars_in_columns_name(df_TU, "")

    return df_TU


def td_sequential_pure_TU_async(df_TU):
    df_TU['mtum_td_seq'] = td_sequential_pure(df_TU['close'])
    return df_TU

def td_sequential_signo_TU_async(df_TU): 
    df_TU['mtum_td_seq_sig'] = td_sequential_signo(df_TU[['close']])
    return df_TU

def get_clould_Ichimoku_TU_async(df_TU):
    df_Ichi = get_clould_Ichimoku(df_TU)
    return df_Ichi

def get_Renko_2_TU_async(df_TU):
    df_renko = get_Renko_2(df_TU)
    df_renko = df_renko.drop(columns=['tend_renko_ATR'])
    return df_renko



ALL_PANDAS_TU = [
    td_sequential_pure_TU_async, 
    td_sequential_signo_TU_async,
    get_clould_Ichimoku_TU_async,  
    get_Renko_2_TU_async
]
