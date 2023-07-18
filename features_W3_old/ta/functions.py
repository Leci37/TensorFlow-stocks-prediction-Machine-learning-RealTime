import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
#https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html
#https://github.com/CanerIrfanoglu/medium/blob/master/candle_stick_recognition/identify_candlestick.py
import talib
import pandas as pd


#Overlap Studies Functions
#Overlap Studies Functions
#BBANDS - Bollinger Bands
from features_W3_old.ta.utils import *


def get_moving_average_indicator(close):
    df = pd.DataFrame()
    # # DEMA - Double Exponential Moving Average
    # df["ma_DEMA"] = talib.DEMA(close, timeperiod=30)
    # # EMA - Exponential Moving Average
    # df["ma_EMA"] = talib.EMA(close, timeperiod=30)
    # # KAMA - Kaufman Adaptive Moving Average
    # df["ma_KAMA"] = talib.KAMA(close, timeperiod=30)
    # # MA - Moving average
    # # df["olap_MA"] = talib.MA(close, timeperiod=30, matype=0) Sobra porque es la agrupacion de todas las demas MA
    # # SMA - Simple Moving Average
    # df["ma_SMA"] = talib.SMA(close, timeperiod=30)
    # # T3 - Triple Exponential Moving Average (T3)
    # df["ma_T3"] = talib.T3(close, timeperiod=5, vfactor=0)
    # # TEMA - Triple Exponential Moving Average
    # df["ma_TEMA"] = talib.TEMA(close, timeperiod=30)
    # # TRIMA - Triangular Moving Average
    # df["ma_TRIMA"] = talib.TRIMA(close, timeperiod=30)
    # # WMA - Weighted Moving Average
    # df["ma_WMA"] = talib.WMA(close, timeperiod=30)
    # # MAMA - MESA Adaptive Moving Average
    # # df["olap_MAMA"], df["olap_FAMA"] = talib.MAMA(close)
    # # MAVP - Moving average with variable period
    # # df["olap_MAVP"] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
    list_timeperiod = [5, 10, 20, 50, 100]#, 200]
    for time_p in list_timeperiod:
        #EMA - Exponential Moving Average
        df["ma_EMA_"+str(time_p)] = talib.EMA(close, timeperiod=time_p)
        #KAMA - Kaufman Adaptive Moving Average
        df["ma_KAMA_"+str(time_p)] = talib.KAMA(close, timeperiod=time_p)
        #MA - Moving average
        #df["olap_MA_"+str(time_p)] = talib.MA(close, timeperiod=time_p, matype=0) Sobra porque es la agrupacion de todas las demas MA
        #SMA - Simple Moving Average
        df["ma_SMA_"+str(time_p)] = talib.SMA(close, timeperiod=time_p)
        #TRIMA - Triangular Moving Average
        df["ma_TRIMA_"+str(time_p)] = talib.TRIMA(close, timeperiod=time_p)
        #WMA - Weighted Moving Average
        df["ma_WMA_"+str(time_p)] = talib.WMA(close, timeperiod=time_p)
    # Remove because need too many previous value > 700 for 'ma_TEMA_50', 'ma_DEMA_100', 'ma_TEMA_100', 'ma_T3_100', 'ma_T3_50'
    list_timeperiod = [5, 10, 20, 50 ]
    for time_p in list_timeperiod:
        #DEMA - Double Exponential Moving Average
        df["ma_DEMA_"+str(time_p)] = talib.DEMA(close, timeperiod=time_p)

    list_timeperiod = [5, 10, 20 ]
    for time_p in list_timeperiod:
        #TEMA - Triple Exponential Moving Average
        df["ma_TEMA_"+str(time_p)] = talib.TEMA(close, timeperiod=time_p)
        #T3 - Triple Exponential Moving Average (T3)
        df["ma_T3_"+str(time_p)] = talib.T3(close, timeperiod=time_p )
        #MAMA - MESA Adaptive Moving Average
            #df["olap_MAMA_"+str(time_p)], df["olap_FAMA_"+str(time_p)] = talib.MAMA(close)
        #MAVP - Moving average with variable period
            #df["olap_MAVP_"+str(time_p)] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)


    return df

def get_overlap_indicator( high, low, close):
    df = pd.DataFrame()

    # reliably restored by inspect Bollinger Bands
    if cos_cols is None or "olap_BBAND_UPPER"  in cos_cols or "olap_BBAND_LOWER" in cos_cols or "olap_BBAND_UPPER_crash" in cos_cols or "olap_BBAND_LOWER_crash" in cos_cols or "olap_BBAND_dif" in cos_cols :
        #BBAND_MIDDLE is repeted in ma_SMA_20 , DONT use IT
        df["olap_BBAND_UPPER"], BBAND_MIDDLE_dont_use, df["olap_BBAND_LOWER"] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['col_cierre'] = close
        df = get_crash_points(df, 'olap_BBAND_UPPER', 'col_cierre', col_result="olap_BBAND_UPPER_crash")
        df = get_crash_points(df, 'olap_BBAND_LOWER', 'col_cierre', col_result="olap_BBAND_LOWER_crash")
        df = df.drop(columns=['col_cierre'])
        df["olap_BBAND_dif"] = df["olap_BBAND_UPPER"] - df["olap_BBAND_LOWER"]
    #HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    if cos_cols is None or "olap_HT_TRENDLINE" in cos_cols:
        df["olap_HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
    #MIDPOINT - MidPoint over period
    if cos_cols is None or "olap_MIDPOINT"  in cos_cols:
        df["olap_MIDPOINT"] = talib.MIDPOINT(close, timeperiod=14)
    #MIDPRICE - Midpoint Price over period
    if cos_cols is None or "olap_MIDPRICE"  in cos_cols:
        df["olap_MIDPRICE"] = talib.MIDPRICE(high, low, timeperiod=14)
    #SAR - Parabolic SAR   Parabolic Stop-and-Reverse
    if cos_cols is None or "olap_SAR"  in cos_cols:
        df["olap_SAR"] = talib.SAR(high, low) #, acceleration=0, maximum=0)
    #SAREXT - Parabolic SAR - Extended
    if cos_cols is None or "olap_SAREXT"  in cos_cols:
        df["olap_SAREXT"] = talib.SAREXT(high, low)#, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    return df

#Momentum Indicator Functions
#Momentum Indicator Functions
def get_momentum_indicator(openS, high, low, close, volume):
    df = pd.DataFrame()

    #ADX - Average Directional Movement Index
    if cos_cols is None or "mtum_ADX"  in cos_cols:
        df["mtum_ADX"] = talib.ADX(high, low, close, timeperiod=14)
    #ADXR - Average Directional Movement Index Rating
    if cos_cols is None or "mtum_ADXR"  in cos_cols:
        df["mtum_ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
    #APO - Absolute Price Oscillator
    if cos_cols is None or "mtum_APO"  in cos_cols:
        df["mtum_APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    #AROON - Aroon
    if cos_cols is None or "mtum_AROON_down" in cos_cols or "mtum_AROON_up" :
        df["mtum_AROON_down"], df["mtum_AROON_up"]= talib.AROON(high, low, timeperiod=14)
    #AROONOSC - Aroon Oscillator
    if cos_cols is None or "mtum_AROONOSC"  in cos_cols:
        df["mtum_AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
    #BOP - Balance Of Power
    if cos_cols is None or "mtum_BOP"  in cos_cols:
        df["mtum_BOP"] = talib.BOP(openS, high, low, close)
    #CCI - Commodity Channel Index
    if cos_cols is None or "mtum_CCI"  in cos_cols:
        df["mtum_CCI"] = talib.CCI(high, low, close, timeperiod=14)
    #CMO - Chande Momentum Oscillator
    if cos_cols is None or "mtum_CMO"  in cos_cols:
        df["mtum_CMO"] = talib.CMO(close, timeperiod=14)
    #DX - Directional Movement Index
    if cos_cols is None or "mtum_DX"  in cos_cols:
        df["mtum_DX"] = talib.DX(high, low, close, timeperiod=14)
    #MACD - Moving Average Convergence/Divergence
    if cos_cols is None or "mtum_MACD" in cos_cols or "mtum_MACD_signal" in cos_cols or "mtum_MACD_list" in cos_cols or "mtum_MACD_crash" in cos_cols:
        df["mtum_MACD"], df["mtum_MACD_signal"] ,df["mtum_MACD_list"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df = get_crash_points(df, 'mtum_MACD', 'mtum_MACD_signal', col_result="mtum_MACD_crash")
    #MACDEXT - MACD with controllable MA type
    if cos_cols is None or "mtum_MACD_ext" in cos_cols or "mtum_MACD_ext_signal" in cos_cols or "mtum_MACD_ext_list" in cos_cols or "mtum_MACD_ext_crash" in cos_cols:
        df["mtum_MACD_ext"], df["mtum_MACD_ext_signal"] ,df["mtum_MACD_ext_list"] = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        df = get_crash_points(df, 'mtum_MACD_ext', 'mtum_MACD_ext_signal', col_result="mtum_MACD_ext_crash")
    #MACDFIX - Moving Average Convergence/Divergence Fix 12/26
    if cos_cols is None or "mtum_MACD_fix" in cos_cols or "mtum_MACD_fix_signal" in cos_cols or "mtum_MACD_fix_list" in cos_cols or "mtum_MACD_fix_crash" in cos_cols:
        df["mtum_MACD_fix"], df["mtum_MACD_fix_signal"] ,df["mtum_MACD_fix_list"] = talib.MACDFIX(close, signalperiod=9)
        df = get_crash_points(df, 'mtum_MACD_fix', 'mtum_MACD_fix_signal', col_result="mtum_MACD_fix_crash")
    #MFI - Money Flow Index
    if cos_cols is None or "mtum_MFI"  in cos_cols:
        df["mtum_MFI"] = talib.MFI(high, low, close, volume, timeperiod=14)
    #MINUS_DI - Minus Directional Indicator
    if cos_cols is None or "mtum_MINUS_DI"  in cos_cols:
        df["mtum_MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    #MINUS_DM - Minus Directional Movement
    if cos_cols is None or "mtum_MINUS_DM"  in cos_cols:
        df["mtum_MINUS_DM"] = talib.MINUS_DM(high, low, timeperiod=14)
    #MOM - Momentum
    if cos_cols is None or "mtum_MOM"  in cos_cols:
        df["mtum_MOM"] = talib.MOM(close, timeperiod=10)
    #PLUS_DI - Plus Directional Indicator
    if cos_cols is None or "mtum_PLUS_DI"  in cos_cols:
        df["mtum_PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    #PLUS_DM - Plus Directional Movement
    if cos_cols is None or "mtum_PLUS_DM"  in cos_cols:
        df["mtum_PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)
    #PPO - Percentage Price Oscillator
    if cos_cols is None or "mtum_PPO"  in cos_cols:
        df["mtum_PPO"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    #ROC - Rate of change : ((price/prevPrice)-1)*100
    if cos_cols is None or "mtum_ROC"  in cos_cols:
        df["mtum_ROC"] = talib.ROC(close, timeperiod=10)
    #ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    if cos_cols is None or "mtum_ROCP"  in cos_cols:
        df["mtum_ROCP"] = talib.ROCP(close, timeperiod=10)
    #ROCR - Rate of change ratio: (price/prevPrice)
    if cos_cols is None or "mtum_ROCR"  in cos_cols:
        df["mtum_ROCR"] = talib.ROCR(close, timeperiod=10)
    #ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    if cos_cols is None or "mtum_ROCR100"  in cos_cols:
        df["mtum_ROCR100"] = talib.ROCR100(close, timeperiod=10)
    #RSI - Relative Strength Index
    if cos_cols is None or "mtum_RSI"  in cos_cols:
        df["mtum_RSI"] = talib.RSI(close, timeperiod=14)
    #STOCH - Stochastic
    if cos_cols is None or "mtum_STOCH_k" in cos_cols or "mtum_STOCH_d" in cos_cols or "mtum_STOCH_kd" in cos_cols or "mtum_STOCH_crash"  :
        df["mtum_STOCH_k"], df["mtum_STOCH_d"] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df["mtum_STOCH_kd"] =  df["mtum_STOCH_k"] - df["mtum_STOCH_d"]
        df = get_crash_points(df, 'mtum_STOCH_k', 'mtum_STOCH_d', col_result="mtum_STOCH_crash")
    #STOCHF - Stochastic Fast
    if cos_cols is None or "mtum_STOCH_Fa_k" in cos_cols or "mtum_STOCH_Fa_d" in cos_cols or "mtum_STOCH_Fa_kd" in cos_cols or "mtum_STOCH_Fa_crash" :
        df["mtum_STOCH_Fa_k"], df["mtum_STOCH_Fa_d"] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        df["mtum_STOCH_Fa_kd"] = df["mtum_STOCH_Fa_k"] - df["mtum_STOCH_Fa_d"]
        df = get_crash_points(df, 'mtum_STOCH_Fa_k', 'mtum_STOCH_Fa_d', col_result="mtum_STOCH_Fa_crash")
    #STOCHRSI - Stochastic Relative Strength Index
    if cos_cols is None or "mtum_STOCH_RSI_k" in cos_cols or "mtum_STOCH_RSI_d" in cos_cols or "mtum_STOCH_RSI_kd" in cos_cols or "mtum_STOCH_RSI_crash" :
        df["mtum_STOCH_RSI_k"], df["mtum_STOCH_RSI_d"] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df["mtum_STOCH_RSI_kd"] = df["mtum_STOCH_RSI_k"] - df["mtum_STOCH_RSI_d"]
        df = get_crash_points(df, 'mtum_STOCH_RSI_k', 'mtum_STOCH_RSI_d', col_result="mtum_STOCH_RSI_crash")
    #TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    if cos_cols is None or "mtum_TRIX"  in cos_cols:
        df["mtum_TRIX"] = talib.TRIX(close, timeperiod=30)
    #ULTOSC - Ultimate Oscillator
    if cos_cols is None or "mtum_ULTOSC"  in cos_cols:
        df["mtum_ULTOSC"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    #WILLR - Williams' %R
    if cos_cols is None or "mtum_WILLIAMS_R"  in cos_cols:
        df["mtum_WILLIAMS_R"] = talib.WILLR(high, low, close, timeperiod=14)
    return df


#volume Indicator Functions
#volume Indicator Functions
def get_volume_indicator(high, low, close,volume):
    df = pd.DataFrame()

    #AD - Chaikin A/D Line
    if cos_cols is None or "volu_Chaikin_AD"  in cos_cols:
        df["volu_Chaikin_AD"] = talib.AD(high, low, close, volume)
    #ADOSC - Chaikin A/D Oscillator
    if cos_cols is None or "volu_Chaikin_ADOSC"  in cos_cols:
        df["volu_Chaikin_ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    #OBV - On Balance volume
    if cos_cols is None or "volu_OBV"  in cos_cols:
        df["volu_OBV"] = talib.OBV(close, volume)
    return df

#Volatility Indicator Functions
#Volatility Indicator Functions
def get_volatility_indicator(high, low, close):
    df = pd.DataFrame()

    #ATR - Average True Range
    if cos_cols is None or "vola_ATR"  in cos_cols:
        df["vola_ATR"] = talib.ATR(high, low, close, timeperiod=14)
    #NATR - Normalized Average True Range
    if cos_cols is None or "vola_NATR"  in cos_cols:
        df["vola_NATR"] = talib.NATR(high, low, close, timeperiod=14)
    #TRANGE - True Range
    if cos_cols is None or "vola_TRANGE"  in cos_cols:
        df["vola_TRANGE"] = talib.TRANGE(high, low, close)
    return df


#Cycle Indicator Functions
#Cycle Indicator Functions
def get_cycle_indicator(close):
    df = pd.DataFrame()

    #HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    if cos_cols is None or "cycl_DCPERIOD"  in cos_cols:
        df["cycl_DCPERIOD"] = talib.HT_DCPERIOD(close)
    #HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    if cos_cols is None or "cycl_DCPHASE"  in cos_cols:
        df["cycl_DCPHASE"] = talib.HT_DCPHASE(close)
    #HT_PHASOR - Hilbert Transform - Phasor Components
    if cos_cols is None or "cycl_PHASOR_quad"  in cos_cols or "cycl_PHASOR_inph"  in cos_cols:
        df["cycl_PHASOR_inph"],    df["cycl_PHASOR_quad"] = talib.HT_PHASOR(close)
    #HT_SINE - Hilbert Transform - SineWave
    if cos_cols is None or "cycl_SINE_sine" in cos_cols or "cycl_SINE_lead" in cos_cols:
        df["cycl_SINE_sine"] , df["cycl_SINE_lead"] = talib.HT_SINE(close)
    #HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    if cos_cols is None or "cycl_HT_TRENDMODE"  in cos_cols:
        df["cycl_HT_TRENDMODE"] = talib.HT_TRENDMODE(close)
    return df


# Pattern Recognition Functions
# Pattern Recognition Functions
def get_candle_pattern(openS, high, low, close):
    #  iMAGENES https://github.com/LastAncientOne/Stock_Analysis_For_Quant/tree/master/Python_Stock/Candlestick_Patterns
    df = pd.DataFrame()

    #CDL2CROWS - Two Crows
    if cos_cols is None or "cdl_2CROWS"  in cos_cols:
        df["cdl_2CROWS"] = talib.CDL2CROWS(openS, high, low, close)
    #CDL3BLACKCROWS - Three Black Crows
    if cos_cols is None or "cdl_3BLACKCROWS"  in cos_cols:
        df["cdl_3BLACKCROWS"] = talib.CDL3BLACKCROWS(openS, high, low, close)
    #CDL3INSIDE - Three Inside Up/Down
    if cos_cols is None or "cdl_3INSIDE"  in cos_cols:
        df["cdl_3INSIDE"] = talib.CDL3INSIDE(openS, high, low, close)
    #CDL3LINESTRIKE - Three-Line Strike
    if cos_cols is None or "cdl_3LINESTRIKE"  in cos_cols:
        df["cdl_3LINESTRIKE"] = talib.CDL3LINESTRIKE(openS, high, low, close)
    #CDL3OUTSIDE - Three Outside Up/Down
    if cos_cols is None or "cdl_3OUTSIDE"  in cos_cols:
        df["cdl_3OUTSIDE"] = talib.CDL3OUTSIDE(openS, high, low, close)
    #CDL3STARSINSOUTH - Three Stars In The South
    if cos_cols is None or "cdl_3STARSINSOUTH"  in cos_cols:
        df["cdl_3STARSINSOUTH"] = talib.CDL3STARSINSOUTH(openS, high, low, close)
    #CDL3WHITESOLDIERS - Three Advancing White Soldiers
    if cos_cols is None or "cdl_3WHITESOLDIERS"  in cos_cols:
        df["cdl_3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(openS, high, low, close)
    #CDLABANDONEDBABY - Abandoned Baby
    if cos_cols is None or "cdl_ABANDONEDBABY"  in cos_cols:
        df["cdl_ABANDONEDBABY"] = talib.CDLABANDONEDBABY(openS, high, low, close)
    #CDLADVANCEBLOCK - Advance Block
    if cos_cols is None or "cdl_ADVANCEBLOCK"  in cos_cols:
        df["cdl_ADVANCEBLOCK"] = talib.CDLADVANCEBLOCK(openS, high, low, close)
    #CDLBELTHOLD - Belt-hold
    if cos_cols is None or "cdl_BELTHOLD"  in cos_cols:
        df["cdl_BELTHOLD"] = talib.CDLBELTHOLD(openS, high, low, close)
    #CDLBREAKAWAY - Breakaway
    if cos_cols is None or "cdl_BREAKAWAY"  in cos_cols:
        df["cdl_BREAKAWAY"] = talib.CDLBREAKAWAY(openS, high, low, close)
    #CDLCLOSINGMARUBOZU - Closing Marubozu
    if cos_cols is None or "cdl_CLOSINGMARUBOZU"  in cos_cols:
        df["cdl_CLOSINGMARUBOZU"] = talib.CDLCLOSINGMARUBOZU(openS, high, low, close)
    #CDLCONCEALBABYSWALL - Concealing Baby Swallow
    if cos_cols is None or "cdl_CONCEALBABYSWALL"  in cos_cols:
        df["cdl_CONCEALBABYSWALL"] = talib.CDLCONCEALBABYSWALL(openS, high, low, close)
    #CDLCOUNTERATTACK - Counterattack
    if cos_cols is None or "cdl_COUNTERATTACK"  in cos_cols:
        df["cdl_COUNTERATTACK"] = talib.CDLCOUNTERATTACK(openS, high, low, close)
    #CDLDARKCLOUDCOVER - Dark Cloud Cover
    if cos_cols is None or "cdl_DARKCLOUDCOVER"  in cos_cols:
        df["cdl_DARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(openS, high, low, close)
    #CDLDOJI - Doji
    if cos_cols is None or "cdl_DOJI"  in cos_cols:
        df["cdl_DOJI"] = talib.CDLDOJI(openS, high, low, close)
    #CDLDOJISTAR - Doji Star
    if cos_cols is None or "cdl_DOJISTAR"  in cos_cols:
        df["cdl_DOJISTAR"] = talib.CDLDOJISTAR(openS, high, low, close)
    #CDLDRAGONFLYDOJI - Dragonfly Doji
    if cos_cols is None or "cdl_DRAGONFLYDOJI"  in cos_cols:
        df["cdl_DRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(openS, high, low, close)
    #CDLENGULFING - Engulfing Pattern
    if cos_cols is None or "cdl_ENGULFING"  in cos_cols:
        df["cdl_ENGULFING"] = talib.CDLENGULFING(openS, high, low, close)
    #CDLEVENINGDOJISTAR - Evening Doji Star
    if cos_cols is None or "cdl_EVENINGDOJISTAR"  in cos_cols:
        df["cdl_EVENINGDOJISTAR"] = talib.CDLEVENINGDOJISTAR(openS, high, low, close)
    #CDLEVENINGSTAR - Evening Star
    if cos_cols is None or "cdl_EVENINGSTAR"  in cos_cols:
        df["cdl_EVENINGSTAR"] = talib.CDLEVENINGSTAR(openS, high, low, close)
    #CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    if cos_cols is None or "cdl_GAPSIDESIDEWHITE"  in cos_cols:
        df["cdl_GAPSIDESIDEWHITE"] = talib.CDLGAPSIDESIDEWHITE(openS, high, low, close)
    #CDLGRAVESTONEDOJI - Gravestone Doji
    if cos_cols is None or "cdl_GRAVESTONEDOJI"  in cos_cols:
        df["cdl_GRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(openS, high, low, close)
    #CDLHAMMER - Hammer
    if cos_cols is None or "cdl_HAMMER"  in cos_cols:
        df["cdl_HAMMER"] = talib.CDLHAMMER(openS, high, low, close)
    #CDLHANGINGMAN - Hanging Man
    if cos_cols is None or "cdl_HANGINGMAN"  in cos_cols:
        df["cdl_HANGINGMAN"] = talib.CDLHANGINGMAN(openS, high, low, close)
    #CDLHARAMI - Harami Pattern
    if cos_cols is None or "cdl_HARAMI"  in cos_cols:
        df["cdl_HARAMI"] = talib.CDLHARAMI(openS, high, low, close)
    #CDLHARAMICROSS - Harami Cross Pattern
    if cos_cols is None or "cdl_HARAMICROSS"  in cos_cols:
        df["cdl_HARAMICROSS"] = talib.CDLHARAMICROSS(openS, high, low, close)
    #CDLHIGHWAVE - high-Wave Candle
    if cos_cols is None or "cdl_HIGHWAVE"  in cos_cols:
        df["cdl_HIGHWAVE"] = talib.CDLHIGHWAVE(openS, high, low, close)
    #CDLHIKKAKE - Hikkake Pattern
    if cos_cols is None or "cdl_HIKKAKE"  in cos_cols:
        df["cdl_HIKKAKE"] = talib.CDLHIKKAKE(openS, high, low, close)
    #CDLHIKKAKEMOD - Modified Hikkake Pattern
    if cos_cols is None or "cdl_HIKKAKEMOD"  in cos_cols:
        df["cdl_HIKKAKEMOD"] = talib.CDLHIKKAKEMOD(openS, high, low, close)
    #CDLHOMINGPIGEON - Homing Pigeon
    if cos_cols is None or "cdl_HOMINGPIGEON"  in cos_cols:
        df["cdl_HOMINGPIGEON"] = talib.CDLHOMINGPIGEON(openS, high, low, close)
    #CDLIDENTICAL3CROWS - Identical Three Crows
    if cos_cols is None or "cdl_IDENTICAL3CROWS"  in cos_cols:
        df["cdl_IDENTICAL3CROWS"] = talib.CDLIDENTICAL3CROWS(openS, high, low, close)
    #CDLINNECK - In-Neck Pattern
    if cos_cols is None or "cdl_INNECK"  in cos_cols:
        df["cdl_INNECK"] = talib.CDLINNECK(openS, high, low, close)
    #CDLINVERTEDHAMMER - Inverted Hammer
    if cos_cols is None or "cdl_INVERTEDHAMMER"  in cos_cols:
        df["cdl_INVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(openS, high, low, close)
    #CDLKICKING - Kicking
    if cos_cols is None or "cdl_KICKING"  in cos_cols:
        df["cdl_KICKING"] = talib.CDLKICKING(openS, high, low, close)
    #CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    if cos_cols is None or "cdl_KICKINGBYLENGTH"  in cos_cols:
        df["cdl_KICKINGBYLENGTH"] = talib.CDLKICKINGBYLENGTH(openS, high, low, close)
    #CDLLADDERBOTTOM - Ladder Bottom
    if cos_cols is None or "cdl_LADDERBOTTOM"  in cos_cols:
        df["cdl_LADDERBOTTOM"] = talib.CDLLADDERBOTTOM(openS, high, low, close)
    #CDLLONGLEGGEDDOJI - Long Legged Doji
    if cos_cols is None or "cdl_LONGLEGGEDDOJI"  in cos_cols:
        df["cdl_LONGLEGGEDDOJI"] = talib.CDLLONGLEGGEDDOJI(openS, high, low, close)
    #CDLLONGLINE - Long Line Candle
    if cos_cols is None or "cdl_LONGLINE"  in cos_cols:
        df["cdl_LONGLINE"] = talib.CDLLONGLINE(openS, high, low, close)
    #CDLMARUBOZU - Marubozu
    if cos_cols is None or "cdl_MARUBOZU"  in cos_cols:
        df["cdl_MARUBOZU"] = talib.CDLMARUBOZU(openS, high, low, close)
    #CDLMATCHINGLOW - Matching low
    if cos_cols is None or "cdl_MATCHINGLOW"  in cos_cols:
        df["cdl_MATCHINGLOW"] = talib.CDLMATCHINGLOW(openS, high, low, close)
    #CDLMATHOLD - Mat Hold
    if cos_cols is None or "cdl_MATHOLD"  in cos_cols:
        df["cdl_MATHOLD"] = talib.CDLMATHOLD(openS, high, low, close)
    #CDLMORNINGDOJISTAR - Morning Doji Star
    if cos_cols is None or "cdl_MORNINGDOJISTAR"  in cos_cols:
        df["cdl_MORNINGDOJISTAR"] = talib.CDLMORNINGDOJISTAR(openS, high, low, close)
    #CDLMORNINGSTAR - Morning Star
    if cos_cols is None or "cdl_MORNINGSTAR"  in cos_cols:
        df["cdl_MORNINGSTAR"] = talib.CDLMORNINGSTAR(openS, high, low, close)
    #CDLONNECK - On-Neck Pattern
    if cos_cols is None or "cdl_ONNECK"  in cos_cols:
        df["cdl_ONNECK"] = talib.CDLONNECK(openS, high, low, close)
    #CDLPIERCING - Piercing Pattern
    if cos_cols is None or "cdl_PIERCING"  in cos_cols:
        df["cdl_PIERCING"] = talib.CDLPIERCING(openS, high, low, close)
    #CDLRICKSHAWMAN - Rickshaw Man
    if cos_cols is None or "cdl_RICKSHAWMAN"  in cos_cols:
        df["cdl_RICKSHAWMAN"] = talib.CDLRICKSHAWMAN(openS, high, low, close)
    #CDLRISEFALL3METHODS - Rising/Falling Three Methods
    if cos_cols is None or "cdl_RISEFALL3METHODS"  in cos_cols:
        df["cdl_RISEFALL3METHODS"] = talib.CDLRISEFALL3METHODS(openS, high, low, close)
    #CDLSEPARATINGLINES - Separating Lines
    if cos_cols is None or "cdl_SEPARATINGLINES"  in cos_cols:
        df["cdl_SEPARATINGLINES"] = talib.CDLSEPARATINGLINES(openS, high, low, close)
    #CDLSHOOTINGSTAR - Shooting Star
    if cos_cols is None or "cdl_SHOOTINGSTAR"  in cos_cols:
        df["cdl_SHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(openS, high, low, close)
    #CDLSHORTLINE - Short Line Candle
    if cos_cols is None or "cdl_SHORTLINE"  in cos_cols:
        df["cdl_SHORTLINE"] = talib.CDLSHORTLINE(openS, high, low, close)
    #CDLSPINNINGTOP - Spinning Top
    if cos_cols is None or "cdl_SPINNINGTOP"  in cos_cols:
        df["cdl_SPINNINGTOP"] = talib.CDLSPINNINGTOP(openS, high, low, close)
    #CDLSTALLEDPATTERN - Stalled Pattern
    if cos_cols is None or "cdl_STALLEDPATTERN"  in cos_cols:
        df["cdl_STALLEDPATTERN"] = talib.CDLSTALLEDPATTERN(openS, high, low, close)
    #CDLSTICKSANDWICH - Stick Sandwich
    if cos_cols is None or "cdl_STICKSANDWICH"  in cos_cols:
        df["cdl_STICKSANDWICH"] = talib.CDLSTICKSANDWICH(openS, high, low, close)
    #CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
    if cos_cols is None or "cdl_TAKURI"  in cos_cols:
        df["cdl_TAKURI"] = talib.CDLTAKURI(openS, high, low, close)
    #CDLTASUKIGAP - Tasuki Gap
    if cos_cols is None or "cdl_TASUKIGAP"  in cos_cols:
        df["cdl_TASUKIGAP"] = talib.CDLTASUKIGAP(openS, high, low, close)
    #CDLTHRUSTING - Thrusting Pattern
    if cos_cols is None or "cdl_THRUSTING"  in cos_cols:
        df["cdl_THRUSTING"] = talib.CDLTHRUSTING(openS, high, low, close)
    #CDLTRISTAR - Tristar Pattern
    if cos_cols is None or "cdl_TRISTAR"  in cos_cols:
        df["cdl_TRISTAR"] = talib.CDLTRISTAR(openS, high, low, close)
    #CDLUNIQUE3RIVER - Unique 3 River
    if cos_cols is None or "cdl_UNIQUE3RIVER"  in cos_cols:
        df["cdl_UNIQUE3RIVER"] = talib.CDLUNIQUE3RIVER(openS, high, low, close)
    #CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    if cos_cols is None or "cdl_UPSIDEGAP2CROWS"  in cos_cols:
        df["cdl_UPSIDEGAP2CROWS"] = talib.CDLUPSIDEGAP2CROWS(openS, high, low, close)
    #CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    if cos_cols is None or "cdl_XSIDEGAP3METHODS"  in cos_cols:
        df["cdl_XSIDEGAP3METHODS"] = talib.CDLXSIDEGAP3METHODS(openS, high, low, close)
    return  df

#Statistic Functions
#Statistic Functions
def get_static_funtions(high, low, close):
    df = pd.DataFrame()

    #BETA - Beta
    if cos_cols is None or "sti_BETA"  in cos_cols:
        df["sti_BETA"] = talib.BETA(high, low, timeperiod=5)
    #CORREL - Pearson's Correlation Coefficient (r)
    if cos_cols is None or "sti_CORREL"  in cos_cols:
        df["sti_CORREL"] = talib.CORREL(high, low, timeperiod=30)
    #LINEARREG - Linear Regression
    if cos_cols is None or "sti_LINEARREG"  in cos_cols:
        df["sti_LINEARREG"] = talib.LINEARREG(close, timeperiod=14)
    #LINEARREG_ANGLE - Linear Regression Angle
    if cos_cols is None or "sti_LINEARREG_ANGLE"  in cos_cols:
        df["sti_LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    #LINEARREG_INTERCEPT - Linear Regression Intercept
    if cos_cols is None or "sti_LINEARREG_INTERCEPT"  in cos_cols:
        df["sti_LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(close, timeperiod=14)
    #LINEARREG_SLOPE - Linear Regression Slope
    if cos_cols is None or "sti_LINEARREG_SLOPE"  in cos_cols:
        df["sti_LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    #STDDEV - Standard Deviation
    if cos_cols is None or "sti_STDDEV"  in cos_cols:
        df["sti_STDDEV"] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    #TSF - Time Series Forecast
    if cos_cols is None or "sti_TSF"  in cos_cols:
        df["sti_TSF"] = talib.TSF(close, timeperiod=14)
    #VAR - Variance
    if cos_cols is None or "sti_VAR"  in cos_cols:
        df["sti_VAR"] = talib.VAR(close, timeperiod=5, nbdev=1)
    return df

cos_cols = None
def gel_all_TALIB_funtion(df, custom_columns=None):
    global cos_cols
    cos_cols = custom_columns
    # print("gel_all_TALIB_funtion...")
    #TODOS LOS PAtrones aplicados https://github.com/LastAncientOne/Stock_Analysis_For_Quant/blob/master/Python_Stock/Technical_Indicators/1_Technical_Indicators_talib.ipynb
    #https://github.com/LastAncientOne/Stock_Analysis_For_Quant/tree/master/Python_Stock/Technical_Indicators
    # siempre ordenada la fecha de mas a menos TODO exception
    df_o = get_overlap_indicator(df["high"], df["low"], df["close"])
    df = pd.concat([df, df_o], axis=1)
    df_m = get_momentum_indicator(df["open"], df["high"], df["low"], df["close"], df["volume"])
    df = pd.concat([df, df_m], axis=1)
    df_vol = get_volume_indicator(df["high"], df["low"], df["close"], df["volume"])
    df = pd.concat([df, df_vol], axis=1)
    df_vola = get_volatility_indicator(df["high"], df["low"], df["close"])
    df = pd.concat([df, df_vola], axis=1)
    df_in = get_cycle_indicator(df["close"])
    df = pd.concat([df, df_in], axis=1)
    df_cld = get_candle_pattern(df["open"], df["high"], df["low"], df["close"])
    df = pd.concat([df, df_cld], axis=1)
    df_st = get_static_funtions(df["high"], df["low"], df["close"])
    df = pd.concat([df, df_st], axis=1)
    df_ma = get_moving_average_indicator(df["close"])
    df = pd.concat([df, df_ma], axis=1)

    #df = df.round(3)
    return df


def get_overlap_indicator_async(df): return get_overlap_indicator(df["high"], df["low"], df["close"])
def get_momentum_indicator_async(df): return get_momentum_indicator(df["open"], df["high"], df["low"], df["close"], df["volume"])
def get_volume_indicator_async(df): return get_volume_indicator(df["high"], df["low"], df["close"], df["volume"])
def get_volatility_indicator_async(df): return get_volatility_indicator(df["high"], df["low"], df["close"])
def get_cycle_indicator_async(df): return get_cycle_indicator(df["close"])
def get_candle_pattern_async(df): return get_candle_pattern(df["open"], df["high"], df["low"], df["close"])
def get_static_funtions_async(df): return get_static_funtions(df["high"], df["low"], df["close"])
def get_moving_average_indicator_async(df): return get_moving_average_indicator(df["close"])

ALL_TALIB_FUNCTIONS = [
    get_overlap_indicator_async,
    get_momentum_indicator_async,
    get_volume_indicator_async,
    get_volatility_indicator_async,
    get_cycle_indicator_async,
    get_candle_pattern_async,
    get_static_funtions_async,
    get_moving_average_indicator_async
]
