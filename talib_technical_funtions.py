#https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html
#https://github.com/CanerIrfanoglu/medium/blob/master/candle_stick_recognition/identify_candlestick.py
import talib
import pandas as pd


#Overlap Studies Functions
#Overlap Studies Functions
#BBANDS - Bollinger Bands
import Utils_Yfinance


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
        #DEMA - Double Exponential Moving Average
        df["ma_DEMA_"+str(time_p)] = talib.DEMA(close, timeperiod=time_p)
        #EMA - Exponential Moving Average
        df["ma_EMA_"+str(time_p)] = talib.EMA(close, timeperiod=time_p)
        #KAMA - Kaufman Adaptive Moving Average
        df["ma_KAMA_"+str(time_p)] = talib.KAMA(close, timeperiod=time_p)
        #MA - Moving average
        #df["olap_MA_"+str(time_p)] = talib.MA(close, timeperiod=time_p, matype=0) Sobra porque es la agrupacion de todas las demas MA
        #SMA - Simple Moving Average
        df["ma_SMA_"+str(time_p)] = talib.SMA(close, timeperiod=time_p)
        #T3 - Triple Exponential Moving Average (T3)
        df["ma_T3_"+str(time_p)] = talib.T3(close, timeperiod=time_p )
        #TEMA - Triple Exponential Moving Average
        df["ma_TEMA_"+str(time_p)] = talib.TEMA(close, timeperiod=time_p)
        #TRIMA - Triangular Moving Average
        df["ma_TRIMA_"+str(time_p)] = talib.TRIMA(close, timeperiod=time_p)
        #WMA - Weighted Moving Average
        df["ma_WMA_"+str(time_p)] = talib.WMA(close, timeperiod=time_p)
        #MAMA - MESA Adaptive Moving Average
            #df["olap_MAMA_"+str(time_p)], df["olap_FAMA_"+str(time_p)] = talib.MAMA(close)
        #MAVP - Moving average with variable period
            #df["olap_MAVP_"+str(time_p)] = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)

    return df

def get_overlap_indicator( high, low, close):
    df = pd.DataFrame()

    # reliably restored by inspect Bollinger Bands
    df["olap_BBAND_UPPER"], df["olap_BBAND_MIDDLE"], df["olap_BBAND_LOWER"] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['col_cierre'] = close
    df = Utils_Yfinance.get_crash_points(df, 'olap_BBAND_UPPER', 'col_cierre', col_result="olap_BBAND_UPPER_crash")
    df = Utils_Yfinance.get_crash_points(df, 'olap_BBAND_LOWER', 'col_cierre', col_result="olap_BBAND_LOWER_crash")
    df = df.drop(columns=['col_cierre'])
    #HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    df["olap_HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
    #MIDPOINT - MidPoint over period
    df["olap_MIDPOINT"] = talib.MIDPOINT(close, timeperiod=14)
    #MIDPRICE - Midpoint Price over period
    df["olap_MIDPRICE"] = talib.MIDPRICE(high, low, timeperiod=14)
    #SAR - Parabolic SAR   Parabolic Stop-and-Reverse
    df["olap_SAR"] = talib.SAR(high, low) #, acceleration=0, maximum=0)
    #SAREXT - Parabolic SAR - Extended
    df["olap_SAREXT"] = talib.SAREXT(high, low)#, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    return df

#Momentum Indicator Functions
#Momentum Indicator Functions
def get_momentum_indicator(openS, high, low, close, volume):
    df = pd.DataFrame()

    #ADX - Average Directional Movement Index
    df["mtum_ADX"] = talib.ADX(high, low, close, timeperiod=14)
    #ADXR - Average Directional Movement Index Rating
    df["mtum_ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
    #APO - Absolute Price Oscillator
    df["mtum_APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    #AROON - Aroon
    df["mtum_AROON_down"], df["mtum_AROON_up"]= talib.AROON(high, low, timeperiod=14)
    #AROONOSC - Aroon Oscillator
    df["mtum_AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
    #BOP - Balance Of Power
    df["mtum_BOP"] = talib.BOP(openS, high, low, close)
    #CCI - Commodity Channel Index
    df["mtum_CCI"] = talib.CCI(high, low, close, timeperiod=14)
    #CMO - Chande Momentum Oscillator
    df["mtum_CMO"] = talib.CMO(close, timeperiod=14)
    #DX - Directional Movement Index
    df["mtum_DX"] = talib.DX(high, low, close, timeperiod=14)
    #MACD - Moving Average Convergence/Divergence
    df["mtum_MACD"], df["mtum_MACD_signal"] ,df["mtum_MACD_list"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df = Utils_Yfinance.get_crash_points(df, 'mtum_MACD', 'mtum_MACD_signal', col_result="mtum_MACD_crash")
    #MACDEXT - MACD with controllable MA type
    df["mtum_MACD_ext"], df["mtum_MACD_ext_signal"] ,df["mtum_MACD_ext_list"] = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df = Utils_Yfinance.get_crash_points(df, 'mtum_MACD_ext', 'mtum_MACD_ext_signal', col_result="mtum_MACD_ext_crash")
    #MACDFIX - Moving Average Convergence/Divergence Fix 12/26
    df["mtum_MACD_fix"], df["mtum_MACD_fix_signal"] ,df["mtum_MACD_fix_list"] = talib.MACDFIX(close, signalperiod=9)
    df = Utils_Yfinance.get_crash_points(df, 'mtum_MACD_fix', 'mtum_MACD_fix_signal', col_result="mtum_MACD_fix_crash")
    #MFI - Money Flow Index
    df["mtum_MFI"] = talib.MFI(high, low, close, volume, timeperiod=14)
    #MINUS_DI - Minus Directional Indicator
    df["mtum_MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    #MINUS_DM - Minus Directional Movement
    df["mtum_MINUS_DM"] = talib.MINUS_DM(high, low, timeperiod=14)
    #MOM - Momentum
    df["mtum_MOM"] = talib.MOM(close, timeperiod=10)
    #PLUS_DI - Plus Directional Indicator
    df["mtum_PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    #PLUS_DM - Plus Directional Movement
    df["mtum_PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)
    #PPO - Percentage Price Oscillator
    df["mtum_PPO"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    #ROC - Rate of change : ((price/prevPrice)-1)*100
    df["mtum_ROC"] = talib.ROC(close, timeperiod=10)
    #ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    df["mtum_ROCP"] = talib.ROCP(close, timeperiod=10)
    #ROCR - Rate of change ratio: (price/prevPrice)
    df["mtum_ROCR"] = talib.ROCR(close, timeperiod=10)
    #ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    df["mtum_ROCR100"] = talib.ROCR100(close, timeperiod=10)
    #RSI - Relative Strength Index
    df["mtum_RSI"] = talib.RSI(close, timeperiod=14)
    #STOCH - Stochastic
    df["mtum_STOCH_k"], df["mtum_STOCH_d"] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df["mtum_STOCH_kd"] =  df["mtum_STOCH_k"] - df["mtum_STOCH_d"]
    df = Utils_Yfinance.get_crash_points(df, 'mtum_STOCH_k', 'mtum_STOCH_d', col_result="mtum_STOCH_crash")
    #STOCHF - Stochastic Fast
    df["mtum_STOCH_Fa_k"], df["mtum_STOCH_Fa_d"] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df["mtum_STOCH_Fa_kd"] = df["mtum_STOCH_Fa_k"] - df["mtum_STOCH_Fa_d"]
    df = Utils_Yfinance.get_crash_points(df, 'mtum_STOCH_Fa_k', 'mtum_STOCH_Fa_d', col_result="mtum_STOCH_Fa_crash")
    #STOCHRSI - Stochastic Relative Strength Index
    df["mtum_STOCH_RSI_k"], df["mtum_STOCH_RSI_d"] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df["mtum_STOCH_RSI_kd"] = df["mtum_STOCH_RSI_k"] - df["mtum_STOCH_RSI_d"]
    df = Utils_Yfinance.get_crash_points(df, 'mtum_STOCH_RSI_k', 'mtum_STOCH_RSI_d', col_result="mtum_STOCH_RSI_crash")
    #TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    df["mtum_TRIX"] = talib.TRIX(close, timeperiod=30)
    #ULTOSC - Ultimate Oscillator
    df["mtum_ULTOSC"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    #WILLR - Williams' %R
    df["mtum_WILLIAMS_R"] = talib.WILLR(high, low, close, timeperiod=14)
    return df


#Volume Indicator Functions
#Volume Indicator Functions
def get_volume_indicator(high, low, close,volume):
    df = pd.DataFrame()

    #AD - Chaikin A/D Line
    df["volu_Chaikin_AD"] = talib.AD(high, low, close, volume)
    #ADOSC - Chaikin A/D Oscillator
    df["volu_Chaikin_ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    #OBV - On Balance Volume
    df["volu_OBV"] = talib.OBV(close, volume)
    return df

#Volatility Indicator Functions
#Volatility Indicator Functions
def get_volatility_indicator(high, low, close):
    df = pd.DataFrame()

    #ATR - Average True Range
    df["vola_ATR"] = talib.ATR(high, low, close, timeperiod=14)
    #NATR - Normalized Average True Range
    df["vola_NATR"] = talib.NATR(high, low, close, timeperiod=14)
    #TRANGE - True Range
    df["vola_TRANGE"] = talib.TRANGE(high, low, close)
    return df


#Cycle Indicator Functions
#Cycle Indicator Functions
def get_cycle_indicator(close):
    df = pd.DataFrame()

    #HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    df["cycl_DCPERIOD"] = talib.HT_DCPERIOD(close)
    #HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    df["cycl_DCPHASE"] = talib.HT_DCPHASE(close)
    #HT_PHASOR - Hilbert Transform - Phasor Components
    df["cycl_PHASOR_inph"], df["cycl_PHASOR_quad"] = talib.HT_PHASOR(close)
    #HT_SINE - Hilbert Transform - SineWave
    df["cycl_SINE_sine"], df["cycl_SINE_lead"] = talib.HT_SINE(close)
    #HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    df["cycl_HT_TRENDMODE"] = talib.HT_TRENDMODE(close)
    return df


# Pattern Recognition Functions
# Pattern Recognition Functions
def get_candle_pattern(openS, high, low, close):
    df = pd.DataFrame()

    #CDL2CROWS - Two Crows
    df["cdl_2CROWS"] = talib.CDL2CROWS(openS, high, low, close)
    #CDL3BLACKCROWS - Three Black Crows
    df["cdl_3BLACKCROWS"] = talib.CDL3BLACKCROWS(openS, high, low, close)
    #CDL3INSIDE - Three Inside Up/Down
    df["cdl_3INSIDE"] = talib.CDL3INSIDE(openS, high, low, close)
    #CDL3LINESTRIKE - Three-Line Strike
    df["cdl_3LINESTRIKE"] = talib.CDL3LINESTRIKE(openS, high, low, close)
    #CDL3OUTSIDE - Three Outside Up/Down
    df["cdl_3OUTSIDE"] = talib.CDL3OUTSIDE(openS, high, low, close)
    #CDL3STARSINSOUTH - Three Stars In The South
    df["cdl_3STARSINSOUTH"] = talib.CDL3STARSINSOUTH(openS, high, low, close)
    #CDL3WHITESOLDIERS - Three Advancing White Soldiers
    df["cdl_3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(openS, high, low, close)
    #CDLABANDONEDBABY - Abandoned Baby
    df["cdl_ABANDONEDBABY"] = talib.CDLABANDONEDBABY(openS, high, low, close)
    #CDLADVANCEBLOCK - Advance Block
    df["cdl_ADVANCEBLOCK"] = talib.CDLADVANCEBLOCK(openS, high, low, close)
    #CDLBELTHOLD - Belt-hold
    df["cdl_BELTHOLD"] = talib.CDLBELTHOLD(openS, high, low, close)
    #CDLBREAKAWAY - Breakaway
    df["cdl_BREAKAWAY"] = talib.CDLBREAKAWAY(openS, high, low, close)
    #CDLCLOSINGMARUBOZU - Closing Marubozu
    df["cdl_CLOSINGMARUBOZU"] = talib.CDLCLOSINGMARUBOZU(openS, high, low, close)
    #CDLCONCEALBABYSWALL - Concealing Baby Swallow
    df["cdl_CONCEALBABYSWALL"] = talib.CDLCONCEALBABYSWALL(openS, high, low, close)
    #CDLCOUNTERATTACK - Counterattack
    df["cdl_COUNTERATTACK"] = talib.CDLCOUNTERATTACK(openS, high, low, close)
    #CDLDARKCLOUDCOVER - Dark Cloud Cover
    df["cdl_DARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(openS, high, low, close)
    #CDLDOJI - Doji
    df["cdl_DOJI"] = talib.CDLDOJI(openS, high, low, close)
    #CDLDOJISTAR - Doji Star
    df["cdl_DOJISTAR"] = talib.CDLDOJISTAR(openS, high, low, close)
    #CDLDRAGONFLYDOJI - Dragonfly Doji
    df["cdl_DRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(openS, high, low, close)
    #CDLENGULFING - Engulfing Pattern
    df["cdl_ENGULFING"] = talib.CDLENGULFING(openS, high, low, close)
    #CDLEVENINGDOJISTAR - Evening Doji Star
    df["cdl_EVENINGDOJISTAR"] = talib.CDLEVENINGDOJISTAR(openS, high, low, close)
    #CDLEVENINGSTAR - Evening Star
    df["cdl_EVENINGSTAR"] = talib.CDLEVENINGSTAR(openS, high, low, close)
    #CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    df["cdl_GAPSIDESIDEWHITE"] = talib.CDLGAPSIDESIDEWHITE(openS, high, low, close)
    #CDLGRAVESTONEDOJI - Gravestone Doji
    df["cdl_GRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(openS, high, low, close)
    #CDLHAMMER - Hammer
    df["cdl_HAMMER"] = talib.CDLHAMMER(openS, high, low, close)
    #CDLHANGINGMAN - Hanging Man
    df["cdl_HANGINGMAN"] = talib.CDLHANGINGMAN(openS, high, low, close)
    #CDLHARAMI - Harami Pattern
    df["cdl_HARAMI"] = talib.CDLHARAMI(openS, high, low, close)
    #CDLHARAMICROSS - Harami Cross Pattern
    df["cdl_HARAMICROSS"] = talib.CDLHARAMICROSS(openS, high, low, close)
    #CDLHIGHWAVE - High-Wave Candle
    df["cdl_HIGHWAVE"] = talib.CDLHIGHWAVE(openS, high, low, close)
    #CDLHIKKAKE - Hikkake Pattern
    df["cdl_HIKKAKE"] = talib.CDLHIKKAKE(openS, high, low, close)
    #CDLHIKKAKEMOD - Modified Hikkake Pattern
    df["cdl_HIKKAKEMOD"] = talib.CDLHIKKAKEMOD(openS, high, low, close)
    #CDLHOMINGPIGEON - Homing Pigeon
    df["cdl_HOMINGPIGEON"] = talib.CDLHOMINGPIGEON(openS, high, low, close)
    #CDLIDENTICAL3CROWS - Identical Three Crows
    df["cdl_IDENTICAL3CROWS"] = talib.CDLIDENTICAL3CROWS(openS, high, low, close)
    #CDLINNECK - In-Neck Pattern
    df["cdl_INNECK"] = talib.CDLINNECK(openS, high, low, close)
    #CDLINVERTEDHAMMER - Inverted Hammer
    df["cdl_INVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(openS, high, low, close)
    #CDLKICKING - Kicking
    df["cdl_KICKING"] = talib.CDLKICKING(openS, high, low, close)
    #CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    df["cdl_KICKINGBYLENGTH"] = talib.CDLKICKINGBYLENGTH(openS, high, low, close)
    #CDLLADDERBOTTOM - Ladder Bottom
    df["cdl_LADDERBOTTOM"] = talib.CDLLADDERBOTTOM(openS, high, low, close)
    #CDLLONGLEGGEDDOJI - Long Legged Doji
    df["cdl_LONGLEGGEDDOJI"] = talib.CDLLONGLEGGEDDOJI(openS, high, low, close)
    #CDLLONGLINE - Long Line Candle
    df["cdl_LONGLINE"] = talib.CDLLONGLINE(openS, high, low, close)
    #CDLMARUBOZU - Marubozu
    df["cdl_MARUBOZU"] = talib.CDLMARUBOZU(openS, high, low, close)
    #CDLMATCHINGLOW - Matching Low
    df["cdl_MATCHINGLOW"] = talib.CDLMATCHINGLOW(openS, high, low, close)
    #CDLMATHOLD - Mat Hold
    df["cdl_MATHOLD"] = talib.CDLMATHOLD(openS, high, low, close)
    #CDLMORNINGDOJISTAR - Morning Doji Star
    df["cdl_MORNINGDOJISTAR"] = talib.CDLMORNINGDOJISTAR(openS, high, low, close)
    #CDLMORNINGSTAR - Morning Star
    df["cdl_MORNINGSTAR"] = talib.CDLMORNINGSTAR(openS, high, low, close)
    #CDLONNECK - On-Neck Pattern
    df["cdl_ONNECK"] = talib.CDLONNECK(openS, high, low, close)
    #CDLPIERCING - Piercing Pattern
    df["cdl_PIERCING"] = talib.CDLPIERCING(openS, high, low, close)
    #CDLRICKSHAWMAN - Rickshaw Man
    df["cdl_RICKSHAWMAN"] = talib.CDLRICKSHAWMAN(openS, high, low, close)
    #CDLRISEFALL3METHODS - Rising/Falling Three Methods
    df["cdl_RISEFALL3METHODS"] = talib.CDLRISEFALL3METHODS(openS, high, low, close)
    #CDLSEPARATINGLINES - Separating Lines
    df["cdl_SEPARATINGLINES"] = talib.CDLSEPARATINGLINES(openS, high, low, close)
    #CDLSHOOTINGSTAR - Shooting Star
    df["cdl_SHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(openS, high, low, close)
    #CDLSHORTLINE - Short Line Candle
    df["cdl_SHORTLINE"] = talib.CDLSHORTLINE(openS, high, low, close)
    #CDLSPINNINGTOP - Spinning Top
    df["cdl_SPINNINGTOP"] = talib.CDLSPINNINGTOP(openS, high, low, close)
    #CDLSTALLEDPATTERN - Stalled Pattern
    df["cdl_STALLEDPATTERN"] = talib.CDLSTALLEDPATTERN(openS, high, low, close)
    #CDLSTICKSANDWICH - Stick Sandwich
    df["cdl_STICKSANDWICH"] = talib.CDLSTICKSANDWICH(openS, high, low, close)
    #CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
    df["cdl_TAKURI"] = talib.CDLTAKURI(openS, high, low, close)
    #CDLTASUKIGAP - Tasuki Gap
    df["cdl_TASUKIGAP"] = talib.CDLTASUKIGAP(openS, high, low, close)
    #CDLTHRUSTING - Thrusting Pattern
    df["cdl_THRUSTING"] = talib.CDLTHRUSTING(openS, high, low, close)
    #CDLTRISTAR - Tristar Pattern
    df["cdl_TRISTAR"] = talib.CDLTRISTAR(openS, high, low, close)
    #CDLUNIQUE3RIVER - Unique 3 River
    df["cdl_UNIQUE3RIVER"] = talib.CDLUNIQUE3RIVER(openS, high, low, close)
    #CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    df["cdl_UPSIDEGAP2CROWS"] = talib.CDLUPSIDEGAP2CROWS(openS, high, low, close)
    #CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    df["cdl_XSIDEGAP3METHODS"] = talib.CDLXSIDEGAP3METHODS(openS, high, low, close)

    return  df

#Statistic Functions
#Statistic Functions
def get_static_funtions(high, low, close):
    df = pd.DataFrame()

    #BETA - Beta
    df["sti_BETA"] = talib.BETA(high, low, timeperiod=5)
    #CORREL - Pearson's Correlation Coefficient (r)
    df["sti_CORREL"] = talib.CORREL(high, low, timeperiod=30)
    #LINEARREG - Linear Regression
    df["sti_LINEARREG"] = talib.LINEARREG(close, timeperiod=14)
    #LINEARREG_ANGLE - Linear Regression Angle
    df["sti_LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    #LINEARREG_INTERCEPT - Linear Regression Intercept
    df["sti_LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(close, timeperiod=14)
    #LINEARREG_SLOPE - Linear Regression Slope
    df["sti_LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    #STDDEV - Standard Deviation
    df["sti_STDDEV"] = talib.STDDEV(close, timeperiod=5, nbdev=1)
    #TSF - Time Series Forecast
    df["sti_TSF"] = talib.TSF(close, timeperiod=14)
    #VAR - Variance
    df["sti_VAR"] = talib.VAR(close, timeperiod=5, nbdev=1)

    return df


def gel_all_TALIB_funtion(df):
    # siempre ordenada la fecha de mas a menos TODO exception
    df_o = get_overlap_indicator(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, df_o], axis=1)
    df_m = get_momentum_indicator(df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    df = pd.concat([df, df_m], axis=1)
    df_vol = get_volume_indicator(df["High"], df["Low"], df["Close"], df["Volume"])
    df = pd.concat([df, df_vol], axis=1)
    df_vola = get_volatility_indicator(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, df_vola], axis=1)
    df_in = get_cycle_indicator(df["Close"])
    df = pd.concat([df, df_in], axis=1)
    df_cld = get_candle_pattern(df["Open"], df["High"], df["Low"], df["Close"])
    df = pd.concat([df, df_cld], axis=1)
    df_st = get_static_funtions(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, df_st], axis=1)
    df_ma = get_moving_average_indicator(df["Close"])
    df = pd.concat([df, df_ma], axis=1)

    #df = df.round(3)
    return df
