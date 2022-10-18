from enum import Enum

import Utils_plotter
import a_manage_stocks_dict
import yhoo_history_stock
import seaborn as sns # crear los plots
from matplotlib import pyplot as plt

import pandas as pd

from yhoo_external_raw_factors import DICT_EXTERNAL_FACTORS

stockId = 'MELI'
opion = yhoo_history_stock.Option_Historical.MONTH_3

#TODO news news_get_data_NUTS.get_json_news_sentimet(stockId)


def generate_png_correlation(df):
    columns_array = "Date", "Open", "High", "Low", "Close", "Volume", "per_Close", "per_Volume", "per_preMarket", "olap_BBAND_UPPER", "olap_BBAND_MIDDLE", "olap_BBAND_LOWER", "olap_HT_TRENDLINE", "olap_MIDPOINT", "olap_MIDPRICE", "mtum_ADX", "mtum_ADXR", "mtum_APO", "mtum_AROON_down", "mtum_AROON_up", "mtum_AROONOSC", "mtum_BOP", "mtum_CCI", "mtum_CMO", "mtum_DX", "mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list", "mtum_MACD_ext", "mtum_MACD_ext_signal", "mtum_MACD_ext_list", "mtum_MACD_fix", "mtum_MACD_fix_signal", "mtum_MACD_fix_list", "mtum_MFI", "mtum_MINUS_DI", "mtum_MINUS_DM", "mtum_MOM", "mtum_PLUS_DI", "mtum_PLUS_DM", "mtum_PPO", "mtum_ROC", "mtum_ROCP", "mtum_ROCR", "mtum_ROCR100", "mtum_RSI", "mtum_STOCH_k", "mtum_STOCH_d", "mtum_STOCHF_k", "mtum_STOCHF_d", "mtum_STOCHRSI_k", "mtum_STOCHRSI_d", "mtum_TRIX", "mtum_ULTOSC", "mtum_WILLIAMS_R", "volu_Chaikin_AD", "volu_Chaikin_ADOSC", "volu_OBV", "vola_ATR", "vola_NATR", "vola_TRANGE", "cycl_DCPERIOD", "cycl_DCPHASE", "cycl_PHASOR_inph", "cycl_PHASOR_quad", "cycl_SINE_sine", "cycl_SINE_lead", "cycl_HT_TRENDMODE", "cdl_2CROWS", "cdl_3BLACKCROWS", "cdl_3INSIDE", "cdl_3LINESTRIKE", "cdl_3OUTSIDE", "cdl_3STARSINSOUTH", "cdl_3WHITESOLDIERS", "cdl_ABANDONEDBABY", "cdl_ADVANCEBLOCK", "cdl_BELTHOLD", "cdl_BREAKAWAY", "cdl_CLOSINGMARUBOZU", "cdl_CONCEALBABYSWALL", "cdl_COUNTERATTACK", "cdl_DARKCLOUDCOVER", "cdl_DOJI", "cdl_DOJISTAR", "cdl_DRAGONFLYDOJI", "cdl_ENGULFING", "cdl_EVENINGDOJISTAR", "cdl_EVENINGSTAR", "cdl_GAPSIDESIDEWHITE", "cdl_GRAVESTONEDOJI", "cdl_HAMMER", "cdl_HANGINGMAN", "cdl_HARAMI", "cdl_HARAMICROSS", "cdl_HIGHWAVE", "cdl_HIKKAKE", "cdl_HIKKAKEMOD", "cdl_HOMINGPIGEON", "cdl_IDENTICAL3CROWS", "cdl_INNECK", "cdl_INVERTEDHAMMER", "cdl_KICKING", "cdl_KICKINGBYLENGTH", "cdl_LADDERBOTTOM", "cdl_LONGLEGGEDDOJI", "cdl_LONGLINE", "cdl_MARUBOZU", "cdl_MATCHINGLOW", "cdl_MATHOLD", "cdl_MORNINGDOJISTAR", "cdl_MORNINGSTAR", "cdl_ONNECK", "cdl_PIERCING", "cdl_RICKSHAWMAN", "cdl_RISEFALL3METHODS", "cdl_SEPARATINGLINES", "cdl_SHOOTINGSTAR", "cdl_SHORTLINE", "cdl_SPINNINGTOP", "cdl_STALLEDPATTERN", "cdl_STICKSANDWICH", "cdl_TAKURI", "cdl_TASUKIGAP", "cdl_THRUSTING", "cdl_TRISTAR", "cdl_UNIQUE3RIVER", "cdl_UPSIDEGAP2CROWS", "cdl_XSIDEGAP3METHODS", "sti_BETA", "sti_CORREL", "sti_LINEARREG", "sti_LINEARREG_ANGLE", "sti_LINEARREG_INTERCEPT", "sti_LINEARREG_SLOPE", "sti_STDDEV", "sti_TSF", "sti_VAR", "ma_DEMA_5", "ma_EMA_5", "ma_KAMA_5", "ma_SMA_5", "ma_T3_5", "ma_TEMA_5", "ma_TRIMA_5", "ma_WMA_5", "ma_DEMA_10", "ma_EMA_10", "ma_KAMA_10", "ma_SMA_10", "ma_T3_10", "ma_TEMA_10", "ma_TRIMA_10", "ma_WMA_10", "ma_DEMA_20", "ma_EMA_20", "ma_KAMA_20", "ma_SMA_20", "ma_T3_20", "ma_TEMA_20", "ma_TRIMA_20", "ma_WMA_20", "ma_DEMA_50", "ma_EMA_50", "ma_KAMA_50", "ma_SMA_50", "ma_T3_50", "ma_TEMA_50", "ma_TRIMA_50", "ma_WMA_50", "ma_DEMA_100", "ma_EMA_100", "ma_KAMA_100", "ma_SMA_100", "ma_T3_100", "ma_TEMA_100", "ma_TRIMA_100", "ma_WMA_100", "ma_DEMA_200", "ma_EMA_200", "ma_KAMA_200", "ma_SMA_200", "ma_T3_200", "ma_TEMA_200", "ma_TRIMA_200", "ma_WMA_200", "trad_s3", "trad_s2", "trad_s1", "trad_pp", "trad_r1", "trad_r2", "trad_r3", "clas_s3", "clas_s2", "clas_s1", "clas_pp", "clas_r1", "clas_r2", "clas_r3", "fibo_s3", "fibo_s2", "fibo_s1", "fibo_pp", "fibo_r1", "fibo_r2", "fibo_r3", "wood_s3", "wood_s2", "wood_s1", "wood_pp", "wood_r1", "wood_r2", "wood_r3", "demark_s1", "demark_pp", "demark_r1", "cama_s3", "cama_s2", "cama_s1", "cama_pp", "cama_r1", "cama_r2", "cama_r3", "ti_ACC_DIST", "ti_CHAIKIN(10,3)", "ti_CHOPPINESS(14)", "ti_COPPOCK(14,11,10)", "ti_DONCHIAN_LOWER(20)", "ti_DONCHIAN_CENTER(20)", "ti_DONCHIAN_UPPER(20)", "ti_EASE_OF_MOVEMENT(14)", "ti_FORCE_INDEX(13)", "ti_HMA(20)", "ti_KELT(20)_LOWER", "ti_KELT(20)_UPPER", "ti_MASS_INDEX(9,25)", "ti_SUPERTREND(20)", "ti_VORTEX_POS(5)", "ti_VORTEX_NEG(5)"
    columns_array = ["Open", "High", "Low", "Close"]
    for c in range(0, len(columns_array), 3):
        # bolean values are no alone
        a = [columns_array[c], columns_array[c + 1], columns_array[c + 2]]
        print(a)
        sns_plot = sns.pairplot(data=df, vars=a, hue='buy_sell_point',
                                kind="reg", palette="husl")
        name = stockId + "_correlation_" + str(opion.name)
        plt.savefig("d_price/correlations/" + name + "_".join(a) + ".png")
        print("d_price/correlations/" + name + "_".join(a) + ".png")



CSV_NAME = "@VOLA"
list_companys_FVOLA = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]

df_download = yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_list(list_companys_FVOLA, CSV_NAME, opion)

# df_raw_local = pd.read_csv("d_price/wRAW_FAV_history_20220929__20220707.csv", index_col=False, sep='\t')
# df_local = yhoo_history_stock.get_favs_SCALA_csv_tocks_history_Local(df_raw_local,CSV_NAME,opion)
#df_local.to_csv("d_price/" + csv_name + "_SCALA_local_stock_history_" + str(opion.name) + ".csv", sep='\t', index=None)

#get_favs_SCALA_csv_stocks_history_Download(list_companys_FAV, csv_name =CSV_NAME)


df_l = pd.read_csv("d_price/" + CSV_NAME + "_SCALA_stock_history_" + str(opion.name) + "_sep.csv", index_col=False, sep='\t')


df_l['Date'] = pd.to_datetime(df_l['Date']).map(pd.Timestamp.timestamp)
#df_l = df_l.sort_values(by=['Date', 'ticker'], ascending=True)
#pd.to_datetime(df_his['Date'], unit='s', origin='unix')


# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
# summarize the shape of the dataset
print(df_l.shape)
# summarize each variable
print(df_l.describe())
# histograms of the variables
print(df_l.info)

df_l['buy_sell_point'].replace([101, -101], [100, -100], inplace=True)
Utils_plotter.plot_pie_countvalues(df_l, colum_count ='buy_sell_point', stockid= stockId, opion = str(opion.name))
