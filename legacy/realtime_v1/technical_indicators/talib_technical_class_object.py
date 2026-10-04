import _KEYS_DICT
from Utils import Utils_Yfinance, Utils_buy_sell_points

from technical_indicators import (
    talib_technical_crash_points,
    talib_technical_funtions,
    talib_technical_PY_TI,
    talib_technical_pandas_TA,
    talib_technical_pandas_TU
)

class TechData:

    df_his = None
    df_ohlcV = None
    costum_columns = []
    def __init__(self, df_ohlc, costum_columns):
        self.df_ohlcV = df_ohlc
        self.costum_columns = costum_columns
        self.df_his = df_ohlc

    def get_ALL_tech_data(self):
        return get_ALL_tech_data_local(self.df_his, self.costum_columns)


def get_ALL_tech_data_local(df_his, costum_columns=None):
    #uncompatible get_buy_sell_points_Roll with get_buy_sell_points_HT_pp() use one o the other to get GT
    df_his = Utils_buy_sell_points.get_buy_sell_points_Roll(df_his)#'buy_sell_point'
    # df_his = Utils_buy_sell_points.get_buy_sell_points_HT_pp(df_his, _KEYS_DICT.BACHT_SIZE_LOOKBACK + 2, (_KEYS_DICT.BACHT_SIZE_LOOKBACK + 2) * 2)
    df_his = Utils_Yfinance.add_variation_percentage(df_his) #'\'per_Close\', \'per_Volume\'

    # RETIRAR DATOS PREMARKET
    df_his.iloc[-1, df_his.columns.get_loc('Volume')] = 1 # quitar todos los Volume = 0 menos el ultimo, el ultimo lo pongo a 1.  alert no funciona df_his.iloc[-1:]['Volume'] = 1
    df_his = df_his[df_his['Volume'] != 0]#.append(df_his.tail(1),ignore_index=True)  # quitar todos los Volume = 0 menos el ultimo,
    df_his.reset_index(drop=True, inplace=True)
    df_his.iloc[-1, df_his.columns.get_loc('Volume')] = df_his.iloc[-2, df_his.columns.get_loc('Volume')] #rellenar la ultima con la penultima el volumen

    df_his = Utils_Yfinance.add_pre_market_percentage(df_his) #'has_preMarket' , 'per_preMarket'

    # df_his['sell_arco'] = Utils_buy_sell_points.get_buy_sell_points_Arcos(df_his)
    # Utils_plotter.plotting_financial_chart_buy_points_serial(df_his, df_his['buy_sell_point'], stockId,str(opion.name) )
    df_his = talib_technical_funtions.gel_all_TALIB_funtion(df_his, costum_columns)  # siempre ordenada la fecha de mas a menos TODO exceptio
    df_his = talib_technical_PY_TI.get_all_pivots_points(df_his, costum_columns) #no se pasa costum_columns ya que se necesitan para calcular los pp crash
    df_his = talib_technical_PY_TI.get_py_TI_indicator(df_his, costum_columns)
    df_his = talib_technical_pandas_TA.get_all_pandas_TA_tecnical(df_his, costum_columns)
    df_his = talib_technical_pandas_TU.get_all_pandas_TU_tecnical(df_his, costum_columns)
    df_his = talib_technical_crash_points.get_ALL_CRASH_funtion(df_his, costum_columns)
    df_his = df_his.round(6)


    return df_his
