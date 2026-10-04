#https://github.com/tlpcap/py_ti/blob/main/py_ti/py_ti.py
'''py_ti
A collection of 48 technical indicators. Suggestions are welcome.
https://github.com/tlpcap/py_ti

def trad_pivots(df, add_col=False, return_struct='numpy'):
def classic_pivots(df, add_col=False, return_struct='numpy'):
def fibonacci_pivots(df, add_col=False, return_struct='numpy'):
def woodie_pivots(df, add_col=False, return_struct='numpy'):
def demark_pivots(df, add_col=False, return_struct='numpy'):
def camarilla_pivots(df, add_col=False, return_struct='numpy'):
'''
#import py_ti
import pandas as pd
from py_ti import py_ti
from Utils import UtilsL, Utils_Yfinance


# stockId = "MSFT"
# stockId = "MELI"
#
# df_stocks = yhoo_history_stock.get_historial_data_3y(stockId)
#
# yho_stk = yf.Ticker(stockId)
# df_m = yho_stk.history(period="7d", prepost=True, interval="5m") #yhoo_history_stock.get_historial_data_1_month(stockId)


def get_all_pivots_points(df_stocks, costum_columns = None):
    df_trad = pd.DataFrame()
    df_clasi = pd.DataFrame()
    df_fibo = pd.DataFrame()
    df_wood = pd.DataFrame()
    df_dema = pd.DataFrame()
    df_cama = pd.DataFrame()

    if costum_columns is None or any("trad_" in co for co in costum_columns):
        df_trad = py_ti.trad_pivots(df_stocks, return_struct='pandas')
        df_trad = UtilsL.add_rename_all_columns_df(df_trad, prefix="trad_")

    if costum_columns is None or any("clas_" in co for co in costum_columns):
        df_clasi = py_ti.classic_pivots(df_stocks, return_struct='pandas')
        df_clasi = UtilsL.add_rename_all_columns_df(df_clasi, prefix="clas_")

    if costum_columns is None or any("fibo_" in co for co in costum_columns):
        df_fibo = py_ti.fibonacci_pivots(df_stocks, return_struct='pandas')
        df_fibo = UtilsL.add_rename_all_columns_df(df_fibo, prefix="fibo_")

    if costum_columns is None or any("wood_" in co for co in costum_columns):
        df_wood = py_ti.woodie_pivots(df_stocks, return_struct='pandas')
        df_wood = UtilsL.add_rename_all_columns_df(df_wood, prefix="wood_")

    if costum_columns is None or any("demark_" in co for co in costum_columns):
        df_dema = py_ti.demark_pivots(df_stocks, return_struct='pandas')
        df_dema = UtilsL.add_rename_all_columns_df(df_dema, prefix="demark_")

    if costum_columns is None or any("cama_" in co for co in costum_columns):
        df_cama = py_ti.camarilla_pivots(df_stocks, return_struct='pandas')
        df_cama = UtilsL.add_rename_all_columns_df(df_cama, prefix="cama_")

    df_list = [df_trad, df_clasi, df_fibo, df_wood, df_dema, df_cama]
    df = df_stocks
    for dfi in df_list:
        df = pd.concat([df, dfi], axis=1)

    return df



def get_py_TI_indicator(df_stocks, cos_cols = None):
    df_acc = pd.DataFrame()
    df_chaikin = pd.DataFrame()
    df_chopp = pd.DataFrame()
    df_coppock = pd.DataFrame()
    df_donchian = pd.DataFrame()
    df_ease_move = pd.DataFrame()
    df_force_index = pd.DataFrame()

    df_keltner = pd.DataFrame()
    df_mass_index = pd.DataFrame()
    df_supertrend = pd.DataFrame()
    df_vortex = pd.DataFrame()
    df_vortex2 = pd.DataFrame()
    df_konk = pd.DataFrame()

    if cos_cols is None or "ti_acc_dist"  in cos_cols:
        df_acc = py_ti.acc_dist(df_stocks, return_struct='pandas') #vol This provides insight into how strong a trend is.
    if cos_cols is None or "ti_chaikin_10_3" in cos_cols:
        df_chaikin = py_ti.chaikin_oscillator(df_stocks, return_struct='pandas') #mtum  Cuanto más cerca esté al máximo el nivel de cierre de la acción o del índice, más activa será la acumulación.
    if cos_cols is None or "ti_choppiness_14" in cos_cols:
        df_chopp = py_ti.choppiness(df_stocks, return_struct='pandas') #vola El Índice Choppiness (CHOP) es un indicador diseñado para determinar si el mercado es variable (negociaciones transversales) o no variable (negociaciones dentro de una tendencia en cualquier dirección).
    if cos_cols is None or "ti_coppock_14_11_10" in cos_cols:
        df_coppock = py_ti.coppock(df_stocks, return_struct='pandas') #mtum Una pregunta recurrente que me hacen es, ¿cómo puedo saber cuándo van a caer «las bolsas»?

    if cos_cols is None or "ti_donchian_lower_20" in cos_cols or "ti_donchian_center_20" in cos_cols or "ti_donchian_upper_20" in cos_cols :
        df_donchian = py_ti.donchian_channels(df_stocks, return_struct='pandas')#Vola El canal de Donchian es un indicador útil para ver la volatilidad de un precio de mercado. Si un precio es estable, el canal de Donchian será relativamente estrecho. Si el precio fluctúa mucho, el canal Donchian será más ancho.
    if cos_cols is None or "ti_ease_of_movement_14" in cos_cols:
        df_ease_move = py_ti.ease_of_movement(df_stocks, return_struct='pandas')#Vola  La intención es usar este valor para discernir si los precios pueden subir o bajar, con poca resistencia en el movimiento direccional.
    if cos_cols is None or "ti_force_index_13" in cos_cols:
        df_force_index = py_ti.force_index(df_stocks, return_struct='pandas')#mtum El índice de fuerza es un indicador utilizado en el análisis técnico para ilustrar qué tan fuerte es la presión real de compra o venta. Los valores positivos altos significan que hay una fuerte tendencia al alza, y los valores bajos significan una fuerte tendencia a la baja.
    #if cos_cols is None or "ti_hma_20" in cos_cols: es necesaria para las medias
    df_ma_hull = py_ti.hma(df_stocks, return_struct='pandas') #ma Hull Moving Average

    if cos_cols is None or "ti_kelt_20_lower" in cos_cols or "ti_kelt_20_upper" in cos_cols:
        df_keltner = py_ti.keltner_channels(df_stocks, return_struct='pandas')#volatilidad  los precios se mantienen en la zona comprendida entre la línea superior e inferior del canal de Keltner gran parte del tiempo.
    if cos_cols is None or "ti_mass_index_9_25" in cos_cols:
        df_mass_index = py_ti.mass_index(df_stocks, return_struct='pandas')#mtum que se utiliza en el análisis técnico para predecir cambios de tendencia. Se basa en la noción de que existe una tendencia a la reversión cuando el rango de precios se amplía y, por lo tanto, compara los rangos comerciales anteriores.
    if cos_cols is None or "ti_supertrend_20" in cos_cols:
        df_supertrend = py_ti.supertrend(df_stocks, return_struct='pandas')#sti El indicador SuperTendencia es un excelente indicador de tendencia que se fundamenta en los precios. Diferencia con claridad la tendencia ascendente y descendente del mercado. También puede indicar niveles de soporte y resistencia. Pero veámoslo con mayor detalle.
    if cos_cols is None or "ti_vortex_pos_5" in cos_cols or "ti_vortex_neg_5" in cos_cols:
        df_vortex = py_ti.vortex(df_stocks,n=5, return_struct='pandas')# sti El Indicador Vortex (VI) está compuesto de 2 líneas que muestran tanto el movimiento de tendencia positivo (VI+) como el negativo (VI-). El indicador se ideó fruto de una fuente de inspiración basada en ciertos movimientos del agua y fue desarrollado por Etienne Botes y Douglas Siepman. El indicador Vortex tiene una aplicación relativamente simple: los traders lo utilizan para identificar el inicio de una tendencia.
    if cos_cols is None or "ti_vortex_pos_14" in cos_cols or "ti_vortex_neg_14" in cos_cols:
        df_vortex2 = py_ti.vortex(df_stocks,n=14, return_struct='pandas')
    # https://es.tradingview.com/script/lLFiT3av/
    if cos_cols is None or "ti_konk_blue" in cos_cols or "ti_konk_brown" in cos_cols or "ti_konk_green" in cos_cols or "ti_konk_rest"  in cos_cols or "ti_konk_bl_avg_crash" in cos_cols or "ti_konk_gre_avg_crash" in cos_cols or "ti_konk_gre_bl_crash" in cos_cols or "ti_konk_avg":
        val_blue, val_brown, val_green, val_avg = get_konkorde_params(df_stocks)
        df_konk = pd.DataFrame({'konk_bl': val_blue, 'konk_bro': val_brown, 'konk_gre': val_green, 'konk_avg': val_avg})
        df_konk['konk_rest'] =  df_konk['konk_bl'] - df_konk['konk_gre']
        df_konk = Utils_Yfinance.get_crash_points(df_konk, 'konk_bl', 'konk_avg', col_result="konk_bl_avg_crash")
        df_konk = Utils_Yfinance.get_crash_points(df_konk, 'konk_gre', 'konk_avg', col_result="konk_gre_avg_crash")
        df_konk = Utils_Yfinance.get_crash_points(df_konk, 'konk_gre', 'konk_bl', col_result="konk_gre_bl_crash")


    df_list = [df_acc, df_chaikin, df_chopp, df_coppock, df_donchian, df_ease_move, df_force_index, df_ma_hull, df_keltner, df_mass_index, df_supertrend, df_vortex,df_vortex2, df_konk]
    df = df_stocks
    for dfi in df_list:
        dfi = UtilsL.replace_bat_chars_in_columns_name(dfi)
        # for c in dfi.columns.values:
        #     dfi = UtilsL.replace_bat_chars_in_columns_name(dfi, str(c))
        # dfi.columns = map(str.upper, dfi.columns)
        # [UtilsL.replace_bat_chars_in_columns(dfi, str(col)) for col in dfi.columns.values]
        dfi = UtilsL.add_rename_all_columns_df(dfi, prefix="ti_")
        df = pd.concat([df, dfi], axis=1)
    #df_trad = UtilsL.add_rename_all_columns_df(df_trad, prefix="trad_")
    # df.columns = map(UtilsL.replace_bat_chars_in_columns,df, df.columns)
    # [UtilsL.replace_bat_chars_in_columns(dfi, str(col)) for col in dfi.columns.values]
    return df



import talib
# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
def get_konkorde_params(df_stocks):
    # df['calc_nvi'] =  df.ta.nvi( cumulative=True, append=False) #calc_nvi(df)
    # tprice=ohlc4
    tprice = (df_stocks['Open'] + df_stocks['High'] + df_stocks['Low'] + df_stocks['Close']) / 4
    # lengthEMA = input(255, minval=1)
    # pvi = calc_pvi()
    # df['calc_pvi'] = df.ta.pvi( cumulative=True, append=False) #calc_pvi(df)
    pvi = df_stocks.ta.nvi(cumulative=True, append=False)  # calc_pvi(df)
    m = 15
    # pvim = ema(pvi, m)
    pvim = talib.EMA(pvi, timeperiod=m)
    # pvimax = highest(pvim, 90)
    # pvimax = lowest(pvim, 90)
    pvimax = pvim.rolling(window=90).max()  # .shift(-89)
    pvimin = pvim.rolling(window=90).min()  # .shift(-89)
    # oscp = (pvi - pvim) * 100/ (pvimax - pvimin)
    oscp = (pvi - pvim) * 100 / (pvimax - pvimin)
    # nvi =calc_nvi()
    # nvim = ema(nvi, m)
    # nvimax = highest(nvim, 90)
    # nvimin = lowest(nvim, 90)
    # azul = (nvi - nvim) * 100/ (nvimax - nvimin)
    nvi = df_stocks.ta.nvi(cumulative=True, append=False)  # calc_nvi(df)
    nvim = talib.EMA(nvi, timeperiod=15)
    nvimax = nvim.rolling(window=90).max()  # .shift(-89)
    nvimin = nvim.rolling(window=90).min()  # .shift(-89)
    val_blue = (nvi - nvim) * 100 / (nvimax - nvimin)
    xmf = talib.MFI(df_stocks['High'], df_stocks['Low'], df_stocks['Close'], df_stocks['Volume'], timeperiod=14)
    # mult=input(2.0)
    basis = talib.SMA(tprice, 25)
    dev = 2.0 * talib.STDDEV(tprice, 25)
    upper = basis + dev
    lower = basis - dev
    # OB1 = (upper + lower) / 2.0
    # OB2 = upper - lower
    # BollOsc = ((tprice - OB1) / OB2 ) * 100
    # xrsi = rsi(tprice, 14)
    OB1 = (upper + lower) / 2.0
    OB2 = upper - lower
    BollOsc = ((tprice - OB1) / OB2) * 100
    xrsi = talib.RSI(tprice, 14)

    # calc_stoch(src, length,smoothFastD ) =>
    #     ll = lowest(low, length)
    #     hh = highest(high, length)
    #     k = 100 * (src - ll) / (hh - ll)
    #     sma(k, smoothFastD)
    def calc_stoch(src, length, smoothFastD):
        ll = df_stocks['Low'].rolling(window=length).min()
        hh = df_stocks['High'].rolling(window=length).max()
        k = 100 * (src - ll) / (hh - ll)
        return talib.SMA(k, smoothFastD)

    stoc = calc_stoch(tprice, 21, 3)
    # stoc = py_ti.stochastic(tprice, 21, 3)
    val_brown = (xrsi + xmf + BollOsc + (stoc / 3)) / 2
    val_green = val_brown + oscp
    val_avg = talib.EMA(val_brown, timeperiod=m)
    return val_blue, val_brown, val_green,val_avg

