import os.path
import numpy
import pandas as pd
import requests
from datetime import datetime

import Model_predictions_handle_Nrows
import yfinance as yf
from LogRoot.Logging import Logger
from _KEYS_DICT import DICT_WEBULL_ID
from Utils import Utils_col_sele, UtilsL
# from talib_technical_class_object import TechData
from technical_indicators.talib_technical_class_object import TechData


#YAHOO API
def crate_OLHCV_df_for_enque_yhooApi(col_v, date_15min, int_unlast):
    Logger.logr.info("   La accion tiene volumen actualizado Stock: " + col_v + " Volume: " + str(int_unlast) + " RealTime: " + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    stock_columns = [x for x in date_15min.columns if x[1] == col_v]
    df_S_raw = date_15min[stock_columns]

    df_S_raw.columns = ['{}'.format(x[0]) for x in df_S_raw.columns]  # se retira el ticker  y se elimina el tuple columns
    df_S_raw = df_S_raw.dropna()
    df_S_raw.reset_index(inplace=True)
    df_S_raw = df_S_raw.rename(columns={'Datetime': 'Date'})
    df_S_raw['Date'] = df_S_raw['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')  # pd.to_datetime(df_S_raw['Date'])
    return df_S_raw


def enque_if_has_realtime_volume_yhooApi(df_15min, list_pro, queue):
    drop_columns = [x for x in df_15min.columns if x[0] == 'Adj Close']
    df_15min = df_15min.drop(columns=drop_columns, errors='ignore')
    vol_columns = [x for x in df_15min.columns if x[0] == 'Volume']
    for col_v in vol_columns:
        #int_last = df_15min[col_v].dropna().iloc[-1]  # last
        int_unlast = df_15min[col_v].dropna().iloc[-2]# unlast
        date_int_unlast = df_15min[col_v].dropna().index[-2]

        if numpy.isnan(int_unlast) or int_unlast <= 0:
            Logger.logr.info("   El volumnen es 0 o Nan , Seguir buscando actualización en la siguiente consulta Stock: " +col_v[1] + " Date: " + str(date_int_unlast))
        else:
            df_S_raw = crate_OLHCV_df_for_enque_yhooApi(col_v[1], df_15min, int_unlast)
            Logger.logr.info("  DF encolado Stock: " + col_v[1] + " Shape_DF: " + str(df_S_raw.shape) + " RealTime: " + str(date_int_unlast))
            list_pro.remove(col_v[1])
            queue.pop(col_v[1])
            queue.set(col_v[1], df_S_raw)


startswith_str_in_colum='NQ_'
def get_tech_data_nasq(S, df_S, df_nasq):
    custom_columns_POS_NEG = Model_predictions_handle_Nrows.get_columns_to_download(S)
    df_his = TechData(df_S, custom_columns_POS_NEG).get_ALL_tech_data()
    df_his = pd.merge(df_his, df_nasq, how='left')
    cols_NQ = [col for col in df_his.columns if col.startswith(startswith_str_in_colum)]
    df_his[cols_NQ] = df_his[cols_NQ].fillna(method='ffill')
    df_his[cols_NQ] = df_his[cols_NQ].fillna(method='bfill')
    df_his['Date'] = df_his['Date'].astype(str)
    df_his.reset_index(drop=True, inplace=True)
    df_his = df_his.drop(columns=Utils_col_sele.COLUMNS_DELETE_NO_ENOGH_DATA, errors='ignore')
    return df_his


#YAHOO API
def df_yhoo_(S, inter, path = None ):
    date_15min = yf.download(tickers=S, period='6d', interval=inter, prepos=False)
    date_15min.index = date_15min.index.tz_convert(None)#location zone adapt to current zone
    date_15min.reset_index(inplace=True)
    date_15min = date_15min.rename(columns={'Datetime': 'Date'})
    # date_15min['Date'] = date_15min['Date'] + pd.Timedelta(hours=5)
    date_15min['Date'] = date_15min['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    date_15min = date_15min.sort_values('Date', ascending=False).round(2)
    if path is not None:
        date_15min[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']].to_csv(path, sep="\t", index=None)
    return date_15min[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
#WEBULL API
def get_df_webull_realTime(INTERVAL, S,path = None):
    if S.endswith("-USD"):
        return None, None

    we_bull_id = DICT_WEBULL_ID[S]
    URL = "https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=" + INTERVAL + "&tickerIds=" + str(we_bull_id)
    Logger.logr.info(S + " ================== " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "   " + URL)
    response = requests.get(URL).json()
    csv_data = response[0]['data']
    df_S_raw = pd.DataFrame([x.split(',') for x in csv_data],columns=['Date', 'Open', 'Close', 'High', 'Low', 'Close_avg', 'Volume', "Adj"])
    df_S_raw = df_S_raw.fillna(0).replace("null", 0).astype(float)
    df_S_raw['Date'] = pd.to_datetime(df_S_raw['Date'], unit="s")
    df_S_raw = df_S_raw[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]

    if path is not None:
        Logger.logr.debug("  File: "+path)
        df_S_raw.to_csv(path, sep="\t", index=None)

    df_S_raw = df_S_raw.sort_values('Date', ascending=False)
    df_S_raw = UtilsL.union_3last_rows_to_one_OLHLV(df_S_raw)

    # df_S_raw[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']].to_csv("Test/Compare_yhoo_webull/" + S + "_5m_webull.csv", sep="\t", index=None)
    SUFFIXES_DATE_ENDS = ("00:00", "15:00", "30:00", "45:00")
    df_S_raw['Date'] = df_S_raw['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_primeros = df_S_raw[:2]
    df_S_raw = df_S_raw[df_S_raw['Date'].str.endswith(SUFFIXES_DATE_ENDS)]
    #Simepre despues de hacer la union de los 5min a 15min
    df_S_raw = df_S_raw.sort_values('Date', ascending=False).round(2)
    if path is not None:
        Logger.logr.debug("  File: "+path)
        df_S_raw.to_csv(path.replace("_d5","_d15"), sep="\t", index=None)
    return df_S_raw, df_primeros


def merge_dataframes_bull_yhoo(S, df_S_raw, df_primeros, df_yhoo):
    if df_primeros is None or 1 >= len( df_primeros['Volume']) or 1 > df_primeros['Volume'][0] or 1 > df_primeros['Volume'][1]:
        Logger.logr.info(
            "No volume data has been obtained from API bull, (probably a crypto) Only Yahoo data will be used. Stock: " + S)
        df = df_yhoo
    else:
        df = (pd.concat([df_S_raw, df_yhoo], ignore_index=True, sort=False).drop_duplicates(['Date'], keep='first'))
        df = (pd.concat([df_primeros, df], ignore_index=True, sort=False).drop_duplicates(['Date'], keep='first'))

    df = df.sort_values('Date', ascending=True, ignore_index=True)
    return df


#WEBULL API