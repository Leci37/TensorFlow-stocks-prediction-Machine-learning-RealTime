from threading import Thread
import threading
from random import randint
import os.path

import requests
import traceback
from sklearn.preprocessing import MinMaxScaler

import Model_predictions_handle_Nrows
import ztelegram_send_message
from LogRoot.Logging import Logger
from Utils import Utils_col_sele, UtilsL
import yhoo_history_stock
from Utils.Utils_QueueMap import QueueMap
from a_manage_stocks_dict import Op_buy_sell, Option_Historical, DICT_WEBULL_ID, DICT_COMPANYS
from predict_POOL_load_stocks_names import get_list_models_to_use
from talib_technical_class_object import TechData

import pandas as pd
import numpy
from datetime import datetime
import yfinance as yf


import time

from ztelegram_send_message import send_alert_and_register, send_exception

list_models_pos_neg = get_list_models_to_use()
list_pos = [x.replace("_"+Op_buy_sell.POS.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.POS.name)]
list_neg = [x.replace("_"+Op_buy_sell.NEG.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.NEG.name)]
list_stocks =  set(list_pos +list_neg)

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

def add_min_max_Scaler( df_S, S):
    if os.path.exists("d_price/min_max/" + S + "_min_max_stock_" + str(Option_Historical.MONTH_3.name) + ".csv"):
        df_min_max = pd.read_csv("d_price/min_max/" + S + "_min_max_stock_" + str(Option_Historical.MONTH_3.name) + ".csv", index_col=0, sep='\t')
    else:
        df_min_max = pd.read_csv("d_price/min_max/" + S + "_min_max_stock_" + str(Option_Historical.MONTH_3_AD.name) + ".csv", index_col=0,sep='\t')
    df_min_max = df_min_max[df_S.columns]
    df_S = pd.concat([df_min_max, df_S], ignore_index=True)

    aux_date_save = df_S['Date']  # despues se añade , hay que pasar el sc.fit_transform
    df_S['Date'] = 0
    array_stock = sc.fit_transform(df_S)
    df_S = pd.DataFrame(array_stock, columns=df_S.columns)
    df_S.insert(loc=1, column='ticker', value=S)
    df_S['Date'] = aux_date_save  # to correct join

    return df_S


#YAHOO API
def df_yhoo_(S, inter, path = None ):
    date_15min = yf.download(tickers=S, period='6d', interval=inter, prepos=False)
    date_15min.reset_index(inplace=True)
    date_15min = date_15min.rename(columns={'Datetime': 'Date'})
    # df['Date'] = pd.DatetimeIndex(df['Date']) + pd.Timedelta(hours=4)
    date_15min['Date'] = date_15min['Date'] + pd.Timedelta(hours=4)
    date_15min['Date'] = date_15min['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    date_15min = date_15min.sort_values('Date', ascending=False).round(2)
    if path is not None:
        date_15min[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']].to_csv(path, sep="\t", index=None)
    return date_15min[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
#WEBULL API
def get_df_webull_realTime(INTERVAL, S,path = None):
    we_bull_id = DICT_WEBULL_ID[S]
    URL = "https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=" + INTERVAL + "&tickerIds=" + str(we_bull_id)
    Logger.logr.info(S + " ================== " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "   " + URL)
    response = requests.get(URL).json()
    csv_data = response[0]['data']
    df_S_raw = pd.DataFrame([x.split(',') for x in csv_data],columns=['Date', 'Open', 'Close', 'High', 'Low', 'Close_avg', 'Volume', "Adj"])
    df_S_raw = df_S_raw.astype(float)
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



#WEBULL API



#CONSUMER MANAGE
def scaler_and_send_predit(S, df_S, df_nasq):
    df_S = get_tech_data_nasq(S, df_S, df_nasq)
    df_S = df_S[-NUM_LAST_REGISTERS_PER_STOCK:]
    df_tech = add_min_max_Scaler(df_S, S)
    df_compar, df_vender = Model_predictions_handle_Nrows.get_df_comprar_vender_predictions(df_tech, S)
    if df_compar is not None:
        send_alert_and_register(S, df_compar, Op_buy_sell.POS)
    if df_vender is not None:
        send_alert_and_register(S, df_vender, Op_buy_sell.NEG)

# timeout variable can be omitted, if you use specific value in the while condition
TIME_OUT_PRODUCER = 5 * 60   # [seconds]





# generate work
def producer():
    global queue
    Logger.logr.info(' Producer: Running Start '+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    while True:
        Logger.logr.debug(' Producer: Running Start ' + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        list_pro = list_stocks.copy()
        timeout_start = time.time()

        while time.time() < timeout_start + TIME_OUT_PRODUCER:
            # date_15min = yf.download(tickers=list_pro, period='6d', interval='15m' ,prepos=False )
            # enque_if_has_realtime_volume_yhooApi(date_15min, list_pro, queue)
            INTERVAL = "d5"  # ""d1" #
            for S in list_stocks:
                try:
                    time.sleep(randint(1, 7))#esperar en 1 y 10s , por miedo a baneo
                    df_S_raw, df_primeros = get_df_webull_realTime(INTERVAL, S,None)# path= "d_price/weBull/weBull_"+S+"_"+INTERVAL+".csv")
                    #retiramos las primeras 40 para que no se solapen
                    df_yhoo = df_yhoo_(S, "15m")[5:]#, path= "d_price/weBull/yhoo_"+S+"_15m.csv")[40:] #
                    df = (pd.concat([df_S_raw, df_yhoo], ignore_index=True, sort =False).drop_duplicates(['Date'], keep='first'))
                    df = (pd.concat([df_primeros, df], ignore_index=True, sort=False).drop_duplicates(['Date'], keep='first'))
                    df = df.sort_values('Date', ascending=True,ignore_index=True)
                    Logger.logr.info(" DF encolado Stock: " + S + " Shape_DF: " + str(df.shape) + " RealTime: " + str(df.iloc[-1]['Date']) + " Volume: "+ str(df.iloc[1]['Volume']) )
                    list_pro.remove(S)#para yhoo API
                    queue.pop(S)
                    queue.set(S, df)
                except Exception as ex:
                    Logger.logr.warning(" Exception Stock: " + S + "  Exception: " + traceback.format_exc())
                    send_exception(ex, "[PRO] Exception Stock: " + S +"\n"+traceback.format_exc())

            if not list_pro:
                Logger.logr.info(" sleep(60 * 2) List all stock download empty")
                time.sleep( 60 * 2.5 )
                break
            else:
                Logger.logr.info(" Sleep(60) There are still values left in the Values queue list:  "+ " ".join(list_pro))
                time.sleep(20)

            if  "30:00" in  datetime.today().strftime('%Y-%m-%d %H:%M:%S') or "00:00" in  datetime.today().strftime('%Y-%m-%d %H:%M:%S'):
                ztelegram_send_message.send_mesage_all_people("<pre> RUNING it is alive: " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "</pre>")
        Logger.logr.info(' Producer: Running End ' + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    Logger.logr.info(' Producer: Done')


sc = MinMaxScaler(feature_range=(-100, 100))
NUM_LAST_REGISTERS_PER_STOCK = 6

COUNT_THREAD = 4
# consume work
def consumer(int_thread):
    global queue
    Logger.logr.debug("  Consumer: Running")
    list_pro = list_stocks.copy()
    # consume work
    while True:
        df_nasq = yhoo_history_stock.get_NASDAQ_data(exter_id_NQ = "NQ=F", interval='15m' , opion=Option_Historical.DAY_6, remove_str_in_colum = "=F")
        Logger.logr.debug("  cycle started   Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        # list_recorrer = list_pro.copy()
        # if int_thread == 2:
        #     list_recorrer = list_pro.copy()[-1]
        for S in list_pro:
            df_S = queue.pop(S)
            if df_S is not None:
                Logger.logr.info("  Stock: " + S + "  Volume unlast: " + str(df_S.iloc[-2]['Volume']) + " Volume last: " + str(df_S.iloc[-1]['Volume'])+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
                try:
                    scaler_and_send_predit(S, df_S, df_nasq)
                except Exception as ex:
                    if 'not in index' in str(ex) or 'No objects to concatenate' in str(ex)or 'inputs are all ' in str(ex): #No objects to concatenate
                        #Todo manage this exceptions
                        # raw_df[columns_selection] las columns_selection no han sido calculadas en el df_tech , o han desaparecido
                        Logger.logr.warning(" Exception raw_df = raw_df[columns_selection] the columns_selection have not been calculated in the df_tech , or have disappeared  " + str(ex))
                    else:
                        Logger.logr.warning(" Exception: " + traceback.format_exc())
                        #send_exception(ex, "[CON] [" + str(int_thread) * 4 + "]Exception Stock: " + S + "\n <pre>" + traceback.format_exc()+"</pre>")

            # print("[CON] end " + S)

        Logger.logr.info(" completed cycle    Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S') + " stoks: "+ " ".join(list_pro))
        time.sleep(int_thread *15)
    Logger.logr.info(" Consumer: Done"+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))



# #**DOCU**
# 5.3 Sending real-time alerts Telegram
# The criteria to send alert or not , is defined in the method ztelegram_send_message.will_send_alert() .If more than half of models have a score greater than 93% or TF models have a score greater than 93%, alert is sent to the consuming users.
# Run predict_POOL_inque_Thread.py
# This class has 2 types of threads
# Producer , constantly asks for OHLCV data, once it is obtained, it enters it in a queue.
# Consumer (2 threads run simultaneously) they get the OHLCV data from the queue, calculate the technical parameters, make the prediction of the models, register them in a_manage_stocks_dict.PATH_REGISTER_RESULT_REAL_TIME, and if they meet the requirements they are sent by telegram.

# create the shared queue
queue = QueueMap()
# start the producer
ztelegram_send_message.send_mesage_all_people("<pre> START: "+datetime.today().strftime('%Y-%m-%d %H:%M:%S') +" </pre>\nStocks actions to be monitored: \n"+" ".join(list_stocks)+" ")
producer = Thread(target=producer, args=(), name='PROD')
producer.start()
time.sleep(2)

# start the consumer
# Creating 3 threads that execute the same function with different parameters
consumer_1 = threading.Thread(target=consumer, args=(1,), name='CONS_1')
consumer_2 = threading.Thread(target=consumer, args=(2,), name='CONS_2')
# # consumer_3 = threading.Thread(target=consumer, args=("[3333]",))
# # Start the threads
consumer_1.start()
consumer_2.start()
# consumer_3.start()

