# SuperFastPython.com
# example of using the queue
from time import sleep
from threading import Thread
import threading
from queue import Queue
from sklearn.preprocessing import MinMaxScaler

import Model_predictions_Nrows
import Utils_col_sele
import yfinance as yf
import yhoo_history_stock
from Model_predictions_Nrows import get_columns_to_download
from Utils_QueueMap import QueueMap
from a_manage_stocks_dict import Op_buy_sell, Option_Historical
from talib_technical_class_object import TechData
from yhoo_POOL_stocks import get_list_models_to_use
import pandas as pd
import numpy
from datetime import datetime

import time

from ztelegram_send_message import send_alert

list_models_pos_neg = get_list_models_to_use()
list_pos = [x.replace("_"+Op_buy_sell.POS.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.POS.name)]
list_neg = [x.replace("_"+Op_buy_sell.NEG.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.NEG.name)]
list_stocks =  set(list_pos +list_neg)

#YAHOO API
def crate_OLHCV_df_dor_enque_yhooApi(col_v, date_15min, int_unlast):
    print("[PRO] La accion tiene volumen actualizado Stock: " + col_v + " Volume: " + str(
        int_unlast) + " RealTime: " + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
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
            print("[PRO] El volumnen es 0 o Nan , Seguir buscando actualización en la siguiente consulta Stock: " +col_v[1] + " Date: " + str(date_int_unlast))
        else:
            df_S_raw = crate_OLHCV_df_dor_enque_yhooApi(col_v[1], df_15min, int_unlast)
            print("[PRO] DF encolado Stock: " + col_v[1] + " Shape_DF: " + str(df_S_raw.shape) + " RealTime: " + str(date_int_unlast))
            list_pro.remove(col_v[1])
            queue.pop(col_v[1])
            queue.set(col_v[1], df_S_raw)


startswith_str_in_colum='NQ_'
def get_tech_data_nasq(S, df_S, df_nasq):
    custom_columns_POS_NEG = get_columns_to_download(S)
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
    df_min_max = pd.read_csv("d_price/min_max/" + S + "_min_max_stock_" + str(Option_Historical.MONTH_3_AD.name) + ".csv", index_col=0, sep='\t')
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
#CONSUMER MANAGE
def scaler_and_send_predit(S, df_S, df_nasq):
    df_S = get_tech_data_nasq(S, df_S, df_nasq)
    df_S = df_S[-NUM_LAST_REGISTERS_PER_STOCK:]
    df_tech = add_min_max_Scaler(df_S, S)
    df_compar, df_vender = Model_predictions_Nrows.get_df_comprar_vender_predictions(df_tech, S)
    if df_compar is not None:
        send_alert(S, df_compar, Op_buy_sell.POS)
    if df_vender is not None:
        send_alert(S, df_vender, Op_buy_sell.NEG)

# timeout variable can be omitted, if you use specific value in the while condition
TIME_OUT_PRODUCER = 5 * 60   # [seconds]





# generate work
def producer():
    global queue
    print('[PRO] Producer: Running Start '+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    while True:
        print('[PRO] Producer: Running Start ' + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        list_pro = list_stocks.copy()
        timeout_start = time.time()

        while time.time() < timeout_start + TIME_OUT_PRODUCER:
            date_15min = yf.download(tickers=list_pro, period='6d', interval='15m' ,prepos=False )
            enque_if_has_realtime_volume_yhooApi(date_15min, list_pro, queue)

            if not list_pro:
                print("[PRO] sleep(60 * 2) List all stock download empty")
                time.sleep(80 )
                break
            else:
                print("[PRO] Sleep(60) Quedan Valores en la lista de encolar Valores:  "+ " ".join(list_pro))
                time.sleep(20)

        print('[PRO]Producer: Running End '+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print('[PRO] Producer: Done')


sc = MinMaxScaler(feature_range=(-100, 100))
NUM_LAST_REGISTERS_PER_STOCK = 6

COUNT_THREAD = 4
# consume work
def consumer(int_thread):
    global queue
    print("[CON] [" + str(int_thread) * 4 + "] Consumer: Running")
    list_pro = list_stocks.copy()
    # consume work
    while True:
        df_nasq = yhoo_history_stock.get_NASDAQ_data(exter_id_NQ = "NQ=F", interval='15m' , opion=Option_Historical.DAY_6, remove_str_in_colum = "=F")
        print("[CON] [" + str(int_thread) * 4 + "] ciclo iniciado   Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        for S in list_pro:
            df_S = queue.pop(S)
            if df_S is not None:
                print("[CON] [" + str(int_thread) * 4 + "] Stock: " + S + "  Volume unlast: " + str(df_S.iloc[-2]['Volume']) + " Volume last: " + str(df_S.iloc[-1]['Volume'])+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
                try:
                    scaler_and_send_predit(S, df_S, df_nasq)
                except Exception as e:
                    print("[CON] [" + str(int_thread) * 4 + "]Exception Stock: " + S + "  Exception: " + str(e))

            # print("[CON] end " + S)

        print("[CON] [" + str(int_thread) * 4 + "] ciclo finalizado " + S+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(int_thread *15)
    print("[CON] [" + str(int_thread) * 4 + "] Consumer: Done"+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))






# create the shared queue
queue = QueueMap()
# start the producer
producer = Thread(target=producer, args=())
producer.start()
time.sleep(2)

# start the consumer
# Creating 3 threads that execute the same function with different parameters
consumer_1 = threading.Thread(target=consumer, args=(1,))
consumer_2 = threading.Thread(target=consumer, args=(2,))
# consumer_3 = threading.Thread(target=consumer, args=("[3333]",))
# Start the threads
consumer_1.start()
consumer_2.start()
# consumer_3.start()

