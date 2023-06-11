"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
import logging
import re
from threading import Thread
import threading
from random import randint
from datetime import datetime
import pandas as pd
import time

import traceback

import Model_predictions_handle_Nrows
import Model_predictions_handle_Multi_Nrows
import ztelegram_send_message
from LogRoot.Logging import Logger
import yhoo_history_stock
from Utils.Utils_QueueMap import QueueMap
from _KEYS_DICT import Op_buy_sell, Option_Historical, DICT_WEBULL_ID, DICT_COMPANYS
# from api_twitter import twi_
from predict_POOL_handle import get_tech_data_nasq, get_df_webull_realTime, df_yhoo_, merge_dataframes_bull_yhoo
from ztelegram_send_message import send_alert_and_register, send_exception
LOGGER = logging.getLogger()
LOGGER.disabled = False


# list_models_pos_neg = get_list_models_to_use()
# list_pos = [x.replace("_"+Op_buy_sell.POS.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.POS.name)]
# list_neg = [x.replace("_"+Op_buy_sell.NEG.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.NEG.name)]
# list_stocks =  set(list_pos +list_neg)

# df_result = pd.read_csv("Models/TF_multi/_RESULTS_multi_all.csv", index_col=0,sep='\t')
df_result = pd.read_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", index_col=0,sep='\t')


list_pos = [x for x in df_result.columns if  "_" + Op_buy_sell.POS.value + "_" in x and  not x.endswith("_per") ]
list_neg = [x for x in df_result.columns if  "_" + Op_buy_sell.NEG.value + "_" in x and  not x.endswith("_per") ]
list_stocks_models = set(list_pos +list_neg)
regex_S = "TFm_([A-Z]{1,5}|[A-Z]{1,5}-USD)_(pos|neg)_"
list_stocks = [re.search(regex_S, x, re.IGNORECASE).group(1) for x in list_stocks_models]
list_stocks = set(list_stocks)
print("Stoscks loaded: "+ ", ".join(list_stocks))

# for r in ["ZEC-USD",'BCH-USD']:
#     list_stocks.remove(r)

#Cripto
# list_cripto = [x for x in list_stocks if x.endswith("-USD")]
# list_stocks = list_cripto





#CONSUMER MANAGE
def scaler_and_send_predit(S, df_S, df_nasq, is_multidimension = False ):
    df_S = get_tech_data_nasq(S, df_S, df_nasq)
    df_tech = df_S[-NUM_LAST_REGISTERS_PER_STOCK:]

    if is_multidimension:
        df_eval_multi = Model_predictions_handle_Multi_Nrows.get_df_Multi_comprar_vender_predictions(df_tech, S, path_result_eval=None)
        if df_eval_multi is not None:
            ztelegram_send_message.send_MULTI_alert_and_register(S, df_eval_multi)
    else:
        df_tech = Model_predictions_handle_Nrows.add_min_max_(df_S, S)
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
        timeout_start = time.time()

        while time.time() < timeout_start + TIME_OUT_PRODUCER:
            list_pro = list_stocks.copy()
            INTERVAL = "d5"  # ""d1" #
            for S in list_stocks:
                try:
                    time.sleep(randint(1, 7))#esperar en 1 y 10s , por miedo a baneo
                    df_S_raw, df_primeros = get_df_webull_realTime(INTERVAL, S,None)# path= "d_price/weBull/weBull_"+S+"_"+INTERVAL+".csv")
                    #retiramos las primeras 40 para que no se solapen
                    df_yhoo = df_yhoo_(S, "15m")[1:]#, path= "d_price/weBull/yhoo_"+S+"_15m.csv")[40:] #
                    df = merge_dataframes_bull_yhoo(S, df_S_raw, df_primeros, df_yhoo)
                    Logger.logr.info(" DF encolado Stock: " + S + " Shape_DF: " + str(df.shape) + " RealTime: " + str(df.iloc[-1]['Date']) + " Volume: "+ str(df.iloc[1]['Volume']) )
                    list_pro.remove(S)#para yhoo API
                    queue.pop(S)
                    queue.set(S, df)
                except Exception as ex:
                    Logger.logr.warning(" Exception Stock: " + S + "  Exception: " + traceback.format_exc())
                    send_exception(ex, "[PRO] Exception Stock: " + S +"\n"+traceback.format_exc())

            if not list_pro:
                Logger.logr.info(" sleep(60 * 2) List all stock download empty")
                # from XTB_api import xAPIConnector_trade
                # xAPIConnector_trade.updates_tp_sp()
                time.sleep( 60 * 2 )
                break
            else:
                Logger.logr.info(" Sleep(60) There are still values left in the Values queue list:  "+ " ".join(list_pro))
                time.sleep(20)

            if  "30:00" in  datetime.today().strftime('%Y-%m-%d %H:%M:%S') or "00:00" in  datetime.today().strftime('%Y-%m-%d %H:%M:%S'):
                ztelegram_send_message.send_mesage_all_people("<pre> RUNING it is alive: " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "</pre>")
        Logger.logr.info(' Producer: Running End ' + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    Logger.logr.info(' Producer: Done')



NUM_LAST_REGISTERS_PER_STOCK = 32

COUNT_THREAD = 4
# consume work
def consumer(int_thread):
    global queue
    Logger.logr.debug("  Consumer: Running")
    list_pro = list_stocks.copy()
    # consume work
    while True:
        df_nasq = yhoo_history_stock.get_NASDAQ_data(exter_id_NQ = "NQ=F", interval='15m' , opion=Option_Historical.DAY_6, remove_str_in_colum = "=F")
        LOGGER = logging.getLogger()
        LOGGER.disabled = False
        Logger.logr.debug("  cycle started   Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        # list_recorrer = list_pro.copy()
        # if int_thread == 2:
        #     list_recorrer = list_pro.copy()[-1]
        for S in list_pro:
            df_S = queue.pop(S)
            if df_S is not None:
                Logger.logr.info("  Stock: " + S + "  Volume unlast: " + str(df_S.iloc[-2]['Volume']) + " Volume last: " + str(df_S.iloc[-1]['Volume'])+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
                try:
                    scaler_and_send_predit(S, df_S, df_nasq, is_multidimension = True)
                except Exception as ex:
                    if 'not in index' in str(ex) or 'No objects to concatenate' in str(ex)or 'inputs are all ' in str(ex): #No objects to concatenate
                        #Todo manage this exceptions
                        # raw_df[columns_selection] las columns_selection no han sido calculadas en el df_tech , o han desaparecido
                        Logger.logr.warning(" Exception raw_df = raw_df[columns_selection] the columns_selection have not been calculated in the df_tech , or have disappeared  " + str(ex))
                    else:
                        Logger.logr.warning(" Exception:  Stock: " + S +  traceback.format_exc())
                        #send_exception(ex, "[CON] [" + str(int_thread) * 4 + "]Exception Stock: " + S + "\n <pre>" + traceback.format_exc()+"</pre>")

            # print("[CON] end " + S)

        Logger.logr.info(" completed cycle    Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S') + " stoks: "+ " ".join(list_pro))
        time.sleep(int_thread *15)
        # UtilsL.thread_list_is_alive([producer_thr, consumer_thr_1, consumer_thr_2],producer,consumer )
    Logger.logr.info(" Consumer: Done"+ " Date: "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))



# #**DOCU**
# 5.3 Sending real-time alerts Telegram
# The criteria to send alert or not , is defined in the method ztelegram_send_message.will_send_alert() .If more than half of models have a score greater than 93% or TF models have a score greater than 93%, alert is sent to the consuming users.
# Run predict_POOL_inque_Thread.py
# This class has 2 types of threads
# Producer , constantly asks for OHLCV data, once it is obtained, it enters it in a queue.
# Consumer (2 threads run simultaneously) they get the OHLCV data from the queue, calculate the technical parameters, make the prediction of the models, register them in _KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME, and if they meet the requirements they are sent by telegram.

# create the shared queue
queue = QueueMap()
# start the producer
ztelegram_send_message.send_mesage_all_people("<pre> START: "+datetime.today().strftime('%Y-%m-%d %H:%M:%S') +" </pre>\nStocks to be monitored: \n"+" ".join(list_stocks)+" ")
# twi_.create_simple_tweet("START: "+datetime.today().strftime('%Y-%m-%d %H:%M:%S') +"\nStocks to be monitored:\n"+" ".join(list_stocks)+" ")
producer_thr = Thread(target=producer, args=(), name='PROD')
producer_thr.start()
time.sleep(2)

# start the consumer
# Creating 3 threads that execute the same function with different parameters
consumer_thr_1 = threading.Thread(target=consumer, args=(1,), name='CONS_1')
consumer_thr_2 = threading.Thread(target=consumer, args=(2,), name='CONS_2')
# # consumer_3 = threading.Thread(target=consumer, args=("[3333]",))
# # Start the threads
consumer_thr_1.start()
# consumer_thr_2.start()
# consumer_3.start()

# while True:
#     time.sleep(20)
#     thread_list_is_alive([producer_thr, consumer_thr_1, consumer_thr_2])
#
# producer_thr.join()
# consumer_thr_1.join()
# consumer_thr_2.join()
