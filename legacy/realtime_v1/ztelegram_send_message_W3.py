#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2
import logging
import threading

import requests
import pandas as pd
import os.path
from datetime import datetime

from telegram.constants import ParseMode

import _KEYS_DICT
import yhoo_history_stock
from LogRoot.Logging import Logger
from Utils import Utils_send_message
from Utils.Plot_scrent_shot import get_traderview_screem_shot
from Utils.Utils_send_message import DICT_SCORE_RATE
from _KEYS_DICT import Option_Historical, DICT_COMPANYS , Op_buy_sell
from ztelegram_send_message_handle import URL_TELE, send_mesage_all_people, send_exception

# from useless.XTB_api import xtb_api

logging.config.fileConfig(r"LogRoot/logging.conf")
# LOGGER = logging.getLogger()
# LOGGER.disabled = False
logging.root.manager.loggerDict['root'].disabled = False

message = "hello from your telegram bot"



# url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
# print(requests.get(url).json())  # this sends the message
all_cols = ['Date' 'buy_sell_point' 'Close' 'has_preMarket' 'Volume' 'sum_r_88', 'sum_r_93' 'have_to_oper' 'sum_r_TF' 'have_to_oper_TF' 'num_models']
prety_cols = ['Date' , 'sum_r_88', 'sum_r_93', 'have_to_oper', 'sum_r_TF', 'have_to_oper_TF', 'num_models']
BOT = None


NUM_MIN_MODLES  = 4
NUM_MIN_MODLES_TF = 2

def __get_runtime_value_date_by_yhoo(S):
    date_detect, value_detect = "n", "n"
    try:
        df_value = yhoo_history_stock.__select_dowload_time_config(interval="15m", opion=Option_Historical.DAY_1,
                                                                   prepost=False,stockId=S)  # interval, opion, prepost, stockId
        date_detect = df_value.tail(1)[['Date', 'Close']].values[0][0]
        value_detect = df_value.tail(1)[['Date', 'Close']].values[0][1]
    except Exception:
        message_aler = "** Exception **" + str(Exception)
        print(message_aler)
    return date_detect, value_detect

THORSHOOL_EVALUATE_NOT_TF = "93" #it could be "88" "95"
def will_send_alert(df_b_s, stotck_id_pos ):
    has_to_send_last_TF = False
    num_models = df_b_s.iloc[[-1]]['num_models'].values[0]
    # ultima
    has_to_send_last = df_b_s.iloc[[-1]]['sum_r_'+THORSHOOL_EVALUATE_NOT_TF].values[0]  > ((num_models / 2) *1.1 )

    # penultima
    # has_to_send_unlast = df_b_s.iloc[[-2]]['sum_r_93'].values[0] > ((num_models / 2) )
    # has_to_send_unlast_TOP = df_b_s.iloc[[-2]]['sum_r_93'].values[0] > ((num_models / 2) * 1.1)

    Logger.logr.info("\tPUNTUACIONES r_93 Stock: "+stotck_id_pos+" Penultima: "+ str(df_b_s.iloc[[-2]]['sum_r_'+THORSHOOL_EVALUATE_NOT_TF].values[0])
                     +"/"+ str(num_models) +"  Ultima: "+ str(df_b_s.iloc[[-1]]['sum_r_'+THORSHOOL_EVALUATE_NOT_TF].values[0]) +"/"+ str(num_models)  )


    modles_evaluated_TF = [col for col in df_b_s.columns if col.startswith('br_TF') and col.endswith("_"+THORSHOOL_EVALUATE_NOT_TF)]
    if len(modles_evaluated_TF) > NUM_MIN_MODLES_TF:
        tf_93 = df_b_s[modles_evaluated_TF][list(df_b_s[modles_evaluated_TF].filter(regex='_93$'))].sum(axis=1)
        tf_95 = df_b_s[modles_evaluated_TF][list(df_b_s[modles_evaluated_TF].filter(regex='_95$'))].sum(axis=1)
        has_to_send_last_TF = tf_93.iloc[-1] >= (len(modles_evaluated_TF) / 2) or tf_95.iloc[-1] >= 1
        # has_to_send_unlast_TF = tf_93.iloc[-2] >= ( len(modles_evaluated_TF) / 2 )
        Logger.logr.info("\tPUNTUACIONES multiple TF r_93 Stock: " + stotck_id_pos + " Penultima: " + str(tf_93.iloc[-2])
                         + "/" + str(len(modles_evaluated_TF)) + "  Ultima: " + str(tf_93.iloc[-1])+ "/" + str(len(modles_evaluated_TF)))
    elif len(modles_evaluated_TF) == NUM_MIN_MODLES_TF:
        tf_95 =df_b_s[modles_evaluated_TF][list(df_b_s[modles_evaluated_TF].filter(regex='_95$'))].sum(axis=1)
        has_to_send_last_TF = tf_95.iloc[-1] >= ( len(modles_evaluated_TF) / 2   )
        # has_to_send_unlast_TF = tf_95.iloc[-2] >= ( len(modles_evaluated_TF) / 2 )
        Logger.logr.info("\tPUNTUACIONES unique TF r_95 Stock: " + stotck_id_pos + " Penultima: " + str(tf_95.iloc[-2])
                         + "/" + str(len(modles_evaluated_TF)) + "  Ultima: " + str(tf_95.iloc[-1])+ "/" + str(len(modles_evaluated_TF)))

    return has_to_send_last  or (has_to_send_last_TF)
    #return has_to_send_unlast_TOP or (has_to_send_last and has_to_send_unlast) or (has_to_send_unlast_TF)#  or has_to_send_unlast_TF



COL_GANAN = ["Date", "stock", "type_buy_sell","value_start", "message" ]
df_registre = pd.DataFrame(columns=COL_GANAN)

# def send_alert_and_register(S, df_b_s, type_b_s):
#     global df_registre
#     modles_evaluated = [col for col in df_b_s.columns if col.startswith('br_')]
#     modles_evaluated_TF = [col for col in df_b_s.columns if col.startswith('br_TF')]
#     dict_predictions = df_b_s.T.to_dict()
#
#     #register data each time UNLAST
#     Utils_send_message.register_in_zTelegram_Registers(S, dict_predictions[list(dict_predictions.keys())[-2]], modles_evaluated, type_b_s, path =_KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME)  #unlast row
#     #LAST
#     Utils_send_message.register_in_zTelegram_Registers(S, dict_predictions[list(dict_predictions.keys())[-1]], modles_evaluated, type_b_s, path =_KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME)  #last row
#
#     i = -1
#     if will_send_alert(df_b_s, S+"_"+type_b_s.name):
#         dict_predict_last = dict_predictions[list(dict_predictions.keys())[i]]
#
#         date_detect, value_detect = __get_runtime_value_date_by_yhoo(S)
#
#         message_aler , alert_message_without_tags = Utils_send_message.get_string_alert_message(S, dict_predict_last, modles_evaluated,
#                                                                                                 type_b_s, date_detect, value_detect)
#
#         df_registre = pd.concat([df_registre, pd.DataFrame([[date_detect, S, type_b_s.name, "{:.2f}".format(value_detect) , alert_message_without_tags ]], columns=COL_GANAN)  ],ignore_index=True)  # añadir fila add row
#         df_registre.to_csv("zSent_Telegram_Registers.csv", sep="\t", index=None , mode='a', header=False)
#
#         send_mesage_all_people(message_aler)
#
#
#
# def send_MULTI_alert_and_register(S, df_mul_r):
#     list_models_to_predict_POS = [x for x in df_mul_r.columns if x.startswith("Acert_TFm_" + S + "_" + Op_buy_sell.POS.value)]
#     list_models_to_predict_NEG = [x for x in df_mul_r.columns if x.startswith("Acert_TFm_" + S + "_" + Op_buy_sell.NEG.value)]
#
#     if (not list_models_to_predict_POS) and (not list_models_to_predict_NEG):
#         Logger.logr.warning("There are no models of class columns_json in the list_good_params list, columns_json, optimal enough, we pass to the next one. Stock: " + S)
#         return
#
#     df_mul = df_mul_r[['Date', 'buy_sell_point', 'Close', 'Volume'] + list_models_to_predict_POS + list_models_to_predict_NEG].iloc[-2:]
#     # df_mul[list_models_to_predict_POS] = df_mul[list_models_to_predict_POS].map(lambda x: x.replace("%", "").replace(" ", "0")).astype('int')
#     # df_mul[list_models_to_predict_NEG] = df_mul[list_models_to_predict_NEG].map(lambda x: x.replace("%", "").replace(" ", "0")).astype('int')
#     for c in list_models_to_predict_POS + list_models_to_predict_NEG:
#         df_mul[c] = df_mul[c].str.replace("%", "").replace(" ", "0").astype('int')
#
#     LIST_COLS_TO_CSV = ['Date',"ticker", 'buy_sell_point', 'Close', 'Volume', "POS_score","POS_num",  "NEG_score", "NEG_num","POS_score_list","NEG_score_list", "POS_models_name","NEG_moldel_name" ]
#
#     df_mul = add_col_df_mul_to_csv_result_predictions(S, df_mul, list_models_to_predict_NEG, list_models_to_predict_POS)
#     # df_mul["NEG_score"] = str(df_mul[list_models_to_predict_NEG].sum(axis=1)) + "/" + str(len(list_models_to_predict_NEG))
#     # ['Date', 'buy_sell_point', 'Close', 'Volume', "POS_score", "NEG_score"]
#
#     Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul[LIST_COLS_TO_CSV], path = _KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME)
#
#     if len(list_models_to_predict_POS) > 0 or True:
#         score_POS = DICT_SCORE_RATE[str(len(list_models_to_predict_POS))]
#         if (df_mul["POS_score"][1:] >= score_POS).any() and any((df_mul[list_models_to_predict_POS][1:] >= 91).any()) :# or 1 == 1 :
#
#             # from XTB_api import xtb_api
#             # thr_xtb = threading.Thread(target=xtb_api.xtb_operate_Lock_thread, args=(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS), name='XTB_POS')
#             # thr_xtb.start()
#
#             # xtb_api.xtb_operate_Lock_thread(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS)
#             if not is_already_sent_alert(df_mul):
#                 thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul,  Op_buy_sell.POS, list_models_to_predict_POS), name='SENDING_POS')
#                 thr_sending.start()
#                 # send_image_full_alersts(S, df_mul, op_buy_sell=Op_buy_sell.POS,list_models_to_predict=list_models_to_predict_POS)
#
#             Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul[LIST_COLS_TO_CSV], path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)
#
#
#     if len(list_models_to_predict_NEG) > 0:
#         score_NEG = DICT_SCORE_RATE[str(len(list_models_to_predict_NEG))]
#         if (df_mul["NEG_score"][1:] >= score_NEG).any() and any((df_mul[list_models_to_predict_NEG][1:] >= 91).any()) :# or 1 == 1 :
#
#             # from XTB_api import xtb_api
#             # thr_xtb = threading.Thread(target=xtb_api.xtb_operate_Lock_thread, args=(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.NEG, list_models_to_predict_NEG), name='XTB_NEG')
#             # thr_xtb.start()
#
#             # xtb_api.xtb_operate_Lock_thread(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.NEG, list_models_to_predict_NEG)
#             if not is_already_sent_alert(df_mul):
#                 thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul,  Op_buy_sell.NEG, list_models_to_predict_NEG), name='SENDING_NEG')
#                 thr_sending.start()
#                 # send_image_full_alersts(S, df_mul,op_buy_sell = Op_buy_sell.NEG, list_models_to_predict = list_models_to_predict_NEG)
#
#             Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul[LIST_COLS_TO_CSV], path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)
#
#     print("START ciclo ")

def send_MULTI_W3_alert_and_register(S, df_mul):
    # list_models_to_predict_POS = [x for x in df_mul_r.columns if x.startswith("Acert_TFm_" + S + "_" + Op_buy_sell.POS.value)]
    # list_models_to_predict_NEG = [x for x in df_mul_r.columns if x.startswith("Acert_TFm_" + S + "_" + Op_buy_sell.NEG.value)]
    #
    # if (not list_models_to_predict_POS) and (not list_models_to_predict_NEG):
    #     Logger.logr.warning("There are no models of class columns_json in the list_good_params list, columns_json, optimal enough, we pass to the next one. Stock: " + S)
    #     return

    # df_mul = df_mul_r[['Date', 'buy_sell_point', 'Close', 'Volume'] + list_models_to_predict_POS + list_models_to_predict_NEG].iloc[-2:]
    # # df_mul[list_models_to_predict_POS] = df_mul[list_models_to_predict_POS].map(lambda x: x.replace("%", "").replace(" ", "0")).astype('int')
    # # df_mul[list_models_to_predict_NEG] = df_mul[list_models_to_predict_NEG].map(lambda x: x.replace("%", "").replace(" ", "0")).astype('int')
    # for c in list_models_to_predict_POS + list_models_to_predict_NEG:
    #     df_mul[c] = df_mul[c].str.replace("%", "").replace(" ", "0").astype('int')

    LIST_COLS_TO_CSV = ['Date',"ticker", 'buy_sell_point', 'Close', 'Volume', "POS_score","POS_num",  "NEG_score", "NEG_num","POS_score_list","NEG_score_list", "POS_models_name","NEG_moldel_name" ]
    SCORE_MIN_POS = 0.65
    SCORE_MIN_NEG = 0.65
    # df_mul = add_col_df_mul_to_csv_result_predictions(S, df_mul, list_models_to_predict_NEG, list_models_to_predict_POS)
    # df_mul["NEG_score"] = str(df_mul[list_models_to_predict_NEG].sum(axis=1)) + "/" + str(len(list_models_to_predict_NEG))
    # ['Date', 'buy_sell_point', 'Close', 'Volume', "POS_score", "NEG_score"]
    df_mul_last = df_mul[-2:]
    df_mul_last.insert(loc=0, column="ticker" , value=S)
    Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path = _KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME)
    #
    if df_mul_last["score_1"][-1] >= SCORE_MIN_POS  :
        # from XTB_api import xtb_api
        # thr_xtb = threading.Thread(target=xtb_api.xtb_operate_Lock_thread, args=(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS), name='XTB_POS')
        # thr_xtb.start()
        # xtb_api.xtb_operate_Lock_thread(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS)
        if not is_already_sent_alert(df_mul_last):
            thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul_last,  Op_buy_sell.POS, ['rem'] ), name='SENDING_POS')
            thr_sending.start()
            # send_image_full_alersts(S, df_mul, op_buy_sell=Op_buy_sell.POS,list_models_to_predict=list_models_to_predict_POS)

        Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)

    if df_mul_last["score_2"][-1] >= SCORE_MIN_NEG : # or 1 == 1 :
        # from XTB_api import xtb_api
        # thr_xtb = threading.Thread(target=xtb_api.xtb_operate_Lock_thread, args=(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS), name='XTB_POS')
        # thr_xtb.start()
        # xtb_api.xtb_operate_Lock_thread(S, df_mul[1:].to_dict('list'), _KEYS_DICT.Op_buy_sell.POS, list_models_to_predict_POS)
        if not is_already_sent_alert(df_mul_last):
            thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul_last,  Op_buy_sell.NEG, ['rem'] ), name='SENDING_NEG')
            thr_sending.start()
            # send_image_full_alersts(S, df_mul, op_buy_sell=Op_buy_sell.POS,list_models_to_predict=list_models_to_predict_POS)

        Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)

    print("START ciclo ")


def send_MULTI_F1_alert_and_register(S, df_mul, df_f1):

    LIST_COLS_TO_CSV = ['Date',"ticker", 'buy_sell_point', 'Close', 'Volume', "POS_score","POS_num",  "NEG_score", "NEG_num","POS_score_list","NEG_score_list", "POS_models_name","NEG_moldel_name" ]
    SCORE_MIN_POS = 0.74
    SCORE_MIN_NEG = 0.74
    # df_mul = add_col_df_mul_to_csv_result_predictions(S, df_mul, list_models_to_predict_NEG, list_models_to_predict_POS)
    # df_mul["NEG_score"] = str(df_mul[list_models_to_predict_NEG].sum(axis=1)) + "/" + str(len(list_models_to_predict_NEG))
    # ['Date', 'buy_sell_point', 'Close', 'Volume', "POS_score", "NEG_score"]
    df_mul_last = df_mul[-2:]
    df_mul_last.insert(loc=0, column="ticker" , value=S)
    df_mul_last[['f1_buy_sell_score','precision_1','precision_2','recall_1','recall_2','f1_score_1','f1_score_2', 'REF_key_model']] = df_f1[['f1_buy_sell_score','precision_1','precision_2','recall_1','recall_2','f1_score_1','f1_score_2','REF_key_model']].round(3).values[0]
    Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path = _KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME)
    #
    if  df_mul_last["score_1"][-1] >= SCORE_MIN_POS:#  or 1==1 :  #df_mul_last["predict"][-1] == 1  and
        if not is_already_sent_alert(df_mul_last):
            thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul_last,  Op_buy_sell.POS, ['foo'] ), name='SENDING_POS')
            thr_sending.start()
            # send_image_full_alersts(S, df_mul, op_buy_sell=Op_buy_sell.POS,list_models_to_predict=list_models_to_predict_POS)

        Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)

    if df_mul_last["score_2"][-1] >= SCORE_MIN_NEG :
        if not is_already_sent_alert(df_mul_last):
            thr_sending = threading.Thread(target=send_image_full_alersts, args=(S, df_mul_last,  Op_buy_sell.NEG, ['foo'] ), name='SENDING_NEG')
            thr_sending.start()
            # send_image_full_alersts(S, df_mul, op_buy_sell=Op_buy_sell.POS,list_models_to_predict=list_models_to_predict_POS)

        Utils_send_message.register_MULTI_in_zTelegram_Registers(S, df_mul_last.round(3), path=_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT)

    print("START ciclo ")

def send_image_full_alersts(S, df_mul, op_buy_sell : _KEYS_DICT.Op_buy_sell ,  list_models_to_predict :list):
    df_send = df_mul.copy()
    df_send['Date'] = df_send.index
    Logger.logr.info("It will send alert  Stock: " + S+ "_"+op_buy_sell.value)
    url_trader_view = Utils_send_message.get_traderview_url(S)
    path_imgs_tech, path_imgs_finan = get_traderview_screem_shot(url_trader_view,_KEYS_DICT.PATH_PNG_TRADER_VIEW + "" + S, will_stadistic_png = False)

    message_aler, alert_message_without_tags = Utils_send_message.get_MULTI_W3_string_alert_message(S, df_send[1:].to_dict(
        'list'), op_buy_sell, list_models_to_predict, list_png=[path_imgs_tech],url_trader_view=url_trader_view )

    send_mesage_all_people(message_aler, list_png=[path_imgs_tech ])

    #https://mothereff.in/twitalics type letters TWITTER FALLO
    from api_twitter import twi_ #"𝘾𝙤𝙣𝙛𝙞𝙙𝙚𝙣𝙘𝙚 𝙤𝙛 𝙢𝙤𝙙𝙚𝙡𝙨:""""📈 𝗕𝗨𝗬 📈    𝗦𝗘𝗟𝗟 𝘾𝙤𝙣𝙛𝙞𝙙𝙚𝙣𝙘𝙚 𝙤𝙛 𝙢𝙤𝙙𝙚𝙡𝙨:📊𝙉𝙖𝙢𝙚𝙨:"""
    tweet_text = alert_message_without_tags.replace(" BUY " , "𝗕𝗨𝗬").replace( " SELL "  , "𝗦𝗘𝗟𝗟").replace("Confidence of models:"  , "𝙈𝙤𝙙𝙚𝙡 𝙏𝙧𝙪𝙨𝙩:").replace("📊⚙Model names:"  , "📊⚙𝙉𝙖𝙢𝙚𝙨: #𝙩𝙧𝙖𝙙𝙚𝙧") #UNICODE:   𝙈𝙤𝙙𝙚𝙡 𝙣𝙖𝙢𝙚𝙨:
    tweet_text = tweet_text.replace("\t\t", ' ').replace("_mult_", '_mu_').replace(".00%", '.0%').replace("0%", '%').replace("f1:", '\t')[:280]# [:280+32] MAX 280 tweet limit 20 por cada url
    twi_.put_tweet_with_images(tweet_text   ,list_images_path = [path_imgs_tech])
    Logger.logr.info("It has sent alert  Stock: " + S + "_" + op_buy_sell.value)


def is_already_sent_alert(df_mul):
    if not os.path.exists(_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT):
        return False

    df_sent = pd.read_csv(_KEYS_DICT.PATH_REGISTER_RESULT_MULTI_REAL_TIME_SENT, index_col=0, sep='\t')
    #TODO check also pos neg
    date_alert_str = pd.to_datetime(str(df_mul.index[1:].values[0])).strftime("%Y-%m-%d %H:%M:%S")
    num_is_in_the_sent_list = df_sent[(df_sent['ticker'] == df_mul["ticker"][1:].values[0]) & (df_sent.index == date_alert_str)].shape[0]
    if num_is_in_the_sent_list > 0:
        return True
    else:
        return False


def add_col_df_mul_to_csv_result_predictions(S, df_mul, list_models_to_predict_NEG, list_models_to_predict_POS):
    df_mul.insert(loc=1, column="ticker", value=S)
    df_mul['ticker'] = df_mul['ticker'].map(lambda x: x.ljust(8 - len(x)))  # all columns with 6 char

    for L in ["NEG_moldel", "POS_models", "NEG_num", "NEG_score", "POS_num", "POS_score"]:
        df_mul.insert(loc=4, column=L, value=-1)

    df_mul["NEG_score"] = df_mul[list_models_to_predict_NEG].sum(axis=1)
    df_mul["POS_score"] = df_mul[list_models_to_predict_POS].sum(axis=1)
    df_mul["NEG_num"] = str(len(list_models_to_predict_NEG)) + ']'
    df_mul["POS_num"] = str(len(list_models_to_predict_POS)) + ']'
    df_mul["NEG_score_list"] = pd.DataFrame([r.astype(str).str.cat(sep="%") for i, r in df_mul[list_models_to_predict_NEG].iterrows()], index=df_mul.index )
    df_mul["POS_score_list"] = pd.DataFrame([r.astype(str).str.cat(sep="%") for i, r in df_mul[list_models_to_predict_POS].iterrows()], index=df_mul.index )
    df_mul["NEG_moldel_name"] = ", ".join(list_models_to_predict_NEG).replace("Acert_TFm_", '')
    df_mul["POS_models_name"] = ", ".join(list_models_to_predict_POS).replace("Acert_TFm_", '')

    return df_mul

# def monitor_all_stocks_and_send_alerts():
#     print("START ciclo ")
#     print("START ciclo " + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
#     print("START ciclo ")
#     for S in DICT_COMPANYS["@FOLO3"] : #list_stocks:  # list_stocks: # [ "UBER","U",  "TWLO", "TSLA", "SNOW", "SHOP", "PINS", "NIO", "MELI" ]:#list_stocks:
#         try:
#             print("Company: "+S)
#             df_compar, df_vender = Model_predictions_handle_Nrows.get_RealTime_buy_seel_points(S, Option_Historical.DAY_6, NUM_LAST_REGISTERS_PER_STOCK=3)
#             # df_compar = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
#             if df_compar is not None:
#                 send_alert(S, df_compar, Op_buy_sell.POS)
#
#             # df_vender = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
#             if df_vender is not None:
#                 send_alert(S, df_vender, Op_buy_sell.NEG)
#         except Exception as ex:
#             send_exception(ex)
#
#     df_res = pd.DataFrame([[datetime.today().strftime('%Y-%m-%d %H:%M:%S'), '----', '----','----','----', '----%', '----%', '----%', '----%', datetime.today().strftime('%Y-%m-%d %H:%M:%S')]], columns=['Date', 'Stock', 'buy_sell','Close', '88%', '93%', '95%', 'TF%', "Models_names"])
#     df_res.to_csv(_KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME, sep="\t", index=None, mode='a', header=False)
#     print("END ciclo ")
#     print("END ciclo "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
#     print("END ciclo ")




# from datetime import datetime
#Para ejecutar algo cada 15 minutos después de la hora.
# while 1:
#     dt = datetime.now() + timedelta(minutes=15)
#     dt = dt.replace(minute=0)
#
#     while datetime.now() < dt:
#         time.sleep(1)
#     diez_minutos = 10*60
#     monitor_all_stocks_and_send_alerts()
#     time.sleep(diez_minutos) #cada_minuto = (60.0 - ((time.time() - starttime) % 60.0))
#
# monitor_all_stocks_and_send_alerts()
# from apscheduler.schedulers.blocking import BlockingScheduler
# scheduler = BlockingScheduler()
# #https://stackoverflow.com/questions/66662408/how-can-i-run-task-every-10-minutes-on-the-5s-using-blockingscheduler
# scheduler.add_job(monitor_all_stocks_and_send_alerts, trigger='cron', minute='4,19,34,49', second=1) #Next wakeup is due at 2022-10-24 16:45:58+02:00 (in 827.089182 seconds)
# scheduler.start()










