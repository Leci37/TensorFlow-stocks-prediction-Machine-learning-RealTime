# Done! Congratulations on your new bot. You will find it at t.me/Whale_Hunter_Alertbot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.
#
# Use this token to access the HTTP API:
# 5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY
# Keep your token secure and store it safely, it can be used by anyone to control your bot.
#
# For a description of the Bot API, see this page: https://core.telegram.org/bots/api

#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2
import requests
import pandas as pd
from datetime import datetime

from telegram.constants import ParseMode

import yhoo_history_stock
from LogRoot.Logging import Logger
from Model_predictions_Nrows import get_RealTime_buy_seel_points
from Utils import Utils_send_message
from a_manage_stocks_dict import Option_Historical, DICT_COMPANYS , Op_buy_sell

TOKEN = "5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"


chat_idLUISL = "5058733760" # LUIS
chat_idIVAN = "563544091"   #Ivan
chat_idARIZT = "2073387965" # tania dosil
chat_idTANIA= "495451714"   # tania dosil
LIST_PEOPLE_IDS_CHAT = [chat_idLUISL, chat_idARIZT, chat_idTANIA ]

message = "hello from your telegram bot"



feedback = requests.get(url).json()
for dict_userId in  feedback['result']:
    print(dict_userId)



print(feedback)
# url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
# print(requests.get(url).json())  # this sends the message
all_cols = ['Date' 'buy_sell_point' 'Close' 'has_preMarket' 'Volume' 'sum_r_88', 'sum_r_93' 'have_to_oper' 'sum_r_TF' 'have_to_oper_TF' 'num_models']
prety_cols = ['Date' , 'sum_r_88', 'sum_r_93', 'have_to_oper', 'sum_r_TF', 'have_to_oper_TF', 'num_models']
BOT = None


CSV_NAME = "@VOLA"
list_stocks = DICT_COMPANYS[CSV_NAME]

NUM_MIN_MODLES  = 3
NUM_MIN_MODLES_TF = 1

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

def send_alert(S, df_b_s, type_b_s):
    global df_registre
    modles_evaluated = [col for col in df_b_s.columns if col.startswith('br_')]
    modles_evaluated_TF = [col for col in df_b_s.columns if col.startswith('br_TF')]
    dict_predictions = df_b_s.T.to_dict()

    #register data each time UNLAST
    Utils_send_message.register_in_zTelegram_Registers(S, dict_predictions[list(dict_predictions.keys())[-2]], modles_evaluated, type_b_s, path ="zTelegram_Registers.csv")  #unlast row
    #LAST
    Utils_send_message.register_in_zTelegram_Registers(S, dict_predictions[list(dict_predictions.keys())[-1]], modles_evaluated, type_b_s, path ="zTelegram_Registers.csv")  #last row

    i = -1
    if will_send_alert(df_b_s, S+"_"+type_b_s.name):
        dict_predict_last = dict_predictions[list(dict_predictions.keys())[i]]

        date_detect, value_detect = __get_runtime_value_date_by_yhoo(S)

        message_aler , alert_message_without_tags = Utils_send_message.get_string_alert_message(S, dict_predict_last, modles_evaluated,
                                                                                                type_b_s, date_detect, value_detect)

        df_registre = pd.concat([df_registre, pd.DataFrame([[date_detect, S, type_b_s.name, "{:.2f}".format(value_detect) , alert_message_without_tags ]], columns=COL_GANAN)  ],ignore_index=True)  # añadir fila add row
        df_registre.to_csv("zSent_Telegram_Registers.csv", sep="\t", index=None , mode='a', header=False)

        send_mesage_all_people(message_aler)


def send_mesage_all_people(message_aler, parse_type = ParseMode.HTML):
    botsUrl = f"https://api.telegram.org/bot{TOKEN}"  # + "/sendMessage?chat_id={}&text={}".format(chat_idLUISL, message_aler, parse_mode=ParseMode.HTML)
    # url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(chat_idLUISL, message_aler,parse_mode=ParseMode.MARKDOWN_V2)
    print("Enviar alerta todo el mundo: " + message_aler)
    for people_id in LIST_PEOPLE_IDS_CHAT:
        url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(people_id, message_aler,parse_mode=parse_type)
        print(requests.get(url).json())

def monitor_all_stocks_and_send_alerts():
    print("START ciclo ")
    print("START ciclo " + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print("START ciclo ")
    for S in DICT_COMPANYS["@FOLO3"] : #list_stocks:  # list_stocks: # [ "UBER","U",  "TWLO", "TSLA", "SNOW", "SHOP", "PINS", "NIO", "MELI" ]:#list_stocks:
        try:
            print("Company: "+S)
            df_compar, df_vender = get_RealTime_buy_seel_points(S, Option_Historical.DAY_6, NUM_LAST_REGISTERS_PER_STOCK=3)
            # df_compar = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
            if df_compar is not None:
                send_alert(S, df_compar, Op_buy_sell.POS)

            # df_vender = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
            if df_vender is not None:
                send_alert(S, df_vender, Op_buy_sell.NEG)
        except Exception as ex:
            send_exception(ex)

    df_res = pd.DataFrame([[datetime.today().strftime('%Y-%m-%d %H:%M:%S'), '----', '----','----','----', '----%', '----%', '----%', '----%', datetime.today().strftime('%Y-%m-%d %H:%M:%S')]], columns=['Date', 'Stock', 'buy_sell','Close', '88%', '93%', '95%', 'TF%', "Models_names"])
    df_res.to_csv( "zTelegram_Registers.csv", sep="\t", index=None, mode='a', header=False)
    print("END ciclo ")
    print("END ciclo "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print("END ciclo ")


def send_exception(ex, extra_mes = ""):
    message_aler = extra_mes + "   ** Exception **" + str(ex)
    print(message_aler)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_idLUISL}&text={message_aler}"
    print("Enviar alerta: " + message_aler)
    print(requests.get(url).json())


list_most_probable = ["UPST", "MELI", "TWLO", "RIVN", "SNOW", "LYFT", "UBER","QCOM", "PYPL",  "RUN", "GTLB", #=['EPAM'] #
 "MDB", "NVDA",  "ADSK", "BABA",    "ABNB", "TSLA", "META",
 "PTON",  "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "AFRM", "DOCN",
 "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]


#https://github.com/python-telegram-bot/python-telegram-bot/wiki/Introduction-to-the-API

# async def main():
#     BOT = telegram.Bot(TOKEN)
#     async with BOT:
#         print(await BOT.get_me())



# if __name__ == '__main__':

print("START ciclo ")
print("START ciclo "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
print("START ciclo ")

# starttime = time.time()
# while True:
#     print("tick")

from datetime import datetime
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










