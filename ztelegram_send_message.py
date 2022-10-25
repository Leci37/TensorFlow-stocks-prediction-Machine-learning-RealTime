# Done! Congratulations on your new bot. You will find it at t.me/Whale_Hunter_Alertbot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.
#
# Use this token to access the HTTP API:
# 5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY
# Keep your token secure and store it safely, it can be used by anyone to control your bot.
#
# For a description of the Bot API, see this page: https://core.telegram.org/bots/api

#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2
import requests
import asyncio
import pandas as pd
import json
from datetime import datetime

from telegram.constants import ParseMode

import UtilsL
import a_manage_stocks_dict
import yhoo_history_stock
from LogRoot.Logging import Logger
from Model_predictions_Nrows import get_RealTime_buy_seel_points
import Utils_send_message
from a_manage_stocks_dict import Option_Historical, DICT_COMPANYS , Op_buy_sell


from telegram import Bot , constants

TOKEN = "5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"


chat_idLUISL = "5058733760"# LUIS
chat_idIVAN = "563544091" #Ivan
chat_idARIZT = "2073387965"
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
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]



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

SUFFIXES_DATE_ENDS = ("00:00", "15:00", "30:00", "45:00")
def will_send_alert(df_b_s):
    num_models = df_b_s.iloc[[-1]]['num_models'].values[0]
    has_to_send_last = df_b_s.iloc[[-1]]['sum_r_93'].values[0]  >= ((num_models / 2) + 0.5) #ultima
    has_to_send_unlast = df_b_s.iloc[[-2]]['sum_r_93'].values[0] >= ((num_models / 2) + 0.5) #penultima
    #has_to_send_oclck =  df_b_s.iloc[[i]]['Date'].values[0].endswith(SUFFIXES_DATE_ENDS)

    return has_to_send_last and has_to_send_unlast
    # df_b_s.iloc[[i]]['have_to_oper'].values[0] == True or df_b_s.iloc[[i]]['sum_r_TF'].values[0] > 4 or \
    #        df_b_s.iloc[[i]]['sum_r_88'].values[0] >= 12

COL_GANAN = ["Date", "stock", "type_buy_sell","value_start" ]
df_registre = pd.DataFrame(columns=COL_GANAN)

def send_alert(S, df_b_s, type_b_s):
    global df_registre
    modles_evaluated = [col for col in df_b_s.columns if col.startswith('br_')]
    modles_evaluated_TF = [col for col in df_b_s.columns if col.startswith('br_TF')]
    dict_predictions = df_b_s.T.to_dict()
    df_b_s = df_b_s.drop(columns=modles_evaluated)

    #for i in [-2, -1]: #ultima y penultima
    i = -1
    if True or will_send_alert(df_b_s):

        date_detect, value_detect = __get_runtime_value_date_by_yhoo(S)
        dict_predict_1 = dict_predictions[list(dict_predictions.keys())[i]]
        df_registre = pd.concat([df_registre, pd.DataFrame([[date_detect, S, type_b_s.name, "{:.2f}".format(value_detect) ]], columns=COL_GANAN)  ],ignore_index=True)  # añadir fila add row
        # df_registre.to_csv("zTelegram_Registers.csv", sep="\t", index=None , mode='a', header=False)

        message_aler = Utils_send_message.get_string_alert_message(S, dict_predict_1, modles_evaluated, modles_evaluated_TF, type_b_s, date_detect, value_detect)

        botsUrl = f"https://api.telegram.org/bot{TOKEN}" #+ "/sendMessage?chat_id={}&text={}".format(chat_idLUISL, message_aler, parse_mode=ParseMode.HTML)
        # url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(chat_idLUISL, message_aler,parse_mode=ParseMode.MARKDOWN_V2)
        url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(chat_idLUISL, message_aler,parse_mode=ParseMode.HTML)
        #url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_idLUISL}&text={message_aler}"
        print("Enviar alerta: "+message_aler)
        print(requests.get(url).json())
        url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(chat_idARIZT, message_aler, parse_mode=ParseMode.HTML)
        print(requests.get(url).json())


def monitor_all_stocks_and_send_alerts():
    print("START ciclo ")
    print("START ciclo " + datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print("START ciclo ")
    for S in DICT_COMPANYS["@FOLO3"] : #list_stocks:  # list_stocks: # [ "UBER","U",  "TWLO", "TSLA", "SNOW", "SHOP", "PINS", "NIO", "MELI" ]:#list_stocks:
        try:
            print("Company: "+S)
            df_compar, df_vender = get_RealTime_buy_seel_points(S, Option_Historical.DAY_6, NUM_LAST_REGISTERS_PER_STOCK=6)
            # df_compar = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
            if df_compar is not None:
                send_alert(S, df_compar, Op_buy_sell.POS)

            # df_vender = pd.read_csv("Models/LiveTime_results/" + S + "_" + type_buy_sell.value + "_" + "_.csv",index_col=False, sep='\t')
            if df_vender is not None:
                send_alert(S, df_vender, Op_buy_sell.NEG)
        except Exception as ex:
            message_aler = "** Exception **"+str(ex)
            print(message_aler)
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_idLUISL}&text={message_aler}"
            print("Enviar alerta: "+message_aler)
            print(requests.get(url).json())


    print("END ciclo ")
    print("END ciclo "+ datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print("END ciclo ")


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

import time
# starttime = time.time()
# while True:
#     print("tick")

from datetime import datetime, timedelta
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

monitor_all_stocks_and_send_alerts()
from apscheduler.schedulers.blocking import BlockingScheduler
scheduler = BlockingScheduler()
#https://stackoverflow.com/questions/66662408/how-can-i-run-task-every-10-minutes-on-the-5s-using-blockingscheduler
scheduler.add_job(monitor_all_stocks_and_send_alerts, trigger='cron', minute='4,19,34,49', second=1) #Next wakeup is due at 2022-10-24 16:45:58+02:00 (in 827.089182 seconds)
scheduler.start()










