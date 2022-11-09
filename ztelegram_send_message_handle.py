# Done! Congratulations on your new bot. You will find it at t.me/Whale_Hunter_Alertbot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

# Use this token to access the HTTP API:
# Keep your token secure and store it safely, it can be used by anyone to control your bot.
# For a description of the Bot API, see this page: https://core.telegram.org/bots/api

#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2

#Get from telegram
#See tutorial https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token
import requests
import pandas as pd
from datetime import datetime

from telegram.constants import ParseMode

TOKEN = "00000000xxxxxxx"
URL_TELE = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
#**DOCU**
# 5.2 Configure chatID and tokes in Telegram
# Once the token has been obtained, the chatId of the users and the administrator must be obtained.
# The users only receive purchase and startup alerts, while the administrator receives the alerts of the users as well as possible problems.
# To get the chatId of each user run ztelegram_send_message_UptateUser.py and then write any message to the bot, the chadID appears both in the execution console and to the user.
# [>>> BOT] Message Send on 2022-11-08 22:30:31
# 	Text: You "User nickname " send me:
# "Hello world""
#  ChatId: "5058733760"
# 	From: Bot name
# 	Message ID: 915
# 	CHAT ID: 500000760
# -----------------------------------------------
# Pick up CHAT ID: 500000760
# With the chatId of the desired users, add them to the list LIST_PEOPLE_IDS_CHAT

chat_idADMIN = "5050000000"
chat_idUser1 = "563000000"
chat_idUser2 = "207000000"
chat_idUser3= "495000000"
LIST_PEOPLE_IDS_CHAT = [chat_idUser1, chat_idUser2, chat_idUser2]


if TOKEN == "00000000xxxxxxx":
    raise ValueError("There is no value for the telegram TOKEN, telegram is required to telegram one, see tutorial: https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token")



def send_exception(ex, extra_mes = ""):
    message_aler = extra_mes + "   ** Exception **" + str(ex)
    print(message_aler)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_idADMIN}&text={message_aler}"
    print("Enviar alerta: " + message_aler)
    print(requests.get(url).json())


def send_mesage_all_people(message_aler, parse_type = ParseMode.HTML):
    botsUrl = f"https://api.telegram.org/bot{TOKEN}"  # + "/sendMessage?chat_id={}&text={}".format(chat_idLUISL, message_aler, parse_mode=ParseMode.HTML)
    # url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(chat_idLUISL, message_aler,parse_mode=ParseMode.MARKDOWN_V2)
    print("Enviar alerta todo el mundo: " + message_aler)
    for people_id in LIST_PEOPLE_IDS_CHAT:
        url = botsUrl + "/sendMessage?chat_id={}&text={}&parse_mode={parse_mode}".format(people_id, message_aler,parse_mode=parse_type)
        print(requests.get(url).json())