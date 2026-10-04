# Done! Congratulations on your new bot. You will find it at t.me/Whale_Hunter_Alertbot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.
#
# Use this token to access the HTTP API:
# Keep your token secure and store it safely, it can be used by anyone to control your bot.
#
# For a description of the Bot API, see this page: https://core.telegram.org/bots/api

#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2
import requests
from ztelegram_send_message_handle import TOKEN
import urllib.request as request
from urllib.error import HTTPError
from http.client import HTTPResponse
from typing import Dict, List, Union
import json
from datetime import datetime
import signal
import os

signal.signal(signal.SIGINT, signal.SIG_DFL)

# https://pythonprogramming.org/making-a-telegram-bot-using-python/
class TelegramEcho:
    def __init__(self, TG_KEY: str):
        self.TG_URL = "https://api.telegram.org/bot{key}/{method}"
        self.TG_KEY = TG_KEY

        self.__last = None
        self.__last_time = None
        pass

    def run(self):
        """
        method to handle the incoming message and the send echo message to the user
        """
        while True:
            try:
                # getting the incoming data
                incoming = self.__handle_incoming()

                # checking if incoming message_id is same as of last, then skip
                if self.__last == incoming["message"]["message_id"]:
                    continue
                else:
                    self.__last = incoming["message"]["message_id"]

                # adding more validation to prevent messaging the last message whenever the polling starts
                if not self.__last_time:
                    self.__last_time = incoming["message"]["date"]
                    continue
                elif self.__last_time < incoming["message"]["date"]:
                    self.__last_time = incoming["message"]["date"]
                else:
                    continue

                # finally printing the incoming message
                self.__print_incoming(incoming)

                # now sending the echo message
                user_id = incoming["message"]["chat"]["id"]
                user_name = incoming["message"]["from"].get("first_name", "") + " " +incoming["message"]["from"].get("last_name", "")
                message = "You \"" +user_name +"\" send me: \n\""+ incoming["message"]["text"] + "\""+"\"\n ChatId: \""+ str(user_id) + "\""
                outgoing = self.__handle_outgoing(
                    user_id ,
                    message  )

                # finally printing the outgoing message
                self.__print_outgoing(outgoing)

                pass
            except (HTTPError, IndexError):
                continue
            pass
        pass

    def __handle_incoming(self) -> Dict[str, Union[int, str]]:
        """
        method fetch the recent messages
        """

        # getting all messages
        getUpdates = request.urlopen(
            self.TG_URL.format(key=self.TG_KEY, method="getUpdates"))

        # parsing results
        results: List[Dict[str, Union[int, str]]] = json.loads(
            getUpdates.read().decode())["result"]

        # getting the last error
        return results[-1]

    def __print_incoming(self, incoming: Dict[str, Union[int, str]]):
        """
        method to print the incoming message on console
        """
        print("[<<< CLIENT] Message Recieved on %s" % datetime.fromtimestamp(
            incoming["message"]["date"]).strftime("%Y-%m-%d %H:%M:%S"))
        print("\tText: %s" % incoming["message"]["text"])
        print("\tFrom: %s" %
              incoming["message"]["from"].get("first_name", "") + " " +
              incoming["message"]["from"].get("last_name", ""))
        print("\tMessage ID: %d" % incoming["message"]["message_id"])
        print("-" *  70) #os.get_terminal_size().columns)
        pass

    def __handle_outgoing(self, chat_id: int,
                          message_txt: str) -> Dict[str, Union[int, str]]:
        """
        method to send the echo message to the same chat
        """

        # making the post data
        _data: Dict[str, Union[int, str]] = {
            "chat_id":
            chat_id,
            "text":
            message_txt#"You sent me \"{MESSAGE_TEXT}\"".format(MESSAGE_TEXT=message_txt)
        }

        # creating the request
        _request: request.Request = request.Request(
            self.TG_URL.format(key=self.TG_KEY, method="sendMessage"),
            data=json.dumps(_data).encode('utf8'),
            headers={"Content-Type": "application/json"})

        # sending HTTP request, for sending message to the user
        sendMessage: HTTPResponse = request.urlopen(_request)
        result: Dict[str, Union[int, str]] = json.loads(
            sendMessage.read().decode())["result"]
        return result

    def __print_outgoing(self, outgoing):
        """
        method to print outgoing data on the console
        """
        print("[>>> BOT] Message Send on %s" % datetime.fromtimestamp(
            outgoing["date"]).strftime("%Y-%m-%d %H:%M:%S"))
        print("\tText: %s" % outgoing["text"])
        print("\tFrom: %s" % outgoing["from"]["first_name"])
        print("\tMessage ID: %d" % outgoing["message_id"])
        print("\tCHAT ID: %d" % outgoing['chat']['id'])
        print("-" *  70) #os.get_terminal_size().columns)
        pass

    pass

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

if __name__ == "__main__":
    tg = TelegramEcho(TOKEN)
    tg.run()