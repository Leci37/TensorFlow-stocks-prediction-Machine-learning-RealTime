# Done! Congratulations on your new bot. You will find it at t.me/Whale_Hunter_Alertbot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.
#
# Use this token to access the HTTP API:
# 5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY
# Keep your token secure and store it safely, it can be used by anyone to control your bot.
#
# For a description of the Bot API, see this page: https://core.telegram.org/bots/api

#https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2
import requests


TOKEN = "5452553430:AAH8ARcZlQZFZHxckJuY0eWnUK08IKo6QnY"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"


feedback = requests.get(url).json()


for dict_userId in  feedback['result']:
    print(dict_userId)
print(feedback)