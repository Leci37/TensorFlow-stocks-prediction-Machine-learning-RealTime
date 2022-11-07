import numpy as np
import pandas as pd
import requests
from datetime import datetime

#INTERVALO INTRA DIA 1min
#https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=d1&tickerIds=913323930
#INTERVALO INTRA DIA 5min
#https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=d5&tickerIds=913323930
import a_manage_stocks_dict
import re

def regex_id_weBull(response_text):
    try:
        id_find = re.search(r'\"tickerId\":(.*?),\"exchangeId\"', response_text).group(1)
        return id_find
    except:
        # print("Error in regex_id_weBull decompress")
        return None


WEBULL_ID = 913323930 #MELI


CSV_NAME = "@CHIC"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
# Para sacar los DICT_WEBULL_ID = { nuevos

for S in list_stocks:
    id_find = "None"
    #S = 'MELI'
    URL_1 = "https://www.webull.com/quote/nyse-"+ str(S).lower()
    #URL_1 = "https://www.webull.com/quote/nasdaq-"+ str(S).lower()
    #print(URL_1)
    response = requests.get(URL_1 )
    id_find = regex_id_weBull(response.text)

    if id_find == None:
        URL_2 = "https://www.webull.com/quote/nasdaq-" + str(S).lower()
        #print(URL_2)
        response = requests.get(URL_2)
        id_find = regex_id_weBull(response.text)
        if id_find == None:
            print("FAIL status code "+S)
    print("\"" +S +"\" : "+id_find +",")


print("END")