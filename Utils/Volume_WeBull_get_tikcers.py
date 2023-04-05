import requests


#INTERVALO INTRA DIA 1min
#https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=d1&tickerIds=913323930
#INTERVALO INTRA DIA 5min
#https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=d5&tickerIds=913323930
import _KEYS_DICT
import re

def regex_id_weBull(response_text):
    try:
        id_find = re.search(r'\"tickerId\":(.*?),\"exchangeId\"', response_text).group(1)
        return id_find
    except:
        # print("Error in regex_id_weBull decompress")
        return None
def regex_id_weBull_cripto(response_text):
    try:
        id_find = re.search(r'\"tickerId\":(.*?),\"tickerRtInfo\"', response_text).group(1)
        return id_find
    except:
        # print("Error in regex_id_weBull decompress")
        return None

print("The result of this .py , are the webull.com IDs for each stock, to get the volume and price reading in real time. place in _KEYS_DICT.py , to make it work yhoo_POOL_enque_Thread.py")
# Example result:
# "ATHE" : 913323301,
# "MU" : 913324077,
# "CRM" : 913255140,

CSV_NAME = "@CHIC"
list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks_cripto = ['BNBUSD', 'XRPUSD', 'ADAUSD', 'HEXUSD', 'SOLUSD', 'AVAXUSD', 'LUNA1USD', 'DOGEUSD', 'DOTUSD', 'DOTUSD', 'SHIBUSD', 'MATICUSD']
list_stocks_cripto = ["ETCUSD", "APIUSD", "CROUSD", "SKLUSD", "IMXUSD", "ADAUSD", "MIOTAUSD", "XLMUSD", "EOSUSD", "NEOUSD", "TRXUSD", "ZECUSD", "BNBUSD", "XTZUSD", "DOTUSD", "MKRUSD", "COMPUSD", "LINKUSD", "UNIUSD", "YFIUSD", "DOGEUSD", "AAVEUSD", "FILUSD", "ALGOUSD", "ATOMUSD", "MANAUSD", "APEUSD", "LRCUSD", "ENJUSD", "BICOUSD", "BATUSD", "BNTUSD", "OGNUSD", "MATICUSD", "FLRUSD", "GALAUSD", "ALICEUSD", "CHZUSD", "HBARUSD", "DYDXUSD", "SOLUSD", "THETAUSD", "FTMUSD", "GRTUSD"]
#['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'HEX-USD', 'SOL-USD', 'AVAX-USD', 'LUNA1-USD', 'DOGE-USD', 'DOT-USD', 'DOT-USD', 'SHIB-USD', 'MATIC-USD']
# CSV_NAME = "@CHIC"
# list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# To get the DICT_WEBULL_ID = { new

for S in list_stocks_cripto:
    id_find = "None"
    # URL_1 = "https://www.webull.com/quote/nyse-"+ str(S).lower()
    #
    # response = requests.get(URL_1 )
    # id_find = regex_id_weBull(response.text)
    #
    # if id_find == None:
    #     URL_2 = "https://www.webull.com/quote/nasdaq-" + str(S).lower()
    #
    #
    #     response = requests.get(URL_2)
    #     id_find = regex_id_weBull(response.text)
    #     if id_find == None:
    URL_3_cripto = "https://www.webull.com/quote/bitfinex-" + str(S).lower()
    # print(URL_3_cripto)
    response = requests.get(URL_3_cripto)
    id_find = regex_id_weBull_cripto(response.text)


    if id_find == None:
        URL_4_cripto ="https://www.webull.com/quote/ccc-" + str(S).lower()
        response = requests.get(URL_3_cripto)
        id_find = regex_id_weBull_cripto(response.text)
        if id_find == None:
            URL_4_cripto = "https://www.webull.com/ticker/ccc-" + str(S).lower()
            response = requests.get(URL_3_cripto)
            id_find = regex_id_weBull_cripto(response.text)
            if id_find == None:
                print("FAIL status code " + S)
    if id_find != None:
        print("\"" +S +"\" : "+id_find +",")

print("USE IT RESULT IN _KEYS_DICT.py DICT_WEBULL_ID")
print("The result of this .py , are the webull.com IDs for each stock, to get the volume and price reading in real time. place in _KEYS_DICT.py , to make it work yhoo_POOL_enque_Thread.py")
print("Example of get data real time using webull.com web : \t https://quotes-gw.webullfintech.com/api/quote/charts/queryMinutes?period=d5&tickerIds=913256192" )
print("END")