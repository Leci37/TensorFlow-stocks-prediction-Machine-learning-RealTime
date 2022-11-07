#https://www.alphavantage.co/documentation/ Intraday (Extended History)
import csv
import io
from time import sleep
from random import randint

import requests
import pandas as pd

# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MELI&interval=15min&outputsize=full&apikey=BCOVDP6GILNXKZG9
# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year20&adjusted=false&apikey=demo
import a_manage_stocks_dict

API_ALPHA_KEY = "BCOVDP6GILNXKZG9"
API_LIST_ALPHA_KEY_2 = ["FXZ0","FXZ1","FXZ2","FXZ3","FXZ4","FXZ5","FXZ6","FXZ7","FXZ8","FXZ9","BCOVDP6GILNXKZG9", "NCLXRBHC77ABT2R7", "D29SLGXA2MSSL0EJ" , "2Q6DAQVVEHSDW9D2" , "YSZM7M3FA8EV632Q" , "T0BOX3F5S5BYFH8O", "PAQXEWTKRDJK4UGT", "713TN3DP9AXEC8QQ", "5DCMSRWZVJROU690", "4IYYMEPKRJSO7PCK", "0YVW8B4STUVL8ENF"]
ALL_TIME_OPTIONS = "year1month1", "year1month2", "year1month3", "year1month4", "year1month5", "year1month6", "year1month7", "year1month8", "year1month9",\
                   "year1month10", "year1month11", "year1month12", "year2month1", "year2month2", "year2month3", "year2month4", "year2month5", "year2month6", \
                   "year2month7", "year2month8", "year2month9", "year2month10", "year2month11", "year2month12"

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=demo'
S = "MELI"
intelval = "15min"
#time_opcion = "year2month12"

CSV_NAME = "@CHIC"
list_companys_FVOLA = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]

#El resto de una división se calcula en Python con el operador %
count_int_try = 1


def get_api_key():
    global count_int_try
    resto = count_int_try % (len(API_LIST_ALPHA_KEY_2)) -1
    if resto == 0 and count_int_try != 1:
        print("Excedidas las llamadas esperar 5min CountID: "+str(count_int_try))
        sleep(60)
        key = API_LIST_ALPHA_KEY_2[resto]
        count_int_try += 1
    else:
        key = API_LIST_ALPHA_KEY_2[resto]
    # print("obtenida key: "+key)
    return key


def do_request():
    global count_int_try
    sleep(randint(2, 7))
    api_key = get_api_key()
    CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=' + S + '&interval=' + intelval + '&slice=' + time_op + '&apikey=' + api_key
    print(S + " ==== " + CSV_URL)
    raw_response = requests.get(CSV_URL)
    return raw_response


def do_request_conunt_adder():
    global count_int_try
    count_int_try += 1
    print("[1] Excedidas las llamadas , cambiar API_KEY Nueva: " )
    raw_response_2 = do_request()
    if "for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day" in raw_response_2.text:
        print("[2] Excedidas las llamadas , cambiar API_KEY Nueva: " )
        raw_response_2 = do_request_conunt_adder()
    return raw_response_2


for S in ["ABNB"] + list_companys_FVOLA:
    df_S_all = pd.DataFrame()
    for time_op in ALL_TIME_OPTIONS:
        raw_response = do_request()

        if "for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day" in raw_response.text:
            raw_response = do_request_conunt_adder()
            # if "for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day" in raw_response.text:
            #     count_int_try += 1
            #     print("[2] Excedidas las llamadas , cambiar API_KEY Nueva: " + get_api_key())
            #     raw_response = do_request()
        df_a_time = pd.read_csv(io.StringIO(raw_response.text), index_col=False, sep=',')
        if df_a_time.shape[0] == 0:
            print("Continue break")
            break
        #"Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency."

        print(S + " df: " + str(df_a_time.shape)  )
        df_S_all = pd.concat([df_S_all, df_a_time], ignore_index=True)

    if df_a_time.shape[0] == 0:
        print("Continue break")
        continue
    df_S_all = df_S_all.rename(columns={'time': 'Date', 'open': 'Open','high': 'High', 'close': 'Close','low': 'Low', 'volume': 'Volume'})
    df_S_all = df_S_all.sort_values(['Date'], ascending=True)
    df_S_all = df_S_all.drop_duplicates(subset=['Date'],keep="first")
    df_S_all = df_S_all.dropna(how='any')
    df_S_all.reset_index(drop=True, inplace=True)

    df_S_all['Date'] = df_S_all['Date'].astype(str)
    max_recent_date = df_S_all['Date'].max()[:10].replace('-','') # pd.to_datetime().strftime("%Y%m%d")
    min_recent_date = df_S_all['Date'].min()[:10].replace('-','') #pd.to_datetime( ).strftime("%Y%m%d")
    print("../d_price/RAW/alpha_"+S+'_'+intelval+"_"+ max_recent_date + "__" + min_recent_date + ".csv")
    df_S_all.to_csv("../d_price/RAW_alpha/alpha_"+S+'_'+intelval+"_"+ max_recent_date + "__" + min_recent_date + ".csv", sep="\t", index=None)

# df_S_raw = pd.DataFrame([x.split(',') for x in ],
#                         columns=['Date', 'Open', 'Close', 'High', 'Low', 'Close_avg', 'Volume', "Adj"])

