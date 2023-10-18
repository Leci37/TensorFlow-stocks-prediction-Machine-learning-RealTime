"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user
@Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty,all rights reserved """
#https://www.alphavantage.co/documentation/ Intraday (Extended History)
import csv
from datetime import datetime
import io
from time import sleep
from random import randint

import requests
import pandas as pd

# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MELI&interval=15min&outputsize=full&apikey=BCOVDP6GILNXKZG9
# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year20&adjusted=false&apikey=demo
import _KEYS_DICT


API_LIST_ALPHA_FREE_KEYS = ["FXZ0", "FXZ1", "FXZ2", "FXZ3", "FXZ4", "FXZ5", "FXZ6", "FXZ7", "FXZ8", "FXZ9", "BCOVDP6GILNXKZG9", "NCLXRBHC77ABT2R7", "D29SLGXA2MSSL0EJ" , "2Q6DAQVVEHSDW9D2" , "YSZM7M3FA8EV632Q" , "T0BOX3F5S5BYFH8O", "PAQXEWTKRDJK4UGT", "713TN3DP9AXEC8QQ", "5DCMSRWZVJROU690", "4IYYMEPKRJSO7PCK", "0YVW8B4STUVL8ENF"]

ALL_TIME_OPTIONS = []


def get_historical_month_to_extract_realtime_date():
    CurrentMonth = datetime.now().month
    CurrentYear = datetime.now().year
    # PRESONALIZE the historical data date thresholds here
    for year in range(CurrentYear - 1, CurrentYear + 1):  # UPTATE
        if CurrentYear == year:
            end_month_count = CurrentMonth
        else:
            end_month_count = 12
        for month in range(1, end_month_count + 1):
            ALL_TIME_OPTIONS.append(str(year) + "-" + "{:02d}".format(month))  # FORMAT 2019-01
    return ALL_TIME_OPTIONS


ALL_TIME_OPTIONS = get_historical_month_to_extract_realtime_date()

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=demo'
intelval = "15min"
#time_opcion = "year2month12"

CSV_NAME = "@CHILL"
list_companys = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]

#The remainder of a division is calculated in Python with the % operator
count_int_try = 1


def get_api_key():
    global count_int_try
    resto = count_int_try % (len(API_LIST_ALPHA_FREE_KEYS)) - 1
    if resto == 0 and count_int_try != 1:
        print("Exceeded calls wait for 5min CountID: "+str(count_int_try))
        sleep(60)
        key = API_LIST_ALPHA_FREE_KEYS[resto]
        count_int_try += 1
    else:
        key = API_LIST_ALPHA_FREE_KEYS[resto]
    # print("obtenida key: "+key)
    return key


def do_request(time_op):
    global count_int_try
    sleep(randint(2, 7))
    api_key = get_api_key()
    #OLD system https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-TensorFlow/issues/21  CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=' + S + '&interval=' + intelval + '&slice=' + time_op + '&apikey=' + api_key
    CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + S + '&interval=' + intelval + '&month=' + time_op + '&outputsize=full&apikey=' + api_key
    print(S + " ==== " + CSV_URL)
    raw_response = requests.get(CSV_URL)
    return raw_response


def do_request_conunt_adder(time_op):
    global count_int_try
    count_int_try += 1
    api_key = get_api_key()
    print("[1] Exceeded calls, change API_KEY New: \t", api_key )
    raw_response_2 = do_request(time_op)
    if "Our standard API call frequency is 5 calls per minute and 100 calls per day" in raw_response_2.text or "You have reached the 100 requests/day limit"  in raw_response.text or "https://www.alphavantage.co/premium/"   in raw_response.text :
        sleep(20)
        print("[2] Exceeded calls, change API_KEY New: \t", api_key )
        raw_response_2 = do_request_conunt_adder(time_op)
    return raw_response_2

#**DOCU**
#1.0 Data collection
# 1.0 (Recommended) The yfinance API, if you want price to price intervals in 15min intervals is limited to 2 months, to get more time data up to 2 years back (more data better predictive models) use the free version of the API https://www.alphavantage.co/documentation/
#
# Run Utils/API_alphavantage_get_old_history.py
# The class is customizable: action intervals, months to ask and action ID.
# Note: being the free version, there is a portrait between request and request, to get a single 2 years history it takes 2-3 minutes per action.
# Once executed, the d_price/RAW_alpha folder will be filled with OHLCV historical .csv files of stock prices. These files will be read in the next step. Example name: alpha_GOOG_15min_20221031__20201112.csv
print("Get the history of the stock 2 years back through the use of free alphavantage.co API keys")
for S in list_companys:
    df_S_all = pd.DataFrame()
    for time_op in ALL_TIME_OPTIONS:
        print("Month: "+ time_op)
        raw_response = do_request(time_op)

        if "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 100 calls per day" in raw_response.text or "You have reached the 100 requests/day limit"  in raw_response.text or "https://www.alphavantage.co/premium/"   in raw_response.text :
            raw_response = do_request_conunt_adder(time_op)
            # if "for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day" in raw_response.text:
            #     count_int_try += 1
            #     print("[2] Excedidas las llamadas , cambiar API_KEY Nueva: " + get_api_key())
            #     raw_response = do_request()

        dict_response = dict(eval( raw_response.text) )
        dict_response_time_series = dict_response[list(dict_response.keys())[1]]  # 'Time Series (15min)'
        # df_a_time = pd.read_csv(io.StringIO(dict_response_time_series), index_col=False, sep=',')
        df_a_time =  pd.DataFrame(dict_response_time_series).T
        if df_a_time.shape[0] == 0:
            print("Continue break")
            break
        #"Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency."

        print(S + " df.shape: " + str(df_a_time.shape)  )
        df_a_time['Date'] = df_a_time.index
        df_S_all = pd.concat([df_S_all, df_a_time], ignore_index=True)

    if df_a_time.shape[0] == 0:
        print("Continue break")
        continue
    df_S_all = df_S_all.rename(columns={'time': 'Date', 'open': 'Open','high': 'High', 'close': 'Close','low': 'Low', 'volume': 'Volume'})
    df_S_all = df_S_all.rename(columns={ '1. open': 'Open', '2. high': 'High', '4. close': 'Close', '3. low': 'Low', '5. volume': 'Volume'})
    df_S_all = df_S_all.sort_values(['Date'], ascending=True)
    df_S_all = df_S_all.drop_duplicates(subset=['Date'],keep="first")
    df_S_all = df_S_all.dropna(how='any')
    df_S_all.reset_index(drop=True, inplace=True)

    df_S_all['Date'] = df_S_all['Date'].astype(str)
    max_recent_date = df_S_all['Date'].max()[:10].replace('-','') # pd.to_datetime().strftime("%Y%m%d")
    min_recent_date = df_S_all['Date'].min()[:10].replace('-','') #pd.to_datetime( ).strftime("%Y%m%d")
    print("d_price/RAW_alpha/alpha_"+S+'_'+intelval+"_"+ max_recent_date + "__" + min_recent_date + ".csv")
    df_S_all.to_csv("d_price/RAW_alpha/alpha_"+S+'_'+intelval+"_"+ max_recent_date + "__" + min_recent_date + ".csv", sep="\t", index=None)