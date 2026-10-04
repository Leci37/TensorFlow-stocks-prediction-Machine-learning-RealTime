# from yahoo_fin.stock_info import get_data
from datetime import datetime

import investpy
import pandas as pd

from Utils import  UtilsL
from LogRoot.Logging import Logger
import requests
import lxml.html as LH
import numpy as np
# from pandas_datareader import data as pdr#https://github.com/ranaroussi/yfinance


# a = yf.pdr_override() # <== that's all it takes :-)


# download dataframe
# data = yf.pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")


def get_news_opinion_yahoo(stockId):
    st = yf.Ticker(stockId)
    news_y = st.get_news()
    # r = st.get_recommendations()
    dates = []
    head_lines = []
    for n in news_y:
        date_y = datetime.fromtimestamp(n['providerPublishTime']).strftime("%Y-%m-%d")
        dates.append(date_y)
        head_line_y = n['title']
        head_lines.append(head_line_y)

    df_header = pd.DataFrame(columns=['Date', 'Headline'])
    df_header['Date'] = dates
    df_header['Headline'] = head_lines
    Logger.logr.info("get_news_opinion_yahoo Stock: "+ stockId+ " Numbers of news: "+ str(len(df_header.index)))
    return df_header


def get_stock_value(stockid, startdate, enddate):
    data = yf.download(stockid, start=startdate, end=enddate)
    per_stc_day = ((data['Close'] * 100) / data['Open']) - 100
    per_stc_max_day = ((data['High'] * 100) / data['Low']) - 100
    data['per_stc_day'] = per_stc_day
    data['per_stc_max_day'] = per_stc_max_day
    data['Ticker'] = stockid

    data['Open'] = data['Open'].apply(lambda x: round(x, 2))
    data['Close'] = data['Close'].apply(lambda x: round(x, 2))
    data['Low'] = data['Low'].apply(lambda x: round(x, 2))
    data['High'] = data['High'].apply(lambda x: round(x, 2))
    data['per_stc_day'] = data['per_stc_day'].apply(lambda x: round(x, 2))
    data['per_stc_max_day'] = data['per_stc_max_day'].apply(lambda x: round(x, 2))
    data['Adj Close'] = data['Adj Close'].apply(lambda x: round(x, 1))

    Logger.logr.info("Get stock data Stock: "+ stockid+ " Date Start: "+ startdate+ "Date End: "+ enddate)
    data = data.reset_index(level=0)  # put Date as another column
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
    return data


# get_news_opinion_yahoo('MSFT')

import yfinance as yf


def aux():
    tickerStrings = ['AAPL', 'MSFT']
    df_list = list()
    for ticker in tickerStrings:
        data = yf.download(ticker, group_by="Ticker", period='2d')
        data['ticker'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
        df_list.append(data)

    # combine all dataframes into a single dataframe
    df = pd.concat(df_list)

    # save to csv
    df.to_csv('ticker.csv')

headers = {
        "User-Agent": investpy.utils.extra.random_user_agent(),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        #"referrer": "https://www.investing.com/equities/twilio-inc-a-technical",
}
headersAux ={
    "User-Agent": investpy.utils.extra.random_user_agent(),
    "accept": "*/*",
    "accept-language": "es,ca;q=0.9,en;q=0.8",
    "content-type": "application/x-www-form-urlencoded",
    "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"101\", \"Google Chrome\";v=\"101\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-requested-with": "XMLHttpRequest",
    #"referrer": "https://www.investing.com/equities/twilio-inc-a-technical"
}


def get_GET_root_from_url(url):
    try:
        r = requests.get(url=url, headers=headers)
        if r.status_code != 200:
            Logger.logr.warn(" GET status request code is NOT 200  r.status_code: " + str(r.status_code) + " url: " + url)
        root = LH.fromstring(r.content)
        return root
    except Exception as e:
        # Logger.logr.warning(e)
        Logger.logr.warn(" GET url: " + url + "requests :" + str(e))
        return None

#session = requests.Session()
#import time

def get_POST_root_from_url(url ,bodyP, headersP=headers):
    try:
        #gggg = session.get("https://es.investing.com/", headers=headers)
        #gggg = session.get("https://www.investing.com/equities/rivian-automotive-technical", headers=headers)
        #time.sleep(3)
        #print(session.cookies.get_dict())
        #r = requests.post(url=url, headers=headersP, data=bodyP)


        r = requests.post(url=url, headers=headers, data=bodyP)
        #print(requests.cookies.get_dict())
        if r.status_code != 200:
            Logger.logr.warn(" POST status request code is NOT 200  r.status_code: " + str(r.status_code) + " url: " + url)
        root = LH.fromstring(r.content)
        return root
    except Exception as e:
        # Logger.logr.warning(e)
        Logger.logr.warn(" POST url: " + url + "requests :" + str(e))
        return None


def merge_all_df_of_the_list(list_df, stockID):
    df_mer = None
    for df in list_df:
        try:
            if (df is None) or (len(df) == 0) or (not isinstance(df, pd.DataFrame)):  # todo NO FUNCIONA
                Logger.logr.info("error type data Stock: " + stockID)
                pass
            else:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                if df_mer is None:
                    df_mer = df
                else:
                    df_mer = pd.merge(df_mer, df, on=['Date'], how='outer')
        except Exception as e:
            Logger.logr.debug(str(e)+ " Exception stockID: "+  stockID)
    return df_mer

def prepare_df_to_json_by_date(df_full):
    df_full = df_full.replace(' ', np.nan)
    df_full = df_full.replace('', np.nan)
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full[(df_full['Date'] > '2019-01-01') & (df_full['Date'] <= '2025-01-01')]
    df_full.reset_index(drop=True, inplace=True)
    df_full.dropna(subset=["Date"], inplace=True)
    df_full.drop_duplicates(inplace=True)
    df_full = df_full.groupby('Date', as_index=False).agg(list)
    df_full = df_full.sort_values('Date', ascending=False)

    df_full['Date'] = df_full['Date'].dt.strftime('%Y-%m-%d')

    df_full.set_index('Date', inplace=True)
    dict_j = df_full.to_dict('index')  # df.to_dict('records') .to_dict('index'
    dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in
              dict_j.items()}
    dict_j = UtilsL.replace_list_in_sub_keys_dicts(dict_j)
    # dict_j = UtilsL.dict_drop_duplicate_subs_elements(dict_j)
    return dict_j


def get_crash_points(df, col_name_A, col_name_B, col_result, highlight_result_in_next_cell = 1 ):
    df["diff"] = df[col_name_A] - df[col_name_B]
    df[col_result] = 0

    df.loc[((df["diff"] >= 0) & (df["diff"].shift() < 0)), col_result] = 1
    df.loc[((df["diff"] <= 0) & (df["diff"].shift() > 0)), col_result] = -1
    #TODO test with oder numer than 1
    if highlight_result_in_next_cell > 0:
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == 1)), col_result] = 1
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == -1)), col_result] = -1

    df = df.drop(columns=['diff'])

    return df




def add_variation_percentage(df_stock, prefix = ""):
    df_stock[prefix+'per_Close'] = (df_stock['Close'] * 100 ) / df_stock['Close'].shift(1)  - 100
    df_stock[prefix+'per_Volume'] = (df_stock[df_stock['Volume'] != 0]['Volume'] * 100 ) / df_stock[df_stock['Volume'] != 0]['Volume'].shift(1) - 100
    df_stock[prefix+'per_Volume'] = df_stock[prefix+'per_Volume'].fillna(0)
    df_stock[prefix+'per_Close'] = df_stock[prefix+'per_Close'].fillna(0)
    return df_stock

def add_pre_market_percentage(df_his):
    df_his.insert(loc=len(df_his.columns), column='has_preMarket', value=False)
    df_his.insert(loc=len(df_his.columns), column='per_preMarket', value=0)

    for i in range(1, len(df_his)):
        if pd.to_datetime(df_his['Date'][i], errors='coerce').day != pd.to_datetime(df_his['Date'][i - 1],
                                                                                    errors='coerce').day:
            # df_his.at[i, 'pre_market'] = df_his.iloc[i]['Open'] - df_his.iloc[i - 1]['Close']
            df_his.at[i, 'has_preMarket'] = True
            df_his.at[i, 'per_preMarket'] = (df_his.iloc[i]['Open'] - df_his.iloc[i - 1]['Close']) * 100 / \
                                            df_his.iloc[i - 1]['Close']
    return df_his


#aaa = get_root_from_url("https://www.investing.com/search/?q=RIVN")
#from lxml.etree import tostring
#inner_html = tostring(aaa)
#sto = "VWDRY"# "Rivn" "Twlo"   ses_id upa
#TODO aqui esta el pairId pairId":985558
'''
window.allResultsQuotesDataArray = [{"pairId":985558,"name":"Twilio Inc","flag":"USA","link":"\\/equities\\/twilio-inc-a","symbol":"TWLO","type":"Stock - NYSE","pair_type_raw":"Equities","pair_type":"equities","countryID":5,"sector":20,"region":1,"industry":154,"isCrypto":false,"exchange":"NYSE","exchangeID":1},{"pairId":997526,"name":"Twilio Inc","flag":"Mexico","link":"\\/equities\\/twilio-inc-a?cid=997526","symbol":"TWLO","type":"Stock - Mexico","pair_type_raw":"Equities","pair_type":"equities","countryID":7,"sector":20,"region":2,"industry":154,"isCrypto":false,"exchange":"Mexico","exchangeID":53},{"pairId":1183161,"name":"Twilio Inc","flag":"Russian_Federation","link":"\\/equities\\/twilio-inc-a?cid=1183161","symbol":"TWLO-RM","type":"Stock - Moscow","pair_type_raw":"Equities","pair_type":"equities","countryID":56,"sector":20,"region":6,"industry":154,"isCrypto":false,"exchange":"Moscow","exchangeID":40}]

sto = "VWDRY"# "Rivn" "Twlo"

'''