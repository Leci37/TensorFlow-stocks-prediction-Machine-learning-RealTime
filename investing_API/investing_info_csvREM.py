import numpy as np
import pandas as pd
import investpy
import inspect

from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger
from lxml.etree import tostring
import datetime

# URL_OPTIONS = "company-profile","chart","advanced-chart","news","opinion","financial-summary","income-statement","balance-sheet","cash-flow","ratios",\
#               "dividends","earnings","technical","candlestick","consensus-estimates","commentary","scoreboard","user-rankings","historical-data",\
#               "options","relatedindices"
#
# URL_OPTIONS = {
#         "company-profile" : ["XXXX", "xxxx"],
#         "chart" : "XXXX",# TODO GRAFICO COMPLICADO
#         "advanced-chart" : "XXXX",# TODO GRAFICO COMPLICADO
#         "news" : "XXXX",#news_investing_analy_opi_sentiment.py
#         "news/2" : "XXXX",#news_investing_analy_opi_sentiment.py
#         "opinion" : "XXXX",#news_investing_analy_opi_sentiment.py
#         "financial-summary" : "XXXX",
#         "income-statement" : "XXXX",
#         "balance-sheet" : "XXXX",
#         "cash-flow" : "XXXX",
#         "ratios" : "XXXX",
#         "dividends" : "XXXX",
#         "earnings" : "XXXX",
#         "technical" : "XXXX",
#         "candlestick" : "XXXX",
#         "consensus-estimates" : "XXXX",
#         "commentary" : "XXXX",
#         "scoreboard" : "XXXX",
#         "user-rankings" : "XXXX",
#         "historical-data" : "XXXX",
#         "options": "XXXX",
#         "relatedindices" : "XXXX"
# }
#
# URL_STOCK = "https://www.investing.com/equities/apple-computer-inc-"

#for u in URL_OPTIONS.keys()[0:1]:

headers = {
        "User-Agent": investpy.utils.extra.random_user_agent(),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
}
#
#
# def get_root_from_url(url):
#         try:
#                 r = requests.get(url=url, headers=headers)
#                 if r.status_code != 200:
#                         Logger.logr.warn(" GET status request code is NOT 200  r.status_code: "+ str(r.status_code)+ " url: "+ url)
#                 root = LH.fromstring(r.content)
#                 return root
#         except Exception as e:
#                 #Logger.logr.warning(e)
#                 Logger.logr.warn(" GET url: "+url +"requests :"+str(e))
#                 return None
#


def recur_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def get_investing_company_profile(stockid, country ="US" ):

        url = UtilsL.Url_stocks_pd.get_url(stockid, country)
        url = url  + "-company-profile"
        Logger.logr.debug(url)
        #get_Params = {'q': postalCode, 'country': country}

        root_html = Utils_Yfinance.get_GET_root_from_url(url)

        if root_html is None:
                Logger.logr.error("ERROR company_profile in get_root_from_url")
                return

        inner_html = tostring(root_html)

        value_day_ago = root_html.xpath('//*[@id="last_last"]')[0].text
        value_day_ago_increase = root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[1]/span[2]')[0].text
        value_day_ago_per = root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[1]/span[4]/text()')[0].replace("(","").replace(")","").replace("%","")

        is_preMarket = True
        premarket_dict = {}
        try:
            pre_market = root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[3]/div[1]/span')[0].text
            pre_market_local_time =  root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[3]/div[2]/i')[0].text
            my_hour = datetime.datetime.strptime(pre_market_local_time.replace(' ',''), "%X").time()
            pre_market_local_time = datetime.datetime.combine(datetime.date.today(), my_hour)
            pre_market_increase = root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[3]/div[1]/div[1]')[0].text
            pre_market_per = root_html.xpath('//*[@id="quotes_summary_current_data"]/div[1]/div[2]/div[3]/div[1]/div[2]')[0].text.replace("(","").replace(")","").replace("%","")
            premarket_dict  = {
                recur_name(pre_market): pre_market,
                recur_name(pre_market_local_time): pre_market_local_time,
                recur_name(pre_market_increase): pre_market_increase,
                recur_name(pre_market_per): pre_market_per
            }
        except IndexError as e:
            Logger.logr.warning(" GET technical analysis(: " + stockid + "Exception :" + str(e))
            is_preMarket = False

        volumen = root_html.xpath('//*[@id="quotes_summary_secondary_data"]/div/ul/li[1]/span[2]/span')[0].text
        # last_val_4 = root_html.xpath('//*[@id="quotes_summary_secondary_data"]/div/ul/li[2]/span[2]/span[1]')[0].text
        # last_val_5 = root_html.xpath('//*[@id="quotes_summary_secondary_data"]/div/ul/li[2]/span[2]/span[2]')[0].text
        day_range_1 = root_html.xpath('//*[@id="quotes_summary_secondary_data"]/div/ul/li[3]/span[2]/span[1]')[0].text
        day_range_2 = root_html.xpath('//*[@id="quotes_summary_secondary_data"]/div/ul/li[3]/span[2]/span[2]')[0].text

        #k = recur_name(last_value)
        dict_result = {
            recur_name(value_day_ago): value_day_ago,
            recur_name(value_day_ago_increase): value_day_ago_increase,
            recur_name(value_day_ago_per): value_day_ago_per,
            recur_name(volumen): volumen,
            # recur_name(last_val_4): last_val_4,
            # recur_name(last_val_5): last_val_5,
            "day_range": np.array([day_range_1,day_range_2]),
        }
        dict_result =  {**dict_result, **premarket_dict} #join merge two dict
        df_i = pd.DataFrame([dict_result])
        return  df_i


tickers = ["AMD", "GE", "BA", "AMZN", "AAPL", "PYPL", "MELI"]
tickers = ["MELI" ,"VWDRY"]


# df = None
# for s in tickers:
#     df_aux = get_investing_company_profile(s)
#     if df_aux is None:  # si hay fallo en la empres a pasa a la siguiente
#         Logger.logr.error("  dont appper Info stock:  "+ s)
#         pass
#     else:
#         df_aux.to_csv("d_info_profile/" + str(s) + "_financial_inv_dividends.csv", sep="\t")
#         Logger.logr.debug(" Generated File info company profile: d_info_profile/" + str(s) + "_financial_inv_dividends.csv.csv")


