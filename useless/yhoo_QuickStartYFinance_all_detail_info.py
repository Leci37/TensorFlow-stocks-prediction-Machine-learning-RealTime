
import csv
import json

import jsonpickle as jsonpickle
import pandas as pd
from LogRoot.Logging import Logger

import yfinance as yf

# import yfinance as yf

msft = yf.Ticker("MSFT")
oil = yf.Ticker("OIL")
natgas = yf.Ticker("NATGAS")

oilhist = oil.history(period="3y")
natgashist = natgas.history(period="3y")
# get stock info
msft.info


# get historical market data
hist = msft.history(period="2y")

# show actions (dividends, splits)
msft.actions

# show dividends
msft.dividends

# show splits
msft.splits

# show financials
msft.financials
msft.quarterly_financials

# show major holders
msft.major_holders

# show institutional holders
msft.institutional_holders

# show balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet

# show cashflow
msft.cashflow
msft.quarterly_cashflow

# show earnings
msft.earnings
msft.quarterly_earnings

# show sustainability
msft.sustainability

# show analysts recommendations
msft.recommendations

# show next event (earnings, etc)
msft.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options
msft.news
vars(msft)
# show news
# msft.news

# get option chain for specific expiration
# opt = msft.option_chain('YYYY-MM-DD')
opt = msft.option_chain('2022-06-17')
# data available via: opt.calls, opt.puts

list_company = ["RIVN", "VWDRY", "TWLO", "OB","ASML","SNOW","ADBE","LYFT","UBER","ZI","BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L","NATGAS","GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC"]

list_company = ["VWDRY"]

#msft = yf.Ticker("MSFT")
l = dir(msft)
#dict_y = vars(msft)

a1 = msft.option_chain()#contrac symbol
a2 = msft.quarterly_balance_sheet
a3 = msft.quarterly_balancesheet
a4 = msft.quarterly_cashflow
a5 = msft.quarterly_earnings
a6 = msft.quarterly_financials

df =None

s = msft
print(s._balancesheet)
print(s._base_url)
print(s._calendar)
print(s._cashflow)
print(s._download_options)
print(s._earnings)
print(s._expirations)
print(s._financials)
print(s._fundamentals)
print(s._get_fundamentals)
print(s._history)
print(s._info)
print(s._institutional_holders)
print(s._isin)
print(s._major_holders)
print(s._mutualfund_holders)
print(s._news)
print(s._options2df)
print(s._recommendations)
print(s._scrape_url)
print(s._shares)
print(s._sustainability)
print(s.actions)
print(s.analysis)
print(s.balance_sheet)
print(s.balancesheet)
print(s.calendar)
print(s.cashflow)
print(s.dividends)
print(s.earnings)
print(s.financials)
print(s.get_actions)
print(s.get_analysis)
print(s.get_balance_sheet)
print(s.get_balancesheet)
print(s.get_calendar)
print(s.get_cashflow)
print(s.get_dividends)
print(s.get_earnings)
print(s.get_financials)
print(s.get_info)
print(s.get_institutional_holders)
print(s.get_isin)
print(s.get_major_holders)
print(s.get_mutualfund_holders)
print(s.get_news)
print(s.get_recommendations)
print(s.get_shares)
print(s.get_splits)
print(s.get_sustainability)
print(s.history)
print(s.info)
print(s.institutional_holders)
print(s.isin)
print(s.major_holders)
print(s.mutualfund_holders)
print(s.news)
print(s.option_chain)
print(s.options)
print(s.quarterly_balance_sheet)
print(s.quarterly_balancesheet)
print(s.quarterly_cashflow)
print(s.quarterly_earnings)
print(s.quarterly_financials)
print(s.recommendations)
print(s.session)
print(s.shares)
print(s.splits)
print(s.stats)
print(s.sustainability)
print(s.ticker)



# for a in dict_y:
#     try:
#         df_y = pd.DataFrame(dict_y[a])
#         df_y.to_csv("yhoo_"+a+"_yhoo.csv", sep="\t")
#         #df_y.to_json("yhoo_"+a+"_yhoo.json")
#     except Exception as e:
#         Logger.logr.warn(" GET financial: " + str(a) + "Exception :" + str(e))
#
#
# Logger.logr.debug(type(msft))
# var_msft = vars(msft)
#
# stock_json_full = []


def manage_dict_to_json(dict_json):
    return_json = []
    # out_dict = var_msft[x]
    print("AAAAAA ", dict_json.keys())
    if type(dict_json) is dict:  # sub diccionario
        for g in dict_json.keys():
            print("....", g, " => ", type(dict_json[g]))
            sub_out_dict = dict_json[g]
            if type(sub_out_dict) is pd.core.frame.DataFrame:
                Save_Dict_or_PD_in_File(dict_json[g])
                # json_dumpDict = jsonpickle.encode(sub_out_dict, unpicklable=False)
                # return_json.append(json_dumpDict)

            else:
                # json_dumpDict = json.dumps(sub_out_dict)
                # return_json.append(json_dumpDict)
                Save_Dict_or_PD_in_File(dict_json[g])
        return return_json
    else:
        Save_Dict_or_PD_in_File(dict_json)
        # json_dumpDict = json.dumps(dict_json)
        # return  json_dumpDict


def Save_Dict_or_PD_in_File(dictPD, nameFile=""):
    if type(dictPD) is pd.core.frame.DataFrame:
        df = dictPD  # pd.DataFrame.from_dict(dictPD, orient="index")
        Logger.logr.info("Create file: "+   nameFile + ".csv")
        df.to_csv("CSV/" + nameFile + ".csv")

    elif type(dictPD) is dict:
        Logger.logr.info("Create file: "+  nameFile + "_" + list(dictPD.keys())[0] + ".csv")
        with open( nameFile + "_" + list(dictPD.keys())[0] + ".csv", 'w') as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, dictPD.keys())
            w.writeheader()
            w.writerow(dictPD)

        # stock_json_full.append(json_dumpDict)
        # if type(v) is pd.core.frame.DataFrame:
        # sub_out_dict = {str(x): y for x, y in sub_out_dict.items()}
    # df.to_json('temp.json', orient='records', lines=True)
    # print(json_dumpDict)


list_dict = {}

for x in var_msft.keys():
    Logger.logr.debug(x+ " => "+ str(type(var_msft[x])))
    if type(var_msft[x]) is str or type(var_msft[x]) is bool:  # or type(var_msft[x]) is NoneType
        Logger.logr.debug(x+ " is str")
        list_dict[x] = var_msft[x]
        # json_dumpStr = json.dumps(var_msft[x])
        # stock_json_full.append(json_dumpStr)
        # print(json_dumpStr)

    elif type(var_msft[x]) is pd.core.frame.DataFrame:
        Logger.logr.debug(x+ " is DataFrame")
        # df_dict = var_msft[x].to_dict().keys()[0]
        Save_Dict_or_PD_in_File(var_msft[x], x)
        # df_dict = dict((str(k), v) for k, v in df_dict.items())
        # for z in df_dict.keys():
        # print("....",z, " => ",type(df_dict[z]))
        # json_dumpDF = jsonpickle.encode(df_dict[z], unpicklable=False)
        # dict_json = manage_dict_to_json(df_dict[z])
        # stock_json_full.append(json_dumpDF)


    elif type(var_msft[x]) is dict:
        Logger.logr.debug(x+ " is dict")
        dict_json = manage_dict_to_json(var_msft[x])
        stock_json_full.append(dict_json)

    elif type(var_msft[x]) is list:
        Logger.logr.debug(x+ " is list")
        # json_dumpList = json.dumps(var_msft[x])
        # stock_json_full.append(json_dumpList)
        # print(json_dumpList)


