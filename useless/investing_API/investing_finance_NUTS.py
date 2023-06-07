import pandas as pd

from Utils import UtilsL, Utils_Yfinance
import investing_finance_summary_csv
from LogRoot.Logging import Logger
from investing_finance_earnings_csv import get_df_financial_earnings
from investing_finance_ratios_csv import get_df_financial_ratios
from investing_financial_dividends import get_df_financial_dividends


list_company = ["RIVN", "VWDRY", "TWLO", "OB","ASML","SNOW","ADBE","LYFT","UBER","ZI","BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L","NATGAS","GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC"]

list_company = []

df = None
pd.set_option('mode.chained_assignment', None)


def get_investing_4_Finance_df(stockid, country="US", print_subs_csv = False):
    list_df = []
    country = "US"
    url = UtilsL.Url_stocks_pd.get_url(stockid, country)
    if url is None:
        Logger.logr.debug(
            " Not detedted URL investing for stock, pass to the next : " + str(stockid) + " country: " + country)
        return
    country = "US"
    df_e = get_df_financial_earnings(stockid, country)
    if df_e is None:
        Logger.logr.error(" NOT data fount Generated financial_earnings info: " + str(stockid) + " country: " + country)
    else:
        df_e = df_e.rename(columns={'Release_Date_ear': 'Date'})
        if print_subs_csv:
            df_e.to_csv( "d_finance/"+ str(stockid) + "_financial_inv_earn.csv", sep="\t")
        Logger.logr.debug(" Generated File info: d_finance/" + str(stockid) + "_financial_inv_earn.csv")
        list_df.append(df_e)

    country = "US"
    df_r = get_df_financial_ratios(stockid, country)
    if df_r is None:
        Logger.logr.error(" NOT data fount Generated financial_ratios info: " + str(stockid) + " country: " + country)
    else:
        df_r['Date'] = pd.Timestamp("today").strftime("%Y-%m-%d")
        if print_subs_csv:
            df_r.to_csv( "d_finance/"+ str(stockid) + "_financial_inv_ratios.csv", sep="\t")
        Logger.logr.debug(" Generated File info: d_finance/" + str(stockid) + "_financial_inv_ratios.csv")
        list_df.append(df_r)

    country = "united states"
    df_f = investing_finance_summary_csv.get_df_financial_full_date_axis(stockid, country)
    if df_f is None:
        Logger.logr.error(
            " NOT data fount Generated financial_full_date_axis info: " + str(stockid) + " country: " + country)
    else:
        if print_subs_csv:
            df_f.to_csv( "d_finance/"+ str(stockid) + "_financial_inv_summary.csv", sep="\t")
        Logger.logr.debug(" Generated File info: d_finance/" + str(stockid) + "_financial_inv_summary.csv")
        list_df.append(df_f)

    country = "US"
    df_d = get_df_financial_dividends(stockid, country)
    if df_d is None:
        Logger.logr.warn(" NOT data fount Generated financial_dividends info: " + str(stockid) + " country: " + country)
    else:
        df_d = df_d.rename(columns={'ExDividend_Date': 'Date'})
        if print_subs_csv:
            df_d.to_csv( "d_finance/"+ str(stockid) + "_financial_inv_dividends.csv", sep="\t")
        Logger.logr.debug(" Generated File info: d_finance/" + str(stockid) + "_financial_inv_dividends.csv.csv")
        list_df.append(df_d)

    return list_df


def get_json_investing_finance(stockID):
    list_df = get_investing_4_Finance_df(stockID, print_subs_csv=True)
    df_full = Utils_Yfinance.merge_all_df_of_the_list(list_df, stockID)
    if df_full is None or len(df_full) == 0:
        Logger.logr.error(" NOT data fount Generated INVESTING FINANCE info : " + stockID)
        return
    else:
        df_full.to_csv("d_finance/" + stockID + "_inves_finance_full_data.csv", sep="\t")

        Logger.logr.info(
            "Full merge of all Dataframes of INVESTING FINANCE completed shape: " + str(
                df_full.shape) + " Stock: " + stockID)
        dict_j = Utils_Yfinance.prepare_df_to_json_by_date(df_full)

        dict_json = {}
        # dict_json['Date'] = dict_j
        dict_json = dict_j
        import json
        with open("d_finance/" + stockID + "_inves_finance_full_data.json", 'w') as fp:
            json.dump(dict_json, fp, allow_nan=True)
        Logger.logr.info(
            "d_finance/" + stockID + "_inves_finance_full_data.json  Numbres of Keys: " + str(len(dict_json)))

get_json_investing_finance("BA")
#INTERVAL = "weekly"
# for stockID in list_company:
#     #stockid = "MELI" #"VWDRY" # "MELI" #"VWDRY"
#     #country = "united states"
#     get_json_investing_finance(stockID)

    # if df_full is None or len(df_full) == 0:
    #     Logger.logr.error(" NOT data fount Generated YAHOO info : " + stockID)
    #     pass

