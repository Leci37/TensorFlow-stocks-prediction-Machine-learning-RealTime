from enum import Enum

import talib

import UtilsL
#from predict_example import kaggle_stock_market_Tech
import Utils_buy_sell_points
import Utils_plotter
import talib_technical_crash_points
import talib_technical_funtions
import talib_technical_PY_TI
import talib_technical_pandas_TA
import talib_technical_pandas_TU
import yfinance as yf
import pandas as pd
import numpy as np


from LogRoot.Logging import Logger

class Option_Historical(Enum):
    YEARS_3 = 1
    MONTH_3 = 2

list_stocks = ["RIVN", "VWDRY", "TWLO",          "GOOG","ASML","SNOW","ADBE","LYFT","UBER","ZI", "BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L",]
list_stocks = ["GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L","GOOG","ASML","SNOW","ADBE","LYFT","UBER","ZI", "BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC","BBVA.MC","OB"]

# BEGIN_DATE = '2019-01-01'
# END_DATE = '2025-01-01'
STOCKID = "UBER"




def get_historial_data_3y(stockID, prepos=True):
    yho_stk = yf.Ticker(stockID)
    hist = yho_stk.history(period="3y", prepost=prepos)

    df_his = pd.DataFrame(hist)
    df_his.reset_index(inplace=True)
    df_his = df_his.drop(columns=['Dividends', 'Stock Splits'],errors='ignore')

    return df_his

def get_historial_data_3_month(stockID, prepos=True, interva="15m"):
    yho_stk = yf.Ticker(stockID)
    hist = yho_stk.history(period="60d",prepost=prepos, interval=interva)

    df_his = pd.DataFrame(hist)
    df_his.reset_index(inplace=True)
    df_his = df_his.drop(columns=['Dividends', 'Stock Splits'],errors='ignore')

    df_his = df_his.rename(columns={'Datetime': 'Date'})
    df_his['Date'] = df_his['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    #     df_his[c] = df_his[c].round(3)
    #     #df_his = df_his.rename(columns={c: c+"_m"})

    return df_his



def add_variation_percentage(df_stock):
    df_stock['per_Close'] = (df_stock['Close'] * 100 ) / df_stock['Close'].shift(1)  - 100
    df_stock['per_Volume'] = (df_stock['Volume'] * 100 ) / df_stock['Volume'].shift(1) - 100
    return df_stock

def add_pre_market_percentage(df_his):
    df_his['has_preMarket'] = False
    df_his['per_preMarket'] = 0
    for i in range(1, len(df_his)):
        if pd.to_datetime(df_his['Date'][i], errors='coerce').day != pd.to_datetime(df_his['Date'][i - 1],
                                                                                    errors='coerce').day:
            # df_his.at[i, 'pre_market'] = df_his.iloc[i]['Open'] - df_his.iloc[i - 1]['Close']
            df_his.at[i, 'has_preMarket'] = True
            df_his.at[i, 'per_preMarket'] = (df_his.iloc[i]['Open'] - df_his.iloc[i - 1]['Close']) * 100 / \
                                            df_his.iloc[i - 1]['Close']
    return df_his

# def add_per_market_indicator(df_stock):
#     df_stock['is_permarket'] = False
#     df_stock.loc[df_stock['per_Close'] >= 0, 'tendency'] = True
#     df_stock.loc[df_stock['per_Close'] < 0, 'tendency'] = False


def get_json_stock_values_history(stockId, opion, get_technical_data = False, prepost=True, interval="30m", add_stock_id_colum = False):
    df_his = pd.DataFrame()
    if opion.value == Option_Historical.YEARS_3.value:
        df_his = get_historial_data_3y(stockId, prepos = prepost )
    elif opion.value == Option_Historical.MONTH_3.value:
        df_his = get_historial_data_3_month(stockId, prepos = prepost, interva=interval)

    if add_stock_id_colum:
        df_his['ticker'] = stockId

    if df_his is None:
        Logger.logr.debug("d_price/" + stockId + "_stock_history_"+str(opion.name)+".csv  is NONE stock: " + stockId)
    else:
        if get_technical_data:
            df_his = add_variation_percentage(df_his)
            df_his = Utils_buy_sell_points.get_buy_sell_points(df_his)
            df_his = add_pre_market_percentage(df_his)
            #Utils_plotter.plotting_financial_chart_buy_points_serial(df_his, df_his['buy_sell_point'], stockId,str(opion.name) )
            df_his = talib_technical_funtions.gel_all_TALIB_funtion(df_his) #siempre ordenada la fecha de mas a menos TODO exception
            df_his = talib_technical_PY_TI.get_all_pivots_points(df_his)
            df_his = talib_technical_PY_TI.get_py_TI_indicator(df_his)
            df_his = talib_technical_pandas_TA.get_all_pandas_TA_tecnical(df_his)
            df_his = talib_technical_pandas_TU.get_all_pandas_TU_tecnical(df_his)
            df_his = talib_technical_crash_points.get_ALL_CRASH_funtion(df_his)
            # df_his.columns
            # df_his = df_his[UtilsL.ALL_COLUMNS_NAME]

            df_his = df_his.round(6)
        df_his['Date'] = df_his['Date'].astype(str)
        df_his.reset_index(drop=True, inplace=True)
        df_his.to_csv("d_price/" + stockId + "_stock_history_"+str(opion.name)+".csv", sep="\t", index=None)
        Logger.logr.info("d_price/" + stockId + "_stock_history_"+str(opion.name)+".csv  stock: " + stockId + " Shape: " + str(df_his.shape))

        return df_his
        # df_his.to_excel("d_price/" + stockId + "_stock_history_"+str(opion.name) + '.xlsx', sheet_name="Hoja_1", index=False)
        #TODO JSON fine creation
        # df_his = df_his.groupby('Date', as_index=True).agg(list)
        #
        # dict_j = df_his.to_dict('index')  # df.to_dict('records') .to_dict('index'
        # dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in
        #           dict_j.items()}
        # dict_j = UtilsL.replace_list_in_sub_keys_dicts(dict_j)
        # dict_json = {}
        # # dict_json['Date'] = dict_j
        # dict_json = dict_j
        # import json
        #
        # with open("d_price/" + stockId + "_stock_history_"+str(opion.name)+".json", 'w') as fp:
        #     json.dump(dict_json, fp, allow_nan=True)
        # Logger.logr.info("d_price/" + stockId + "_stock_history_"+str(opion.name)+".json Numbres of Keys: " + str(len(dict_json)))



STOCKID = "UPST" # "UPST" #"BA"
# # for s in list_stocks:
# get_json_stock_values_history(STOCKID, Option_Historical.YEARS_3, get_technical_data = True, prepost=False )
# get_json_stock_values_history(STOCKID, Option_Historical.MONTH_3, get_technical_data = True, prepost=False, interval="15m")
# yho_stk = yf.Ticker("MELI")

# hist = yho_stk.history(period="1mo", prepost=True, interval="30m")
#
# print(hist.head())
##PS C:\Users\leci\AppData\Local\Programs\Python\Python38\Scripts> .\jupyter.exe notebook C:\Users\Luis\Desktop\LecTrade