import pickle
from datetime import datetime
import gc
import os
import tensorflow as tf
import pandas as pd
import numpy as np

#YAHOO API
import yfinance as yf
def get_df_yhoo_(S, inter, path = None ):
    date_15min = yf.download(tickers=S, period='6d', interval=inter, prepos=False)
    date_15min.index = date_15min.index.tz_convert(None)#location zone adapt to current zone
    date_15min.reset_index(inplace=True)
    date_15min = date_15min.rename(columns={'Datetime': 'Date'})
    # date_15min['Date'] = date_15min['Date'] + pd.Timedelta(hours=5)
    date_15min['Date'] = date_15min['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    date_15min = date_15min.sort_values('Date', ascending=False).round(2)
    if path is not None:
        date_15min[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']].to_csv(path, sep="\t", index=None)
    return date_15min#[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]

def manage_format_df_c_yh(df_c_yh):
    print("get stock : ", c, "\tshape:", df_c_yh.shape)
    df_c_yh.columns = df_c_yh.columns.droplevel(level=1)
    df_c_yh.insert(0, "date", df_c_yh.index)
    df_c_yh.rename(columns=lambda x: x.lower(), inplace=True)  # to minus
    df_c_yh.rename(columns={'close': 'close_yh'}, inplace=True)
    df_c_yh = df_c_yh[['date', 'close_yh']]
    return df_c_yh

path_preditc_resul = "../d_result/predi_MULTI_real_time_2023_10_19.csv"
df_wl = pd.read_csv(path_preditc_resul, sep='\t', index_col=0)
assert not path_preditc_resul.startswith("sent_"), "the code must receive all the predictions, and filter them. df_mer['predict']  > 0] "

df_wl.insert(0, "date", df_wl.index)
df_wl.reset_index(drop=True, inplace=True)

df_wl = df_wl.groupby(["date", "ticker"]).last().reset_index().sort_values(["date", "ticker"], ascending=True)

df_yh = get_df_yhoo_(list(df_wl['ticker'].unique()), "5m").sort_values(["Date"], ascending=True)
df_yh.index = df_yh["Date"]
# df_yh = df_yh.T
# df_yh['ticker'] = df_yh.index.get_level_values(1)
# df_yh['type'] = df_yh.index.get_level_values(0)
# df_yh.columns = df_yh.columns.map('_'.join)
df_meli = df_yh.iloc[:, df_yh.columns.get_level_values(1)=='MELI']

DOLARS_TO_OPERA = 100




df_full_result = pd.DataFrame()
for c in list(df_wl['ticker'].unique()):
    df_c_yh = df_yh.iloc[:, df_yh.columns.get_level_values(1) == c]
    df_c_yh = manage_format_df_c_yh(df_c_yh)

    df_c_res =  df_wl[df_wl['ticker']==c]
    df_c_res = df_c_res[['date','ticker','close', 'predict'] ]
    df_c_res['date'].max() , df_c_res['date'].min()

    df_c_yh = df_c_yh[ (df_c_yh['date'] <=  df_c_res['date'].max()) & (df_c_yh['date'] >=  df_c_res['date'].min())]
    df_mer = pd.merge(df_c_res, df_c_yh, on=['date'], how='outer')
    if (df_mer['predict'] == 0).all():
        print("No prediction for Stock : ", c)
    else:
        print("YES prediction for Stock : ", c)
        for n in[ 1,2, 3, 6, 12] : #range(0,13):
            # df_mer['close_yh_'+str(n)] = df_mer['close_yh'].shift(-1*n)
            # df_mer['stocks_buy_'+str(n)] = df_mer['close_yh_'+str(n)] * DOLARS_BUY
            stocks_broght = DOLARS_TO_OPERA / df_mer['close_yh']
            dollars_sell = stocks_broght * df_mer['close_yh'].shift(-1 * n)
            df_mer['dolars_win_cdl_' + str(n)] = dollars_sell - DOLARS_TO_OPERA #(stocks_sell - stocks_buy)
        df_full_result = pd.concat([df_full_result, df_mer[df_mer['predict'] > 0] ])


    print("aa")

df_full_result = df_full_result.sort_values(['predict', "date"], ascending=True)

df_result = df_full_result.groupby(['predict', 'ticker']).sum()
df_result['count'] = df_full_result.groupby(['predict', 'ticker']).count()['date']
# df['count'] = df_full_result.groupby('predict', 'ticker').transform('count')


print(   df_result  ) #FINAL

print("FINAL DEL PROYECTO ")