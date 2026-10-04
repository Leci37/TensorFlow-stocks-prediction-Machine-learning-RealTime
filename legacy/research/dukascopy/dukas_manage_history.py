import pandas as pd
import json
import  glob, re
import os

from _KEYS_DICT import DICT_COMPANYS

CSV_NAME = "@FOLO3"
list_stocks = DICT_COMPANYS[CSV_NAME]
PATH_LOCATED = "C:\\Users\\Luis\\Downloads\\"

list_stocks = [l.lower() for l in list_stocks]
list_to_check = [PATH_LOCATED + l+ "ususd-m15-bid-"  for l in list_stocks]

list_dukes_files = glob.glob(PATH_LOCATED + "*.csv")
# list_files_download = [x for x in list_dukes_files if x.startswith( tuple( list_to_check) )]

stocks_csv_paths = {}
for S in list_stocks:
    files_star_with = [x for x in list_dukes_files if x.startswith(PATH_LOCATED + S+ "ususd-m15-bid-")]
    if len(files_star_with) == 0:
        print("WARN Zero file in the folder for one action  Stock: " + S )
        continue
    elif len(files_star_with) > 1:
        print("ERROR more a file in the folder for one action  Stock: "+S +" Files: "+ ", ".join(files_star_with))
        continue
    print("Stock: ", S , " Path: ",files_star_with[0] )
    stocks_csv_paths[S] = files_star_with[0]


for S, path_S in stocks_csv_paths.items():
    df = pd.read_csv(path_S, index_col=None, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp']/1000, unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={ 'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'close': 'Close', 'low': 'Low', 'volume': 'Volume'})
    print("a")


print("aa")
# df_result = pd.read_csv("Models/TF_multi/" + name_model + "_per_score.csv", index_col=0, sep=',')
# df_result_all['Date'] = pd.to_datetime(df_result_all['Date'], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")