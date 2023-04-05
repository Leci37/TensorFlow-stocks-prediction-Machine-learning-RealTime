import warnings

from Utils import UtilsL

warnings.simplefilter(action='ignore', category=FutureWarning)

# import finwiz_get_data
# import investing_finance_NUTS
# import investing_tech_NUTS
# import news_get_data_NUTS
# import yhoo_date_stock_date
# import yhoo_external_raw_factors
# import yhoo_history_stock
import pandas as pd
import numpy as np
import json

import os.path

from LogRoot.Logging import Logger

list_company = ["RIVN", "VWDRY", "TWLO", "OB","ASML","SNOW","ADBE","LYFT","UBER","ZI","BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L","NATGAS","GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC"]

pd.set_option('mode.chained_assignment', None)

stockid = "BA"

list_json = [
    "d_info_profile/" + stockid + "_finviz_data.json",
    "d_finance/" + stockid + "_inves_finance_full_data.json",
    "d_tech/" + stockid + "_inves_tech_full_data.json",
    "d_sentiment/" + stockid + "_stock_news.json",
    "d_info_profile/" + stockid + "_yahoo_full_data.json",
    "d_price/" + stockid + "_stock_history.json",
    "d_external_factors/external_factors_hist.json"
]
list_json_csv = [
    "d_info_profile/" + stockid + "_finviz_data.csv",
    "d_finance/" + stockid + "_inves_finance_full_data.csv",
    "d_tech/" + stockid + "_inves_tech_full_data.csv",
    "d_sentiment/" + stockid + "_stock_news.csv",
    "d_info_profile/" + stockid + "_yahoo_full_data.csv",
    "d_price/" + stockid + "_stock_history.csv",
    "d_external_factors/external_factors_hist.csv"
]


def printDict(inDict):
    #print(k,end=" ")
    for k,v in inDict.items():
        if type(v) == dict:
            printDict(v)
        elif type(v) == list:
            for i in v:
                print(i, end=" ")
        else:
            print(v, end=" ")

# finwiz_get_data.get_json_finwiz_data(stockid)                #"d_info_profile/" + stockid + "_finviz_data.json"
# investing_finance_NUTS.get_json_investing_finance(stockid)   #"d_finance/" + stockID + "_inves_finance_full_data.json"
# investing_tech_NUTS.get_json_investing_tech(stockid)         #"d_sentiment/" + stockId +"_stock_news.json"
# news_get_data_NUTS.get_json_news_sentimet(stockid)           #"d_sentiment/" + stockId + "_stock_news.json"
# yhoo_date_stock_date.get_all_date_info_yhoo(stockid)         #"d_info_profile/" + stockID + "_yahoo_full_data.json"
#yhoo_history_stock.get_json_stock_values_history(stockid)     #"d_price/" + stockID + "_stock_history.json
# yhoo_external_raw_factors.get_raw_stocks_values()            #d_external_factors/external_factors_hist.json


def clean_dict_for_Nan(dict_j):
    for k, v in dict_j.items():
        for a, b in v.items():
            if type(b) == list or type(b) == dict:
                if (not pd.isna(b).all() and not pd.isnull(b).all()):
                    dict_j[k] = {a : [x for x in b if x == x]}
            else:
                if not pd.isna(b) and not pd.isnull(b):
                    dict_j[k] = b
    return dict_j

def prepare_df_to_json_by_date(df_full):
    df_full = df_full.replace(' ', np.nan)
    df_full = df_full.replace('', np.nan)
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full[(df_full['Date'] > '2019-01-01') & (df_full['Date'] <= '2025-01-01')]
    df_full.reset_index(drop=True, inplace=True)
    df_full.dropna(subset=["Date"], inplace=True)
    df_full.drop_duplicates(inplace=True)
    df_full.index = df_full.index.astype(str)
    df_full = df_full.groupby('Date', as_index=True).agg(list)
    df_full = df_full.sort_values('Date', ascending=False)
    #df_full['Date'] = df_full['Date'].dt.strftime('%Y-%m-%d')


    #df_full['Date'] = df_full['Date'].astype(str)
    df_full.index = df_full.index.astype(str)
    #df_full.set_index('Date', inplace=True)
    dict_j = df_full.to_dict('index')  # df.to_dict('records') .to_dict('index'
    dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in dict_j.items()}
    #TODO
    #dict_j = clean_dict_for_Nan(dict_j)
    #dict_j = UtilsL.dict_drop_duplicate_subs_elements(dict_j)
    return dict_j

import datetime as date_dict
def replace_Full_list_in_sub_keys_dicts(dict_sub_list):
    '''
    cambio de los elementos que estan en ["dato"] por "dato" , quitar las listas individuales
    :param dict_sub_list:
    :return:
    '''
    for key in dict_sub_list.keys():
        if type(dict_sub_list[key]) == dict:
            for k2, v2 in dict_sub_list[key].items():
                if type(v2) == list and len(v2) > 1 and all(isinstance(ele, date_dict.datetime) for ele in v2) :#     isinstance(v2, date_dict.datetime):
                    #dict_sub_list[key][k2] = v2[0].strftime('%Y-%m-%d')
                    dict_sub_list[key][k2] = [i.strftime('%Y-%m-%d') for i in v2]
                elif type(v2) == list and len(v2) == 1:
                    try:
                        dict_sub_list[key][k2] = UtilsL.maybe_make_number(v2[0])#cast to float if it is allow
                    except ValueError:
                        dict_sub_list[key][k2] = v2[0]
                elif type(v2) == list and  all(isinstance(item, str) for item in v2) :
                    if UtilsL.all_equal(v2):
                        if (v2[0] is not None) or (str(v2[0]) != 'nan'):
                            dict_sub_list[key][k2] = v2[0]     #si son todos iguales y no es null
                    else:
                        dict_sub_list[key][k2] = list([ e for e in v2 if (str(e) != 'nan') ] ) #quitar string de sobra de la lista
                # elif type(v2) == list and len(v2) > 7:
                #     if UtilsL.all_equal(v2):
                #         if (v2[0] is not None) or (str(v2[0]) != 'nan'):
                #             dict_sub_list[key][k2] = v2[0]     #si son todos iguales y no es null
                #     else:
                #         l_aux = list(set([ e for e in v2 if (e is not None) or (str(e) != 'nan')  ] ))
                #         dict_sub_list[key][k2] = list(map(UtilsL.maybe_make_number, (l_aux)))
                elif type(v2) == list and len(v2) > 1:
                    l_aux = list([e for e in v2 if (str(e) != 'nan')]) # list(set([e for e in v2 if (str(e) != 'nan')]))
                    if UtilsL.all_equal(v2):
                        #if (str(v2[0]) != 'nan'):
                        dict_sub_list[key][k2] = v2[0]     #si son todos iguales y no es null
                    elif len(l_aux) == 1:
                        dict_sub_list[key][k2] = l_aux[0]
                    elif len(l_aux) > 1:
                        dict_sub_list[key][k2] = list(map(UtilsL.maybe_make_number, (l_aux)))



    return  dict_sub_list

df_full = None
for file_path in list_json_csv:
    Logger.logr.debug("Check file Path: "+ file_path + " \t\tExist: "+ str(os.path.exists(file_path)) + " \t\tIsFile: "+  str(os.path.isfile(file_path)) )
    df_aux = pd.read_csv(file_path, sep="\t", index_col=0)#avoid de index Unname
    # df_aux.reset_index(drop=True, inplace=True)
    # df_aux = df_group_all(df_aux)
    if df_full is None:
        df_full = df_aux
    else:
        df_full = pd.concat([df_full , df_aux])
        #df_full = pd.merge(df_full , df_aux,  how='outer')



# df_full.reset_index(drop=True, inplace=True)
# df_full.dropna(subset=["Date"], inplace=True)
# df_full.set_index('Date', inplace=True)
# df_full.drop_duplicates(inplace=True)
# df_full = df_full.groupby('Date', as_index=False).agg(list)
# df_full.reset_index(drop=True, inplace=True)
# dict_j = df_full.to_dict('index')
dict_j = prepare_df_to_json_by_date(df_full)
dict_j = replace_Full_list_in_sub_keys_dicts(dict_j)
# dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in
#           dict_j.items()}
# dict_j = UtilsL.replace_list_in_sub_keys_dicts(dict_j)

# df.to_dict('records') .to_dict('index'
# dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in dict_j.items()}
#dict_j = Utils_Yfinance.prepare_df_to_json_by_date(df_full)
#dict_j = clean_dict_for_Nan(dict_j)
# dict_j = UtilsL.replace_list_in_sub_keys_dicts(dict_j)

dict_json = dict_j
with open("d_result/"+ stockid + "_FULLconcat.json", 'w') as fp:
    json.dump(dict_json, fp, allow_nan=True)
Logger.logr.info("d_result/"+ stockid + "_FULL.json Numbres of Keys: " + str(len(dict_json)))

# df_full.to_csv("d_result/" + stockid + "_FULL.csv", sep="\t")
print("aa")
# i = 0
# dict_j = defaultdict(list)
# df_full = None
# dict_join = {}
#
# dictdump = None
# for file_path in list_json:
#     Logger.logr.debug("Check file Path: "+ file_path + " \t\tExist: "+ str(os.path.exists(file_path)) + " \t\tIsFile: "+  str(os.path.isfile(file_path)) )
#     with open(file_path) as handle:
#         dictdump = json.loads(handle.read())
#
#     dict_join = {**dict_join, **dictdump}
    # unique_items = {}
    # for k, v in dict_join.items():
    #     if v not in unique_items.values():
    #         unique_items[k] = v

    # if df_full is None:
    #     df_full = pd.DataFrame(dictdump)
    # else:
    #     df_aux = pd.DataFrame(dictdump)
    #     df_full = pd.merge(df_full , df_aux, how='outer')
    # df_full.to_csv("d_result/"+str(i)+"_" + stockid + "_FULL.csv")
    #dict_j = {**dict_j, **dictdump}

    # for k, v in dictdump:
    #     if v not in dict_j.values():
    #         dict_j[k].append(v)

    # for d in (dict_j, dictdump):  # you can list as many input dicts as you want here
    #     for key, value in d.items():
    #         # if key in dict_j:
    #         #     dict_j[key].append(value)
    #         # else:
    #         dict_j[key].append(value)
#     i = i+1
#     dict_fu = {}#to avoid circular reference error
#     dict_fu = dict_join#.copy()
#     with open("d_result/"+str(i)+"_" + stockid + "_FULL.json", 'w') as fp:
#         json.dump(dict_fu, fp, allow_nan=True)
#     Logger.logr.info("d_result/" +str(i)+"_"+ stockid + "_FULL.json Numbres of Keys: " + str(len(dict_j)))
# #printDict(dict_j)
# print("aa")



