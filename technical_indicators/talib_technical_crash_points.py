import itertools

from Utils import Utils_Yfinance

list_MA_columns = ["ma_DEMA_5", "ma_EMA_5", "ma_KAMA_5", "ma_SMA_5", "ma_T3_5", "ma_TEMA_5", "ma_TRIMA_5", "ma_WMA_5", "ma_DEMA_10", "ma_EMA_10",
                  "ma_KAMA_10", "ma_SMA_10", "ma_T3_10", "ma_TEMA_10", "ma_TRIMA_10", "ma_WMA_10", "ma_DEMA_20", "ma_EMA_20", "ma_KAMA_20", "ma_SMA_20",
                  "ma_T3_20", "ma_TEMA_20", "ma_TRIMA_20", "ma_WMA_20", "ma_DEMA_50", "ma_EMA_50", "ma_KAMA_50", "ma_SMA_50", "ma_T3_50", "ma_TEMA_50",
                  "ma_TRIMA_50", "ma_WMA_50", "ma_DEMA_100", "ma_EMA_100", "ma_KAMA_100", "ma_SMA_100", "ma_T3_100", "ma_TEMA_100", "ma_TRIMA_100", "ma_WMA_100",
                    "ti_hma_20"]

list_PP_columns = ["trad_s3", "trad_s2", "trad_s1", "trad_pp", "trad_r1", "trad_r2", "trad_r3", "clas_s3", "clas_s2", "clas_s1", "clas_pp", "clas_r1", "clas_r2",
                   "clas_r3", "fibo_s3", "fibo_s2", "fibo_s1", "fibo_pp", "fibo_r1", "fibo_r2", "fibo_r3", "wood_s3", "wood_s2", "wood_s1", "wood_pp", "wood_r1",
                   "wood_r2", "wood_r3", "demark_s1", "demark_pp", "demark_r1", "cama_s3", "cama_s2", "cama_s1", "cama_pp", "cama_r1", "cama_r2", "cama_r3"]

def gel_MA_CRASH_funtion(df):
    # for ma_1_i in range(len(list_MA_columns)):
    #     for ma_2_i in range(len(list_MA_columns)):

    all_combinations = list( itertools.combinations(list_MA_columns, 2) )

    for com in all_combinations:
        ma_1 = com[0]
        ma_2 = com[1]
        #Si los ultimos 2 chars no son los mismos , se procede a comparar , no interesa EJ:  "ma_TEMA_20", "ma_TRIMA_20"
        if(ma_1[-2:] != ma_2[-2:]):
            new_column_name = "mcrh_" + ma_1.replace("ma_", "") + "_" + ma_2.replace("ma_", "")
            if cos_cols is None or new_column_name in cos_cols:
                df = Utils_Yfinance.get_crash_points(df, ma_1, ma_2, col_result=new_column_name)
    return df


def gel_PP_CRASH_funtion(df):
    for pp in list_PP_columns:
        new_column_name = "pcrh_" + pp

        if cos_cols is None or new_column_name in cos_cols:
            df = Utils_Yfinance.get_crash_points(df, "Close", pp, col_result=new_column_name, highlight_result_in_next_cell =0)

    return df

cos_cols = None
def get_ALL_CRASH_funtion(df, costum_columns=None):
    global cos_cols
    cos_cols = costum_columns

    df = gel_PP_CRASH_funtion(df)
    df = gel_MA_CRASH_funtion(df)
    return df


# raw_df = pd.read_csv("d_price/MELI_stock_history_MONTH_3.csv",index_col=False, sep='\t')
#
# df = gel_PP_CRASH_funtion(raw_df)
# df[["Close", "trad_r1", "pcrh_trad_r1"]]
# df = gel_MA_CRASH_funtion(raw_df)
# bbb=
# listCombined = combine( list_MA_columns )
# aaa = get_combinations(list_MA_columns)
# print("FINA")