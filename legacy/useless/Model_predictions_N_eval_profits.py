import pandas as pd

import Model_predictions_handle_Nrows
from Utils import UtilsL, Utils_buy_sell_points

import _KEYS_DICT

Y_TARGET = 'buy_sell_point'



def __get_dates_min_max(df_vender,df_compar ):
    max_recent_date, min_recent_date = "n", "n"
    if df_vender is not None:
        max_recent_date, min_recent_date = UtilsL.get_recent_dates(df_vender)
    elif df_compar is not None:
        max_recent_date, min_recent_date = UtilsL.get_recent_dates(df_compar)

    return max_recent_date, min_recent_date

def __get_A_sum_units_of_predict_models(df_v, profit_xxx_units):
    df_v = df_v[COLS_EVAL]
    list_r = []
    for r_types in ["88", "93", "95", "TF"]:
        if "have_to_oper_" + r_types not in df_v.columns:
            df_v.insert(loc=1, column="have_to_oper_" + r_types, value=False)
        df_v.loc[df_v['sum_r_' + r_types] >= ((df_v["num_models"] / 2) - 0.5), "have_to_oper_" + r_types] = True
        #SUMA
        df_aux = df_v.groupby("have_to_oper_" + r_types).sum()
        if True in df_aux.index:
            list_r.append( df_aux[profit_xxx_units][df_aux.index.get_loc(True)] )
        else:
            list_r.append( 0 )
        #NUMERO DE OPERACIONES
        df_auxcount = df_v.groupby("have_to_oper_" + r_types).count()
        if True in df_auxcount.index:
            list_r.append(df_auxcount[profit_xxx_units][df_auxcount.index.get_loc(True)])
        else:
            list_r.append( 0 )
        #CUANTO SE GANA DE MEDIA CON CADA OPÊRACION
        if True in df_auxcount.index and True in df_aux.index:
            earn_div_count = (df_aux[profit_xxx_units][df_aux.index.get_loc(True)]) / df_auxcount[profit_xxx_units][df_auxcount.index.get_loc(True)]
            list_r.append(earn_div_count)
        else:
            list_r.append(-0.1)

    return list_r #  counnt ese es el orden de salida: "stockid", 'buy_88', 'count_88','buy_count_88', 'buy_93','count_93','buy_count_93' 'buy_95','count_95','buy_count_95' 'buy_TF','count_TF','buy_count_TF'



def add_stock_eval_to_df_eval_earing(stock_id , path_each_stock = None):
    global df_eval_earnings
    if df_compar is not None:
        df_c = Utils_buy_sell_points.get_buy_sell_points_Roll(df_compar, delete_aux_rows=False)
        if path_each_stock is not None:
            df_c[COLS_EVAL].sort_values(['Date'], ascending=True).round(2).to_csv(path_each_stock + stock_id+ "_profit_POS_units.csv", sep='\t', index=None)

        list_b_s = __get_A_sum_units_of_predict_models(df_c, profit_xxx_units='profit_POS_units')
        df_eval_earnings = pd.concat([df_eval_earnings, pd.DataFrame([[stock_id + "_pos"] + list_b_s], columns=COL_GANAN)],ignore_index=True)  # añadir fila add row

        print("END BUY : " + stock_id)

    if df_vender is not None:
        df_v = Utils_buy_sell_points.get_buy_sell_points_Roll(df_vender, delete_aux_rows=False)
        if path_each_stock is not None:
            df_v[COLS_EVAL].sort_values(['Date'], ascending=True).round(2).to_csv(path_each_stock + stock_id+ "_profit_NEG_units.csv", sep='\t', index=None)

        list_b_s = __get_A_sum_units_of_predict_models(df_v, profit_xxx_units='profit_NEG_units')
        df_eval_earnings = pd.concat([df_eval_earnings, pd.DataFrame([[stock_id + "_neg"] + list_b_s], columns=COL_GANAN)],ignore_index=True)
        print("END SELL : " + stock_id)



CSV_NAME = "@FOLO3"
list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# list_stocks = ['UPST']

opion = _KEYS_DICT.Option_Historical.MONTH_3_AD
df_all_generate_history = pd.DataFrame()
NUM_LAST_REGISTERS_PER_STOCK =130 #lastweekend , 135 la ultima semana
COLS_EVAL = ['Date', 'buy_sell_point', 'Close', 'has_preMarket', 'Volume','sum_r_88', 'sum_r_93', 'sum_r_95', 'have_to_oper', 'sum_r_TF',
 'num_models', 'have_to_oper_TF' , 'sell_value_POS', 'sell_value_NEG','profit_POS_units', 'profit_NEG_units']


df_compar = pd.DataFrame()
df_vender = pd.DataFrame()

COL_GANAN = ["stockid", 'buy_88', 'count_88','buy_count_88', 'buy_93','count_93','buy_count_93', 'buy_95','count_95','buy_count_95', 'buy_TF','count_TF','buy_count_TF']
df_eval_earnings = pd.DataFrame(columns=COL_GANAN)


for S in list_stocks: # [ "UBER","U",  "TWLO", "TSLA", "SNOW", "SHOP", "PINS", "NIO", "MELI" ]:#list_stocks:
    try:
        df_compar, df_vender = Model_predictions_handle_Nrows.get_RealTime_buy_seel_points(S, opion, NUM_LAST_REGISTERS_PER_STOCK =NUM_LAST_REGISTERS_PER_STOCK)
    except Exception as e:
        df_compar = None
        df_vender = None
        print(S , " ", str(e))

    add_stock_eval_to_df_eval_earing(S, path_each_stock="Models/eval_Profits/")
    print(S + "   ")

max_recent_date, min_recent_date = __get_dates_min_max(df_vender,df_compar )
print("Models/eval_Profits/_"+CSV_NAME+"_ALL_stock_" + max_recent_date + "__" + min_recent_date + ".csv")

#Para que se pueda hacer round(2)
COL_GANAN.remove("stockid")
df_eval_earnings[COL_GANAN] =  df_eval_earnings[COL_GANAN].astype(float)
df_eval_earnings.sort_values(['buy_88', 'count_88'], ascending=True).round(2).to_csv("Models/eval_Profits/_"+CSV_NAME+"_ALL_stock_" + max_recent_date + "__" + min_recent_date + ".csv", sep='\t', index=None)






