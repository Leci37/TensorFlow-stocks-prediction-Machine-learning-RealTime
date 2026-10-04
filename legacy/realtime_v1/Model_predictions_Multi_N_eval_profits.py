import pandas as pd
import os.path

import Feature_selection_json_columns
import Model_predictions_handle_Multi_Nrows
import _KEYS_DICT
from LogRoot.Logging import Logger
from Utils import UtilsL, Utils_buy_sell_points, Utils_scoring, Utils_model_predict

from _KEYS_DICT import Op_buy_sell, Option_Historical, DICT_COMPANYS, MODEL_TF_DENSE_TYPE_MULTI_DIMENSI, \
    PERCENTAGES_SCORE
from Utils.UtilsL import bcolors

Y_TARGET = 'buy_sell_point'


opion = Option_Historical.MONTH_3

def get_best_proftis_models(df_eval, S, type_buy_sell :Op_buy_sell, df_moldes_S_NP_columns, NUMBER_MACHTING_TEST_BEST = 3):
    # list_models_POS_NEG = [x for x in df_eval.columns if x.startswith("Acert_TFm_" + S + "_" + type_buy_sell.value)]  # es el porcentage

    list_models_POS_NEG = [x for x in df_moldes_S_NP_columns if x.startswith("TFm_" + S + "_" + type_buy_sell.value)]
    df_profits = pd.DataFrame()
    for c in [x for x in list_models_POS_NEG if "_" + type_buy_sell.value + "_" in x and not x.endswith("_per")]:
        float_close_mean = df_eval['Close'].mean().round(2)
        list_per = PERCENTAGES_SCORE
        for per in list_per:
            c_acer = "Acert_" + c
            if type_buy_sell == Op_buy_sell.POS:
                df_profits.loc[per, c] = df_eval[df_eval[c_acer] >= per * 100]['profit_POS_units'].mean()
                df_profits.loc[str(per) + "_n", c] = df_eval[df_eval[c_acer] >= per * 100]['profit_NEG_units'].mean()
                df_profits.loc[str(per) + "_dif", c] = df_eval[df_eval[c_acer] >= per * 100]['profit_POS_units'].mean() - df_eval[df_eval[c_acer] >= per * 100]['profit_NEG_units'].mean()
            elif type_buy_sell == Op_buy_sell.NEG:
                df_profits.loc[per, c] = df_eval[df_eval[c_acer] >= per * 100]['profit_NEG_units'].mean()
                df_profits.loc[str(per) + "_n", c] = df_eval[df_eval[c_acer] >= per * 100]['profit_POS_units'].mean()
                df_profits.loc[str(per) + "_dif", c] = df_eval[df_eval[c_acer] >= per * 100]['profit_NEG_units'].mean() - df_eval[df_eval[c_acer] >= per * 100]['profit_POS_units'].mean()

            df_profits.loc[str(per) + "_dif_per", c] = (df_profits.loc[str(per) + "_dif", c] / float_close_mean *100).round(2)
        print(c + "\t\t 0.93_dif: "+str(df_profits.loc["0.93_dif", c].round(2)) +"\t 0.9_Percentage: "+str(df_profits.loc["0.9_dif_per", c].round(2)) +"% "+"\t 0.93_Percentage: "+str(df_profits.loc["0.93_dif_per", c].round(2)) +"%"  +"\t 0.95_Percentage: "+str(df_profits.loc["0.95_dif_per", c].round(2)) +"%" )

    m_highs_dif_per = df_profits.loc[[  '0.91_dif_per', '0.93_dif_per' ]].mean().nlargest(NUMBER_MACHTING_TEST_BEST)
    m_highs_dif_per = m_highs_dif_per[m_highs_dif_per > 2.2]
    print(bcolors.HEADER + "Stock: "+S+"\tModelos por encima de 2.2% de beneficio: "+ str(m_highs_dif_per)+ bcolors.ENDC)

    # m_highs_dif = df_profits.loc[[ '0.9_dif', '0.91_dif', '0.92_dif', '0.93_dif', '0.94_dif', '0.95_dif']].mean().nlargest(NUMBER_MACHTING_TEST_BEST)

    return list(m_highs_dif_per.index)



def add_models_colum_by_best_Profits(path_csv_file_EVAL , NUMBER_MACHTING_TEST_BEST=2):
    '''
    :param path_csv_file_EVAL:
    :param NUMBER_MACHTING_TEST_BEST:
    :return: evalue los mejores modelos en funcion de los beneficios generados
    '''
    print(path_csv_file_EVAL)
    df_eval = pd.read_csv(path_csv_file_EVAL, index_col=False, sep='\t')
    df_eval = Utils_buy_sell_points.get_buy_sell_points_Roll(df_eval, delete_aux_rows=False)
    list_cols = ['sell_value_POS', 'sell_value_NEG', 'profit_POS_units', 'profit_NEG_units']
    list_cols.reverse()
    for c in list_cols:
        df_eval.insert(4, c, df_eval.pop(c))
    list_models_percent = [x for x in df_eval.columns if x.startswith("Acert_TFm_" + S + "_")]
    df_eval[list_models_percent] = df_eval[list_models_percent].replace("%", "", regex=True).replace(" ", "0").astype(
        'float')
    type_buy_sell = Op_buy_sell.POS
    df_moldes_S_NP, _ = Utils_scoring.get_models_Multi_not_bads(S, type_buy_sell)
    valid_moldel = get_best_proftis_models(df_eval, S, type_buy_sell, df_moldes_S_NP.columns, NUMBER_MACHTING_TEST_BEST=NUMBER_MACHTING_TEST_BEST)
    print("\nModels Selections stock: " + S + "_" + type_buy_sell.value + " Count: " + str(
        len(valid_moldel)) + "  Names: " + "\t".join(valid_moldel) + "\n")
    valid_moldel = valid_moldel + [i + "_per" for i in valid_moldel]
    df_valids[valid_moldel] = df_moldes_S_NP[valid_moldel]
    type_buy_sell = Op_buy_sell.NEG
    df_moldes_S_NP, _ = Utils_scoring.get_models_Multi_not_bads(S, type_buy_sell)
    valid_moldel = get_best_proftis_models(df_eval,S, type_buy_sell, df_moldes_S_NP.columns, NUMBER_MACHTING_TEST_BEST=NUMBER_MACHTING_TEST_BEST)
    print("\nModels Selections stock: " + S + "_" + type_buy_sell.value + " Count: " + str(
        len(valid_moldel)) + "  Names: " + "\t".join(valid_moldel) + "\n")
    valid_moldel = valid_moldel + [i + "_per" for i in valid_moldel]
    df_valids[valid_moldel] = df_moldes_S_NP[valid_moldel]

def eval_profits_per_of_all_models(S):
    global df_eval
    df_eval = pd.DataFrame()
    df_eval_r = pd.DataFrame()
    for type_buy_sell in [Op_buy_sell.POS, Op_buy_sell.NEG]:
        columns_json = Feature_selection_json_columns.JsonColumns(S, type_buy_sell)

        path_csv_file_SCALA = "d_price/" + S + "_PLAIN_stock_history_" + str(opion.name) + ".csv"
        print(path_csv_file_SCALA)
        df_eval = pd.read_csv(path_csv_file_SCALA, index_col=False, sep='\t')
        # df_eval = raw_df[COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET] + columns_json.get_ALL_Good_and_Low()]
        Logger.logr.info("df loaded from .csv Shape: " + str(df_eval.shape) + " Path: " + path_csv_file_SCALA)

        for type_cols, list_cols in columns_json.get_Dict_JsonColumns().items():
            print("START " + S)
            print(S + "_" + type_buy_sell.value + type_cols)

            for type_mo in LIST_MULTI:
                try:
                    model_h5_name = "TFm_" + S + "_" + type_buy_sell.value + type_cols + type_mo.value
                    print(bcolors.OKBLUE + "\t\t EvalProfit " + model_h5_name + bcolors.ENDC)
                    print(MODEL_FOLDER_TF_MULTI + model_h5_name)
                    if not os.path.isfile(MODEL_FOLDER_TF_MULTI + model_h5_name+".h5"):
                        print("El fichero del modelo no existe, check it ModelPath: "+MODEL_FOLDER_TF_MULTI + model_h5_name+".h5")
                        continue
                    df_eval_r = Model_predictions_handle_Multi_Nrows.create_df_eval_prediction(df_eval, columns_json, df_eval_r, model_h5_name)
                    print(MODEL_FOLDER_TF_MULTI + model_h5_name)
                except Exception as e:
                    print("Exception: ", S, " ", str(e))
    df_eval_r = df_eval_r[120:]
    # df_eval_r = Utils_buy_sell_points.get_buy_sell_points_Roll(df_eval_r, delete_aux_rows=False)
    df_eval_r.to_csv("Models/Eval_multi/Eval_profict_" + S + ".csv", sep='\t', index=None)
    df_eval_r = None




MODEL_FOLDER_TF_MULTI = "Models/TF_multi/"
LIST_MULTI = MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.list()

CSV_NAME = "@CRT"
list_stocks_crt = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@CHIC"
list_stocks_chic =  _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@FOLO3"
list_stocks =  _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# list_stocks = list_stocks + list_stocks_chic
list_stocks = list_stocks + list_stocks_chic + list_stocks_crt # list_stocks +


opion = _KEYS_DICT.Option_Historical.MONTH_3_AD
Y_TARGET = 'buy_sell_point'
COLS_TO_EVAL_FIRSH = ["Close", 'Volume'] # "per_Close", 'has_preMarket',


#EVALUAR EL BENEFICIO DE LOS MODELOS CREAER Models/Eval_multi/Eval_profict_"+S+".csv"
#EVALUAR EL BENEFICIO DE LOS MODELOS CREAER Models/Eval_multi/Eval_profict_"+S+".csv"
for S in  list_stocks:
    eval_profits_per_of_all_models(S)
    print("END"+S)


#SELECIONAR LOS MEJORES MODELOS
#SELECIONAR LOS MEJORES MODELOS
df_valids = pd.DataFrame()

for S in  list_stocks:
    path_csv_file_EVAL = "Models/Eval_multi/Eval_profict_" + S + ".csv"
    add_models_colum_by_best_Profits(path_csv_file_EVAL, NUMBER_MACHTING_TEST_BEST= 3)

df_valids.to_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", sep='\t')
print("\n\nModels/TF_multi/_RESULTS_profit_multi_all.csv")


