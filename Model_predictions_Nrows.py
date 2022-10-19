import pandas as pd

import Feature_selection_get_columns_json
import Model_predictions_handle
import Utils_model_predict

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
import a_manage_stocks_dict
import yhoo_history_stock

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
#TO CONFIGURE
MODELS_EVAL_RESULT = "Models/all_results/"

#model_folder_TF = "Models/TF_in_balance/"
MODEL_FOLDER_TF = "Models/TF_balance/"
path= "d_price/@VOLA_SCALA_stock_history_MONTH_3_sep" #"FAV_SCALA_stock_history_L_MONTH_3_sep.csv"#
columns_aux_to_evaluate = ["Close", "per_Close", 'has_preMarket', 'Volume'] #necesario para Utils_buy_sell_points.check_buy_points_prediction

df_result_all_isValid_to_buy = None
NUM_LAST_REGISTERS_PER_STOCK = 5



def get_columns_to_download(stock_id):
    columns_json_POS = Feature_selection_get_columns_json.JsonColumns(stock_id, a_manage_stocks_dict.Op_buy_sell.POS)
    columns_json_NEG = Feature_selection_get_columns_json.JsonColumns(stock_id, a_manage_stocks_dict.Op_buy_sell.NEG)
    custom_col_POS_NEG = set(columns_json_POS.get_ALL_Good_and_Low() + columns_json_NEG.get_ALL_Good_and_Low())
    return list(custom_col_POS_NEG)


def get_has_to_buy_sell_points(df_tech, S, type_buy_sell, path_csv_result = None):

    columns_json = Feature_selection_get_columns_json.JsonColumns(S, type_buy_sell)
    list_good_models = Model_predictions_handle.get_dict_scoring_evaluation(S, type_buy_sell)['list_good_params']

    df_b_s = None
    for type_cols, funtion_cols in columns_json.get_Dict_JsonColumns().items():
        model_name_type = S + "_" + type_buy_sell.value + type_cols
        if any(model_name_type in co for co in list_good_models):

            columns_selection_predict = ['Date', Y_TARGET] + funtion_cols + columns_aux_to_evaluate + ['ticker']
            print("model_name_type: " + model_name_type + " Columns Selected:" + ', '.join(columns_selection_predict))

            df_to_predict = Utils_model_predict.load_and_clean__buy_sell_atack(df_tech, columns_selection_predict, type_buy_sell,Y_TARGET)
            df_ignore , df_b_s = Model_predictions_handle.get_df_predictions_from_all_models_by_Selection(df_to_predict, model_name_type, list_good_models, df_result_A=df_b_s)
        else:
            print("No hay en la lista list_good_params, modelos de clase columns_json, sufientemente optimos, se pasa al siguiente. Modeles serie: " + model_name_type)

    # Model_creation_scoring.__get_df_Evaluation_from_all_models(df_A, S + "_" + type_detect + "_")
    df_threshold = pd.read_csv("Models/Scoring/" + S + "_" + type_buy_sell.value + "_" + "_when_model_ok_threshold.csv",
                               index_col=0,sep='\t')  # How to set in pandas the first column and row as index? index_col=0,

    df_b_s = Model_predictions_handle.is_predict_buy_point(df_b_s, df_threshold, None)
    df_b_s['Date'] = pd.to_datetime(df_b_s['Date'], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    df_b_s.insert(loc=10, column="num_models", value=len(list_good_models))

    if path_csv_result is not None:
        print(path_csv_result )
        df_b_s.round(4).to_csv(path_csv_result, sep='\t',index=None)

    return df_b_s



def get_RealTime_buy_seel_points(stock_id):
    custom_columns_POS_NEG = get_columns_to_download(stock_id)
    df_HOLC_ign, df_tech = yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_One(df_all_generate_history, stock_id,
                                                                                             opion,
                                                                                             generate_csv_a_stock=False,
                                                                                             costum_columns=custom_columns_POS_NEG)
    df_tech = df_tech[-NUM_LAST_REGISTERS_PER_STOCK:]
    type_buy_sell = a_manage_stocks_dict.Op_buy_sell.POS
    df_compar = get_has_to_buy_sell_points(df_tech, stock_id, type_buy_sell,path_csv_result="Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    print("END : " + stock_id + "_" + type_buy_sell.value + "   Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")

    type_buy_sell = a_manage_stocks_dict.Op_buy_sell.NEG
    df_vender = get_has_to_buy_sell_points(df_tech, stock_id, type_buy_sell,path_csv_result="Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    print("END : " + stock_id + "_" + type_buy_sell.value + "   Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")

    return df_compar, df_vender




CSV_NAME = "@VOLA"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]

opion = yhoo_history_stock.Option_Historical.MONTH_3
df_all_generate_history = pd.DataFrame()


df_compar = pd.DataFrame()
df_vender = pd.DataFrame()
# for S in list_stocks: # [ "UBER","U",  "TWLO", "TSLA", "SNOW", "SHOP", "PINS", "NIO", "MELI" ]:#list_stocks:
#     df_compar, df_vender = get_RealTime_buy_seel_points(S)






