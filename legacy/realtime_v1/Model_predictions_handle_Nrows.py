import pandas as pd

import Feature_selection_json_columns
import Model_predictions_handle
import _KEYS_DICT
from Utils import Utils_model_predict

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
from _KEYS_DICT import Op_buy_sell, DICT_COMPANYS
import yhoo_history_stock
from LogRoot.Logging import Logger

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
#TO CONFIGURE
MODELS_EVAL_RESULT = "Models/all_results/"

#model_folder_TF = "Models/TF_in_balance/"
MODEL_FOLDER_TF = "Models/TF_balance/"
columns_aux_to_evaluate = ["Close", "per_Close", 'has_preMarket', 'Volume'] #necesario para Utils_buy_sell_points.check_buy_points_prediction

df_result_all_isValid_to_buy = None
NUM_MIN_MODLES  = 3
NUM_MIN_MODLES_TF = 1



def get_columns_to_download(stock_id):
    columns_json_POS = Feature_selection_json_columns.JsonColumns(stock_id, Op_buy_sell.POS)
    columns_json_NEG = Feature_selection_json_columns.JsonColumns(stock_id, Op_buy_sell.NEG)
    custom_col_POS_NEG = set(columns_json_POS.get_ALL_Good_and_Low() + columns_json_NEG.get_ALL_Good_and_Low())
    return list(custom_col_POS_NEG)


def __get_has_to_seelBuy_by_Predictions_models(df_tech, S, type_buy_sell, path_csv_result = None):
    dict_score = Model_predictions_handle.get_dict_scoring_evaluation(S, type_buy_sell)
    #existe list_good_params_down y list_good_params_up , la list_good_params_down contiene ambas
    list_good_models =  dict_score['list_good_params_down']
    list_good_models_TF = [col for col in list_good_models if col.startswith('r_TF')]

    if type_buy_sell == Op_buy_sell.POS and ( len(list_good_models) <= NUM_MIN_MODLES and len(list_good_models_TF) < NUM_MIN_MODLES_TF )  :
        Logger.logr.info("The POS models present are less than: "+  str(NUM_MIN_MODLES) + ", STOCK prediction is not performed: "+  S + " Len: "+str(len(list_good_models)) +" Modeles: "+str(list_good_models))
        return None
    if type_buy_sell == Op_buy_sell.NEG and ( len(list_good_models) <= NUM_MIN_MODLES and len(list_good_models_TF) < NUM_MIN_MODLES_TF )  :
        Logger.logr.info("The NEG models present are less than: "+  str(NUM_MIN_MODLES) + ", STOCK prediction is not performed: "+  S + "Len: "+str(len(list_good_models)) +" Modeles: "+str(list_good_models))
        return None

    columns_json = Feature_selection_json_columns.JsonColumns(S, type_buy_sell)

    df_b_s = None
    for type_cols, funtion_cols in columns_json.get_Dict_JsonColumns().items():
        model_name_type = S + "_" + type_buy_sell.value + type_cols
        if any(model_name_type in co for co in list_good_models):

            columns_selection_predict = ['Date', Y_TARGET] + funtion_cols + columns_aux_to_evaluate + ['ticker']
            print("model_name_type: " + model_name_type + " Columns Selected:" + ', '.join(columns_selection_predict))
            # try:
            #     df_tech[columns_selection_predict ]
            # except Exception as ex:
            #     Logger.logr.warning(" Exception Stock: " + S + "  Exception: " + traceback.format_exc())

            df_to_predict = Utils_model_predict.load_and_clean__buy_sell_atack(df_tech, columns_selection_predict, type_buy_sell, Y_TARGET)
            df_ignore , df_b_s = Model_predictions_handle.get_df_predictions_from_all_models_by_Selection(df_to_predict, model_name_type, list_good_models, df_result_A=df_b_s)
        else:
            print("There are no models of class columns_json in the list_good_params list, columns_json, optimal enough, we pass to the next one. Serial models: " + model_name_type)

    # Model_creation_scoring.__get_df_Evaluation_from_all_models(df_A, S + "_" + type_detect + "_")
    df_threshold = pd.read_csv("Models/Scoring/" + S + "_" + type_buy_sell.value + "_" + "_when_model_ok_threshold.csv",
                               index_col=0,sep='\t')  # How to set in pandas the first column and row as index? index_col=0,

    df_b_s = Model_predictions_handle.is_predict_buy_point_bt_scoring_csv(df_b_s, df_threshold, None)
    df_b_s['Date'] = pd.to_datetime(df_b_s['Date'], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    df_b_s.insert(loc=10, column="num_models", value=len(list_good_models))

    if path_csv_result is not None:
        print(path_csv_result )
        df_b_s.round(4).to_csv(path_csv_result, sep='\t',index=None)

    return df_b_s



def get_RealTime_buy_seel_points(stock_id, opion_real_time, NUM_LAST_REGISTERS_PER_STOCK, df_all_generate_history= pd.DataFrame()):

    list_score_POS = Model_predictions_handle.get_dict_scoring_evaluation(stock_id, Op_buy_sell.POS)['list_good_params_down']
    list_score_POS_TF = [col for col in list_score_POS if col.startswith('r_TF')]
    list_score_NEG = Model_predictions_handle.get_dict_scoring_evaluation(stock_id, Op_buy_sell.NEG)['list_good_params_down']
    list_score_NEG_TF = [col for col in list_score_NEG if col.startswith('r_TF')]
    if ( len(list_score_NEG) <= NUM_MIN_MODLES and len(list_score_POS) <= NUM_MIN_MODLES ) \
        or ( len(list_score_POS_TF) < NUM_MIN_MODLES_TF and len(list_score_NEG_TF) < NUM_MIN_MODLES_TF ):
        Logger.logr.info("The POS-NEG models present are less than:"+  str(NUM_MIN_MODLES) + ", or the TFs are less than "+str(NUM_MIN_MODLES_TF)+"  STOCK prediction is not performed: "+  stock_id  )
        return None, None


    custom_columns_POS_NEG = get_columns_to_download(stock_id)
    df_HOLC_ign, df_tech = yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_One(df_all_generate_history, stock_id,
                                                                                             opion_real_time,
                                                                                             generate_csv_a_stock=False,
                                                                                             costum_columns=custom_columns_POS_NEG, add_min_max_values_to_scaler = True)
    df_tech = df_tech[-NUM_LAST_REGISTERS_PER_STOCK:]
    return get_df_comprar_vender_predictions(df_tech, stock_id)


def get_df_comprar_vender_predictions(df_tech, stock_id):
    type_buy_sell = Op_buy_sell.POS
    #error ValueError: Found array with 0 sample(s) (shape=(0, 66)) while a minimum of 1 is required by StandardScaler.
    df_compar = __get_has_to_seelBuy_by_Predictions_models(df_tech, stock_id, type_buy_sell, path_csv_result="Models/LiveTime_results/lastweek_" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    # if df_compar is not None:
    #     Model_predictions_handle.how_much_each_entry_point_earns(df_compar,stock_id,type_buy_sell )
    print("df_compar : " + stock_id + "_" + type_buy_sell.value + "   Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    type_buy_sell = Op_buy_sell.NEG
    df_vender = __get_has_to_seelBuy_by_Predictions_models(df_tech, stock_id, type_buy_sell, path_csv_result="Models/LiveTime_results/lastweek_" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    print("df_vender : " + stock_id + "_" + type_buy_sell.value + "   Models/LiveTime_results/" + stock_id + "_" + type_buy_sell.value + "_" + "_.csv")
    # para evaluar en metodos
    if df_compar is not None:
        df_compar['Close'] = df_tech['Close']
    if df_vender is not None:
        df_vender['Close'] = df_tech['Close']
    return df_compar, df_vender
