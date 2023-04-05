import pandas as pd
import numpy as np
import json
import os

import Model_predictions_TF_sklearn_XGB
from Utils import Utils_buy_sell_points, Utils_model_predict
import _KEYS_DICT

from LogRoot.Logging import Logger
from Utils.UtilsL import remove_column_name_repeted_last_one
from Utils.UtilsL import bcolors
from Utils import Utils_model_predict

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
#TO CONFIGURE

#raw_df = raw_df[ Columns ]

MODELS_EVAL_RESULT = "Models/all_results/"





MODEL_FOLDER_TF = "Models/TF_balance/"
COLS_TO_EVAL_R = ["Close", "per_Close", 'has_preMarket', 'Volume'] #necesario para Utils_buy_sell_points.check_buy_points_prediction



def __prepare_df_to_predict(df_in_pure, BACHT_SIZE_LOOKBACK = None):
    df_result = df_in_pure.drop(columns=Y_TARGET).copy()
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    if not (all( x in df_in_pure.columns for x in  COLS_TO_EVAL_R) ):
        Logger.logr.warning("Auxiliary measuring columns are NOT df.  Aux df COLS_TO_EVAL_R: " + " ".join(COLS_TO_EVAL_R))
        raise "Auxiliary measuring columns are NOT df.  Aux df COLS_TO_EVAL_R: " + " ".join(COLS_TO_EVAL_R)

    # Borras desde la penultima cuatro para atras , la posicion de columns_aux_to_evaluate
    if set(df_in_pure.columns[-5:-1]) == set(COLS_TO_EVAL_R):
        df = df_in_pure.iloc[:, 0:len(df_in_pure.columns) - 5]  # elimina las 5 ulitmas columnas siempre serán  "Close", "per_Close", 'has_preMarket', 'Volume', ticker
        df['ticker'] = df_in_pure.iloc[:,len(df_in_pure.columns) - 1]  # coge la ultima columna por indice de columna, sea cual sea el nombre

    else:
        print("Multiple values for @ticker, before predicting ")
        df, num_times_is_repeted = remove_column_name_repeted_last_one(COLS_TO_EVAL_R, df_in_pure)
        if num_times_is_repeted != 1:
            Logger.logr.warning("Evaluation columns are not present or are present more than once. Number of times:   "+ str(num_times_is_repeted) + " ".join(COLS_TO_EVAL_R) )
            raise "Evaluation columns are not present or are present more than once. Number of times:   "+ str(num_times_is_repeted) + " ".join(COLS_TO_EVAL_R)

    # Form np arrays of labels and features.
    if BACHT_SIZE_LOOKBACK is not None:
        test_labels, test_features = Utils_model_predict.df_to_df_multidimension_array_3D(df.copy(), BACHT_SIZE_LOOKBACK = 8)
    else:
        test_features, test_labels = Utils_model_predict.scaler_Raw_TF_onbalance(df.copy(), Y_TARGET)

    X_test = df.drop(columns=Y_TARGET).copy() # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    return X_test, df_result, test_features


def get_df_predictions_from_all_models(df_in_pure, model_name_type, df_result_A = None, plot_cm = False, ):

    if df_result_A is None:
        df_result_A  = Utils_model_predict.fill_first_time_df_result_all(df_in_pure)

    # X_test_multi, df_result, test_features_multi = __prepare_df_to_predict(df_in_pure, BACHT_SIZE_LOOKBACK = 8)
    #
    # for type_mo in [ _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_LSTM ] : # _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.list(): #_KEYS_DICT.MODEL_TF_DENSE_TYPE.list():
    #     model_h5_name_k = "TF_" + model_name_type + type_mo.value+'.h5'
    #     print(bcolors.OKBLUE + "\t\t Predict "+model_h5_name_k + bcolors.ENDC)
    #     multi_predicts = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features_multi, MODEL_FOLDER_TF, model_h5_name_k)
    #     #despues de hacer mutidimension , test_features_multi, se pierden unas +- 15 en los registros
    #     df_result.insert(loc=df_result.shape[1], column='r_TF_mul_' + model_name_type + type_mo.value, value=0)
    #     df_result['r_TF_mul_' + model_name_type + type_mo.value][(df_result.shape[0] - multi_predicts.shape[0]): ] = multi_predicts.T[0]
    #
    # return

    X_test, df_result, test_features = __prepare_df_to_predict(df_in_pure)
    for type_mo in _KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list():
        model_h5_name_k = "TF_" + model_name_type + type_mo.value+'.h5'
        print("\t\t "+model_h5_name_k )
        df_result['r_TF_' + model_name_type + type_mo.value] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_h5_name_k)
        if plot_cm:
            p_tolerance = 1 * 2
            Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_TF_' + model_name_type + type_mo.value ,
                                                              path_cm=MODELS_EVAL_RESULT + "_TF_balance_CM_" + model_name_type + type_mo.value + "_" + str(p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance, generate_CM_for_each_ticker = False)


    #[columns_selection_predict.remove(Y_TARGET)] el remove y la seleccion no parecen necesario pero de desea perservar el orden de las columnas al entrar al modelo
    df_result['r_gbr_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_GradientBoostingRegressor(X_test, model_name_type)
    if plot_cm:
        p_tolerance = 0.5 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_gbr_' + model_name_type, path_cm=MODELS_EVAL_RESULT + "_Gradient_CM_" + model_name_type + "_" + str(
                                                          p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance)


    df_result['r_xgb_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_XGBClassifier(X_test, model_name_type)
    if plot_cm:
        p_tolerance = 0.24 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_xgb_' + model_name_type,
                                                          path_cm=MODELS_EVAL_RESULT + "_XGB_CM_" + model_name_type + "_" + str(p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance)


    df_result['r_rf_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_Random_Forest(X_test, model_name_type)
    # if plot_cm:
    p_tolerance = 0.249 * 2
    df_result['isValid_' + model_name_type] = Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_rf_' + model_name_type,
                                                                                                path_cm=MODELS_EVAL_RESULT + "_RamdonFo_CM_" + model_name_type + "_" + str(p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance)

    list_TF_simple_models = ['r_TF_' + model_name_type + x for x in _KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list_values()]
    col_result = ['isValid_' + model_name_type]+ list_TF_simple_models +['r_gbr_' + model_name_type,'r_xgb_' + model_name_type, 'r_rf_' + model_name_type]
    df_result_A[col_result] = df_result[col_result]

    return df_result, df_result_A

def get_df_predictions_from_all_models_by_Selection(df_in_pure, model_name_type,list_good_models, df_result_A = None,):

    if df_result_A is None:
        df_result_A  = Utils_model_predict.fill_first_time_df_result_all(df_in_pure)

    X_test, df_result, test_features = __prepare_df_to_predict(df_in_pure)

    if any( [co.startswith( 'r_TF_' + model_name_type )  for co in list_good_models] ) :
        for type_mo in _KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list():
            if 'r_TF_' + model_name_type+ type_mo.value in list_good_models:
                model_h5_name = "TF_" + model_name_type + type_mo.value+'.h5'
                print(model_h5_name)
                df_result['r_TF_' + model_name_type + type_mo.value] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_h5_name)

    if 'r_gbr_' + model_name_type in list_good_models:
        df_result['r_gbr_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_GradientBoostingRegressor(X_test, model_name_type)

    if 'r_xgb_' + model_name_type in list_good_models:
        df_result['r_xgb_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_XGBClassifier(X_test, model_name_type)

    if 'r_rf_' + model_name_type in list_good_models:
        df_result['r_rf_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_Random_Forest(X_test, model_name_type)

    result_cols = [col for col in df_result.columns if col.startswith('r_')]

    df_result_A[result_cols] = df_result[result_cols]

    return df_result, df_result_A


def get_dict_scoring_evaluation(stockId , type_buy_sell : _KEYS_DICT.Op_buy_sell, folder = "Models/Scoring/", extension =".json", contains = "_" +Y_TARGET):

    #TODO mejorar no me gusta , la manera de obtner el nombre del fichero , y la gestiion del error
    prefixed = [filename for filename in os.listdir(folder) if
                filename.startswith(stockId + "_" + type_buy_sell.value) and filename.endswith(extension) and (contains in filename)]
    if len(prefixed) != 1:
        Logger.logr.error("Not found ONE (0 or more than 2) file with the following settings " + "".join(prefixed))
        raise ValueError("Not found ONE (0 or more than 2) file with the following settings " + "".join(prefixed))

    path_json = folder + prefixed[0]
    Logger.logr.info(path_json)
    with open(path_json) as json_file:
        dict_scores = json.load(json_file)
    return dict_scores


def is_predict_buy_point_bt_scoring_csv(df_Ab, df_threshold, list_valid_params):
    THO_DOWN = 88
    THO_UP = 93
    THO_UPH = 95

    if list_valid_params is None:
        list_valid_params = [col for col in df_Ab.columns if col.startswith('r_')]

    df_list_r = Utils_model_predict.get_df_for_list_of_result(df_Ab)

    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_" + str(THO_DOWN), value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_" + str(THO_UP), value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_" + str(THO_UPH), value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="have_to_oper", value=False)
    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_TF", value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="have_to_oper_TF", value=False)
    count_models_eval = 0
    count_models_eval_TF = 0
    for col_r_ in list_valid_params:
        Threshold_MIN_88 = df_threshold[col_r_][str(THO_DOWN) + "%"].round(4)
        Threshold_MIN_93 = df_threshold[col_r_][str(THO_UP) + "%"].round(4)
        Threshold_MIN_95 = df_threshold[col_r_][str(THO_UPH) + "%"].round(4)

        df_list_r["b"+col_r_ + "_" + str(THO_DOWN)] = (df_Ab[col_r_] > Threshold_MIN_88).astype(int)
        df_list_r["b"+col_r_ + "_" + str(THO_UP)] = (df_Ab[col_r_] > Threshold_MIN_93).astype(int)
        df_list_r["b" + col_r_ + "_" + str(THO_UPH)] = (df_Ab[col_r_] > Threshold_MIN_95).astype(int)

        if col_r_.startswith('r_TF'):
            df_list_r["sum_r_TF"] = df_list_r["sum_r_TF"] + df_list_r["b"+col_r_ + "_" + str(THO_DOWN)]+ df_list_r["b"+col_r_ + "_" + str(THO_UP)]
            count_models_eval_TF = count_models_eval_TF + 1

        df_list_r["sum_r_" + str(THO_DOWN)] = df_list_r["sum_r_" + str(THO_DOWN)] + df_list_r["b"+col_r_ + "_" + str(THO_DOWN)]
        df_list_r["sum_r_" + str(THO_UP)] = df_list_r["sum_r_" + str(THO_UP)] + df_list_r["b"+col_r_ + "_" + str(THO_UP)]
        df_list_r["sum_r_" + str(THO_UPH)] = df_list_r["sum_r_" + str(THO_UPH)] + df_list_r["b" + col_r_ + "_" + str(THO_UPH)]
        count_models_eval = count_models_eval + 1

    print("La evaluación del punto have_to_oper se hará por encima de Down "+ str(THO_DOWN)+ "% : "+ str(int(count_models_eval / 2 +1)) +" Up "+ str(THO_UP)+ "% : "+ str(int(count_models_eval / 2 -1 )))

    df_list_r["have_to_oper"] = (df_list_r["sum_r_" + str(THO_DOWN)] > int(count_models_eval / 2) +1 ) & (df_list_r["sum_r_" + str(THO_UP)] > int(count_models_eval / 2 - 1) )
    if count_models_eval_TF > 0:
        df_list_r["have_to_oper_TF"] = df_list_r["sum_r_TF"] >= count_models_eval_TF

    result_cols_binary = [col for col in df_list_r.columns if col.startswith('br_')]

    df_list_r = df_list_r[['Date', 'buy_sell_point', 'Close', 'has_preMarket', 'Volume', "sum_r_" + str(THO_DOWN), "sum_r_" + str(THO_UP),"sum_r_" + str(THO_UPH), 'have_to_oper', 'sum_r_TF', 'have_to_oper_TF'] + result_cols_binary ]
    return df_list_r




def how_much_each_entry_point_earns(df_r, stock_id, type_buy_sell : _KEYS_DICT.Op_buy_sell, NUM_LAST_ROWS = -400):
    df_final_list_stocks = pd.DataFrame()
    # aqui es solo evalucion solo hay que tener en cuenta los profit_NEG_units y profit_POS_units
    df_r = Utils_buy_sell_points.get_buy_sell_points_Roll(df_r, delete_aux_rows=False).drop(columns=Y_TARGET)
    df_r = df_r.dropna(how='any')

    if type_buy_sell == _KEYS_DICT.Op_buy_sell.POS or type_buy_sell == _KEYS_DICT.Op_buy_sell.BOTH:
        df_final_list_stocks['bought_' + stock_id + "_" + type_buy_sell.value + "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper")['profit_POS_units'].sum()
        df_final_list_stocks['boughtTF_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper_TF")['profit_POS_units'].sum()

    if type_buy_sell == _KEYS_DICT.Op_buy_sell.NEG or type_buy_sell == _KEYS_DICT.Op_buy_sell.BOTH:
        df_final_list_stocks['bought_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper")['profit_NEG_units'].sum()
        df_final_list_stocks['boughtTF_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper_TF")['profit_NEG_units'].sum()

    return df_final_list_stocks