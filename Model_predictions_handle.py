import pandas as pd
import json

import Model_predictions_TF_sklearn_XGB
import Utils_buy_sell_points
import Utils_model_predict
import a_manage_stocks_dict

from LogRoot.Logging import Logger

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
#TO CONFIGURE

#raw_df = raw_df[ Columns ]

MODELS_EVAL_RESULT = "Models/all_results/"





MODEL_FOLDER_TF = "Models/TF_balance/"
columns_aux_to_evaluate = ["Close", "per_Close", 'has_preMarket', 'Volume'] #necesario para Utils_buy_sell_points.check_buy_points_prediction



def __prepare_df_to_predict(df_in_pure):
    df_result = df_in_pure.drop(columns=Y_TARGET).copy()
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
    # Borras desde la penultima cuatro para atras , la posicion de columns_aux_to_evaluate
    if set(df_in_pure.columns[-5:-1]) != set(columns_aux_to_evaluate):
        Logger.logr.warning(
            "Auxiliary measuring columns are not in the expected position of the df. Aux df columns: " + " ".join(
                df_in_pure.columns[-5:-1]))
    df = df_in_pure.iloc[:, 0:len(df_in_pure.columns) - 5]  # elimina las 5 ulitmas columnas
    df['ticker'] = df_in_pure.iloc[:,
                   len(df_in_pure.columns) - 1]  # coge la ultima columna por indice de columna, sea cual sea el nombre
    test_features, test_labels = Utils_model_predict.scaler_Raw_TF_onbalance(df.copy(), Y_TARGET)
    X = df.drop(columns=Y_TARGET)  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]

    X_test = X.copy()

    return X_test, df_result, test_features


def get_df_predictions_from_all_models(df_in_pure, model_name_type, df_result_A = None, plot_cm = False, ):

    if df_result_A is None:
        df_result_A  = Utils_model_predict.fill_first_time_df_result_all(df_in_pure)

    X_test, df_result, test_features = __prepare_df_to_predict(df_in_pure)


    model_h5_name = 'TF_' + model_name_type + '28.h5'
    df_result['r_TF_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_h5_name)
    if plot_cm:
        p_tolerance = 1 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(),result_column_name='r_TF_' + model_name_type,
                                                      path_cm=MODELS_EVAL_RESULT + "_TF_balance_CM_" + model_name_type + "_" + str(p_tolerance) + ".png",SUM_RESULT_2_VALID=p_tolerance, generate_CM_for_each_ticker = False)


    model_64_h5_name_k = "TF_" + model_name_type + '64.h5'
    df_result['r_TF64_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_64_h5_name_k)
    if plot_cm:
        p_tolerance = 1 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(),result_column_name='r_TF_' + model_name_type,
                                                      path_cm=MODELS_EVAL_RESULT + "_TF_balance_CM_" + model_name_type + "_" + str(p_tolerance) + ".png",SUM_RESULT_2_VALID=p_tolerance, generate_CM_for_each_ticker = False)


    #[columns_selection_predict.remove(Y_TARGET)] el remove y la seleccion no parecen necesario pero de desea perservar el orden de las columnas al entrar al modelo
    df_result['r_gbr_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_GradientBoostingRegressor(X_test, model_name_type)
    if plot_cm:
        p_tolerance = 0.5 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_gbr_' + model_name_type,path_cm=MODELS_EVAL_RESULT + "_Gradient_CM_" + model_name_type + "_" + str(
                                                          p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance)


    df_result['r_xgb_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_XGBClassifier(X_test, model_name_type)
    if plot_cm:
        p_tolerance = 0.24 * 2
        Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name='r_xgb_' + model_name_type,
                                                      path_cm=MODELS_EVAL_RESULT + "_XGB_CM_" + model_name_type + "_" + str(p_tolerance) + ".png", SUM_RESULT_2_VALID=p_tolerance)


    df_result['r_rf_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_Random_Forest(X_test, model_name_type)
    # if plot_cm:
    p_tolerance = 0.249 * 2
    df_result['isValid_' + model_name_type] = Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(),result_column_name='r_rf_' + model_name_type,
                                                                                  path_cm=MODELS_EVAL_RESULT + "_RamdonFo_CM_" + model_name_type + "_" + str(p_tolerance) + ".png",SUM_RESULT_2_VALID=p_tolerance)

    df_result_A[['isValid_' + model_name_type, 'r_TF_' + model_name_type, 'r_TF64_' + model_name_type,  'r_gbr_' + model_name_type,
                   'r_xgb_' + model_name_type, 'r_rf_' + model_name_type]] = df_result[
        ['isValid_' + model_name_type, 'r_TF_' + model_name_type, 'r_TF64_' + model_name_type, 'r_gbr_' + model_name_type,
         'r_xgb_' + model_name_type, 'r_rf_' + model_name_type]]

    return df_result, df_result_A

def get_df_predictions_from_all_models_by_Selection(df_in_pure, model_name_type,list_good_models, df_result_A = None,):

    if df_result_A is None:
        df_result_A  = Utils_model_predict.fill_first_time_df_result_all(df_in_pure)

    X_test, df_result, test_features = __prepare_df_to_predict(df_in_pure)


    model_h5_name = 'TF_' + model_name_type + '28.h5'
    if 'r_TF_' + model_name_type in list_good_models:
        df_result['r_TF_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_h5_name)

    model_64_h5_name_k = "TF_" + model_name_type + '64.h5'
    if 'r_TF64_' + model_name_type in list_good_models:
        df_result['r_TF64_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_64_h5_name_k)

    if 'r_gbr_' + model_name_type in list_good_models:
        df_result['r_gbr_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_GradientBoostingRegressor(X_test, model_name_type)

    if 'r_xgb_' + model_name_type in list_good_models:
        df_result['r_xgb_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_XGBClassifier(X_test, model_name_type)

    if 'r_rf_' + model_name_type in list_good_models:
        df_result['r_rf_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_Random_Forest(X_test, model_name_type)

    result_cols = [col for col in df_result.columns if col.startswith('r_')]

    df_result_A[result_cols] = df_result[result_cols]

    return df_result, df_result_A


def get_dict_scoring_evaluation(stockId , type_buy_sell : a_manage_stocks_dict.Op_buy_sell, folder = "Models/Scoring/", extension =".json", contains = "_" +Y_TARGET):
    import os
    #TODO mejorar no me gusta , la manera de obtner el nombre del fichero , y la gestiion del error
    prefixed = [filename for filename in os.listdir(folder) if
                filename.startswith(stockId + "_" + type_buy_sell.value) and filename.endswith(extension) and (contains in filename)]
    if len(prefixed) != 1:
        Logger.logr.error("No se encuntra UN solo (0 o más de 2)  fichero con la configuracion ")
        raise ValueError("No se encuntra UN solo (0 o más de 2)  fichero con la configuracion ")

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

    print("La evaluación del punto have_to_oper se hara por encima de Down "+ str(THO_DOWN)+ "% : "+ str(int(count_models_eval / 2 +1)) +" Up "+ str(THO_UP)+ "% : "+ str(int(count_models_eval / 2 -1 )))

    df_list_r["have_to_oper"] = (df_list_r["sum_r_" + str(THO_DOWN)] > int(count_models_eval / 2) +1 ) & (df_list_r["sum_r_" + str(THO_UP)] > int(count_models_eval / 2 - 1) )
    if count_models_eval_TF > 0:
        df_list_r["have_to_oper_TF"] = df_list_r["sum_r_TF"] >= count_models_eval_TF

    result_cols_binary = [col for col in df_list_r.columns if col.startswith('br_')]

    df_list_r = df_list_r[['Date', 'buy_sell_point', 'Close', 'has_preMarket', 'Volume', "sum_r_" + str(THO_DOWN), "sum_r_" + str(THO_UP),"sum_r_" + str(THO_UPH), 'have_to_oper', 'sum_r_TF', 'have_to_oper_TF'] + result_cols_binary ]
    return df_list_r




def how_much_each_entry_point_earns(df_r, stock_id, type_buy_sell : a_manage_stocks_dict.Op_buy_sell, NUM_LAST_ROWS = -400):
    df_final_list_stocks = pd.DataFrame()
    # aqui es solo evalucion solo hay que tener en cuenta los profit_NEG_units y profit_POS_units
    df_r = Utils_buy_sell_points.get_buy_sell_points_Roll(df_r, delete_aux_rows=False).drop(columns=Y_TARGET)
    df_r = df_r.dropna(how='any')

    if type_buy_sell == a_manage_stocks_dict.Op_buy_sell.POS or type_buy_sell == a_manage_stocks_dict.Op_buy_sell.BOTH:
        df_final_list_stocks['bought_' + stock_id + "_" + type_buy_sell.value + "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper")['profit_POS_units'].sum()
        df_final_list_stocks['boughtTF_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper_TF")['profit_POS_units'].sum()

    if type_buy_sell == a_manage_stocks_dict.Op_buy_sell.NEG or type_buy_sell == a_manage_stocks_dict.Op_buy_sell.BOTH:
        df_final_list_stocks['bought_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper")['profit_NEG_units'].sum()
        df_final_list_stocks['boughtTF_' + stock_id + "_" + type_buy_sell.value+ "_units"] = df_r[NUM_LAST_ROWS:].groupby("have_to_oper_TF")['profit_NEG_units'].sum()

    return df_final_list_stocks