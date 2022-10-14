import numpy as np
import pandas as pd
import json

import Feature_selection_get_columns_json
import Model_predictions_TF_sklearn_XGB
import Utils_buy_sell_points
import Utils_model_predict


#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
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
columns_order_entre_to_model = []
df_result_all = None
df_result_all_isValid_to_buy = None



def prepare_df_to_predict(df_in_pure):
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


def __get_df_predictions_from_all_models(df_in_pure, model_name_type,df_result_A = None ,  plot_cm = False, ):

    if df_result_A is None:
        df_result_A  = Utils_model_predict.fill_first_time_df_result_all(df_in_pure)

    X_test, df_result, test_features = prepare_df_to_predict(df_in_pure)


    model_h5_name = 'TF_' + model_name_type + '.h5'
    df_result['r_TF_' + model_name_type] = Model_predictions_TF_sklearn_XGB.predict_TF_onBalance(test_features, MODEL_FOLDER_TF,                                                                            model_h5_name)
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

    df_result_A[['isValid_' + model_name_type, 'r_TF_' + model_name_type, 'r_gbr_' + model_name_type,
                   'r_xgb_' + model_name_type, 'r_rf_' + model_name_type]] = df_result[
        ['isValid_' + model_name_type, 'r_TF_' + model_name_type, 'r_gbr_' + model_name_type,
         'r_xgb_' + model_name_type, 'r_rf_' + model_name_type]]

    return df_result, df_result_A



def __get_df_Evaluation_from_all_models(df_result_all, name_type_result):

    print(MODELS_EVAL_RESULT + "_RESULTS_modles_isValid_" + name_type_result + ".csv")
    name_col_is_valid = [col for col in df_result_all.columns if col.startswith('isValid_')][1:][0]  # get the last one [-1:][0]
    df_result_all.rename(columns={name_col_is_valid: "is_valid"}, inplace=True)
    name_cols_is_valid = [col for col in df_result_all.columns if col.startswith('isValid_')]
    for c in name_cols_is_valid:
        df_result_all.drop(columns=c, inplace=True)

    df_result_all['Date'] = pd.to_datetime(df_result_all['Date'], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    if 'ticker' in df_result_all.columns:
        df_result_all.sort_values(['ticker', 'Date'], ascending=True).round(4).to_csv(MODELS_EVAL_RESULT + "_RESULTS_" + name_type_result + "_all.csv", sep='\t', index=None)
    else:
        df_result_all.sort_values(['Date'], ascending=True).round(4).to_csv(MODELS_EVAL_RESULT + "_RESULTS_" + name_type_result + "_all.csv", sep='\t', index=None)
    pd.to_datetime(df_result_all['Date']).map(pd.Timestamp.timestamp)

    #Responde a ¿cuantas operaciones son buenas? (solo buenas)
    get_list_good_models(df_result_all,groupby_colum =Y_TARGET ,  path= "Models/Scoring/"+name_type_result)
    # Responde a ¿cuantas operaciones no son malas? (buenas y regulares buenas)
    get_list_good_models(df_result_all, groupby_colum="is_valid" ,  path="Models/Scoring/" + name_type_result)

    # df_list_r = df_result_all[['Date', Y_TARGET,"is_valid"] + columns_aux_to_evaluate]
    df_threshold = df_result_all.describe(percentiles=[0.25, 0.5, 0.7, 0.8,0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    df_threshold.round(4).to_csv("Models/Scoring/"+name_type_result +"_when_model_ok_threshold.csv", sep='\t')


    #is_predict_buy_point(df_list_r, df_result_all, df_threshold, list_valid_params)

    print("Models/Scoring/"+name_type_result +"_when_model_ok.csv")




def get_list_good_models(df_result_all ,groupby_colum , path = None):
    rows_test_in_df = int(df_result_all.shape[0] * 0.35)
    df_eval_result_paramans = df_result_all[-rows_test_in_df:].groupby(groupby_colum).mean().T

    if df_eval_result_paramans.shape[1] != 2:
        Logger.logr.warn("El formato despues del group by no corresponde a 2. groupby_colum: "+ groupby_colum+ " Format: "+ str(df_eval_result_paramans.shape)+ " path: "+ path)
        return
    df_eval_result_paramans["r_eval"] = df_eval_result_paramans[1] - df_eval_result_paramans[0]

    #df_json.set_index('count', inplace=True)
    if path is not None:
        list_r_params = [col for col in df_eval_result_paramans.index.values if col.startswith('r_')]
        df_json = df_eval_result_paramans["r_eval"][list_r_params].round(4)
        scoring_sum_only_pos = df_eval_result_paramans["r_eval"][list_r_params][ df_eval_result_paramans["r_eval"][list_r_params] > 0 ].sum()

        list_r_TF_params = [col for col in list_r_params if col.startswith('r_TF_')]
        scoring_sum_TF = df_eval_result_paramans["r_eval"][list_r_TF_params].sum()
        list_r_TF_not_params = [col for col in list_r_params if not col.startswith('r_TF_')]
        scoring_sum_not_TF = df_eval_result_paramans["r_eval"][list_r_TF_not_params].sum()
        scoring_sum = scoring_sum_TF + scoring_sum_not_TF

        dict_json = df_json.to_dict()
        dict_json["score_sum"] = scoring_sum.round(4)
        dict_json["score_sum_only_pos"] = scoring_sum_only_pos.round(4)
        dict_json["score_sum_TF"] = scoring_sum_TF.round(4)
        dict_json["score_sum_not_TF"] = scoring_sum_not_TF.round(4)
        dict_json["rows_num_eval"] = rows_test_in_df
        path = path +"_groupby_"+groupby_colum+"_"+str(int(scoring_sum * 100))+".json"

        list_valid_params = df_eval_result_paramans[df_eval_result_paramans["r_eval"] >= 0].index.values
        list_valid_params = [col for col in list_valid_params if col.startswith('r_')]
        dict_json["list_good_params"] = list_valid_params

        with open(path, 'w') as fp:
            json.dump(dict_json, fp, allow_nan=True, indent=3)
            print("\tget_json_ Scoring path: ", path)
        print(path)

    return list_valid_params , scoring_sum


def get_dict_scoring_evaluation(folder = "Models/Scoring/", extension =".json", contains = "_" +Y_TARGET):
    import os
    #TODO mejorar no me gusta , la manera de obtner el nombre del fichero , y la gestiion del error
    prefixed = [filename for filename in os.listdir(folder) if
                filename.startswith(S + "_" + type_detect) and filename.endswith(extension) and (contains in filename)]
    if len(prefixed) != 1:
        print("No se encuntra UN solo (0 o más de 2)  fichero con la configuracion ")
    path_json = folder + prefixed[0]
    with open(path_json) as json_file:
        dict_scores = json.load(json_file)
    return dict_scores


def is_predict_buy_point(df_result_all, df_threshold, list_valid_params):
    tho_down = 88
    tho_up = 93

    if list_valid_params is None:
        list_valid_params = [col for col in df_A.columns if col.startswith('r_')]

    df_list_r = Utils_model_predict.get_df_for_list_of_result(df_result_all)

    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_" + str(tho_down), value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_" + str(tho_up), value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="have_to_oper", value=False)
    df_list_r.insert(loc=len(df_list_r.columns), column="sum_r_TF", value=0)
    df_list_r.insert(loc=len(df_list_r.columns), column="have_to_oper_TF", value=False)
    count_models_eval = 0
    count_models_eval_TF = 0
    for col_r_ in list_valid_params:
        Threshold_MIN_88 = df_threshold[col_r_][str(tho_down) + "%"].round(4)
        Threshold_MIN_93 = df_threshold[col_r_][str(tho_up) + "%"].round(4)

        df_list_r[col_r_ + "_" + str(tho_down)] = (df_result_all[col_r_] > Threshold_MIN_88).astype(int)
        df_list_r[col_r_ + "_" + str(tho_up)] = (df_result_all[col_r_] > Threshold_MIN_93).astype(int)

        if col_r_.startswith('r_TF'):
            df_list_r["sum_r_TF"] = df_list_r["sum_r_TF"] + df_list_r[col_r_ + "_" + str(tho_down)]+ df_list_r[col_r_ + "_" + str(tho_up)]
            count_models_eval_TF = count_models_eval_TF + 1

        df_list_r["sum_r_" + str(tho_down)] = df_list_r["sum_r_" + str(tho_down)] + df_list_r[col_r_ + "_" + str(tho_down)]
        df_list_r["sum_r_" + str(tho_up)] = df_list_r["sum_r_" + str(tho_up)] + df_list_r[col_r_ + "_" + str(tho_up)]
        count_models_eval = count_models_eval + 1

    print("La evalucion de punto have to oper se hara por encima de Down "+ str(tho_down)+ "% : "+ str(int(count_models_eval / 2 +1)) +" Up "+ str(tho_down)+ "% : "+ str(int(count_models_eval / 2 -1 )))
    df_list_r["have_to_oper"] = (df_list_r["sum_r_" + str(tho_down)] > int(count_models_eval / 2) +1 ) & (df_list_r["sum_r_" + str(tho_up)] > int(count_models_eval / 2 - 1) )
    if count_models_eval_TF > 0:
        df_list_r["have_to_oper_TF"] = df_list_r["sum_r_TF"] >= count_models_eval_TF

    df_list_r = df_list_r[['Date', 'buy_sell_point', 'Close', 'has_preMarket', 'Volume', "sum_r_" + str(tho_down), "sum_r_" + str(tho_up), 'have_to_oper', 'sum_r_TF', 'have_to_oper_TF']]
    return df_list_r



CSV_NAME = "@VOLA"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
type_detect = 'pos'
df_r_list_stocks = pd.DataFrame()

for S in list_stocks : #["SHOP", "PINS", "NIO", "UBER","U",  "TWLO", "TSLA", "SNOW",  "MELI" ]:#list_stocks:
    columns_json = Feature_selection_get_columns_json.JsonColumns(S, type_detect)

    path_csv_file_SCALA = "d_price/" + S + "_SCALA_stock_history_MONTH_3.csv"

    #se puede dar el caso de que se repitan columns_aux_to_evaluate , no eliminar duplicados, el codigo lo gestiona en __prepare_get_all_result_df
    df_A = None
    for type_cols , funtion_cols in  columns_json.get_Dict_JsonColumns().items():
        model_name_type = S + "_" + type_detect + type_cols
        columns_selection_predict = ['Date', Y_TARGET] + funtion_cols + columns_aux_to_evaluate + ['ticker']
        print("model_name_type: " + model_name_type + " Columns Selected:" + ', '.join(columns_selection_predict))

        df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv_file_SCALA,columns_selection_predict)
        df_result, df_A = __get_df_predictions_from_all_models(df, model_name_type, df_result_A=df_A, plot_cm=False)

    __get_df_Evaluation_from_all_models(df_A, S + "_" + type_detect + "_")

    df_threshold = pd.read_csv("Models/Scoring/"+S + "_" + type_detect + "_" +"_when_model_ok_threshold.csv", index_col=0, sep='\t')#How to set in pandas the first column and row as index? index_col=0,

    df_r =  is_predict_buy_point(df_A, df_threshold, None)
    df_r['Close']  = df_r['Close'] + 100
    df_r = Utils_buy_sell_points.get_buy_sell_points_Roll(df_r, delete_aux_rows=False)
    df_r = df_r.dropna(how='any')
    df_r['per_PROFIT_POS'] = df_r['per_PROFIT_POS'].astype(float)


    df_r_list_stocks['ticker_'+S ] = df_r[-400:].groupby("have_to_oper")['per_PROFIT_POS'].sum()
    df_r_list_stocks['tickTF_'+S ] = df_r[-400:].groupby("have_to_oper_TF")['per_PROFIT_POS'].sum()
    print("END")

df_r_list_stocks.T.round(4).to_csv("Models/Scoring/_All_pos_predict_result.csv", sep='\t')
print("END")
