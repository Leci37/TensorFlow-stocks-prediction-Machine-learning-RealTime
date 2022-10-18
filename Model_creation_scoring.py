import numpy as np
import pandas as pd
import json

import Feature_selection_get_columns_json
import Model_predictions_TF_sklearn_XGB
import Model_predictions_handle
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

def __get_df_Evaluation_from_all_models_model_ok_thresholdCSV(df_result_all, name_type_result):

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
    df_threshold = df_result_all.describe(percentiles=[0.25, 0.5, 0.7, 0.8,0.88,0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    df_threshold.round(4).to_csv("Models/Scoring/"+name_type_result +"_when_model_ok_threshold.csv", sep='\t')


    #is_predict_buy_point(df_list_r, df_result_all, df_threshold, list_valid_params)

    print("Models/Scoring/"+name_type_result +"_when_model_ok.csv")



def get_list_good_models(df_result_all ,groupby_colum , path = None):
    thorshool_is_valid_model = 0.1 # tiene que tener mas que esto para ser valido

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
        scoring_sum_only_pos = df_eval_result_paramans["r_eval"][list_r_params][ df_eval_result_paramans["r_eval"][list_r_params] > thorshool_is_valid_model ].sum()

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

        list_valid_params = df_eval_result_paramans[df_eval_result_paramans["r_eval"] >= thorshool_is_valid_model].index.values
        list_valid_params = [col for col in list_valid_params if col.startswith('r_')]
        dict_json["list_good_params"] = list_valid_params

        with open(path, 'w') as fp:
            json.dump(dict_json, fp, allow_nan=True, indent=3)
            print("\tget_json_ Scoring path: ", path)
        print(path)

    return list_valid_params , scoring_sum






CSV_NAME = "@VOLA"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
#type_detect = a_manage_stocks_dict.Op_buy_sell.POS #for type_buy_sell in [ a_manage_stocks_dict.Op_buy_sell.NEG , a_manage_stocks_dict.Op_buy_sell.POS  ]:
df_final_list_stocks = pd.DataFrame()

for type_buy_sell in [a_manage_stocks_dict.Op_buy_sell.NEG , a_manage_stocks_dict.Op_buy_sell.POS]:
    for S in list_stocks : #["SHOP", "PINS", "NIO", "UBER","U",  "TWLO", "TSLA", "SNOW",  "MELI" ]:#list_stocks:
        path_csv_file_SCALA = "d_price/" + S + "_SCALA_stock_history_MONTH_3.csv"
        print(" START STOCK scoring: ", S,  " type: ", type_buy_sell.value, " \t path: ", path_csv_file_SCALA)
        columns_json = Feature_selection_get_columns_json.JsonColumns(S, type_buy_sell)


        #se puede dar el caso de que se repitan columns_aux_to_evaluate , no eliminar duplicados, el codigo lo gestiona en __prepare_get_all_result_df
        df_A = None
        for type_cols , funtion_cols in  columns_json.get_Dict_JsonColumns().items():
            model_name_type = S + "_" + type_buy_sell.value + type_cols
            columns_selection_predict = ['Date', Y_TARGET] + funtion_cols + columns_aux_to_evaluate + ['ticker']
            print("model_name_type: " + model_name_type + " Columns Selected:" + ', '.join(columns_selection_predict))

            df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv_file_SCALA,  type_buy_sell , columns_selection_predict)
            df_result, df_A = Model_predictions_handle.get_df_predictions_from_all_models(df, model_name_type, df_result_A=df_A, plot_cm=False)

        __get_df_Evaluation_from_all_models_model_ok_thresholdCSV(df_A, S + "_" + type_buy_sell.value + "_")

        df_threshold = pd.read_csv("Models/Scoring/" + S + "_" + type_buy_sell.value + "_" + "_when_model_ok_threshold.csv", index_col=0, sep='\t')#How to set in pandas the first column and row as index? index_col=0,

        df_r =  Model_predictions_handle.is_predict_buy_point(df_A, df_threshold, None)
        df_r['Close']  = df_r['Close'] + 100.1
        df_r = Utils_buy_sell_points.get_buy_sell_points_Roll(df_r, delete_aux_rows=False)
        df_r = df_r.dropna(how='any')
        if type_buy_sell == a_manage_stocks_dict.Op_buy_sell.POS:
            df_r['per_PROFIT_POS'] = df_r['per_PROFIT_POS'].astype(float)
            df_final_list_stocks['ticker_' + S+ "_" + type_buy_sell.value] = df_r[-400:].groupby("have_to_oper")['per_PROFIT_POS'].sum()
            df_final_list_stocks['tickTF_' + S+ "_" + type_buy_sell.value] = df_r[-400:].groupby("have_to_oper_TF")['per_PROFIT_POS'].sum()

        elif type_buy_sell == a_manage_stocks_dict.Op_buy_sell.NEG:
            df_r['per_PROFIT_NEG'] = df_r['per_PROFIT_NEG'].astype(float)
            df_final_list_stocks['ticker_' + S+ "_" + type_buy_sell.value] = df_r[-400:].groupby("have_to_oper")['per_PROFIT_NEG'].sum()
            df_final_list_stocks['tickTF_' + S+ "_" + type_buy_sell.value] = df_r[-400:].groupby("have_to_oper_TF")['per_PROFIT_NEG'].sum()
    print("END")

    print("Models/Scoring/_All_" + type_buy_sell.value + "_predict_result.csv")
    df_final_list_stocks.T.round(4).to_csv("Models/Scoring/_All_" + type_buy_sell.value + "_predict_result.csv", sep='\t')
    df_final_list_stocks = pd.DataFrame()
print("END")
