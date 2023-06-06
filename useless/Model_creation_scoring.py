import pandas as pd
import json
import  glob, re
import os

from sklearn.preprocessing import MinMaxScaler

import Feature_selection_json_columns
import Model_predictions_handle
from Utils import Utils_model_predict

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
import _KEYS_DICT
from LogRoot.Logging import Logger


Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
MODELS_EVAL_RESULT = "Models/all_results/"

MODEL_FOLDER_TF = "Models/TF_balance/"
 #necesario para Utils_buy_sell_points.check_buy_points_prediction
COLS_TO_EVAL_R = ["Close", "per_Close", 'has_preMarket', 'Volume']
columns_order_entre_to_model = []
df_result_all = None
df_result_all_isValid_to_buy = None

def __get_df_Evaluation_from_all_models_model_ok_thresholdCSV_json(df_result_all, name_type_result):

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
    get_list_good_models_json(df_result_all, groupby_colum =Y_TARGET, path_json="Models/Scoring/" + name_type_result)
    # Responde a ¿cuantas operaciones no son malas? (buenas y regulares buenas)
    #get_list_good_models(df_result_all, groupby_colum="is_valid" ,  path="Models/Scoring/" + name_type_result)

    # df_list_r = df_result_all[['Date', Y_TARGET,"is_valid"] + columns_aux_to_evaluate]
    df_threshold = df_result_all.describe(percentiles=[0.25, 0.5, 0.7, 0.8,0.88,0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    df_threshold.round(4).to_csv("Models/Scoring/"+name_type_result +"_when_model_ok_threshold.csv", sep='\t')
    print("Models/Scoring/"+name_type_result +"_when_model_ok.csv")

def __get_r_eval_TF_accuracy_from_csv_result(df_eval):
    for index, row in df_eval.iterrows():
        if index.startswith('r_TF_') and index.endswith(tuple(_KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list_values())):
            h5_name = glob.glob("Models/TF_balance/" + index.replace('r_', '') + "*.csv")[0]
            accuracy_per = float(re.search(r'accuracy_(.*)%_', h5_name, re.IGNORECASE).group(1))
            epochs_num = float(re.search(r'epochs_(.*)\[', h5_name, re.IGNORECASE).group(1))
            print(index + " accuracy_per: "+str(accuracy_per) + "% epochs: "+ str(epochs_num))
            df_eval.at[index, 'r_eval'] = accuracy_per / 100
            df_eval.at[index, 'r_epoch'] = epochs_num

    return df_eval


def get_list_good_models_json(df_result_all, groupby_colum, path_json = None):
    THORSHOOL_VALID_MODEL_down = 0.32 # tiene que tener mas que esto para ser valido
    THORSHOOL_VALID_MODEL_up = 0.5
    THORSHOOL_VALID_MODEL_TF_down = 0.18  # 0.25 + 0.5 para indicar que solo se recogen los mayores de 75
    THORSHOOL_VALID_MODEL_TF_up = 0.25
    THORSHOOL_VALID_MODEL_TF_epochs = 19

    rows_test_in_df = int(df_result_all.shape[0] * 0.35)
    df_eval_result_paramans = df_result_all[-rows_test_in_df:].groupby(groupby_colum).mean().T

    if df_eval_result_paramans.shape[1] != 2:
        Logger.logr.warn("El formato despues del group by no corresponde a 2. groupby_colum: " + groupby_colum + " Format: " + str(df_eval_result_paramans.shape) + " path: " + path_json)
        return
    df_eval_result_paramans["r_eval"] = df_eval_result_paramans[1] - df_eval_result_paramans[0]
    df_eval_result_paramans["r_epoch"] = 0
    #Para los TF ya viene dada en el .csv de resultados
    df_eval_result_paramans = __get_r_eval_TF_accuracy_from_csv_result(df_eval_result_paramans)

    #df_json.set_index('count', inplace=True)
    if path_json is not None:
        list_r_params = [col for col in df_eval_result_paramans.index.values if col.startswith('r_')]
        df_json = df_eval_result_paramans["r_eval"][list_r_params].round(4)
        scoring_sum_only_pos = df_eval_result_paramans["r_eval"][list_r_params][ df_eval_result_paramans["r_eval"][list_r_params] > THORSHOOL_VALID_MODEL_down ].sum()

        list_r_TF_params = [col for col in list_r_params if col.startswith('r_TF')]

        df_eval_result_paramans["r_eval"][list_r_TF_params] = df_eval_result_paramans["r_eval"][list_r_TF_params] - 0.5

        scoring_sum_TF = (df_eval_result_paramans["r_eval"][list_r_TF_params]).sum()
        list_r_TF_not_params = [col for col in list_r_params if not col.startswith('r_TF')]
        scoring_sum_not_TF = df_eval_result_paramans["r_eval"][list_r_TF_not_params].sum()
        scoring_sum = scoring_sum_TF + scoring_sum_not_TF

        dict_json = df_json.to_dict()
        dict_json["score_sum"] = scoring_sum.round(4)
        dict_json["score_sum_only_pos"] = scoring_sum_only_pos.round(4)
        dict_json["score_sum_TF"] = scoring_sum_TF.round(4)
        dict_json["score_sum_not_TF"] = scoring_sum_not_TF.round(4)
        dict_json["rows_num_eval"] = rows_test_in_df
        dict_json["THORSHOOL_VALID_MODEL_down__up__TF"] = str(THORSHOOL_VALID_MODEL_down) + "__"+str(THORSHOOL_VALID_MODEL_up)+ "__"+str(THORSHOOL_VALID_MODEL_TF_down)

        list_valid_params_down = df_eval_result_paramans["r_eval"][list_r_TF_not_params][df_eval_result_paramans["r_eval"][list_r_TF_not_params] >= THORSHOOL_VALID_MODEL_down].index.values
        list_valid_params_TF_down = df_eval_result_paramans["r_eval"][list_r_TF_params][df_eval_result_paramans["r_eval"][list_r_TF_params] >= THORSHOOL_VALID_MODEL_TF_down].index.values
        list_valid_params_TF_up = df_eval_result_paramans["r_eval"][list_r_TF_params][df_eval_result_paramans["r_eval"][list_r_TF_params] >= THORSHOOL_VALID_MODEL_TF_up].index.values
        #si tiene muchos pasos de entrenamiento , se baja el humbral de valido .h5 model
        list_valid_params_TF_epochs = df_eval_result_paramans["r_epoch"][list_r_TF_params][(df_eval_result_paramans["r_epoch"][list_r_TF_params] >= THORSHOOL_VALID_MODEL_TF_epochs) & (df_eval_result_paramans["r_eval"][list_r_TF_params] >= (THORSHOOL_VALID_MODEL_TF_down - 0.06))].index.values

        list_valid_params_down = list(set(list(list_valid_params_down) + list(list_valid_params_TF_down) + list(list_valid_params_TF_epochs) ))
        dict_json["list_good_params_down"] = list_valid_params_down
        list_valid_params_up = df_eval_result_paramans[df_eval_result_paramans["r_eval"] >= THORSHOOL_VALID_MODEL_up].index.values
        list_valid_params_up = [col for col in list_valid_params_up if col.startswith('r_')]
        dict_json["list_good_params_up"] = list(set(list(list_valid_params_up) + list(list_valid_params_TF_down)  ))

        #Remove files duplicates
        splited_path = path_json.split('/')
        list_files_duplicates = [filename for filename in os.listdir('/'.join(splited_path[:2])) if filename.startswith(splited_path[2]) and filename.endswith(".json") ]
        [os.remove(os.path.join('/'.join(splited_path[:2]), f)) for f in list_files_duplicates]

        path_json = path_json + "_groupby_" + groupby_colum + "_" + str(int(scoring_sum * 100)) + ".json"
        with open(path_json, 'w') as fp:
            json.dump(dict_json, fp, allow_nan=True, indent=3)
            print("\tget_json_ Scoring path: ", path_json)
        print(path_json)

    # return list_valid_params_down , scoring_sum
#**DOCU**
#4 Evaluate the quality of the predictive models
# From the 36 models created for each OHLCV history of each action, only the best ones will be run in real time, in order to select and evaluate the best ones.
# Run Model_creation_scoring.py
#
# In the Models/Scoring folder
# AMD_yyy__groupby_buy_sell_point_000.json
# AMD_yyy__when_model_ok_threshold.csv
# Check that two have been generated for each action.
CSV_NAME = "@CRT"
list_stocks_crt = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@CHIC"
list_stocks_chic = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@FOLO3"
list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks =  list_stocks + list_stocks_chic#+ list_stocks_chic #list_stocks
# for re in ["ATHE", "O" ,"CARV", 'DXCM','PSEC']:
#     list_stocks.remove(re)
# list_stocks = ["GTLB", "MDB", "NVDA", "AMD" , "ADSK", "AMZN", "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]
opion = _KEYS_DICT.Option_Historical.MONTH_3
df_final_list_stocks = pd.DataFrame()

for type_buy_sell in [_KEYS_DICT.Op_buy_sell.NEG , _KEYS_DICT.Op_buy_sell.POS]:

    for S in  list_stocks :
        path_csv_file_SCALA = "d_price/" + S + "_PLAIN_stock_history_" + str(opion.name) + ".csv"
        print(" START STOCK scoring: ", S,  " type: ", type_buy_sell.value, " \t path: ", path_csv_file_SCALA)
        columns_json = Feature_selection_json_columns.JsonColumns(S, type_buy_sell)


        #se puede dar el caso de que se repitan columns_aux_to_evaluate , no eliminar duplicados, el codigo lo gestiona en __prepare_get_all_result_df
        df_A = None
        for type_cols , funtion_cols in  columns_json.get_Dict_JsonColumns().items():
            model_name_type = S + "_" + type_buy_sell.value + type_cols
            columns_selection_predict = ['Date', Y_TARGET] + funtion_cols + COLS_TO_EVAL_R + ['ticker']
            print("model_name_type: " + model_name_type + " Columns Selected:" + ', '.join(columns_selection_predict))

            df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv_file_SCALA, type_buy_sell, columns_selection_predict)
            sc = MinMaxScaler(feature_range=(_KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER))
            # df_his_stock[df_his_stock.isin([np.nan, np.inf, -np.inf])]
            aux_date_save = df.pop('Date').values  # despues se añade , hay que pasar el sc.fit_transform
            aux_ticker_save = df.pop('ticker').values
            array_stock = sc.fit_transform(df)
            df = pd.DataFrame(array_stock, columns=df.columns)
            # para poner date la primera y ticker la segunda
            df.insert(0, 'ticker', aux_ticker_save)
            df.insert(0, 'Date', aux_date_save)

            df_result, df_A = Model_predictions_handle.get_df_predictions_from_all_models(df, model_name_type, df_result_A=df_A, plot_cm=False)

        __get_df_Evaluation_from_all_models_model_ok_thresholdCSV_json(df_A, S + "_" + type_buy_sell.value + "_")

        df_threshold = pd.read_csv("Models/Scoring/" + S + "_" + type_buy_sell.value + "_" + "_when_model_ok_threshold.csv", index_col=0, sep='\t')#How to set in pandas the first column and row as index? index_col=0,

        df_r =  Model_predictions_handle.is_predict_buy_point_bt_scoring_csv(df_A, df_threshold, None)
        df_f = Model_predictions_handle.how_much_each_entry_point_earns(df_r, S, type_buy_sell, NUM_LAST_ROWS = -1600)
        df_final_list_stocks = pd.concat([df_final_list_stocks, df_f], axis=1)

    print("END")

    print("Models/Scoring/_All_" + type_buy_sell.value + "_predict_result.csv")
    df_final_list_stocks.T.round(4).to_csv("Models/Scoring/_All_" + type_buy_sell.value + "_predict_result.csv", sep='\t')
    df_final_list_stocks = pd.DataFrame()
print("END")
