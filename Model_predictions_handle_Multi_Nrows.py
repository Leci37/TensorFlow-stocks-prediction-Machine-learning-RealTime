import glob
import re
import os
import sys
# I've seen this error occur when a Python script has infinite (or very deep) recursion and the following code is used to increase the recursion limit:
from _KEYS_DICT import Op_buy_sell, Option_Historical, DICT_COMPANYS, MODEL_TF_DENSE_TYPE_MULTI_DIMENSI, BACHT_SIZE_LOOKBACK

sys.setrecursionlimit(6000)
#https://github.com/tensorflow/tensorflow/issues/48545
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
a = tf.zeros([], tf.float32)


import pandas as pd
import numpy as np
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from tensorflow import keras
import Feature_selection_json_columns
import Model_predictions_handle
import Model_predictions_handle_Nrows
import _KEYS_DICT


#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
from Utils import Utils_model_predict
from Utils.UtilsL import bcolors
from Utils.Utils_model_predict import clean_redifine_df_dummy_ticker, get_model_summary_format
from _KEYS_DICT import Op_buy_sell, DICT_COMPANYS
from LogRoot.Logging import Logger

Y_TARGET = 'buy_sell_point'
COLS_TO_EVAL_FIRSH = ["Close", 'Volume'] # "per_Close", 'has_preMarket',

df_result_all_isValid_to_buy = None
NUM_MIN_MODLES  = 3
NUM_MIN_MODLES_TF = 1

# df_RESULT = pd.read_csv("Models/TF_multi/_RESULTS_multi_all.csv", index_col=0, sep='\t')
df_RESULT = None #df_RESULT = None pd.read_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", index_col=0,sep='\t')
df_SCORing = None # pd.read_csv("Models/TF_multi/_SCORE_ALL_T_multi_all.csv", index_col=0, sep='\t')



REGEX_COLUMNS_MODEL = r"Columns TFm_[\w\-]*.h5:\n((\w*[,]?[ ]?)*)"
def check_columns_selections_Raise(cols_model, model):
    h5_name = glob.glob("Models/TF_multi/" + model + "*.csv")[0]
    f = open(h5_name, "r")
    string_columns = re.search(REGEX_COLUMNS_MODEL, f.read()).group(1)
    if not ((cols_model == string_columns.split(", ")).all()):
        Logger.logr.info("The columns selected for prediction are not the same as those required by the Model: " + h5_name)
        raise ValueError("The columns selected for prediction are not the same as those required by the Model: " + h5_name)


def fill_df_eval_r_with_values(arr_predit, df_eval_r, model_name):
    global df_SCORing
    if df_SCORing is None:
        df_SCORing = pd.read_csv("Models/TF_multi/_SCORE_ALL_T_multi_all.csv", index_col=0, sep='\t')
    list_per = _KEYS_DICT.PERCENTAGES_SCORE
    # list_per.sort(reverse=True)
    df_eval_r.insert(len(df_eval_r.columns), 'Acert_' + model_name, " ")
    for per in list_per:
        Threshold_N = df_SCORing[str(int(per * 100)) + "%"].loc[[model_name]][0]
        # Threshold_N = df_RESULT[model_name][str(int(per * 100)) + "%"]
        df_eval_r.loc[(arr_predit >= Threshold_N).reshape(-1), 'Acert_' + model_name] = str(int(per * 100)) + "%"
    df_eval_r['Score_' + model_name] = np.round(arr_predit, 2)
    df_eval_r['Close'] = df_eval_r['Close'].round(2)
    return df_eval_r

def predict_Multi_models(df_S, model_name, shape_imput_3d, scaler_name_file):

    if df_S.shape[0] < BACHT_SIZE_LOOKBACK +4:
        Logger.logr.warning("Input df is too small to predict more than: "+str(BACHT_SIZE_LOOKBACK +4)+" time rows in a sequence.  Model: " + model_name+ " DF.shape: "+str(df_S.shape))
        raise ValueError("Input df is too small to predict more than: "+str(BACHT_SIZE_LOOKBACK +4)+" time rows in a sequence.  Model: " + model_name+ " DF.shape: "+str(df_S.shape))

    Logger.logr.info(" Load model Type MULTI TF:   Path:  "+ MODEL_FOLDER_TF_MULTI + bcolors.OKBLUE + model_name + bcolors.ENDC + ".h5   Shape df to predict: " + str(shape_imput_3d))
    loaded_model = keras.models.load_model(MODEL_FOLDER_TF_MULTI + model_name + ".h5")
    if shape_imput_3d[1] != loaded_model.input_shape[1] or shape_imput_3d[2] != loaded_model.input_shape[2]:
        Logger.logr.warning("Dimensions of the prediction are not the same in model and in runtime.   Runtime: " + str(shape_imput_3d) + " Model: " + str(loaded_model.input_shape) + " Model_Name: " + str(model_name))
        raise ValueError("Dimensions of the prediction are not the same in model and in runtime.   Runtime: " + str(shape_imput_3d) + " Model: " + str(loaded_model.input_shape) + " Model_Name: " + str(model_name))

    # rows_N_original = df_S.shape[0]
    # BACHT_SIZE_LOOKBACK se añade el min max n veces para un correcto scaler
    # num_col_will_add_min_max__add = ((BACHT_SIZE_LOOKBACK//2) +1)*2
    # df_S = Model_predictions_handle_Nrows.add_min_max_(df_S, S, pass_date_to_timeStamp=True,BACHT_SIZE_LOOKBACK=BACHT_SIZE_LOOKBACK)
    arr_mul_labels, arr_mul_features = Utils_model_predict.df_to_df_multidimension_array_2D(df_S.reset_index(drop=True),BACHT_SIZE_LOOKBACK=BACHT_SIZE_LOOKBACK, will_check_reshaped = False)

    x_features  = Utils_model_predict.scaler_min_max_array(arr_mul_features, path_to_load= _KEYS_DICT.PATH_SCALERS_FOLDER+scaler_name_file+".scal")

    # Despues de add_min_max_Scaler para un correcto Scaler
    # 5.1 pasar los labeles Y_TARGET a array 2D requerido para TF
    # x_features = x_features[x_features.shape[0]-rows_N_original:, :]#retirar las filas de mas del 3D
    predict_features = np.array(x_features).reshape(shape_imput_3d)
    # sc.inverse_transform(predict_features[-1])
    # pd.DataFrame(predict_features)[-rows_N_original:]
    # predict_features = predict_features[:-rows_N_original, :] #get last N rows of array


    # Logger.logr.debug(get_model_summary_format(loaded_model))
    BATCH_SIZE = 256
    predictions_resampled = loaded_model.predict(predict_features, batch_size=BATCH_SIZE)
    # pre = loaded_model.predict(predict_features, batch_size=BATCH_SIZE).reshape(-1, )
    return predictions_resampled



MODEL_FOLDER_TF_MULTI = "Models/TF_multi/"
def split_df_predict_and_df_eval(df_in_pure):
    # PREDICT Model_creation_models_for_a_stock.py
    df_in_pure.reset_index(drop=True, inplace=True)
    # Borras desde la penultima cuatro para atras , la posicion de columns_aux_to_evaluate
    if set(df_in_pure.columns[:len(COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET])]) == set(COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET]):
        # Se eleiminan las columnas auxilires de evalucion , ya que son las primeras del df
        df_S = df_in_pure.iloc[:, len(COLS_TO_EVAL_FIRSH):].copy()
        # se recogen las 4 primeras columnas , serán las de evalucion
        df_eval = df_in_pure.iloc[:, :len(COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET])].copy()
        df_eval = df_eval[['Date', Y_TARGET] + COLS_TO_EVAL_FIRSH]
        df_eval = df_eval[BACHT_SIZE_LOOKBACK-1:].reset_index(drop=True)
        if not pd.to_datetime(df_eval['Date'], unit='s',  errors='coerce').isnull().any():
            df_eval['Date'] = pd.to_datetime(df_eval['Date'], unit='s',  errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError("Columns have not been received in the expected order:  " + ", ".join(COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET]))

    if not pd.to_datetime(df_S.eval('Date'), format="%Y-%m-%d %H:%M:%S", errors='coerce').isnull().any():
        df_S['Date'] = pd.to_datetime(df_S['Date']).map(pd.Timestamp.timestamp)
    return df_S, df_eval

def create_df_eval_prediction(df_in_pure, columns_json, df_eval_r, model_name):
    stock, pos_neg, col_select, type_multi = Utils_model_predict.get_config_from_name_model(model_name)
    cols_model = columns_json.get_Dict_JsonColumns()["_" + col_select + "_"]
    cols_model = COLS_TO_EVAL_FIRSH + ['Date', Y_TARGET] + cols_model
    df_in_pure = df_in_pure[cols_model]
    Logger.logr.info('Load model NameModel: ' +model_name+' Neg/Pos: ' + pos_neg + ' df.shape: ' + str(df_in_pure.shape) +" Columns: " + col_select  )

    df_S, df_eval = split_df_predict_and_df_eval(df_in_pure)
    df_eval_r[['Date', Y_TARGET] + COLS_TO_EVAL_FIRSH] = df_eval[['Date', Y_TARGET] + COLS_TO_EVAL_FIRSH]
    check_columns_selections_Raise(df_S.columns, model_name)
    shape_imput_3d = (-1, BACHT_SIZE_LOOKBACK, len(df_S.columns) - 1)
    arr_predit = predict_Multi_models(df_S, model_name, shape_imput_3d=shape_imput_3d,scaler_name_file=stock + "_" + pos_neg + "_" + col_select + "_")

    if df_eval.shape[0] != arr_predit.shape[0]:
        raise ValueError("The number of prediction rows do not coincide with the evaluation rows. Eval: " + str(df_eval.shape[0]) + "Predict: " + str(arr_predit.shape[0]))

    df_eval_r = fill_df_eval_r_with_values(arr_predit, df_eval_r, model_name)
    return df_eval_r


def get_df_Multi_comprar_vender_predictions(df_tech,  stock_id, path_result_eval = None ):
    global df_RESULT
    if df_RESULT is None:
        df_RESULT = pd.read_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", index_col=0, sep='\t')
    list_models_to_predict_POS = [x for x in df_RESULT.columns if x.startswith("TFm_" + stock_id + "_" + Op_buy_sell.POS.value)]
    list_models_to_predict_NEG = [x for x in df_RESULT.columns if x.startswith("TFm_" + stock_id + "_" + Op_buy_sell.NEG.value)]

    if (not list_models_to_predict_POS) and (not list_models_to_predict_NEG):
        Logger.logr.warning("There are no models of class columns_json in the list_good_params list, columns_json, optimal enough, we pass to the next one. Stock: " + stock_id)
        return

    list_models_to_predict_POS_notPer = [x for x in list_models_to_predict_POS if not x.endswith("_per")]
    list_models_to_predict_NEG_notPer = [x for x in list_models_to_predict_NEG if not x.endswith("_per")]
    Logger.logr.info("Stock: " + stock_id + " uses the following models for MULTI prediction: " + stock_id + ":\n\t\tPOS: " +", ".join(list_models_to_predict_POS_notPer) +"\n\t\tNEG: " +", ".join(list_models_to_predict_NEG_notPer) )

    df_eval_r = pd.DataFrame(columns=['Date', Y_TARGET] + COLS_TO_EVAL_FIRSH)

    type_buy_sell = Op_buy_sell.POS
    columns_json = Feature_selection_json_columns.JsonColumns(stock_id, type_buy_sell)
    for model_name in list_models_to_predict_POS_notPer:
        df_eval_r = create_df_eval_prediction(df_tech, columns_json, df_eval_r, model_name)

    type_buy_sell = Op_buy_sell.NEG
    columns_json = Feature_selection_json_columns.JsonColumns(stock_id, type_buy_sell)
    for model_name in list_models_to_predict_NEG_notPer:
        df_eval_r = create_df_eval_prediction(df_tech ,columns_json, df_eval_r, model_name)

    if  path_result_eval is not None:
        df_eval_r.to_csv(path_result_eval, sep='\t')
        print("resiults path: "+ path_result_eval)
    return df_eval_r


def get_PLAINT_df_to_predict_eval_local(path_csv):
    if not "_PLAIN_" in path_csv:
        Logger.logr.error('The input data must not have any scaling on the input to be correctly scaled. Path: ' + path_csv)
        raise ValueError('The input data must not have any scaling on the input to be correctly scaled. Path: ' + path_csv)
    df_in_pure = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv, op_buy_sell=Op_buy_sell.BOTH)
    df_in_pure = df_in_pure[-NUM_LAST_REGISTERS_PER_STOCK:]
    return df_in_pure

NUM_LAST_REGISTERS_PER_STOCK = 32
# NUM_LAST_REGISTERS_PER_STOCK = 1200
# opion = _KEYS_DICT.Option_Historical.MONTH_3
# #"GOOG","MSFT", "TSLA","UPST", "MELI", "TWLO", "RIVN", "SNOW", "LYFT", "ADBE", "UBER", "ZI", "QCOM", "PYPL", "SPOT", "GTLB", "MDB", "NVDA", "AMD" ,
# list_stocks =  [  "ADSK", "AMZN", "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY",  "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]
# # list_stocks =  [  "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]
# CSV_NAME = "@FOLO3"
# list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# for S in list_stocks:
#     df_tech_plain = get_PLAINT_df_to_predict_eval_local( path_csv = "d_price/" + S + "_PLAIN_stock_history_" + str(opion.name) + ".csv")
#     get_df_Multi_comprar_vender_predictions(df_tech_plain, S, path_result_eval="Models/Eval_multi/multi_"+S +".csv")
# print("E")

