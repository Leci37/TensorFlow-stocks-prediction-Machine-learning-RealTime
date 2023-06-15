"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
import pandas as pd
from Utils import Utils_GPU_manage
import traceback
from tensorflow import keras
from keras.backend import set_session


Utils_GPU_manage.compute_resource()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

import Feature_selection_json_columns
import Model_train_TF_onBalance
import Model_train_sklearn_XGB
import Model_train_TF_multi_onBalance
import _KEYS_DICT
from Data_multidimension import Data_multidimension
from _KEYS_DICT import DICT_COMPANYS

from Utils.UtilsL import bcolors


Y_TARGET = 'buy_sell_point'
model_folder = "Models/Sklearn_smote/"


# csv_file_SCALA = "d_price/FAV_SCALA_stock_history_MONTH_3.csv" #"FAV_SCALA_stock_history_L_MONTH_3_sep.csv"#
#TO CONFIGURE
#Columns =['Date', Y_TARGET, 'ticker'] +  MUY_BUENOS_COLUMNAS_TRAINS
#SAV_files_surname = "veryGood_16"
#TO CONFIGURE
#TODO ananalizar en más detalle https://github.com/huseinzol05/Stock-Prediction-Models

'''Para ENTRENAR los distintos tipos de configuracion TF GradientBoost XGBClassifier RandomForestClassifier '''
# def train_model_with_custom_columns(name_model, columns_list, csv_file_SCALA, op_buy_sell : _KEYS_DICT.Op_buy_sell):
#     columns_selection = ['Date', Y_TARGET, 'ticker'] + columns_list
#     print(
#         "GradientBoost XGBClassifier RandomForestClassifier \n DICT_COLUMNS_TYPES: " + name_model + " Columns Selected:" + ', '.join(
#             columns_selection))
#     X_train, X_test, y_train, y_test = Model_train_sklearn_XGB.get_x_y_train_test_sklearn_XGB(columns_selection,path=csv_file_SCALA, op_buy_sell=op_buy_sell)
#
#     for type_mo in _KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list():
#         model_h5_name_k = "TF_" + name_model + type_mo.value+'.h5'
#         print(bcolors.OKBLUE + "\t\t "+model_h5_name_k + bcolors.ENDC)
#         print("START")
#         Model_train_TF_onBalance.train_TF_onBalance_One_dimension(columns_selection, model_h5_name_k,csv_file_SCALA, op_buy_sell=op_buy_sell, type_model_dime=type_mo)
#
#     SAV_surname = name_model
#     print("\nGradientBoost")
#     Model_train_sklearn_XGB.train_GradientBoost(X_train, X_test, y_train, y_test, SAV_surname)
#     print("\nXGBClassifier")
#     Model_train_sklearn_XGB.train_XGBClassifier(X_train, X_test, y_train, y_test, SAV_surname)
#     print("\nRandomForestClassifier")
#     Model_train_sklearn_XGB.train_RandomForestClassifier(X_train, X_test, y_train, y_test, SAV_surname)



def train_MULTI_model_with_custom_columns(name_model, columns_list, csv_file_SCALA, op_buy_sell : _KEYS_DICT.Op_buy_sell):
    columns_selection = ['Date', Y_TARGET] + columns_list

    #LOAD
    multi_data = Data_multidimension(columns_selection, op_buy_sell, path_csv_a= csv_file_SCALA, name_models_stock=name_model)

    df_result = pd.DataFrame()
    list_multi = _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.list()
    # list_multi = [_KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_DENSE2 , _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_DENSE128]
    for type_mo in list_multi : #_KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.list(): #_KEYS_DICT.MODEL_TF_DENSE_TYPE.list():
        model_h5_name_k = "TFm_" + name_model + type_mo.value+'.h5'
        print(bcolors.OKBLUE + "\t\t Train "+model_h5_name_k + bcolors.ENDC)
        try:
            df_result_pers = Model_train_TF_multi_onBalance.train_TF_Multi_dimension_onBalance(multi_data=multi_data,model_h5_name= model_h5_name_k, model_type=type_mo)
            df_result['TFm_' + name_model + type_mo.value] = df_result_pers[0]
            df_result['TFm_' + name_model + type_mo.value+"_per"] = df_result_pers["per_acert"]
        except Exception as ex:
            df_result['r_TFm_' + name_model + type_mo.value] = -1
            print(bcolors.WARNING + "Exception MULTI " + bcolors.ENDC + "_" , ex)
            print(traceback.format_exc(), "\n")

    df_result.to_csv("Models/TF_multi/" + name_model + "_per_score.csv", sep='\t')
    print("Path: Models/TF_multi/" + name_model + "_per_score.csv")
    print("Path: Models/TF_multi/" + name_model + "_per_score.csv")
        # except Exception as ex:
        #     print("Exception MULTI ")
        #     print("Exception MULTI ", ex)
#**DOCU**
#3 train TensorFlow, XGB and Sklearn models
# Train the models, for each action 36 different models are trained.
# 15 minutes per action.
# Run Model_creation_models_for_a_stock.py
#
# For each action the following files are generated:
# Models/Sklearn_smote folder:
# XGboost_AMD_yyy_xxx_.sav
# RandomForest_AMD_yyy_xxx_.sav
# XGboost_AMD_yyy_xxx_.sav
# Models/TF_balance folder:
# TF_AMD_yyy_xxx_zzz.h5
# TF_AMD_yyy_xxx_zzz.h5_accuracy_71.08%__loss_0.59__epochs_10[160].csv
# xxx can take value vgood16 good9 reg4 y low1
# yyy can take value "pos" and "neg".
# zzz can take value s28 s64 and s128
# Check that all files have been generated for each action in the subfolders of /Models.

CSV_NAME = "@CHILL"
list_stocks = DICT_COMPANYS[CSV_NAME]
opion = _KEYS_DICT.Option_Historical.MONTH_3_AD

# CSV_NAME = "@FOLO3"
# list_stocks = [    "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]
# opion = _KEYS_DICT.Option_Historical.MONTH_3_AD
for S in list_stocks:
    path_csv_file_SCALA = "d_price/" + S + "_PLAIN_stock_history_" + str(opion.name) + ".csv"

    for type_buy_sell in [ _KEYS_DICT.Op_buy_sell.NEG , _KEYS_DICT.Op_buy_sell.POS  ]:
        print(" START STOCK: ", S,  " type: ", type_buy_sell, " \t path: ", path_csv_file_SCALA)
        columns_json = Feature_selection_json_columns.JsonColumns(S, type_buy_sell)


        for type_cols, list_cols in columns_json.get_Dict_JsonColumns().items():
            print("START")
            print( S + "_" + type_buy_sell.value + type_cols)
            print("START")
            train_MULTI_model_with_custom_columns(   S + "_" + type_buy_sell.value + type_cols,  list_cols, path_csv_file_SCALA, type_buy_sell )

    # train_model_with_custom_columns(S + "_" + type_detect + _KEYS_DICT.MODEL_TYPE_COLM.VGOOD.value, columns_json.vgood16, path_csv_file_SCALA)
    # train_model_with_custom_columns(S + "_" + type_detect + _KEYS_DICT.MODEL_TYPE_COLM.GOOD.value, columns_json.get_vGood_and_Good(), path_csv_file_SCALA)
    # train_model_with_custom_columns(S + "_" + type_detect + _KEYS_DICT.MODEL_TYPE_COLM.REG.value, columns_json.get_Regulars(), path_csv_file_SCALA)


