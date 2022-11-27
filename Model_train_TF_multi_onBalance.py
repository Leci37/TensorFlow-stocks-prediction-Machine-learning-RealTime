import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
from keras.layers import Flatten

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras

from Model_TF_definitions import ModelDefinition
from Utils import UtilsL, Utils_model_predict, Utils_plotter, LTSM_WindowGenerator, Utils_buy_sell_points
import a_manage_stocks_dict
from Utils.Utils_model_predict import __print_csv_accuracy_loss_models, get_model_summary_format
from Data_multidimension import Data_multidimension

Y_TARGET = 'buy_sell_point'
EPOCHS = 160
BATCH_SIZE = 32
MODEL_FOLDER_TF_MULTI = "Models/TF_multi/"


#DATOS desequilibrados https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def train_TF_Multi_dimension_onBalance(multi_data: Data_multidimension, model_h5_name, model_type : a_manage_stocks_dict.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI):

    array_aux_np, train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels, columns_df = multi_data.get_all_data()
    # TRAIN
    neg, pos = np.bincount(array_aux_np) #(df[Y_TARGET])
    initial_bias = np.log([pos / neg])
    imput_shape = (train_features.shape[1], train_features.shape[2])
    resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
    early_stopping = Utils_model_predict.CustomEarlyStopping(patience=8)

    model = ModelDefinition(shape_inputs_m=imput_shape,num_features_m=len(columns_df)).get_dicts_models_multi_dimension()[model_type]
    model_history = model.fit(
          train_features, train_labels ,
        epochs=EPOCHS,
          steps_per_epoch=resampled_steps_per_epoch,
          callbacks=[early_stopping],  # callbacks=[early_stopping, early_stopping_board],
          validation_data=(val_features, val_labels))


    accuracy , loss , epochs = __print_csv_accuracy_loss_models(MODEL_FOLDER_TF_MULTI, model_h5_name, model_history)
    model.save(MODEL_FOLDER_TF_MULTI + model_h5_name)
    print(" Save model Type MULTI TF: " + model_type.value +"  Path:  ", MODEL_FOLDER_TF_MULTI + model_h5_name)

    pre  = model.predict(test_features).reshape(-1,)
    df_pres = __manage_get_per_results_from_predit_result(pre, test_labels, accuracy, loss, epochs)
    return df_pres
    # df_threshold = df_val_result.describe(percentiles=[0.25, 0.5, 0.7, 0.8, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    # df_threshold.round(4).to_csv("Models/Scoring/" + model_h5_name + "_threshold.csv", sep='\t')
    # print("Models/Scoring/" + model_h5_name + "_threshold.csv")
    # if df_threshold['result']["88%"].round(2) == df_threshold['result']["97%"].round(2) or df_threshold['result']["89%"].round(2) == df_threshold['result']["95%"].round(2):
    #     print("ERROR HUMBRALES NO IGUALES model_type: "+model_type.value + " Model: "+ model_h5_name )
    #     raise "ERROR HUMBRALES NO IGUALES model_type: "+model_type.value + " Model: "+ model_h5_name

    # EXAMPLE


def __manage_get_per_results_from_predit_result(pre, test_labels, accuracy, loss, epochs):
    series_c = pd.Series(pre).describe(percentiles=[0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,0.98]).round(4)
    series_d = pd.Series({'acc_per': accuracy, 'loss': loss, 'epochs': epochs}, name=0)

    df_pres = pd.DataFrame( series_d.append(series_c) )
    df_pres.insert(loc=len(df_pres.columns), column="predict_", value=-1)
    df_pres.insert(loc=len(df_pres.columns), column="acert_", value=-1)
    df_pres.insert(loc=len(df_pres.columns), column="per_acert", value=-1)
    df_ = pd.DataFrame({"validation": test_labels.reshape(-1, ), "result": pd.Series(pre)})
    for idnx_per in [x for x in df_pres.index if x.endswith("%")]:
        rate_score = df_pres.at[idnx_per, 0]
        df_.insert(loc=len(df_.columns), column="predict_" + idnx_per, value=0)
        df_.loc[((df_["result"] > rate_score)), "predict_" + idnx_per] = 1  # & (df_val_result["validation"] == True)
        df_.insert(loc=len(df_.columns), column="acert_" + idnx_per, value=0)
        df_.loc[(df_['validation'] == True) & (df_["result"] > rate_score), "acert_" + idnx_per] = 1

        df_pres.at[idnx_per, "predict_"] = df_["predict_" + idnx_per].value_counts()[1]
        df_pres.at[idnx_per, "acert_"] = df_["acert_" + idnx_per].value_counts()[1]
        df_pres.at[idnx_per, "per_acert"] = (df_pres.at[idnx_per, "acert_"] * 100 / df_pres.at[idnx_per, "predict_"]).round(2)

    df_pres = df_pres.drop(columns=["acert_", "predict_"], errors='ignore')
    return df_pres

