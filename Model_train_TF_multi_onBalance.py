import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
from keras.layers import Flatten

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras

from Model_TF_definitions import ModelDefinition
from Utils import UtilsL, Utils_model_predict, Utils_plotter, LTSM_WindowGenerator, Utils_buy_sell_points
import _KEYS_DICT
from Utils.Utils_model_predict import __print_csv_accuracy_loss_models, get_model_summary_format
from Data_multidimension import Data_multidimension

Y_TARGET = 'buy_sell_point'
EPOCHS = 90
BATCH_SIZE = 64
MODEL_FOLDER_TF_MULTI = "Models/TF_multi/"


#DATOS desequilibrados https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def train_TF_Multi_dimension_onBalance(multi_data: Data_multidimension, model_h5_name, model_type : _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI):

    # 1.0  LOAD the data with TF format (split, 3D, SMOTE and balance)
    array_aux_np, train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = multi_data.get_all_data()
    # TRAIN
    neg, pos = np.bincount(array_aux_np) #(df[Y_TARGET])
    initial_bias = np.log([pos / neg])

    # 2.0  STOP EARLY load and created a CustomEarlyStopping to avoid overfit
    resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
    early_stopping = Utils_model_predict.CustomEarlyStopping(patience=8)

    # 3.0 TRAIN get the model from list of models objects and train
    model = multi_data.get_dicts_models_multi_dimension(model_type)
    model_history = model.fit(
          x=train_features, y=train_labels ,
          epochs=EPOCHS,
          steps_per_epoch=resampled_steps_per_epoch,
          callbacks=[early_stopping],  # callbacks=[early_stopping, early_stopping_board],
          validation_data=(val_features, val_labels),  verbose=0)

    # 3.1 Save the model to reuse into .h5 file
    model.save(MODEL_FOLDER_TF_MULTI + model_h5_name)
    print(" Save model Type MULTI TF: " + model_type.value +"  Path:  ", MODEL_FOLDER_TF_MULTI + model_h5_name)

    # 4.0 Eval de model with test_features this data had splited , and the .h5 model never see it
    predit_test  = model.predict(test_features).reshape(-1,)

    # 4.1 generate stadistic  abaut the test predictions
    df_pres = __manage_get_per_results_stadistic_from_predit_result(predit_test, test_labels)
    #Put prety format to df_pres and generate model_history model
    per_df_test = df_pres["per_acert"].loc[['88%', '89%', '90%', '91%', '92%', '93%']].mean().round(1)
    print("Evaluation with df_test.  Name: " + model_type.value  + '\tPrecision:' + str(per_df_test) + '% \tShape:' + str(test_features.shape) )
    accuracy, loss, epochs = __print_csv_accuracy_loss_models(MODEL_FOLDER_TF_MULTI, model_h5_name, model_history,multi_data.cols_df.values, per_df_test)
    series_d = pd.Series({'acc_per': accuracy, 'loss': loss, 'epochs': epochs, 'df_test_%':per_df_test, 'positives':np.unique(test_labels, return_counts=True)[1][1]  }, name=0)
    df_pres = pd.DataFrame(  pd.concat([series_d, df_pres])).fillna(-1)

    return df_pres
    # df_threshold = df_val_result.describe(percentiles=[0.25, 0.5, 0.7, 0.8, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    # df_threshold.round(4).to_csv("Models/Scoring/" + model_h5_name + "_threshold.csv", sep='\t')
    # print("Models/Scoring/" + model_h5_name + "_threshold.csv")
    # if df_threshold['result']["88%"].round(2) == df_threshold['result']["97%"].round(2) or df_threshold['result']["89%"].round(2) == df_threshold['result']["95%"].round(2):
    #     print("ERROR HUMBRALES NO IGUALES model_type: "+model_type.value + " Model: "+ model_h5_name )
    #     raise "ERROR HUMBRALES NO IGUALES model_type: "+model_type.value + " Model: "+ model_h5_name

    # EXAMPLE


def __manage_get_per_results_stadistic_from_predit_result(pre, test_labels):
    df_re_dftest = pd.DataFrame({"validation": test_labels.reshape(-1, ), "result": pd.Series(pre)})

    df_pres = pd.DataFrame(pre).describe(percentiles=_KEYS_DICT.PERCENTAGES_SCORE).round(4)
    # series_d = pd.Series({'acc_per': accuracy, 'loss': loss, 'epochs': epochs, 'positives':df_re_dftest["validation"].value_counts()[1] }, name=0)
    # df_pres = pd.DataFrame(  pd.concat([series_d, series_c]))
    df_pres.insert(loc=len(df_pres.columns), column="df_test", value="-1")
    df_pres.insert(loc=len(df_pres.columns), column="per_acert", value=-1)
    df_pres.insert(loc=len(df_pres.columns), column="predict_", value=-1)
    df_pres.insert(loc=len(df_pres.columns), column="acert_", value=-1)


    for idnx_per in [x for x in df_pres.index if x.endswith("%")]:
        rate_score = df_pres.at[idnx_per, 0]
        df_re_dftest.insert(loc=len(df_re_dftest.columns), column="predict_" + idnx_per, value=0)
        df_re_dftest.loc[((df_re_dftest["result"] > rate_score)), "predict_" + idnx_per] = 1  # & (df_val_result["validation"] == True)
        df_re_dftest.insert(loc=len(df_re_dftest.columns), column="acert_" + idnx_per, value=0)
        df_re_dftest.loc[(df_re_dftest['validation'] == True) & (df_re_dftest["result"] > rate_score), "acert_" + idnx_per] = 1

        df_pres.at[idnx_per, "predict_"] = df_re_dftest["predict_" + idnx_per].value_counts()[1]
        #To avoid when dont have [1] as result
        if len(df_re_dftest["acert_" + idnx_per].value_counts()) > 1:
            df_pres.at[idnx_per, "acert_"]= df_re_dftest["acert_" + idnx_per].value_counts()[1]
            df_pres.at[idnx_per, "df_test"] = str(df_re_dftest["acert_" + idnx_per].value_counts()[1]) + "_in_" + str(df_re_dftest["predict_" + idnx_per].value_counts()[1])
            df_pres.at[idnx_per, "per_acert"] = (df_pres.at[idnx_per, "acert_"] * 100 / df_pres.at[idnx_per, "predict_"]).round(2)
        else:
            df_pres.at[idnx_per, "acert_"] = 0
            df_pres.at[idnx_per, "df_test"] = str(np.nan) + "_in_" + str(df_re_dftest["predict_" + idnx_per].value_counts()[1])
            df_pres.at[idnx_per, "per_acert"] = 0 # (df_pres.at[idnx_per, "acert_"] * 100 / df_pres.at[idnx_per, "predict_"]).round(2)


    df_pres = df_pres.drop(columns=["acert_", "predict_"], errors='ignore')
    return df_pres

