"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from Model_TF_definitions import ModelDefinition
from Utils import UtilsL, Utils_model_predict, Utils_plotter
import _KEYS_DICT

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='loss' , #'monitor argument of tf.keras.callbacks.EarlyStopping has 4 values: 'loss','accuracy','val_loss','val_accuracy'.
#     verbose=2,
#     patience=9,
#     mode='auto',#min_delta=1 By default, any change in the performance measure, no matter how fractional, will be considered an improvement
#     restore_best_weights=True)
from Utils.Utils_model_predict import __print_csv_accuracy_loss_models, get_model_summary_format


def get_EarlyStopping(model_h5_name):
    monitor_type = 'val_accuracy'#EarlyStopping has 4 values: 'loss','accuracy','val_loss','val_accuracy'.
    # if _KEYS_DICT.MODEL_TYPE_COLM.VGOOD.value in model_h5_name:
    #     monitor_type = 'accuracy'
    # elif _KEYS_DICT.MODEL_TYPE_COLM.GOOD.value in model_h5_name:
    #     monitor_type = 'val_loss'
    # elif _KEYS_DICT.MODEL_TYPE_COLM.REG.value in model_h5_name:
    #     monitor_type = 'val_loss'

    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor_type , #'monitor argument of tf.keras.callbacks.EarlyStopping has 4 values: 'loss','accuracy','val_loss','val_accuracy'.
        verbose=2,
        patience=12,
        mode='auto',#min_delta=1 By default, any change in the performance measure, no matter how fractional, will be considered an improvement
        restore_best_weights=True)

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
# Comando de apertura de tensorFlow board
# tensorboard  --logdir=C:\Users\Luis\Desktop\LecTrade\LecTrade\Models\logs\TF_MELI_pos_low1_28.h5
def get_EarlyStopping_TensorFlowBoard(model_h5_name):
    path_tf_board = "Models/logs/" + model_h5_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_tf_board, histogram_freq=1, write_graph=True,
                                                          write_images=True)
    return tensorboard_callback


Y_TARGET = 'buy_sell_point'
EPOCHS = 160
BATCH_SIZE = 1024
MODEL_FOLDER_TF = "Models/TF_balance/"
#model_h5_name = 'TF_in_balance.h5'
#train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = None


def train_TF_onBalance_One_dimension(columns_selection  , model_h5_name   , path_csv , op_buy_sell : _KEYS_DICT.Op_buy_sell, type_model_dime : _KEYS_DICT.MODEL_TF_DENSE_TYPE_ONE_DIMENSI):
    #LOAD
    global train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels

    df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv, op_buy_sell, columns_selection)
    neg, pos = np.bincount(df[Y_TARGET])
    initial_bias = np.log([pos / neg])

    df = Utils_model_predict.prepare_to_split_SMOTETomek_and_scaler01(df)
    # graficos de relaciones
    # Utils_plotter.plot_relationdist_main_val_and_all_rest_val(df[["mtum_RSI","mtum_STOCH_k","mtum_STOCH_d", Y_TARGET]],Y_TARGET ,path = model_folder+"plot_relationdistplot_")
    train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(
        df, label_name=Y_TARGET, will_shuffle=True)
    # END LOAD

    # TRAIN
    # model = ModelDefinition(shape_inputs_m=train_features.shape[-1], num_features_m=None).get_dicts_models_One_dimension()[type_model_dime]
    # print(get_model_summary_format(model))
    #
    # results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
    # print("Loss: {:0.4f}  without output_bias ".format(results[0]))
    # model.predict(train_features[:10])
    model = ModelDefinition(shape_inputs_m=train_features.shape[-1], num_features_m=None, output_bias_m= initial_bias).get_dicts_models_One_dimension()[type_model_dime]

    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
    print("Loss: {:0.4f} with output_bias".format(results[0]))
    # model.predict(train_features[:10])
    initial_weights = MODEL_FOLDER_TF + "initial_weights/initial_weights_" + model_h5_name
    model.save_weights(initial_weights)
    print("model.save_weights initial_weights: ", initial_weights)
    resampled_ds = Utils_model_predict.get_resampled_ds_onBalance(train_features, train_labels, bool_train_labels,
                                                                  BATCH_SIZE)
    resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
    # Train on the oversampled data
    # Now try training the model with the resampled data set instead of using class weights to see how these methods compare.
    # Note: Because the data was balanced by replicating the positive examples, the total dataset size is larger, and each epoch runs for more training steps.
    resampled_model = ModelDefinition(shape_inputs_m=train_features.shape[-1], num_features_m=None).get_dicts_models_One_dimension()[type_model_dime]
    resampled_model.load_weights(initial_weights)
    print(get_model_summary_format(resampled_model))
    # Reset the bias to zero, since this dataset is balanced.
    output_layer = resampled_model.layers[-1]
    output_layer.bias.assign([0])
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

    #early_stopping = get_EarlyStopping(model_h5_name)
    early_stopping = Utils_model_predict.CustomEarlyStopping(patience=10)
    #early_stopping_board = get_EarlyStopping_TensorFlowBoard(model_h5_name)

    resampled_history = resampled_model.fit(
        resampled_ds,
        epochs=EPOCHS,
        steps_per_epoch=resampled_steps_per_epoch,
        callbacks=[early_stopping], #callbacks=[early_stopping, early_stopping_board],
        validation_data=val_ds)

    __print_csv_accuracy_loss_models(MODEL_FOLDER_TF, model_h5_name, resampled_history)
    resampled_model.save(MODEL_FOLDER_TF + model_h5_name)
    print(" Save model :  ", MODEL_FOLDER_TF + model_h5_name)




# def train_TF_onBalance_64(columns_selection, model_h5_name,path_csv, op_buy_sell : _KEYS_DICT.Op_buy_sell):
#     # LOAD
#     global train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels
#     df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv, op_buy_sell, columns_selection)
#
#     # graficos de relaciones
#     # Utils_plotter.plot_relationdist_main_val_and_all_rest_val(df[["mtum_RSI","mtum_STOCH_k","mtum_STOCH_d", Y_TARGET]],Y_TARGET ,path = model_folder+"plot_relationdistplot_")
#     train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(
#         df, label_name=Y_TARGET)
#     # END LOAD
#
#     # TRAIN
#     neg, pos = np.bincount(df[Y_TARGET])
#     initial_bias = np.log([pos / neg])
#     model = Utils_model_predict.make_model_TF_onbalance_fine_64(shape_features=train_features.shape[-1])
#     print(model.summary())
#     results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
#     print("Loss: {:0.4f}  without output_bias ".format(results[0]))
#     # model.predict(train_features[:10])
#     model = Utils_model_predict.make_model_TF_onbalance_fine_64(shape_features=train_features.shape[-1],
#                                                                 output_bias=initial_bias)
#     results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
#     print("Loss: {:0.4f} with output_bias".format(results[0]))
#     # model.predict(train_features[:10])
#     initial_weights = MODEL_FOLDER_TF + "initial_weights/initial_weights_" + model_h5_name
#     model.save_weights(initial_weights)
#     print("model.save_weights initial_weights: ", initial_weights)
#     resampled_ds = Utils_model_predict.get_resampled_ds_onBalance(train_features, train_labels, bool_train_labels,
#                                                                   BATCH_SIZE)
#     resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
#     # Train on the oversampled data
#     # Now try training the model with the resampled data set instead of using class weights to see how these methods compare.
#     # Note: Because the data was balanced by replicating the positive examples, the total dataset size is larger, and each epoch runs for more training steps.
#     resampled_model = Utils_model_predict.make_model_TF_onbalance_fine_64(shape_features=train_features.shape[-1])
#     resampled_model.load_weights(initial_weights)
#     # Reset the bias to zero, since this dataset is balanced.
#     output_layer = resampled_model.layers[-1]
#     output_layer.bias.assign([0])
#     val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
#     val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)
#
#     #early_stopping = get_EarlyStopping(model_h5_name)
#     early_stopping = Utils_model_predict.CustomEarlyStopping(patience=10)
#     early_stopping_board = get_EarlyStopping_TensorFlowBoard(model_h5_name)
#
#     resampled_history = resampled_model.fit(
#         resampled_ds,
#         epochs=EPOCHS,
#         steps_per_epoch=resampled_steps_per_epoch,
#         callbacks=[early_stopping], # callbacks=[early_stopping, early_stopping_board],
#         validation_data=val_ds)
#
#     __print_csv_accuracy_loss_models(MODEL_FOLDER_TF, model_h5_name, resampled_history)
#     resampled_model.save(MODEL_FOLDER_TF + model_h5_name)
#     print(" Save model :  ", MODEL_FOLDER_TF + model_h5_name)


def predict_TF_onBalance(X_test,  model_folder, model_h5_name):
    print(" \n", model_folder + model_h5_name)
    resampled_model_2 = keras.models.load_model(model_folder + model_h5_name)
    """### Re-check training history"""
    # plot_metrics(resampled_history)
    """### Evaluate metrics"""
    test_predictions_resampled = resampled_model_2.predict(X_test, batch_size=BATCH_SIZE)
    resampled_results = resampled_model_2.evaluate(X_test, test_labels,
                                                   batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(resampled_model_2.metrics_names, resampled_results):
        print(name, ': ', value)
    print()
    p_tolerance = 0.7
    # for to in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  # [0.45,0.47,0.5,0.53,0.56,0.6]:
    #     p_tolerance = to
    Utils_plotter.plot_cm_TF_imbalance(test_labels, test_predictions_resampled,
                                       path=model_folder + "plot_TFbalance_"+model_h5_name.replace(".h5","")+"_CM_"+ str(p_tolerance) + ".png", p=p_tolerance)
    # Utils_plotter.plot_confusion_matrix(cf_matrix, model_folder + "plot_confusion_matrix.png")
    return  test_predictions_resampled

