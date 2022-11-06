import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras


from Utils import UtilsL, Utils_model_predict, Utils_plotter, LTSM_WindowGenerator, Utils_buy_sell_points
import a_manage_stocks_dict
from Utils.Utils_model_predict import __print_csv_accuracy_loss_models


Y_TARGET = 'buy_sell_point'
EPOCHS = 160
BATCH_SIZE = 2048
MODEL_FOLDER_TF = "Models/TF_balance/"


#DATOS desequilibrados https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def train_TF_LSTM_onBalance(columns_selection  , model_h5_name   , path_csv , op_buy_sell : a_manage_stocks_dict.Op_buy_sell):
    #LOAD
    global train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels
    df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path_csv, op_buy_sell, columns_selection)

    train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(
        df, label_name=Y_TARGET, BACHT_SIZE_LOOKBACK=8)
    # END LOAD

    # TRAIN
    neg, pos = np.bincount(df[Y_TARGET])
    initial_bias = np.log([pos / neg])
    imput_shape = (train_features.shape[1], train_features.shape[2])  # number_of_channels = 1
    model = Utils_model_predict.make_model_TF_multidimension_onbalance_fine_28(input_shape_m=imput_shape, num_features=len(df.columns) )  # train_features.shape[-1])
    print(model.summary())

    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
    print("Loss: {:0.4f}  without output_bias ".format(results[0]))
    # model.predict(train_features[:10])
    model = Utils_model_predict.make_model_TF_multidimension_onbalance_fine_28(input_shape_m=imput_shape , output_bias=initial_bias, num_features=len(df.columns) )
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
    resampled_model = Utils_model_predict.make_model_TF_multidimension_onbalance_fine_28(input_shape_m=imput_shape, num_features=len(df.columns) )
    resampled_model.load_weights(initial_weights)
    # Reset the bias to zero, since this dataset is balanced. TODO why?
    output_layer = resampled_model.layers[-1]
    output_layer.bias.assign([0])
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

    #early_stopping = get_EarlyStopping(model_h5_name)
    early_stopping = Utils_model_predict.CustomEarlyStopping(patience=10)
    #early_stopping_board = get_EarlyStopping_TensorFlowBoard(model_h5_name)

    resampled_history = model.fit(
        resampled_ds,
        epochs=EPOCHS,
        steps_per_epoch=resampled_steps_per_epoch,
        callbacks=[early_stopping], #callbacks=[early_stopping, early_stopping_board],
        validation_data=val_ds)

    __print_csv_accuracy_loss_models(MODEL_FOLDER_TF, model_h5_name, resampled_history)
    resampled_model.save(MODEL_FOLDER_TF + model_h5_name)
    print(" Save model :  ", MODEL_FOLDER_TF + model_h5_name)
