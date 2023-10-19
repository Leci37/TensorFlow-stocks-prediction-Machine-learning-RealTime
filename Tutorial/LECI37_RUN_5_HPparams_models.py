import pickle
from datetime import datetime
import gc
import os
from time import sleep

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.backend import set_session

from Utils import Utils_GPU_manage
Utils_GPU_manage.compute_resource()


from tensorboard.plugins.hparams import api as hp
from sklearn.utils.class_weight import compute_class_weight

import _KEYS_DICT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))
physical_devices = tf.config.experimental.list_physical_devices('CPU')


from utilW3 import buy_sell_point, Feature_selection_create_json, get_info_model_evaluation
from utilW3.data import normalise_data, get_window_data, split_data, set_save_class_weight, get_load_class_weight, \
    remove_strong_correlations_columns, get_best_Y_correlations_columns
from features_W3_old.extract import uncorrelate_selection , get_tree_correlation
from CONFIG import *

# Disable Tensorflow useless warnings
from tensorflow import keras
from tensorflow_addons.metrics import F1Score
from keras.callbacks import ModelCheckpoint, EarlyStopping

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#CONFIGURATE
# INDicator_label_3W51560_ = "_3W51560_"
INDicator_label_3W5_ = "_3W5_"
#CONFIGURATE

# list_options_w =  [INDicator_label_3W5_, INDicator_label_3W51560_ ,INDicator_label_3W515_, INDicator_label_3W560_ ]
# stocks_list = [  "U", "DDOG","UPST", "RIVN", "SNOW", "LYFT",  "GTLB"] + [ "MDB", "HUBS", "TTD", "ASAN",    "APPS" , "DOCN", "SHOP", "RBLX", "NIO"]
stocks_list =  ["RIVN"]

# stocks_list =  ["USDJPY_H1" , "AUDUSD_H1" ]

INDI = INDicator_label_3W5_
indicator_timeframe = INDI
# for INDI in list_options_w:
REF_MODEL = INDI+"_WAL_"

import matplotlib
matplotlib.use('TKAgg')  # avoid error could not find or load the Qt platform plugin windows
import matplotlib.pyplot as plt
def plot_history_data_acc_loss_f1(history, path_png):
    # print(history.history.keys())
    plt.figure(1, figsize=(8, 16)).tight_layout()
    # summarize history for accuracy
    plt.subplot(4, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right') #The strings 'upper left', 'upper right', 'lower left', 'lower right'
    # summarize history for loss
    plt.subplot(4, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    # summarize history for f1_score
    # plt.subplot(4, 1, 3)
    # plt.plot(history.history['f1_macro'])
    # plt.plot(history.history['val_f1_macro'])
    # plt.title('model f1')
    plt.ylabel('f1_score')
    plt.xlabel('epoch')
    # plt.legend(['f1_macro', 'val_f1_macro'], loc='lower right')
    # summarize history for lr
    plt.subplot(4, 1, 4)
    plt.plot(history.history['lr'])
    # plt.title('learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper right')
    plt.subplots_adjust(left=0.2, top=0.9, right=0.9, bottom=0.1, hspace=0.5, wspace=0.8)
    print("PLOT data: ", path_png)
    plt.savefig(path_png)
    plt.close()

def plot_metrics_loss_prc_precision_recall(history, path_png):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric],  label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8,1])
        else:
          plt.ylim([0,1])

    plt.legend()
    print("PLOT data: ", path_png)
    plt.savefig(path_png)
    plt.close()

def get_dict_good_params():
    # {'num_units 1': 182, 'num_units 2': 64, 'dropout': 0.2, 'l2 regularizer': 0.0005, 'optimizer': 'adam'} F1_model: 0.445 F1_model: 0.44499999999999995
    hparams_A = {'HP_NUM_UNITS1': 182, 'HP_NUM_UNITS2': 64, 'HP_DROPOUT': 0.2, 'HP_L2': 0.0005, 'HP_OPTIMIZER': 'adam'}
    # {'num_units 1': 256, 'num_units 2': 128, 'dropout': 0.4, 'l2 regularizer': 0.0005, 'optimizer': 'adam'} F1_model: 0.445  F1_model: 0.41500000000000004
    hparams_B = {'HP_NUM_UNITS1': 256, 'HP_NUM_UNITS2': 128, 'HP_DROPOUT': 0.4, 'HP_L2': 0.0005, 'HP_OPTIMIZER': 'adam'}
    # {'num_units 1': 128, 'num_units 2': 64, 'dropout': 0.3, 'l2 regularizer': 0.0005, 'optimizer': 'adam'} F1_model: 0.39  F1_model: 0.42000000000000004
    hparams_C = {'HP_NUM_UNITS1': 128, 'HP_NUM_UNITS2': 64, 'HP_DROPOUT': 0.3, 'HP_L2': 0.001, 'HP_OPTIMIZER': 'adam'}
    # {'num_units 1': 128, 'num_units 2': 128, 'dropout': 0.2, 'l2 regularizer': 0.0005, 'optimizer': 'adam'} F1_model: 0.405   F1_model: 0.39
    hparams_D = {'HP_NUM_UNITS1': 128, 'HP_NUM_UNITS2': 128, 'HP_DROPOUT': 0.2, 'HP_L2': 0.0005, 'HP_OPTIMIZER': 'adam'}
    # {'num_units 1': 128, 'num_units 2': 128, 'dropout': 0.2, 'l2 regularizer': 0.0005, 'optimizer': 'adam'} F1_model: 0.405   F1_model: 0.39
    hparams_E = {'HP_NUM_UNITS1': 64, 'HP_NUM_UNITS2': 128, 'HP_DROPOUT': 0.3, 'HP_L2': 0.001, 'HP_OPTIMIZER': 'adam'}
    return {"A_182" : hparams_A, "B_256" :hparams_B, "C_128" :hparams_C, "D_128" :hparams_D, "E_64" :hparams_E}
# from tensorflow.keras.models import Sequential
def get_model_optimizer(hparams):
    # ACTIVATION_2_ReLU = tf.keras.layers.LeakyReLU(alpha=0.005)
    model = tf.keras.Sequential([
        # keras.layers.Dense(WINDOWS_SIZE*3, activation=ACTIVATION_2_ReLU, input_shape = input_shape ),
        # keras.layers.BatchNormalization(input_shape=input_shape),
        tf.keras.layers.BatchNormalization(input_shape=input_shape),
        # axis=1 For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
        # keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation="relu"),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation="relu"),  # or elu
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.BatchNormalization(),
        # keras.layers.LeakyReLU(alpha=hparams[HP_L2]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS1] // 1.5, kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation="relu"),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS2], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation="relu"),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT] - 0.1),
        # keras.layers.Dense(8, activation="tanh"), no works
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_shape, activation='softmax')  # , bias_initializer = output_bias_k),
    ])
    '''
    if optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=lr)
    if optimizer_type == 'Adamax':
        optimizer = Adamax(learning_rate=lr)
    if optimizer_type == 'AdamW':
        optimizer = AdamW(learning_rate=lr)'''

    optimizer_name = hparams[HP_OPTIMIZER]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_L2] )
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=hparams[HP_L2])
    return model , optimizer

def get_metrics(num_classes=3):
    return [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        F1Score(name='f1_macro', num_classes=num_classes, average='macro'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]

# from sklearn.metrics import precision_recall_fscore_support # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
tf.config.run_functions_eagerly(True) #requerid for f1_macro(y_true, y_pred)

import tensorflow as tf


# CSV_NAME = "@FAV"
# stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# stocks_list = [  "DDOG", "MELI", "TWLO", "UBER", "U","GTLB", "RIVN", "SNOW" ] # "SNOW"  "PYPL", "GTLB", "MDB", "TSLA",
# stocks_list = ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
# stocks_list =  [  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD", "CRSR", "BABA" , "UPST", "AMZN","AMD" , "ADSK" , "ADBE" ]#"LYFT", "QCOM",  "SPOT", "PTON","CRWD", "NVST", "HUBS", "EPAM",    "NVDA",error
stocks_list = []
# stocks_list = stocks_list + [  "U", "DDOG", "MELI", "TWLO", "UBER","GTLB",  "SNOW" , "PYPL",  "MDB", "TSLA",  "ADSK" , "ADBE"  ] # ERROR "RIVN",
stocks_list = stocks_list + [  "ADBE"  ]
stocks_list = stocks_list + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
stocks_list = stocks_list +[ "LYFT", "UBER", "ZI", "QCOM",  "SPOT", "PTON","CRWD", "NVST", "HUBS" ,  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD" ,  "CRSR","UPST"]

df_f1 = pd.DataFrame(columns=['date_train', 'stock', 'f1_buy_sell_score',      'precision_1',       'precision_2','recall_1',          'recall_2',        'f1_score_1','f1_score_2',     'REF_key_model'])
for symbol in stocks_list : #stocks_list:

    features_X_col = []
    ind  = "5min"
    # [1]  Read ohlcv data
    file_rwa_alpha =  f'../d_price/tutorial_pre_{symbol}_{ind}' + ".csv"
    file_rwa_alpha =  f"../d_price/alpaca/alpaca_{symbol}_{ind}_20231004__20211110.csv"
    file_rwa_alpha =  f"../d_price/alpaca/alpaca_{symbol}_{ind}_.csv"
    # file_rwa_alpha = f'../d_price/{symbol}' + ".csv" # EURUSD_H1
    #TIP to get more olhcv data can use 0_API_alphavantage_get_old_history.py (in this project) or alpaca API
    # df_raw = pd.read_csv(file_rwa_alpha, index_col='Date') #EURUSD_H1
    # df_raw.index = pd.to_datetime(df_raw.index, unit='ms', utc=True) #EURUSD_H1

    df_raw = pd.read_csv(file_rwa_alpha, parse_dates=['Date'], index_col='Date', sep='\t')
    for c in df_raw.columns:
        df_raw = df_raw.rename(columns={c: c.lower()})
    df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
    print("\n[1] Read data ohlcv data  path ", file_rwa_alpha, ' Shape_raw: ', df_raw.shape)


    # [2] Get the 292 tech paterns
    print("\n[2] Calculating Technical indicators. stock: ", symbol )
    # df_bars = features_W3.ta.extract_features(df_raw)#TODO optimise, calculate the 830, It just need the +-90 from the .json
    #TODO review why using the 800 indicators reduces the effectiveness of the 270 indicators.
    from features_W3_old import extract_features_v3
    df_bars = extract_features_v3(df_raw, extra_columns=False)  # IT WORKS   Tech indicators Count:  292
    # from features_W3_old.ta import extract_features
    # df_bars = extract_features(df_raw)# IT does NOT WORKS   Tech indicators Count:  800
    # TIP The technical indicators have been cleaned (LIST_TECH_REMOVE) , but it is possible that some of them may be GET data from the future, in order to make the calculation, these must be discarded for a correct functioning in real time.
    # TIP tip avoid using data older than 2 years, old data not useful data. So old data is bad data ??
    df_bars = df_bars.loc[df_bars.index > '2018-01-01 08:00:00']
    print("[2.1] Calculated Technical indicators. stock: ",symbol," Tech indicator count: ",  df_bars.shape )
    df_bars = df_bars.drop(columns=LIST_TECH_REMOVE+ ['Date','date', 'Date.1','date.1', 'Date1','date1'], errors='ignore') #TODO no generate LIST_TECH_REMOVE, it is bad tech paterns
    # see the diferenet bettewn two list set(df_bars.columns) ^ set(df_S_2.columns)

    # [3] Get the target_Y (ground true)
    print("\n[3] Calculate the target_Y (ground true) ,what is the target to detect?  the highest and lowest peaks stock: ", symbol)
    #TODO How do you know that the clustering GroundTrue (buy, sell, nothing) is similar and correct?
    # Is it necessary to cluster with more labels (buy_A, sell_A , nothing_A, buy_B, sell_B , nothing_B, buy_C, sell_C)? How do I know the number of tags needed ?
    # To solve this question you can use the "K-means Clustering" https://neptune.ai/blog/clustering-algorithms
    target_Y , df_l, _= buy_sell_point.get_buy_sell_point_HT_pp(df_bars.copy(), period=TIME_WINDOW_GT, rolling=TIME_WINDOW_GT * 2)
    # TIP the GT detection should be improved, it only detects the maximum and minimum points in time window,
    # in my opinion THE MOST IMPORTANT it should detect when the candles are "sharpened", to detect the entry of the "sharpening".   more info: https://github.com/Leci37/stocks-prediction-Machine-learning-RealTime-telegram#11-data-collection
    print("[3.1]target_Y.shape: " ,target_Y.shape)
    features_X_ALL = df_bars
    # Create correlation matrix https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    print("[3.2] DEBUG All technical indicators Extracted Count: ",
          len(features_X_ALL.columns))  # ," Names : ",",".join( features_X_ALL.columns ))

    # [4] Do selection best tech params
    print("\n[4] Calculation of correlation strength. What are the +-100 best technical indicators and which are noise. Remove strong correlacted columns")
    print("[4.1]uncorrelate_selection path: ", f'data/columns_select/{symbol}_{REF_MODEL}_columns.pkl')
    features_X_ALL = remove_strong_correlations_columns(features_X_ALL,factor=0.85)
    features_X_col_corr = get_best_Y_correlations_columns(features_X_ALL.copy(), target_Y, num_columns=NUMS_FEATURES)
    features_X_col_tree = get_tree_correlation(features_X_ALL.copy(), target_Y, num_features=18)
    # TIP  all these convolutions have been tried to find out which has the strongest relationship in Tutorial/CONFIG.dict_correlations
    #TODO remove columns that have more than 0.9 correlation between them, as they are redundant. More info help: https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/
    columns_json_BOTH = list(set(list(features_X_col_corr) + list(features_X_col_tree)  ))
    Feature_selection_create_json.created_json_feature_selection(list_all_columns=columns_json_BOTH,path_json=f'data/columns_select/{symbol}_{REF_MODEL}_corr.json')
    features_X = features_X_ALL[columns_json_BOTH]
    print("[4.2] Select the best technical patterns to train with features_W3 Extracted Count: ", len(features_X.columns), " Names : ",",".join(features_X.columns))
    print("[4.3] features_W3 index Dates: from ", features_X.index[0]," to ", features_X.index[-1], " Shape: ", features_X.shape)
    target_Y = target_Y[target_Y.index.isin(features_X.index)]
    features_X = features_X[features_X.index.isin(target_Y.index)]

    # [5] Normalizated
    print("\n[5] For correct training, for correct training the values must be normalised to between 0 and 1  ")
    #TIP   how to normalise this data is stored in file 'data/scalers/{symbol}_{REF_MODEL}_scalex.pkl' for use in the future realtime  .
    print("[5.1] Normalise_data path: ", f'data/scalers/{symbol}_{REF_MODEL}_scalex.pkl')
    features_X, scaleX = normalise_data(features_X, SPLIT_RATIO_X_Y, f'data/scalers/{symbol}_{REF_MODEL}_scalex.pkl')

    # [6] Get Window Data
    print("\n[6] Currently you have for each target_Y value a row of technical indicators, you add a 'window'"
          " to make the decision to predict whether the +-48 rows above will be taken (about 4 hours of previous indicators, splited in 5min).")
    X, y, index = get_window_data(features_X, target_Y, window_size=WINDOWS_SIZE)
    # TIP currently only 5min windows are used, but there is also a multi-window model that collects the 5min, 15min and 60min windows, ask the author
    # Example 'INFO created array x_array Created x_array.shape : (54000, 3, 48, 101) ,EX: 54000 rows, 3 clandle times (5min,15mim.60min) , windows size 48, 101tech parents' Maybe add a one minute candle as well. method get_window_data_3Window()
    print("[6.1] get_window_data Shapes Y:" , y.shape, " X: ", X.shape)

    # [7] SPLIT the Data
    print("\n[7] data split between training and validation ")
    X_train, y_train, X_val, y_val, index_train, index_val = split_data(X, y, index, split_ratio=SPLIT_RATIO_X_Y)
    print("[7.1] Shapes: X_train: ",X_train.shape, " y_train:  ", y_train.shape, " index_train:  ", index_train.shape)
    print("[7.2] Shapes: X_VALITATION : ", X_val.shape, " y_train:  ", y_val.shape, " index_train:  ", index_val.shape)
    # TIP  the tests I did by shuffling the data , destroy the result , I don't understand why there are time windows to save the relationship.

    # [8] Get  class_weight
    print("\n[8] Ground True data are unbalanced  Given that there is a lot of 'do nothing' 0, and very little 'do buy' 1 or 'do sell' 2,"
          " weight balancing is required, to give more importance to the minorities.  ")
    class_weight = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    dict_class_weight = dict(enumerate(class_weight))
    print("[8.1] Class weight  path: ", f'data/class_weight/{symbol}_{REF_MODEL}_class_weight.pkl' , " Dict: " , dict_class_weight)
    set_save_class_weight(dict_class_weight, f'data/class_weight/{symbol}_{REF_MODEL}_class_weight.pkl')
    # TIP tip the balancing to check the unbalance is very necessary, I would also like to add that it penalizes more in the training the false positives,
    # there are other methods for balancing as SMOTE, they have been tested without improvement, in my opinion, I don't like them, they destroy the data, if it is applied it must be very delicate.

    # [9] Create a model TF
    print("\n[9] Creation of the TF model architecture. must respect the input_shape and output_shape and the 'softmax' ,"
          " from there EXPRIMENT combinations ")
    input_shape = X_train.shape[1:]
    output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    print("[9.1] arrays to use the TF model  , input_shape: ", input_shape, " output_shape: ",output_shape)
    #se deben entrenar con los mismos datos
    X_train, y_train, X_test2, y_test2, index_train, index_test2 = split_data(X_train, y_train, index_train, split_ratio=0.34)

    with open(f'outputs/val_data/{symbol}_{REF_MODEL}_X_val_y_val_index_val.data', 'wb') as f:
        pickle.dump((X_val, y_val, index_val), f)
    # _, accuracy = model.evaluate(x=X_test, y=y_test)
    print("[9.2] save_val_data : ", f'outputs/val_data/{symbol}_{REF_MODEL}_X_val_y_val_index_val.data')
    # MODEL
    HP_NUM_UNITS1 = hp.HParam('num_units 1', hp.Discrete([64, 128, 182, 256, 312, 372]))
    HP_NUM_UNITS2 = hp.HParam('num_units 2', hp.Discrete([ 64, 128 ]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([ 0.2, 0.3,0.4,0.45,0.5]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([   'adam', 'sgd'])) #, RMS so bad   'adam',
    HP_L2 = hp.HParam('l2 regularizer', hp.Discrete([.0001, .0005,  .001 ]) ) # hp.RealInterval(.001,  .01))
    HP_ACT = hp.HParam('activation', hp.Discrete([ "tanh", "relu", "selu"]))
    METRIC_ACCURACY = 'val_accuracy'
    #reset_random_seeds()
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_DROPOUT, HP_L2, HP_OPTIMIZER],
             metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')])

    #feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    # feature_layer = keras.layers.Dense(WINDOWS_SIZE*3, activation=ACTIVATION_2_ReLU, input_shape = input_shape )
    def train_test_model(hparams, key_model ):
        global df_f1
        model,optimizer = get_model_optimizer(hparams)
        tf.keras.utils.plot_model(model, to_file=f'outputs/plots/{symbol}_{REF_MODEL}_{key_model}_buysell.png')
        print("[9.3] LOOP print diagram of the TF model Path: ", f'outputs/plots/{symbol}_{REF_MODEL}_{key_model}_buysell.png')

        # opt = tf.optimizers.Adam(learning_rate=0.005)  # https://www.kaggle.com/code/viktortaran/ps-s-3-e-6#%E2%9C%855.12.-Keras
        loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1,min_delta=0.005, min_lr=0.000001, mode='min')
        early_stopping = keras.callbacks.EarlyStopping(patience=11, monitor="val_loss", mode='min', verbose=1 , restore_best_weights=True)
        callbacks_save = ModelCheckpoint(filepath=f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5', monitor='val_loss', verbose=0, save_best_only=True )

        # [10] Trian the model
        print("\n[10] Start training TF model "+ bcolors.OKBLUE  , f'outputs/{symbol}_{REF_MODEL}_buysell.h5' , bcolors.ENDC)
        print("[10.1] To adjust and measure the model, callbacks is required, the callbacks will stop before the overfit. ")
        model.compile(optimizer=optimizer, loss=loss , metrics=get_metrics(num_classes=len(target_Y.columns)) )

        history = model.fit( x=X_train, y=y_train, batch_size=512 ,   epochs=280, class_weight=dict_class_weight,  verbose=0,
                             validation_data=( X_test2, y_test2), callbacks=[reduce_lr_loss,early_stopping, callbacks_save]  )
        print("File history : ", f'outputs/model_info/{symbol}_{REF_MODEL}_{key_model}_history_train.csv')
        pd.DataFrame(history.history).round(5).to_csv(f'outputs/model_info/{symbol}_{REF_MODEL}_{key_model}_history_train.csv')
        plot_metrics_loss_prc_precision_recall(history, path_png=f'outputs/plots/{symbol}_{REF_MODEL}_{key_model}_history_PRC.png')
        plot_history_data_acc_loss_f1(history, path_png=f'outputs/plots/{symbol}_{REF_MODEL}_{key_model}_history_Loss.png')


        # _, accuracy = model.evaluate(x=X_test, y=y_test)
        print("[11.1] save_val_data : ", f'outputs/val_data/{symbol}_{REF_MODEL}_{key_model}_X_val_y_val_index_val.data')
        # resampled_model_2 = keras.models.load_model(f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5', custom_objects={"f1_macro": f1_macro })# compile=False) https://stackoverflow.com/questions/68923962/unknown-metric-function-please-ensure-this-object-is-passed-to-the-custom-obje
        # y_pred_1 = resampled_model_2.predict(X_val, batch_size=6)
        # y_pred_1 = np.argmax(np.squeeze(y_pred_1), axis=1)
        # y_test_a = np.argmax(y_val, axis=1)
        # print("[11.2] read the evaluation: ", X_val.shape , "\n")
        #reset_random_seeds()
        # accuracy , _1 , _2 = get_info_model_evaluation.get_info_model_evaluation(y_pred_1, y_test_a, symbol=symbol,indicator_timeframe=f'{REF_MODEL}_{key_model}')

        # K.clear_session() # tf.Session.run() or tf.Tensor.eval(), so your models will become slower and slower to train, and you may also run out of memory. https://stackoverflow.com/questions/50895110/what-do-i-need-k-clear-session-and-del-model-for-keras-with-tensorflow-gpu
        del model, optimizer, history # , X_val, y_val
        gc.collect()
        sleep(20) # para ayudar a la limpieza
        # from numba import cuda CAUSA ERROR EVITAR
        # cuda.select_device(0)CAUSA ERROR EVITAR
        # cuda.close()CAUSA ERROR EVITAR
        # try:
        #     #group 1 precison , 2 recall 3 f1_score
        #     f1_buy_sell_score = np.median([float(_1.group(3)), float(_2.group(3))])
        #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + bcolors.OKBLUE + "\t F1_model: "+ str(f1_buy_sell_score)+ bcolors.ENDC + "\n")
        #     df_f1_a_model = pd.DataFrame([{'date_train': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'stock': symbol,'f1_buy_sell_score': str(f1_buy_sell_score),
        #                                    'path': f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5','precision_1': _1.group(1), 'precision_2': _2.group(1),
        #                                    'recall_1': _1.group(2), 'recall_2': _2.group(2), 'f1_score_1': _1.group(3),'f1_score_2': _2.group(3), 'REF_key_model': f'{REF_MODEL}_{key_model}'}])
        #     df_f1_a_model.set_index('path', inplace=True)
        #     df_f1 = pd.concat([df_f1, df_f1_a_model])
        #     return np.median(   [float(_1.group(3)), float(_2.group(3))] )# want to measure the pos/neg f1
        # except Exception as e:
        #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "Exception: ",  str(e))
        #     return 0
        return 0.23 #history.history['val_f1_macro'][-1] # for run_hpparams


    def run_hpparams(run_dir, hparams, key_model):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams, key_model)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


    DICT_BEST_HPARAMS = get_dict_good_params()
    session_num = 0
    for key_model, sub_dict in DICT_BEST_HPARAMS.items():
        hparams = {
            HP_NUM_UNITS1: sub_dict['HP_NUM_UNITS1'],
            HP_NUM_UNITS2: sub_dict['HP_NUM_UNITS2'],
            HP_DROPOUT: sub_dict['HP_DROPOUT'],
            HP_L2: sub_dict['HP_L2'],
            HP_OPTIMIZER: sub_dict['HP_OPTIMIZER']
        }
        run_name = "run-%d" % session_num
        print(bcolors.BOLD +datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '--- Starting trial:' + run_name + bcolors.ENDC + bcolors.OKGREEN +  "\tTraing.... " + bcolors.ENDC + "\t\tModel_ref: " + key_model)
        print({h.name: hparams[h] for h in hparams})
        run_hpparams('logs/hparam_tuning/' + run_name, hparams, key_model)
        session_num += 1


    # path = "../Tutorial/ALL_F1_score_EVAL.csv"
    # if os.path.isfile(path):
    #     df_f1.to_csv(path, sep="\t", mode='a', header=False)
    # else:
    #     df_f1.to_csv(path, sep="\t")
    # print("Created F1_scores: : " + path)






    print("\n[9] Creation of the TF model architecture. must respect the input_shape and output_shape and the 'softmax' ,"
          " from there EXPRIMENT combinations ")

print("END")


