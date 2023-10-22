import pickle
from datetime import datetime
import gc
import os
import tensorflow as tf
import pandas as pd
import numpy as np


from tensorboard.plugins.hparams import api as hp
from sklearn.utils.class_weight import compute_class_weight

import _KEYS_DICT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))
physical_devices = tf.config.experimental.list_physical_devices('CPU')


from utilW3 import buy_sell_point, Feature_selection_create_json, get_info_model_evaluation
from utilW3.data import normalise_data, get_window_data, split_data, set_save_class_weight, get_load_class_weight, remove_strong_correlations_columns
from features_W3_old.extract import uncorrelate_selection , get_tree_correlation
from CONFIG import *

# Disable Tensorflow useless warnings
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

def get_best_model_column(df_f1):
    df_f1.insert(2, 'best', "__")
    for S in df_f1['stock'].unique():
        # df2.loc[((df2['close'] < df2['ichi_senkou_b']) & (df2['close'] > df2['ichi_senkou_a']))
        df_S = df_f1[(df_f1['stock'] == S) & (df_f1['REF_key_model'].str.startswith(REF_MODEL))]
        # 'precision_BS','recall_BS','f1_BS','precision_avg', 'recall_avg','f1_avg'
        df_S['score_quality'] = df_S[['recall_BS', 'f1_BS', 'recall_avg', 'f1_avg']].astype(float).mean(axis=1)
        ref_model_best = df_S[df_S['score_quality'] == df_S['score_quality'].max()].index[0]
        df_f1.loc[df_f1.index == ref_model_best, 'best'] = "**"
        # df_S = df_S[(df_S['score_quality'] == df_S['score_quality'].max())]
        return df_f1

INDI = INDicator_label_3W5_
indicator_timeframe = INDI
# for INDI in list_options_w:
REF_MODEL = INDI+"_WAL_"
LIST_MODELS_TYPE = ["A_182" , "B_256" , "C_128" , "D_128" , "E_64"]
CSV_NAME = "@FAV"
stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
stocks_list = [] #[  "DDOG", "MELI", "TWLO", "UBER", "U","GTLB", "RIVN", "SNOW" ]  + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN"]# "SNOW"  "PYPL", "GTLB", "MDB", "TSLA",
stocks_list = stocks_list + [  "DDOG", "MELI", "TWLO", "UBER", "U","GTLB",  "SNOW" , "PYPL",  "MDB", "TSLA",  "ADSK" , "ADBE"  ] # ERROR "RIVN",
stocks_list = stocks_list + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
stocks_list = stocks_list +[ "LYFT", "UBER", "ZI", "QCOM",  "SPOT", "PTON","CRWD", "NVST", "HUBS" ,  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD" ,  "CRSR","UPST"]

#DEBUG
stocks_list = [  "U", "DDOG", "MELI", "TWLO", "UBER","GTLB",  "SNOW" , "PYPL",  "MDB", "TSLA",  "ADSK" , "ADBE"  ]
stocks_list = stocks_list + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM"]


df_f1 = pd.DataFrame(columns=['date_train', 'stock', 'precision_BS','recall_BS','f1_BS','precision_avg', 'recall_avg','f1_avg','precision_1','precision_2','recall_1','recall_2','f1_score_1','f1_score_2','REF_key_model', 'val_count'])


for symbol in stocks_list : #stocks_list:
    for key_model in LIST_MODELS_TYPE:
        save_model_path = f'outputs/val_data/{symbol}_{REF_MODEL}_X_val_y_val_index_val.data'
        print("Load Validation data : ", save_model_path)
        # TO WRITE with open(f'outputs/val_data/{symbol}_{REF_MODEL}_X_val_y_val_index_val.data', 'wb') as f:
        #     pickle.dump((X_val, y_val, index_val), f)
        X_val, y_val, index_val = pickle.load(open(save_model_path, 'rb'))

        # _, accuracy = model.evaluate(x=X_test, y=y_test)
        print("load model : ", f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5')
        resampled_model_2 = keras.models.load_model(f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5' , compile=False) #https://stackoverflow.com/questions/68923962/unknown-metric-function-please-ensure-this-object-is-passed-to-the-custom-obje
        y_pred_1_probabily = resampled_model_2.predict(X_val, batch_size=64)
        y_pred_1 = np.argmax(np.squeeze(y_pred_1_probabily), axis=1)
        y_test_a = np.argmax(y_val, axis=1)
        print("[11.2] read the evaluation: ", X_val.shape , "\n")
        # reset_random_seeds()
        accuracy , _1 , _2, per_sta_neg_f1_avg = get_info_model_evaluation.get_info_model_evaluation(y_pred_1, y_test_a, symbol=symbol
                                                                               ,indicator_timeframe=f'{REF_MODEL}_{key_model}', retur_info_predict =True)

        K.clear_session() # tf.Session.run() or tf.Tensor.eval(), so your models will become slower and slower to train, and you may also run out of memory. https://stackoverflow.com/questions/50895110/what-do-i-need-k-clear-session-and-del-model-for-keras-with-tensorflow-gpu
        del  resampled_model_2
        gc.collect()
        try:
            #group 1 precison , 2 recall 3 f1_score
            precision_buy_sell_avg = round(np.average([float(_1.group(1)), float(_2.group(1))]), 3)
            recall_buy_sell_avg = round(np.average([float(_1.group(2)), float(_2.group(2))]), 3)
            f1_buy_sell_avg = round(np.average([float(_1.group(3)), float(_2.group(3))]), 3)
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + bcolors.OKBLUE + "\t F1_model: " + str(f1_buy_sell_avg) + bcolors.ENDC + "\tF1_avg:", per_sta_neg_f1_avg.group(3), "\tCount:", per_sta_neg_f1_avg.group(4))
            df_f1_a_model = pd.DataFrame([{'date_train': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'stock': symbol,
                                           'precision_BS': str(precision_buy_sell_avg), 'recall_BS': str(recall_buy_sell_avg), 'f1_BS': str(f1_buy_sell_avg),
                                           'precision_avg': per_sta_neg_f1_avg.group(1),'recall_avg': per_sta_neg_f1_avg.group(2),'f1_avg': per_sta_neg_f1_avg.group(3),
                                           'path': f'outputs/{symbol}_{REF_MODEL}_{key_model}_buysell.h5','precision_1': _1.group(1), 'precision_2': _2.group(1),
                                           'recall_1': _1.group(2), 'recall_2': _2.group(2), 'f1_score_1': _1.group(3),'f1_score_2': _2.group(3),
                                           'REF_key_model': f'{REF_MODEL}_{key_model}', 'val_count': per_sta_neg_f1_avg.group(4)}])
            df_f1_a_model.set_index('path', inplace=True)
            df_f1 = pd.concat([df_f1, df_f1_a_model])
        except Exception as e:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "Exception: ",  str(e))




df_f1 = get_best_model_column(df_f1)

df_f1 = df_f1.apply(pd.to_numeric, errors='ignore')
path = f"../Tutorial/ALL_F1_score_EVAL_{REF_MODEL}.csv"
if os.path.isfile(path):
    df_f1.to_csv(path, sep="\t", mode='a', header=False)
else:
    df_f1.to_csv(path, sep="\t")
print("Created F1_scores: : " + path)