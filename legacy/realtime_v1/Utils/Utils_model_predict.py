import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from Utils import UtilsL
from Utils import Utils_buy_sell_points
from Utils import Utils_col_sele

import _KEYS_DICT
from LogRoot.Logging import Logger

METRICS_ACCU_PRE = [
      #keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.BinaryAccuracy(name='accuracy')
      #keras.metrics.Precision(name='precision')
]

METRICS_ALL = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

#https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_v_loss = np.Inf
        self.best_v_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        v_acc = logs.get('val_accuracy')
        v_val_acc = logs.get('val_loss')

        # If BOTH the validation loss AND map10 does not improve for 'patience' epochs, stop training early.
        if np.less(v_acc, self.best_v_loss) or np.greater(v_val_acc, self.best_v_acc):
            self.best_v_loss = v_acc
            self.best_v_acc = v_val_acc
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(" Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def get_model_summary_format(model: tf.keras.Model) -> str:
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list).replace('                                                                                \n', "")

def load_and_clean_DF_Train_from_csv(path, op_buy_sell : _KEYS_DICT.Op_buy_sell, columns_selection = [], Y_TARGET ='buy_sell_point',  ):
    # https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook

    #El metodo puede recibir por path, leer.csv o por un df descargado en el momento
    raw_df = pd.read_csv(path, index_col=False, sep='\t')
    Logger.logr.info("df loaded from .csv Shape: "+ str( raw_df.shape ) + " Path: "+path )
    # else:
    #     Logger.logr.info("df has just loaded in memory (no read .csv) Shape: "+ str(  raw_df.shape))

    df = load_and_clean__buy_sell_atack(raw_df, columns_selection, op_buy_sell, Y_TARGET)
    return df


def load_and_clean__buy_sell_atack( raw_df, columns_selection, op_buy_sell : _KEYS_DICT.Op_buy_sell,  Y_TARGET  ='buy_sell_point' ):

    if not columns_selection:
        Logger.logr.info("columns_selection List is empty, works trains with all default columns")
    else:
        Logger.logr.debug("columns_selection List HAS vulues, works trains with: " + ', '.join(columns_selection))
        raw_df = raw_df[columns_selection]
        if 'ti_acc_dist' in raw_df.columns:
            raw_df = UtilsL.fill_last_values_of_colum_with_previos_value(raw_df, "ti_acc_dist")
        if "ti_ease_of_movement_14" in raw_df.columns:
            raw_df = UtilsL.fill_last_values_of_colum_with_previos_value(raw_df, "ti_ease_of_movement_14")
        if "ichi_chikou_span" in raw_df.columns:  # TODO muchas columnas nan eliminar
            raw_df = UtilsL.fill_last_values_of_colum_with_previos_value(raw_df, "ichi_chikou_span")

    # raw_df[Y_TARGET] = raw_df[Y_TARGET].astype(int).replace([101, -101], [100, -100])
    # raw_df[Y_TARGET] = raw_df[Y_TARGET].astype(int).replace(-100, 0)  # Solo para puntos de compra
    raw_df.replace([np.inf, -np.inf], 0, inplace=True) #Todo remove the row with inf
    df = Utils_buy_sell_points.select_work_buy_or_sell_point(raw_df.copy(), op_buy_sell)

    print("COMPRA VENTA PUNTO")
    # Logger.logr.debug(" groupby(Y_TARGET).count() " + str(df[['Date', Y_TARGET]].groupby(Y_TARGET).count()))
    df = cast_Y_label_binary(df, label_name=Y_TARGET)
    df = clean_redifine_df_dummy_ticker(df)
    return df


def cast_Y_label_binary(raw_df, label_name = 'buy_sell_point'):

    Y_target_classes = raw_df[label_name].unique().tolist()
    Y_target_classes.sort(reverse=False)#nothing must be the first
    Logger.logr.debug(f"Label classes: {Y_target_classes}")
    raw_df[label_name] = raw_df[label_name].map(Y_target_classes.index)

    if len(Y_target_classes) == 2:
        neg, pos = np.bincount(raw_df[label_name])
        total = neg + pos
        print('Examples:    Total: {}    Positive: {} ({:.2f}% of total) '.format(
            total, pos, 100 * pos / total))
    else:
        Logger.logr.info(f"Label classes is NOT 2 (pos/neg) Label classes:: {Y_target_classes}")

    return raw_df


def clean_redifine_df_dummy_ticker(raw_df):
    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date']).map(pd.Timestamp.timestamp)
    #cleaned_df = cleaned_df[COLUMNS_VALIDS]
    cleaned_df = cleaned_df.drop(columns= Utils_col_sele.DROPS_COLUMNS, errors='ignore')
    if 'ticker' in cleaned_df.columns:
        cleaned_df = pd.get_dummies(cleaned_df, columns = [ 'ticker'])
        #Si solo hay una accion no hace falta que ponga ticker_U ticker_PYPL
        filter_col = [col for col in cleaned_df if col.startswith('ticker_')]
        if len(filter_col) == 1:
            cleaned_df = cleaned_df.rename({filter_col[0]: 'ticker'}, axis='columns')

    cleaned_df = cleaned_df.dropna()

    return cleaned_df


def scaler_Raw_TF_onbalance(test_df, label_name = 'buy_sell_point'):
    test_labels = np.array(test_df.pop(label_name))
    test_features = np.array(test_df)
    scaler = StandardScaler()
    test_features = scaler.fit_transform(test_features)
    test_features = np.clip(test_features, -5, 5)
    Logger.logr.debug('Test labels shape:'+ str( test_labels.shape) + ' Test features shape:'+ str(  test_features.shape))
    return test_features,  test_labels

dataX, dataY = [], []
def df_to_df_multidimension_array_3D(dataframe, BACHT_SIZE_LOOKBACK):
    global dataY, dataX
    dataX, dataY = [], []
    # https://stackoverflow.com/questions/60736556/pandas-rolling-apply-using-multiple-columns
    def __create_tensor_values(ser):
        global dataY, dataX
        y_label = dataframe.at[ser.index[-1] , Y_TARGET]  # más óptimo que .loc
        dataY.append(y_label)
        x_data = dataframe.loc[ser.index[0]:ser.index[-1], dataframe.columns.drop(Y_TARGET)].values
        dataX.append(x_data)
        return 0

    dataframe[Y_TARGET].shift(-BACHT_SIZE_LOOKBACK).rolling(window=BACHT_SIZE_LOOKBACK).apply(__create_tensor_values,raw=False)
    return_feature = np.array(dataX)
    return_label = np.array(dataY)
    # Our vectorized labels https://stackoverflow.com/questions/48851558/tensorflow-estimator-valueerror-logits-and-labels-must-have-the-same-shape
    return_label = np.asarray(return_label).astype('float32').reshape((-1, 1))
    return return_label, return_feature

dataX, dataY = [], []
def df_to_df_multidimension_array_2D(dataframe, BACHT_SIZE_LOOKBACK, will_check_reshaped = True):
    global dataY, dataX
    dataX, dataY = [], []
    # https://stackoverflow.com/questions/60736556/pandas-rolling-apply-using-multiple-columns
    def __create_tensor_values(ser):
        global dataY, dataX
        y_label = dataframe.at[ser.index[-1] , Y_TARGET]  # más óptimo que .loc
        dataY.append(y_label)
        x_data = dataframe.loc[ser.index[0]:ser.index[-1], dataframe.columns.drop(Y_TARGET)].values
        dataX.append( x_data[::-1].reshape(1, -1) )
        return 0

    # dataframe[Y_TARGET].shift(-BACHT_SIZE_LOOKBACK).rolling(window=BACHT_SIZE_LOOKBACK).apply(__create_tensor_values,raw=False)
    dataframe[Y_TARGET].rolling(window=BACHT_SIZE_LOOKBACK).apply(__create_tensor_values,raw=False)
    return_feature = np.array(dataX)
    return_label = np.array(dataY)
    return_feature = return_feature.reshape(-1 , return_feature.shape[2] )

    # Check the reshape
    if will_check_reshaped:
        for N_row in [return_feature.shape[0]//2, return_feature.shape[0]//3, return_feature.shape[0]//5, return_feature.shape[0]-2]:
            arr_check  = np.array( dataframe.loc[(N_row-BACHT_SIZE_LOOKBACK+1):N_row, dataframe.columns.drop(Y_TARGET)] )[::-1].reshape(1, -1)
            # return_feature[N_row-BACHT_SIZE_LOOKBACK+1]
            if not (arr_check == return_feature[N_row-BACHT_SIZE_LOOKBACK+1]).all():
                Logger.logr.error("data has not been reshaped 2D correctly ")
                raise ValueError("df_to_df_multidimension_array_2D() - data has not been reshaped 2D correctly ")
    # arr_check == return_feature[N_row-BACHT_SIZE_LOOKBACK+1] #.reshape(-1 , return_feature.shape[2] )


    # Our vectorized labels https://stackoverflow.com/questions/48851558/tensorflow-estimator-valueerror-logits-and-labels-must-have-the-same-shape
    return_label = np.asarray(return_label).astype('float32').reshape((-1,1))
    return return_label, return_feature


from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
def prepare_to_split_SMOTETomek_and_scaler01(df):
    smote_tomek = SMOTETomek(sampling_strategy='all', random_state=1)  # minority  majority
    X_smt, y_smt = smote_tomek.fit_resample(df.drop(columns=[Y_TARGET]), df[Y_TARGET])
    X_smt[Y_TARGET] = y_smt
    df = X_smt

    sc = MinMaxScaler(feature_range=(_KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER))
    array_stock = sc.fit_transform(df)
    #np.clip Recorta (limita) los valores de una matriz.
    # Dado un intervalo, los valores fuera del intervalo se recortan a
    # los bordes del intervalo.  Por ejemplo, si se especifica un intervalo de ``[0, 1]``
    array_stock = np.clip(array_stock, _KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER)
    df = pd.DataFrame(array_stock, columns=df.columns)
    return df

#https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html
def prepare_to_split_SMOTETomek_01(df_feature,df_label ):
    smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=1)  # minority  majority
    X_smt, y_smt = smote_tomek.fit_resample(df_feature, df_label)
    # X_smt[Y_TARGET] = y_smt
    # df = X_smt

    return  X_smt, y_smt

import joblib
def scaler_min_max_array(array_raw, path_to_save = None, path_to_load = None):
    sc = None
    if path_to_load is not None:
        # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
        Logger.logr.debug("Scaler will be Loaded Path: " + path_to_load)
        sc = joblib.load(path_to_load)
    else:
        sc = MinMaxScaler(feature_range=(_KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER))
        sc = sc.fit(array_raw)

    array_stock = sc.transform(array_raw)

    if path_to_save is not None:
        Logger.logr.debug("Scaler will be Saved Path: "+path_to_save)
        joblib.dump(sc, path_to_save)

    # np.clip Recorta (limita) los valores de una matriz.
    # Dado un intervalo, los valores fuera del intervalo se recortan a
    # los bordes del intervalo.  Por ejemplo, si se especifica un intervalo de ``[0, 1]``
    array_stock = np.clip(array_stock, _KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER)
    # df = pd.DataFrame(array_stock, columns=df.columns)
    return array_stock.astype('float')


def scaler_split_TF_onbalance(cleaned_df, label_name = 'buy_sell_point', BACHT_SIZE_LOOKBACK = None, will_shuffle = False):

    # Use a utility from sklearn to split and shuffle your dataset.
    train_df, test_df = train_test_split(cleaned_df, test_size = 0.16, shuffle=will_shuffle)
    train_df, val_df = train_test_split(train_df, test_size = 0.32, shuffle=will_shuffle)

    # Form np arrays of labels and features.
    if BACHT_SIZE_LOOKBACK is not None:
        train_labels,train_features = df_to_df_multidimension_array_3D(train_df, BACHT_SIZE_LOOKBACK)
        bool_train_labels = (train_labels != 0).reshape((-1))
        test_labels, test_features = df_to_df_multidimension_array_3D(test_df, BACHT_SIZE_LOOKBACK)
        val_labels, val_features = df_to_df_multidimension_array_3D(val_df, BACHT_SIZE_LOOKBACK)
        log_shapes_trains_val_data(test_features, test_labels, train_features, train_labels, val_features, val_labels)
        return train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels

    bool_train_labels, test_features, test_labels, train_features, train_labels, val_features, val_labels = __cast_numpy_array(label_name, test_df, train_df, val_df)


    log_shapes_trains_val_data(test_features, test_labels, train_features, train_labels, val_features, val_labels)
    return train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels


def __cast_numpy_array(label_name, test_df, train_df, val_df):
    train_labels = np.array(train_df.pop(label_name))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop(label_name))
    test_labels = np.array(test_df.pop(label_name))
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    return bool_train_labels, test_features, test_labels, train_features, train_labels, val_features, val_labels


# root - [INFO]{MainThread} - __log_shapes_trains_val_data() - Training labels shape:(5634,)
# root - [DEBUG]{MainThread} - __log_shapes_trains_val_data() - Validation labels shape:(2192,)
# root - [DEBUG]{MainThread} - __log_shapes_trains_val_data() - Test labels shape:(3044,)
# root - [DEBUG]{MainThread} - __log_shapes_trains_val_data() - Training features shape:(5634, 57)
# root - [DEBUG]{MainThread} - __log_shapes_trains_val_data() - Validation features shape:(2192, 57)
# root - [DEBUG]{MainThread} - __log_shapes_trains_val_data() - Test features shape:(3044, 57)

def log_shapes_trains_val_data(test_features, test_labels, train_features, train_labels, val_features, val_labels):
    Logger.logr.info('Training features shape SMOTE:' + str(train_features.shape) + '\t Labels shape:' + str(train_labels.shape))
    Logger.logr.debug('Validation features shape:' + str(val_features.shape) + '\t Labels shape:' + str(val_labels.shape))
    Logger.logr.debug('Test features shape:' + str(test_features.shape) + '\t\t Labels shape:' + str(test_labels.shape))


def make_model_TF_onbalance(shape_features, metrics=METRICS_ALL, output_bias=None):
  if output_bias is not None:
    output_bias = keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(shape_features,)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model

#DATOS desequilibrados https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def make_model_TF_multidimension_onbalance_fine_28(input_shape_m, metrics=METRICS_ACCU_PRE, output_bias=None, num_features =20):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential([
      keras.layers.Dense(28, activation='relu',input_shape=input_shape_m),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='relu',bias_initializer=output_bias),
    ])

    model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.001),
          loss=keras.losses.BinaryCrossentropy(),
          metrics=metrics)

    return model


def make_model_TF_onbalance_fine_28(shape_features, metrics=METRICS_ACCU_PRE, output_bias=None):
  if output_bias is not None:
    output_bias = keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(28, activation='relu',input_shape=(shape_features,)),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='relu',bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.001),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


def dataset_shapes(dataset):
    try:
        return [x.get_shape().as_list() for x in dataset._tensors]
    except TypeError:
        return dataset._tensors.get_shape().as_list()

def get_resampled_ds_onBalance(train_features, train_labels, bool_train_labels, BATCH_SIZE):
    #Cuantos valores de la minoria en proporcion a la mayoria van  salir
    #Ej: si es 1 , la proporcion entre pos 50% y neg  50%
    # si es 0.8 la proporcion será de +-40% para la minoria , y +-60% para la mayoria
    EQUAL_FACTOR_BALANCE = 1
    # Oversample the minority class
    # A related approach would be to resample the dataset by oversampling the minority class.
    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    # You can balance the dataset manually by choosing the right number of random
    # indices from the positive examples:
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, int(len(neg_features) * EQUAL_FACTOR_BALANCE))
    res_pos_features = pos_features[choices]
    if pos_labels is None or len(pos_labels) == 0:
        print("No hay valores positivos en para predecir ValueError: 'a' cannot be empty unless no samples are taken")
        raise "No hay valores positivos en para predecir ValueError: 'a' cannot be empty unless no samples are taken"
    res_pos_labels = pos_labels[choices]
    # res_pos_features.shape
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    resampled_features.shape
    """#### Using `tf.data`
    If you're using `tf.data` the easiest way to produce balanced examples is to start with a `positive` and a `negative` dataset, 
    and merge them. See [the tf.data guide](../../guide/data.ipynb) for more examples.
    """
    BUFFER_SIZE = 100000
    def make_ds(features, labels):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
        ds = ds.shuffle(BUFFER_SIZE).repeat() # esto tiene sentido??
        return ds

    pos_ds = make_ds(pos_features, pos_labels)
    neg_ds = make_ds(neg_features, neg_labels)
    # Each dataset provides `(feature, label)` pairs:
    # for features, label in pos_ds.take(1):

    # Merge the two together using `tf.data.Dataset.sample_from_datasets`:
    resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
    for features, label in resampled_ds.take(1):
        print("Resampled_ds type: "+str(resampled_ds.take(1).element_spec)  +"  mean():" + str(label.numpy().mean() ))
    # print("tf.dataset_shape: "+dataset_shapes(resampled_ds) )
    return resampled_ds

Y_TARGET = 'buy_sell_point'
def __prepare_get_all_result_df(X_test, y_test):
    df_re = X_test[['Date', "Close", "per_Close", 'has_preMarket', 'Volume']].copy()
    list_ticker_stocks = [col for col in X_test.columns if
                          col.startswith('ticker_')]  # todas las que empiecen por ticker_ , son variables tontas
    if (list_ticker_stocks is not None) and len(list_ticker_stocks) > 0:
        df_re['ticker'] = X_test[list_ticker_stocks].idxmax(axis=1).copy()  # undo dummy variable
        df_re[Y_TARGET] = y_test.copy()
        df_re = df_re[['Date', Y_TARGET, 'ticker', "Close", "per_Close", 'has_preMarket', 'Volume']]
    else:
        df_re[Y_TARGET] = y_test.copy()
        df_re = df_re[['Date', Y_TARGET,  "Close", "per_Close", 'has_preMarket', 'Volume']]

    df_re = df_re.loc[:,~df_re.columns.duplicated()].copy() #quitar remove duplicates duplicadas
    return df_re.copy()


def fill_first_time_df_result_all(df):
    global df_result_all
    X = df.drop(columns=Y_TARGET)  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, shuffle=False)
    X_test = X.copy()
    y_test = y
    df_result = X.copy()
    # PARA organizar la salida en el df_result de los resultados
    return __prepare_get_all_result_df(X_test, y_test)

def get_df_for_list_of_result(df_full):
    df_list_r = pd.DataFrame()
    list_ticker_stocks = [col for col in df_full.columns if
                          col.startswith('ticker_')]  # todas las que empiecen por ticker_ , son variables tontas
    if (list_ticker_stocks is not None) and len(list_ticker_stocks) > 0:
        df_full['ticker'] = df_full[list_ticker_stocks].idxmax(axis=1).copy()  # undo dummy variable
        df_list_r = df_full[['Date', Y_TARGET, 'ticker', "Close", 'has_preMarket', 'Volume']].copy()
    else:
        df_list_r = df_full[['Date', Y_TARGET, "Close", 'has_preMarket', 'Volume']].copy()
    return df_list_r


def __print_csv_accuracy_loss_models(MODEL_FOLDER_TF, model_h5_name, resampled_history, columns :list, per_df_test = None):
    UtilsL.remove_files_starwith(MODEL_FOLDER_TF + model_h5_name + "_")

    text_df_test_per = ""
    if per_df_test is not None:
        text_df_test_per = "_TEST_"+str(per_df_test) +"%_"

    data_hist_model = text_df_test_per+ resampled_history.model.metrics_names[1]  +"_" + "{:.2f}".format(
        resampled_history.history['accuracy'][-1] * 100) + "%__" \
                      + resampled_history.model.metrics_names[0] + "_" + "{:.2f}".format(
        resampled_history.history['loss'][-1]) + "__" \
                      + "epochs_" + str(resampled_history.epoch[-1]) + "[" + str(
        resampled_history.params['epochs']) + "]"

    pd.DataFrame(resampled_history.history).round(3).to_csv(
        MODEL_FOLDER_TF + model_h5_name + "_" + data_hist_model + ".csv", sep="\t", index=None)
    print("Statistics: " +MODEL_FOLDER_TF + model_h5_name + "_" + data_hist_model + ".csv")


    accuracy = "{:.2f}".format(resampled_history.history['accuracy'][-1] * 100)
    loss = "{:.2f}".format(resampled_history.history['loss'][-1])
    epochs = str(resampled_history.epoch[-1])


    #Add columns that model use to predict:
    with open(MODEL_FOLDER_TF + model_h5_name + "_" + data_hist_model + ".csv", 'a') as file:
        file.write("\nColumns "+model_h5_name+":\n" + ", ".join(columns))


    return accuracy , loss , epochs

#Format TFm_MELI_pos_low1_mult_28.h5
def get_config_from_name_model(name_col_model):
    subname = name_col_model.split("_")
    stock = subname[1]
    pos_neg = subname[2]
    columns_type_sele = subname[3]
    multi = subname[4]
    type_multi = subname[5]
    return stock,pos_neg,columns_type_sele, multi+"_"+type_multi