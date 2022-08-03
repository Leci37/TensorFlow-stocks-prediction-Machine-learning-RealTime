import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


ALL_METRICS = [
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
# def ALL_METRICS():
#     return ALL_METRICS

NUMERIC_FEATURE_NAMES = ['buy_sell_point',
                         # "Date",
                         "Open","High","Low","Close","Volume","per_Close","per_Volume","per_preMarket","olap_BBAND_UPPER","olap_BBAND_MIDDLE",
                         "olap_BBAND_LOWER","olap_HT_TRENDLINE","olap_MIDPOINT","olap_MIDPRICE","olap_SAR","olap_SAREXT","mtum_ADX","mtum_ADXR","mtum_APO",
                         "mtum_AROON_down","mtum_AROON_up","mtum_AROONOSC","mtum_BOP","mtum_CCI","mtum_CMO","mtum_DX","mtum_MACD","mtum_MACD_signal","mtum_MACD_list",
                         "mtum_MACD_ext","mtum_MACD_ext_signal","mtum_MACD_ext_list","mtum_MACD_fix","mtum_MACD_fix_signal","mtum_MACD_fix_list","mtum_MFI",
                         "mtum_MINUS_DI","mtum_MINUS_DM","mtum_MOM","mtum_PLUS_DI","mtum_PLUS_DM","mtum_PPO","mtum_ROC","mtum_ROCP","mtum_ROCR","mtum_ROCR100",
                         "mtum_RSI","mtum_STOCH_k","mtum_STOCH_d","mtum_STOCH_kd","mtum_STOCH_Fa_k","mtum_STOCH_Fa_d","mtum_STOCH_Fa_kd","mtum_STOCH_RSI_k",
                         "mtum_STOCH_RSI_d","mtum_STOCH_RSI_kd","mtum_ULTOSC","mtum_WILLIAMS_R","volu_Chaikin_AD","volu_Chaikin_ADOSC","volu_OBV",
                         "vola_ATR","vola_NATR","vola_TRANGE","cycl_DCPERIOD","cycl_DCPHASE","cycl_PHASOR_inph","cycl_PHASOR_quad","cycl_SINE_sine","cycl_SINE_lead",  # "cycl_HT_TRENDMODE",
                         "sti_BETA","sti_CORREL","sti_LINEARREG","sti_LINEARREG_ANGLE","sti_LINEARREG_INTERCEPT",
                         "sti_LINEARREG_SLOPE", "sti_STDDEV", "sti_TSF", "sti_VAR", "ma_DEMA_5", "ma_EMA_5", "ma_KAMA_5", "ma_SMA_5", "ma_T3_5", "ma_TEMA_5", "ma_TRIMA_5",
                         "ma_WMA_5", "ma_DEMA_10", "ma_EMA_10", "ma_KAMA_10", "ma_SMA_10", "ma_T3_10", "ma_TEMA_10", "ma_TRIMA_10", "ma_WMA_10", "ma_DEMA_20", "ma_EMA_20",
                         "ma_KAMA_20", "ma_SMA_20", "ma_TEMA_20", "ma_TRIMA_20", "ma_WMA_20", "ma_EMA_50", "ma_KAMA_50", "ma_SMA_50", "ma_TRIMA_50",
                         # TOO MACH null data in this columns ,"ma_DEMA_50","ma_T3_20","mtum_TRIX"
                         #  "ma_WMA_50","ma_EMA_100","ma_KAMA_100","ma_SMA_100","ma_TRIMA_100","ma_WMA_100",
                         "trad_s3", "trad_s2", "trad_s1", "trad_pp", "trad_r1", "trad_r2",
                         "trad_r3", "clas_s3", "clas_s2", "clas_s1", "clas_pp", "clas_r1", "clas_r2", "clas_r3", "fibo_s3", "fibo_s2", "fibo_s1", "fibo_pp", "fibo_r1", "fibo_r2",
                         "fibo_r3", "wood_s3", "wood_s2", "wood_s1", "wood_pp", "wood_r1", "wood_r2", "wood_r3", "demark_s1", "demark_pp", "demark_r1", "cama_s3", "cama_s2",
                         "cama_s1", "cama_pp", "cama_r1", "cama_r2", "cama_r3", "ti_acc_dist", "ti_chaikin_10_3", "ti_choppiness_14", "ti_coppock_14_11_10",
                         "ti_donchian_lower_20", "ti_donchian_center_20", "ti_donchian_upper_20", "ti_ease_of_movement_14", "ti_force_index_13", "ti_hma_20",
                         "ti_kelt_20_lower", "ti_kelt_20_upper", "ti_mass_index_9_25", "ti_supertrend_20", "ti_vortex_pos_5", "ti_vortex_neg_5", "ti_vortex_pos_14", "ti_vortex_neg_14"]

CANDLE_COLUMNS = ["cdl_2CROWS", "cdl_3BLACKCROWS", "cdl_3INSIDE", "cdl_3LINESTRIKE", "cdl_3OUTSIDE", "cdl_3STARSINSOUTH", "cdl_3WHITESOLDIERS",
                         "cdl_ABANDONEDBABY","cdl_ADVANCEBLOCK","cdl_BELTHOLD","cdl_BREAKAWAY","cdl_CLOSINGMARUBOZU","cdl_CONCEALBABYSWALL","cdl_COUNTERATTACK",
                         "cdl_DARKCLOUDCOVER","cdl_DOJI","cdl_DOJISTAR","cdl_DRAGONFLYDOJI","cdl_ENGULFING","cdl_EVENINGDOJISTAR","cdl_EVENINGSTAR",
                         "cdl_GAPSIDESIDEWHITE","cdl_GRAVESTONEDOJI","cdl_HAMMER","cdl_HANGINGMAN","cdl_HARAMI","cdl_HARAMICROSS","cdl_HIGHWAVE","cdl_HIKKAKE",
                         "cdl_HIKKAKEMOD","cdl_HOMINGPIGEON","cdl_IDENTICAL3CROWS","cdl_INNECK","cdl_INVERTEDHAMMER","cdl_KICKING","cdl_KICKINGBYLENGTH",
                         "cdl_LADDERBOTTOM","cdl_LONGLEGGEDDOJI","cdl_LONGLINE","cdl_MARUBOZU","cdl_MATCHINGLOW","cdl_MATHOLD","cdl_MORNINGDOJISTAR","cdl_MORNINGSTAR",
                         "cdl_ONNECK","cdl_PIERCING","cdl_RICKSHAWMAN","cdl_RISEFALL3METHODS","cdl_SEPARATINGLINES","cdl_SHOOTINGSTAR","cdl_SHORTLINE",
                         "cdl_SPINNINGTOP","cdl_STALLEDPATTERN","cdl_STICKSANDWICH","cdl_TAKURI","cdl_TASUKIGAP","cdl_THRUSTING","cdl_TRISTAR","cdl_UNIQUE3RIVER",
                         "cdl_UPSIDEGAP2CROWS","cdl_XSIDEGAP3METHODS"]

COLUMNS_VALIDS = NUMERIC_FEATURE_NAMES + CANDLE_COLUMNS


def cast_Y_label_binary(raw_df, label_name = 'buy_sell_point'):

    Y_target_classes = raw_df[label_name].unique().tolist()
    Y_target_classes.sort(reverse=True)#nothing must be the first
    print(f"Label classes: {Y_target_classes}")
    raw_df[label_name] = raw_df[label_name].map(Y_target_classes.index)

    neg, pos = np.bincount(raw_df[label_name])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    return raw_df


def clean_redifine_df(raw_df):
    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date']).map(pd.Timestamp.timestamp)
    cleaned_df = cleaned_df[COLUMNS_VALIDS]
    if 'ticker' in cleaned_df.columns:
        cleaned_df = pd.get_dummies(cleaned_df, columns = [ 'ticker'])
    cleaned_df = cleaned_df.dropna()

    return cleaned_df


def scaler_split_TF_onbalance(cleaned_df, label_name = 'buy_sell_point'):
    # Use a utility from sklearn to split and shuffle your dataset.
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop(label_name))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop(label_name))
    test_labels = np.array(test_df.pop(label_name))
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)
    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)
    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)
    return train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels


def make_model_TF_onbalance(shape_features, metrics=ALL_METRICS, output_bias=None):
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


def get_resampled_ds_onBalance(train_features, train_labels, bool_train_labels, BATCH_SIZE):
    # Oversample the minority class
    # A related approach would be to resample the dataset by oversampling the minority class.
    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    # You can balance the dataset manually by choosing the right number of random
    # indices from the positive examples:
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]
    res_pos_features.shape
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    resampled_features.shape
    """#### Using `tf.data`

    If you're using `tf.data` the easiest way to produce balanced examples is to start with a `positive` and a `negative` dataset, and merge them. See [the tf.data guide](../../guide/data.ipynb) for more examples.
    """
    BUFFER_SIZE = 100000

    def make_ds(features, labels):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
        ds = ds.shuffle(BUFFER_SIZE).repeat()
        return ds

    pos_ds = make_ds(pos_features, pos_labels)
    neg_ds = make_ds(neg_features, neg_labels)
    # Each dataset provides `(feature, label)` pairs:
    for features, label in pos_ds.take(1):
        print("Features:\n", features.numpy())
        print()
        print("Label: ", label.numpy())
    # Merge the two together using `tf.data.Dataset.sample_from_datasets`:
    resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
    for features, label in resampled_ds.take(1):
        print(label.numpy().mean())

    return resampled_ds