import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import quantstats as qs
from typing import Dict, List
# from tuner.classes import Function

def import_data(file_path: str):
    bars = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    columns = bars.columns.tolist()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in columns
        
    return bars

def get_feature_window(df: pd.DataFrame, window_size: int, todict=False):
    x = []
    index = []

    for i in range(window_size, len(df)+1):
        x.append(df[i - window_size: i].values)
        index.append(df.index[i-1])

    try: index = np.array(index, np.datetime64)
    except: index = np.array(index)

    assert len(index) == len(x)
    return pd.Series(x, index=index)

def get_window_data(df: pd.DataFrame, target: pd.Series, window_size: int, todict=False):
    x = []
    y = []
    index = []

    if target is None:
        df.insert(loc=0, column="target", value=0)
        df["target"].iloc[1] = 1;
        df["target"].iloc[2] = 2
        target = pd.get_dummies(df['target'])
        df = df.drop(['target'], axis=1)
        #Por que lo hace en lo anteiror
        print("INFO To create the window_data need to create an artificial Useless Y_target  Count: ", len(df.columns) , " Names : ", ",".join(df.columns))
        print("DEBUG features_W3 index Dates:: ", df.index[0], df.index[-1], " Shape: ", df.shape)
        target = target[target.index.isin(df.index)]
        df = df[df.index.isin(target.index)]

    assert len(df) == len(target)

    for i in range(window_size, len(df)+1):
        # Target is at the same row to the feature
        assert df[i - window_size: i].index[-1] == target.index[i-1]

        x.append(df[i - window_size: i].values)
        if isinstance(target, pd.DataFrame): y.append(target.iloc[i-1].values)
        else: y.append(target.iloc[i-1])
        index.append(target.index[i-1])

    x = np.array(x, np.float32)
    try:
        index = np.array(index, np.datetime64)
    except Exception as ex:
        print("Exception: ", ex)
        index = np.array(index)
    y = np.array(y)

    if todict: return { 'X': x, 'y': y, 'index': index }
    else: return x, y, index

def firsh_index_ends_with(df, end_with = ":00:00"):
    df['date_aux'] = df.index
    df['date_aux'] = df['date_aux'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    # df2.loc[((df2['Close'] > df2['ichi_senkou_b']) & (df2['Close'] < df2['ichi_senkou_a'])), "ichi_isin_cloud"] = 1
    firsh_date_correct = df.loc[(df['date_aux'].str.endswith(end_with) )].iloc[[0]].index
    last_date_correct = df.loc[(df['date_aux'].str.endswith(end_with))].iloc[[-1]].index
    df.drop('date_aux', axis=1, inplace=True)
    return firsh_date_correct , last_date_correct

def get_window_data_3Window(df_5min: pd.DataFrame, df_15min: pd.DataFrame, df_60min: pd.DataFrame, target_5min: pd.Series, WINDOW_SIZE: int,OPTIONS_W:list, todict=False ):

    start5 ,last5 = firsh_index_ends_with(df_5min)
    start15, last15 =  firsh_index_ends_with(df_15min)
    start60, last60 =  firsh_index_ends_with(df_60min)
    print("DEBUG First Dates DF : ",start5, " ",start15, " ",start60, " ",)
    print("DEBUG End Dates DF : ", last5, " ", last15, " ", last60, " ", )
    start_date = pd.DataFrame( {"Date" : [start5, start15,start60 ]} ).max()[0].strftime("%Y-%m-%d %H:%M:%S").values[0]
    end_date = pd.DataFrame({"Date": [last5, last15, last60]}).min()[0].strftime("%Y-%m-%d %H:%M:%S").values[0]
    #TODO check if date is inside all df
    # x = [][]
    DICT_TYPE_WIND = {"5min":0, "15min":1, "60min":2}
    df_5min = df_5min[start_date:end_date]
    df_15min = df_15min[start_date:end_date]
    df_60min = df_60min[start_date:end_date]
    target_5min = target_5min[start_date:end_date]

    print("\nINFO Generate model with TIME WINDOWS: ", OPTIONS_W)
    windows_num, len_of_rows = len(OPTIONS_W), len(df_5min)-WINDOW_SIZE
    x = [[0 for x in range(windows_num)] for y in range(len_of_rows)]
    y = []
    index = []

    assert len(df_5min) == len(target_5min)

    for i in range(WINDOW_SIZE, len(df_5min)):
        # Target is at the same row to the feature
        i_index =  i- WINDOW_SIZE
        assert df_5min[i - WINDOW_SIZE: i].index[-1] == target_5min.index[i - 1]
        #5min window
        x[i_index][DICT_TYPE_WIND["5min"]] = (df_5min[i - WINDOW_SIZE: i].values)
        #15min window                    df_5min.iloc[[i ]].index ERROR i = 54261
        index_start_window = df_5min.iloc[[i - WINDOW_SIZE]].index.strftime("%Y-%m-%d %H:%M:%S").values[0]
        # index_end_window = df_5min.iloc[[i]].index.strftime("%Y-%m-%d %H:%M:%S").values[0]
        index_start_window_df15 = index_start_window.replace(":05:00", ":00:00").replace(":10:00", ":00:00").replace(":20:00", ":15:00").\
            replace(":25:00", ":15:00").replace(":35:00", ":30:00").replace(":40:00", ":30:00").replace(":50:00", ":45:00").replace(":55:00", ":45:00")
        index_start_window_df60 = index_start_window.replace(":05:00", ":00:00").replace(":10:00", ":00:00").replace(":15:00", ":00:00").replace(":20:00", ":00:00").\
            replace(":25:00", ":00:00").replace(":30:00", ":00:00").replace(":35:00", ":00:00").replace(":40:00", ":00:00").replace(":45:00", ":00:00").replace(":50:00", ":00:00").replace(":55:00", ":00:00")

        select_indices_15min = np.where(df_15min.index == index_start_window_df15)[0][0] #https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-where-column-matches-certain-value
        select_indices_60min = np.where(df_60min.index == index_start_window_df60)[0][0]
        # print("Dates\t 5min: ",index_start_window, " 15min: ", index_start_window_df15 , " 60min: ", index_start_window_df60 , " pos_15df: ", select_indices_15min, " pos_60df: ", select_indices_60min)
        # try:
        if "15min" in OPTIONS_W:
            x[i_index][DICT_TYPE_WIND["15min"]] = df_15min.iloc[select_indices_15min - WINDOW_SIZE: select_indices_15min].values
        if "15min" not in OPTIONS_W and "60min" in OPTIONS_W:
            x[i_index][1] = df_60min.iloc[select_indices_60min - WINDOW_SIZE: select_indices_60min].values
        elif "60min" in OPTIONS_W:
            x[i_index][DICT_TYPE_WIND["60min"]] = df_60min.iloc[select_indices_60min - WINDOW_SIZE: select_indices_60min].values
        # except Exception as ex:
        #     print("Exception: ", ex)

        # df_15min.loc[ index_strat_window[0].strftime("%Y-%m-%d %H:%M:%S") ]
        if isinstance(target_5min, pd.DataFrame): y.append(target_5min.iloc[i-1].values)
        else: y.append(target_5min.iloc[i-1])
        index.append(target_5min.index[i-1])

    NUM_FIRST_ROW_EMPTY_60MIN = 10 * WINDOW_SIZE
    #need remove the empty start
    #54000 rows, 3 clandle times (5min, 15,im.60min) , windows size 12 , 160 tech parents
    x_array = np.array(x[NUM_FIRST_ROW_EMPTY_60MIN:][:], np.float32)
    print("INFO created array x_array Created x_array.shape : ",x_array.shape,"  \tEX: 54000 rows, 3 clandle times (5min,15mim.60min) , windows size 12 , 160 tech parents ")
    # x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["5min"]] = np.array(x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["5min"]], np.float32)
    # x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["15min"]] = np.array(x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["15min"]], np.float32)
    # x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["60min"]] = np.array(x[NUM_FIRST_ROW_EMPTY_60MIN:][DICT_TYPE_WIND["60min"]], np.float32)
    try:
        index = np.array(index[NUM_FIRST_ROW_EMPTY_60MIN:], np.datetime64)
    except Exception as ex:
        print("Exception: ", ex)
        index = np.array(index[NUM_FIRST_ROW_EMPTY_60MIN:])
    y = np.array(y[NUM_FIRST_ROW_EMPTY_60MIN:])
    assert len(y) == len(index);assert len(x_array) == len(index)

    df_details = pd.DataFrame()
    print("\nTARGET detail : ")
    df_details['count'] = pd.DataFrame(y).value_counts().to_frame()
    df_details['per%'] = pd.DataFrame(y).value_counts(normalize=True).mul(100).round(2)
    print(df_details.to_string())

    if todict:
        return {'X': x_array, 'y': y, 'index': index}
    else: return x_array, y, index


def normalise_data(X, split_ratio, scaler_path, todict=False):
    # Normalise the Data
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    # X = np.float32(X_int)
    if isinstance(split_ratio, bool) and split_ratio == True:
        scaler: MinMaxScaler = pickle.load(open(scaler_path, 'rb'))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_size = int(len(X) * split_ratio)
        scaler = scaler.fit(X[:train_size])
        pickle.dump(scaler, open(scaler_path, 'wb'))

    if isinstance(X, pd.DataFrame):
        columns = X.columns
        index = X.index
        try:
            output = scaler.transform(X)
        except ValueError:
            print("DEBUG {ValueError}The feature names should match those that were passed during fit.")
            output = scaler.transform(X[scaler.feature_names_in_])
        output = pd.DataFrame(output, columns=columns, index=index)
    else:
        output = scaler.transform(X.reshape((-1, X.shape[-1] if len(X.shape) > 1 else 1))).reshape(X.shape)
    
    if todict: return { 'output': output, 'scaler': scaler }
    else: return output, scaler

def set_save_class_weight(dict_class_weight , path_class_weight):
    with open(path_class_weight, 'wb') as f:
        pickle.dump(dict_class_weight, f)

def get_load_class_weight( path_class_weight):
    with open(path_class_weight, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def merge_set(args: Dict):
    X: pd.Series = args['X']
    y: pd.Series = args['y']

    y = y[y.index.isin(X.index)]
    X = X[X.index.isin(y.index)]

    return { 'X': X.values, 'y': y.values, 'index': y.index }

def split_set(args: Dict, split_ratio: float):
    X: np.ndarray = args['X']
    y: np.ndarray = args['y']
    index: pd.Index = args['index']

    assert len(X) == len(y)
    assert len(y) == len(index)

    train_size = int(len(index) * split_ratio)
    X_train, y_train, index_train = X[:train_size], y[:train_size], index[:train_size]
    X_test, y_test, index_test = X[train_size:], y[train_size:], index[train_size:]
    return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'index_train': index_train, 'index_test': index_test }

def split_data(X: np.ndarray, y: np.ndarray, index: np.ndarray, split_ratio: float, todict=False):
    assert len(X) == len(y)
    assert len(index) == len(X)

    train_size = int(len(X) * split_ratio)

    X_train, y_train, index_train = X[:train_size], y[:train_size], index[:train_size]
    X_test, y_test, index_test = X[train_size:], y[train_size:], index[train_size:]

    if todict:
        return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'index_train': index_train, 'index_test': index_test }
    else: return X_train, y_train, X_test, y_test, index_train, index_test

def create_report(trades: pd.DataFrame, export_path: str = 'report.html'):
    qs.extend_pandas()
    trades = trades.copy()

    trades['cashflow'] = trades['profit'].cumsum()
    trades['returns'] = trades['cashflow'].pct_change()
    trades.set_index('openTime', inplace=True)
    trades.index = pd.to_datetime(trades.index)
    qs.reports.html(trades['returns'].dropna(), download_filename=export_path)


def set_save_class_weight(dict_class_weight , path_class_weight):
    with open(path_class_weight, 'wb') as f:
        pickle.dump(dict_class_weight, f)

def get_load_class_weight( path_class_weight):
    with open(path_class_weight, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict



def remove_strong_correlations_columns(df_cor , factor:float):
    # Create correlation matrix https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    corr_matrix = df_cor.corr().abs()  # Select upper triangle of correlation matrix
    upper = corr_matrix.where( np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > factor)]
    print("\tDEBUG Columns more correlated than factor, will be Removed. Factor: ",factor, " Columns number: ", len(to_drop) )
    df_cor.drop(to_drop, axis=1, inplace=True)  # Drop features
    return df_cor

def get_best_Y_correlations_columns(df_cor, target_Y, num_columns:int):
    # Create correlation matrix https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    df_cor['aux_target_Y_1'] = target_Y[1]
    df_cor['aux_target_Y_2'] = target_Y[2]
    df_corr = df_cor.corr()[['aux_target_Y_1','aux_target_Y_2']].abs().head(-2)

    df_corr['sum_strong_corr'] = df_corr['aux_target_Y_1'] + df_corr['aux_target_Y_2']
    df_corr = df_corr.sort_values(['sum_strong_corr'], ascending=False).head(num_columns)
    df_cor.drop(['aux_target_Y_1','aux_target_Y_2'], axis=1, inplace=True)
    return df_corr.index.values

    # corr_matrix = df_cor['aux_target_Y_1','aux_target_Y_2'].corr().abs()  # Select upper triangle of correlation matrix
    # upper = corr_matrix.where( np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Find features with correlation greater than 0.95
    # # df_one_col = upper.stack().reset_index()
    # # df_one_col = df_one_col.sort_values([0], ascending=True)
    # # df_one_col = df_one_col[df_one_col[0] >0.0001] #at lest something
    # #
    # # df_one_col.head(factor)
    #
    # to_drop = [column for column in upper.columns if any(upper[column] > factor)]
    # # print("\tDEBUG Columns best correlated  Num: ",factor, " Columns number: ", len(to_drop) )
    # df_cor.drop(to_drop, axis=1, inplace=True)  # Drop features
    # return df_cor