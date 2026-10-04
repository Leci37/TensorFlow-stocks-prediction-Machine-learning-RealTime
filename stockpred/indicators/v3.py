from .ta import *
from tqdm import tqdm
from pprint import pprint

def extract_features(df: pd.DataFrame,extra_columns =False,  shift=150, debug=False):
    df = df.copy()

    result = None
    # for i in tqdm(range(shift, len(df)+1)):
    #     subset = df[i-shift:i]

    df['Date.1'] = pd.to_datetime(df.index)
    df['day_minute'] = (df['Date.1'].dt.hour * 60 + df['Date.1'].dt.minute)
    df['day_week'] = df['Date.1'].dt.dayofweek

        # Part 1 : Create all technical features
    subset = gel_all_TALIB_funtion(df, custom_columns=None)
    subset = get_all_pivots_points(subset, custom_columns=None)
    subset = get_py_TI_indicator(subset, cos_cols=None)
    subset = get_all_pandas_TA_tecnical(subset, cos_cols=None)
    subset = get_all_pandas_TU_tecnical(subset, cos_cols=None)
    if extra_columns:
        subset = get_ALL_CRASH_funtion(subset, custom_columns=None) #this add +-600 columns
        # if result is None: result = subset[-1:]
        # else: result = pd.concat([result, subset[-1:]], axis=0)
        #
        # if i % 20 == 0 :
        #     pprint("exit() ")
        #     exit()
    result = subset
    original_length = len(df)
    result = result.dropna()
    if debug: print(f'Dropped V3333333 {original_length - len(result)} from {original_length} bars')

    for l in ['mtum_QQEl_14_5_4.236', 'mtum_QQEs_14_5_4.236', 'tend_PSARl_0.02_0.2', 'tend_PSARs_0.02_0.2']:
        if l in df.columns: df[l] = df[l].replace(np.nan, 0)

    # Part 2 : Stationarise all non-stationary data
    # if make_stationary:
    #     result = stationise(result, debug)
    #     result = result.dropna()
    #
    #     if debug: print(f'Finalised with {len(result)} bars')
    return result



def extract_features_A(df: pd.DataFrame, make_stationary=False, shift=150, debug=False):
    df = df.copy()

    result = None
    for i in tqdm(range(shift, len(df)+1)):
        subset = df[i-shift:i]

        # Part 1 : Create all technical features
        subset = gel_all_TALIB_funtion(subset, custom_columns=None)
        subset = get_all_pivots_points(subset, custom_columns=None)
        subset = get_py_TI_indicator(subset, cos_cols=None)
        subset = get_all_pandas_TA_tecnical(subset, cos_cols=None)
        subset = get_all_pandas_TU_tecnical(subset, cos_cols=None)
        if result is None: result = subset[-1:]
        else: result = pd.concat([result, subset[-1:]], axis=0)

        if i % 20 == 0 :
            pprint("exit() ")
            exit()
    original_length = len(df)
    result = result.dropna()
    if debug: print(f'Dropped {original_length - len(result)} from {original_length} bars')

    for l in ['mtum_QQEl_14_5_4.236', 'mtum_QQEs_14_5_4.236', 'tend_PSARl_0.02_0.2', 'tend_PSARs_0.02_0.2']:
        if l in df.columns: df[l] = df[l].replace(np.nan, 0)

    # Part 2 : Stationarise all non-stationary data
    if make_stationary:
        result = stationise(result, debug)
        result = result.dropna()

        if debug: print(f'Finalised with {len(result)} bars')
        return result