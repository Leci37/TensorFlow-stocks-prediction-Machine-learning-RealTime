from pandas import DataFrame

RET_METHODS = {'simple', 'log'}
MA_METHODS = {'sma', 'ema', 'wma', 'hma', 'wilders', 'kama', 'fma'}
RETURN_STRUCTS = {'numpy', 'pandas'}
DDOF = {0, 1}

DF_ERR_MESSAGE = "Error: 'df' must be a Pandas DataFrame"
COLUMN_ERR_MESSAGE = "Invalid Column: column name not found in dataframe"
RET_ERR_MESSAGE = f"Invalid method. Valid methods: {RET_METHODS}"
MA_ERR_MESSAGE = f"Invalid method. Valid methods: {MA_METHODS}"
RETURN_STRUCTS_ERR_MESSAGE = f"Invalid return_struct. Valid return_structs: {RETURN_STRUCTS}"

type_dict = {
    'column': str,
    'n': int,
    'n_er': int,
    'n_fast': int,
    'n_med': int,
    'n_slow': int,
    'n_k': int,
    'n_d': int,
    'n_macd': int,
    'n_sum': int,
    'n_1': int,
    'n_2': int,
    'n_3': int,
    'n_4': int,
    'ma_1': int,
    'ma_2': int,
    'ma_3': int,
    'ma_4': int,
    'fast': int,
    'slow': int,
    'sig': int,
    'constant': float,
    'factor': float,
    'upper_factor': float,
    'lower_factor': float,
    'upper_num_sd': float,
    'lower_num_sd': float,
    'af_step': float,
    'max_af': float,
    'add_col': bool,
    }

def int_err_message(var):
    return f"Error: {var} must be of type int"

def float_err_message(var):
    return f"Error: {var} must be of type float"

def string_err_message(var):
    return f"Error: {var} must be of type string"

def bool_err_message(var):
    return f"Error: {var} must be of type bool"
  
def check_errors(df=None, column=None, ret_method=None, ma_method=None,
                 ddof=None, return_struct=None, **kwargs):

    if df is not None and type(df) is not DataFrame:
        raise Exception(DF_ERR_MESSAGE)

    if column is not None and column not in df.columns:
        raise Exception(COLUMN_ERR_MESSAGE)

    if column is not None and type(column) is not str:
        raise TypeError(string_err_message('column'))

    if ret_method is not None and ret_method not in RET_METHODS:
        raise Exception(RET_ERR_MESSAGE)

    if ma_method is not None and ma_method not in MA_METHODS:
        raise Exception(MA_ERR_MESSAGE)

    if ddof is not None and ddof not in DDOF:
        raise TypeError(int_err_message('ddof') + 'and either 0 or 1')

    if return_struct is not None and return_struct not in RETURN_STRUCTS:
        raise Exception(RETURN_STRUCTS_ERR_MESSAGE)

    for k, v in kwargs.items():
        if type_dict[k] != type(v):
            if type_dict[k] == int:
                raise TypeError(int_err_message(k))
            elif type_dict[k] == float:
                raise TypeError(float_err_message(k))
            elif type_dict[k] == str:
                raise TypeError(string_err_message(k))
            elif type_dict[k] == bool:
                raise TypeError(bool_err_message(k))
                
