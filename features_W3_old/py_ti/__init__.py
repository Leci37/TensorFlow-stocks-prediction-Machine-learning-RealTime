import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from features_W3_old.py_ti.helper_loops import supertrend_loop
from features_W3_old.py_ti.moving_averages import moving_average_mapper
from features_W3_old.py_ti.check_errors import check_errors

sma = moving_average_mapper('sma')
ema = moving_average_mapper('ema')
wma = moving_average_mapper('wma')
hma = moving_average_mapper('hma')
wilders_ma = moving_average_mapper('wilders')
kama = moving_average_mapper('kama')
fma = moving_average_mapper('fma')


def returns(df, column='close', ret_method='simple',
            add_col=False, return_struct='numpy'):
    """ Calculate Returns

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    ret_method : String, optional. The default is 'simple'
        The kind of returns you want returned: 'simple' or 'log'
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to
        True, the function will add a column to the dataframe that was
        passed in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, ret_method=ret_method,
                 add_col=add_col, return_struct=return_struct)

    if ret_method == 'simple':
        returns = df[column].pct_change()
    elif ret_method == 'log':
        returns = np.log(df[column] / df[column].shift(1))

    if add_col == True:
        df[f'{ret_method}_ret'] = returns
        return df
    elif return_struct == 'pandas':
        return returns.to_frame(name=f'{ret_method}_ret')
    else:
        return returns.to_numpy()


def hvol(df, column='close', n=20, ret_method='simple', ddof=1,
         add_col=False, return_struct='numpy'):
    """ Calculate Annualized Historical Volatility

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20
        This is the lookback period for which you want to calculate
        historical volatility.
    ddof : Int, optional. The default is 1
        The degrees of freedom to feed into the standard deviation
        function of pandas: 1 is for sample standard deviation and
        0 is for population standard deviation.
    ret_method : String, optional. The default is 'simple'
        The kind of returns you want returned: 'simple' or 'log'
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to
        True, the function will add a column to the dataframe that was
        passed in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ret_method=ret_method, ddof=ddof,
                 add_col=add_col, return_struct=return_struct)

    rets = returns(df, column=column, ret_method=ret_method)
    _df = pd.DataFrame(rets, columns=[column])
    hvol = _df.rolling(window=n).std(ddof=ddof) * 252 ** 0.5
    hvol.columns = [f'hvol({n})']

    if add_col == True:
        df[f'hvol({n})'] = hvol.to_numpy()
        return df
    elif return_struct == 'pandas':
        return hvol
    else:
        return hvol.to_numpy()


def momentum(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Momentum

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20
        The lookback period.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    mom = df[column].diff(n)

    if add_col == True:
        df[f'mom({n})'] = mom
        return df
    elif return_struct == 'pandas':
        return mom.to_frame(name=f'mom({n})')
    else:
        return mom.to_numpy()


def rate_of_change(df, column='close', n=20,
                   add_col=False, return_struct='numpy'):
    """ Rate of Change

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int.  The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20
        The lookback period.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    roc = df[column].diff(n) / df[column].shift(n) * 100

    if add_col == True:
        df[f'roc({n})'] = roc
        return df
    elif return_struct == 'pandas':
        return roc.to_frame(name=f'roc({n})')
    else:
        return roc.to_numpy()


def true_range(df, add_col=False, return_struct='numpy'):
    """ True Range

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift(1))
    lc = abs(df['low'] - df['close'].shift(1))
    tr = np.nanmax([hl, hc, lc], axis=0)

    if add_col == True:
        df['true_range'] = tr
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(tr, columns=['true_range'], index=df.index)
    else:
        return tr


def atr(df, n=20, ma_method='sma', add_col=False, return_struct='numpy'):
    """ Average True Range

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 20
        The lookback period.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the True Range.  Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    tr = true_range(df, add_col=False, return_struct='pandas')
    tr.columns = ['close']

    _ma = moving_average_mapper(ma_method)
    atr = _ma(tr, n=n)

    if add_col == True:
        df[f'{ma_method}_atr({n})'] = atr
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(atr,
                            columns=[f'{ma_method}_atr({n})'],
                            index=df.index)
    else:
        return atr


def atr_percent(df, column='close', n=20, ma_method='sma',
                add_col=False, return_struct='numpy'):
    """ Average True Range Percent

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to use as the denominator
        of the percentage calculation.
    n : Int, optional. The default is 20
        The lookback period.
    ma_method : String, optional.  The default is 'sma'
        The method of smoothing the True Range. Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    _atr = atr(df, n=n, ma_method=ma_method)
    atr_prcnt = (_atr / df[column]) * 100

    if add_col == True:
        df[f'atr_per({n})'] = atr_prcnt
        return df
    elif return_struct == 'pandas':
        return atr_prcnt.to_frame(name=f'atr_per({n})')
    else:
        return atr_prcnt.to_numpy()


def keltner_channels(df, column='close', n=20, ma_method='sma',
                     upper_factor=2.0, lower_factor=2.0,
                     add_col=False, return_struct='numpy'):
    """ Keltner Channels

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20
        The lookback period.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the True Range. Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    upper_factor : Float, optional. The default is 2.0
        The amount by which to multiply the ATR to create the upper channel.
    lower_factor : Float, optional. The default is 2.0
        The amount by which to multiply the ATR to create the lower channel.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 upper_factor=upper_factor, lower_factor=lower_factor,
                 add_col=add_col, return_struct=return_struct)

    _ma_func = moving_average_mapper(ma_method)

    _ma = _ma_func(df, column=column, n=n)
    _atr = atr(df, n=n, ma_method=ma_method)

    keltner_upper = _ma + (_atr * upper_factor)
    keltner_lower = _ma - (_atr * lower_factor)
    keltner = np.vstack((keltner_lower, keltner_upper)).transpose()

    if add_col == True:
        df[f'kelt({n})_lower'] = keltner_lower
        df[f'kelt({n})_upper'] = keltner_upper
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(keltner,
                            columns=[f'kelt({n})_lower', f'kelt({n})_upper'],
                            index=df.index)
    else:
        return keltner


def bollinger_bands(df, column='close', n=20, ma_method='sma', ddof=1,
                    upper_num_sd=2.0, lower_num_sd=2.0,
                    add_col=False, return_struct='numpy'):
    """ Bollinger Bands

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20
        The lookback period.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the column to obtain the middle band.
        Available smoothing methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    ddof : Int, optional. The default is 1
        The degrees of freedom to feed into the standard deviation
        function of pandas: 1 is for sample standard deviation and
        0 is for population standard deviation.
    upper_num_sd : Float, optional. The default is 2.0
        The amount by which to the standard deviation is multiplied and then
        added to the middle band to create the upper band.
    lower_num_sd : Float, optional. The default is 2.0
        The amount by which to the standard deviation is multiplied and then
        subtracted from the middle band to create the lower band.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 upper_num_sd=upper_num_sd, lower_num_sd=lower_num_sd,
                 add_col=add_col, return_struct=return_struct)

    _ma_func = moving_average_mapper(ma_method)

    price_std = (df[column].rolling(window=n).std(ddof=ddof)).to_numpy()
    mid_bb = _ma_func(df, column=column, n=n)
    lower_bb = mid_bb - (price_std * lower_num_sd)
    upper_bb = mid_bb + (price_std * upper_num_sd)
    bollinger = np.vstack((lower_bb, upper_bb)).transpose()

    if add_col == True:
        df[f'bb({n})_lower'] = lower_bb
        df[f'bb({n})_upper'] = upper_bb
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(bollinger,
                            columns=[f'bb({n})_lower', f'bb({n})_upper'],
                            index=df.index)
    else:
        return bollinger


def rsi(df, column='close', n=14, ma_method='sma',
        add_col=False, return_struct='numpy'):
    """ Relative Strength Index

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 14
        The lookback period.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the average up and average down variables.
        Available smoothing methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
        Traditionally, RSI is calculated using Wilder's moving average.
        This variable enables you to select other moving average types
        such as Simple, Exponential, Weighted, or Hull.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    change = pd.DataFrame(df[column].diff()).fillna(0)
    up, dn = change.copy(), change.copy()
    up[up < 0] = 0
    dn[dn > 0] = 0

    _ma_func = moving_average_mapper(ma_method)

    avg_up = _ma_func(up, column=column, n=n)
    avg_dn = -_ma_func(dn, column=column, n=n)

    rsi = np.where(avg_dn == 0.0, 100, 100.0 - 100.0 / (1 + avg_up / avg_dn))

    if add_col == True:
        df[f'rsi({n})'] = rsi
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(rsi, columns=[f'rsi({n})'], index=df.index)
    else:
        return rsi


def tsi(df, column='close', n=1, slow=25, fast=13, sig=7,
        ma_method='sma', add_col=False, return_struct='numpy'):
    """ True Strength Index

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 1
        The lookback period for the initial momentum calculation.
    slow : Int, optional. The default is 25
        The lookback period for smoothing the momentum calculations.
    fast : Int, optional. The default is 13
        The lookback period for smoothing the slow calculations.
    sig : Int, optional. The default is 7
        The lookback period for smoothing the true strength calculations.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the average up and average down variables.
        Available smoothing methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, slow=slow, fast=fast,
                 sig=sig, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    mom = momentum(df, column=column, n=n, return_struct='pandas')
    abs_mom = abs(mom)

    _ma_func = moving_average_mapper(ma_method)

    _slow = _ma_func(mom, column=f'mom({n})',
                     n=slow, return_struct='pandas')
    _abs_slow = _ma_func(abs_mom, column=f'mom({n})',
                         n=slow, return_struct='pandas')
    _fast = _ma_func(_slow, column=f'{ma_method}({slow})',
                     n=fast, return_struct='pandas')
    _abs_fast = _ma_func(_abs_slow, column=f'{ma_method}({slow})',
                         n=fast, return_struct='pandas')

    tsi = _fast / _abs_fast * 100
    signal = _ma_func(tsi, column=f'{ma_method}({fast})', n=sig)

    tsi_signal = np.vstack((tsi[f'{ma_method}({fast})'], signal)).transpose()

    if add_col == True:
        df[f'tsi({slow},{fast},{sig})'] = tsi_signal[:, 0]
        df['tsi_signal'] = tsi_signal[:, 1]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(tsi_signal,
                            columns=[f'tsi({slow},{fast},{sig})', 'tsi_signal'],
                            index=df.index)
    else:
        return tsi_signal


def adx(df, column='close', n=20, ma_method='sma',
        add_col=False, return_struct='numpy'):
    """ Average Directional Index

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is just a place holder for this function so that dataframes that
        pass to other functions don't encounter errors. It shouldn't be changed.
    n : Int, optional. The default is 20
        The lookback period for the all internal calculations of the ADX.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the internal calculation of the ADX.
        Available smoothing methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    _atr = atr(df, n=n, ma_method=ma_method)

    up = (df['high'] - df['high'].shift(1)).fillna(0)
    dn = (df['low'].shift(1) - df['low']).fillna(0)
    up[up < 0] = 0
    dn[dn < 0] = 0
    pos = pd.DataFrame(((up > dn) & (up > 0)) * up, columns=[column])
    neg = pd.DataFrame(((dn > up) & (dn > 0)) * dn, columns=[column])

    dm_pos = pos[column].rolling(n).sum()
    dm_neg = neg[column].rolling(n).sum()
    di_pos = 100 * (dm_pos / _atr)
    di_neg = 100 * (dm_neg / _atr)
    di_diff = abs(di_pos - di_neg)
    di_sum = di_pos + di_neg
    dx = pd.DataFrame(100 * (di_diff / di_sum), columns=[column]).fillna(0)

    _ma_func = moving_average_mapper(ma_method)
    _adx = _ma_func(dx, column=column, n=n)

    adx = np.vstack((_adx, di_pos, di_neg)).transpose()

    if add_col == True:
        df[f'adx({n})'] = adx[:, 0]
        df['DI+'] = adx[:, 1]
        df['DI-'] = adx[:, 2]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(adx,
                            columns=[f'adx({n})', 'di+', 'di-'],
                            index=df.index)
    else:
        return adx


def parabolic_sar(df, af_step=0.02, max_af=0.2,
                  add_col=False, return_struct='numpy'):
    """ Parabolic Stop-and-Reverse

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    af_step : Float, optional. The default is 0.02
        The acceleration factor.
    max_af : Float, optional. The default is 0.2
        The maximum value the accleration factor can have.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    from helper_loops import psar_loop

    check_errors(df=df, af_step=af_step, max_af=max_af,
                 add_col=add_col, return_struct=return_struct)

    _psar = df['close'].copy().to_numpy()
    high = df['high'].copy().to_numpy()
    low = df['low'].copy().to_numpy()

    psar = psar_loop(_psar, high, low, af_step, max_af)
    #psar = py_ti.helper_loops.psar_loop(_psar, high, low, af_step, max_af)

    if add_col == True:
        df['psar'] = psar
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(psar,
                            columns=['psar'],
                            index=df.index)
    else:
        return psar


def supertrend(df, column='close', n=20, ma_method='sma', factor=2.0,
               add_col=False, return_struct='numpy'):
    """ Supertrend
El indicador SuperTendencia es un excelente indicador de tendencia que se fundamenta en los precios. Diferencia con claridad la tendencia ascendente y descendente del mercado. También puede indicar niveles de soporte y resistencia. Pero veámoslo con mayor detalle.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the column that is sent to the loop to use for calculations.
        While uncommon, using 'open' instead of 'close' could be done.
    n : Int, optional. The default is 20
        This is the lookback period for the ATR that is used in the
        calculation and the beginning value for the loop.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the ATR.
    factor : Float, optional. The default is 2.0
        The value added and subtracted to the basic upper and basic
        lower bands.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    #from helper_loops import supertrend_loop

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 factor=factor, add_col=add_col, return_struct=return_struct)

    _atr = atr(df, n=n, ma_method=ma_method)
    hl_avg = (df['high'] + df['low']) / 2
    close = df[column].to_numpy()
    basic_ub = (hl_avg + factor * _atr).to_numpy()
    basic_lb = (hl_avg - factor * _atr).to_numpy()
    #supertrend = supertrend_loop(close, basic_ub, basic_lb, n)
    supertrend = supertrend_loop(close, basic_ub, basic_lb, n)

    if add_col == True:
        df[f'supertrend({n})'] = supertrend
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(supertrend,
                            columns=[f'supertrend({n})'],
                            index=df.index)
    else:
        return supertrend


def acc_dist(df, add_col=False, return_struct='numpy'):
    """ Accumulation/Distribution
This provides insight into how strong a trend is.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    clv = ((2 * df['close'] - df['high'] - df['low']) /
           (df['high'] - df['low']) * df['volume'])
    ad = clv.cumsum()

    if add_col == True:
        df['acc_dist'] = ad
        return df
    elif return_struct == 'pandas':
        return ad.to_frame(name='acc_dist')
    else:
        return ad.to_numpy()


def obv(df, add_col=False, return_struct='numpy'):
    """ On-Balance volume

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _mask = df['close'].mask(df['close'] >= df['close'].shift(1), other=1)
    mask = _mask.where(_mask == 1, other=-1)
    obv = (df['volume'] * mask).cumsum()

    if add_col == True:
        df['obv'] = obv
        return df
    elif return_struct == 'pandas':
        return obv.to_frame(name='obv')
    else:
        return obv.to_numpy()


def trad_pivots(df, add_col=False, return_struct='numpy'):
    """ Traditional Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.shift(1, axis=0)  # Use yesterday's HLC

    pp = (_df['high'] + _df['low'] + _df['close']) / 3
    r1 = 2 * pp - _df['low']
    s1 = 2 * pp - _df['high']
    # r2 = pp + _df['high'] - _df['low']
    # s2 = pp - _df['high'] - _df['low']  pp – (r1 – s1);
    r2 = pp + (r1 - s1)
    s2 = pp - (r1 - s1)
    r3 = 2 * pp + (_df['high'] - 2 * _df['low'])
    s3 = 2 * pp - (_df['high'] * 2 - _df['low'])

    pps = np.vstack((s3, s2, s1, pp, r1, r2, r3)).transpose()

    if add_col == True:
        df['s3'] = pps[:, 0]
        df['s2'] = pps[:, 1]
        df['s1'] = pps[:, 2]
        df['pp'] = pps[:, 3]
        df['r1'] = pps[:, 4]
        df['r2'] = pps[:, 5]
        df['r3'] = pps[:, 6]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3'],
                            index=df.index)
    else:
        return pps


def classic_pivots(df, add_col=False, return_struct='numpy'):
    """ Classic Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.shift(1, axis=0)  # Use yesterday's HLC

    pp = (_df['high'] + _df['low'] + _df['close']) / 3
    r1 = 2 * pp - _df['low']
    s1 = 2 * pp - _df['high']
    # r2 = pp + _df['high'] - _df['low']
    # s2 = pp - _df['high'] - _df['low']
    r2 = pp + (_df['high'] - _df['low'])
    s2 = pp - (_df['high'] - _df['low'])
    r3 = pp + 2 * (_df['high'] - _df['low'])
    s3 = pp - 2 * (_df['high'] - _df['low'])

    pps = np.vstack((s3, s2, s1, pp, r1, r2, r3)).transpose()

    if add_col == True:
        df['s3'] = pps[:, 0]
        df['s2'] = pps[:, 1]
        df['s1'] = pps[:, 2]
        df['pp'] = pps[:, 3]
        df['r1'] = pps[:, 4]
        df['r2'] = pps[:, 5]
        df['r3'] = pps[:, 6]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3'],
                            index=df.index)
    else:
        return pps


def fibonacci_pivots(df, add_col=False, return_struct='numpy'):
    """ Fibonacci Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.shift(1, axis=0)  # Use yesterday's HLC

    pp = (_df['high'] + _df['low'] + _df['close']) / 3
    r1 = pp + 0.382 * (_df['high'] - _df['low'])
    s1 = pp - 0.382 * (_df['high'] - _df['low'])
    r2 = pp + 0.618 * (_df['high'] - _df['low'])
    s2 = pp - 0.618 * (_df['high'] - _df['low'])
    r3 = pp + (_df['high'] - _df['low'])
    s3 = pp - (_df['high'] - _df['low'])

    pps = np.vstack((s3, s2, s1, pp, r1, r2, r3)).transpose()

    if add_col == True:
        df['s3'] = pps[:, 0]
        df['s2'] = pps[:, 1]
        df['s1'] = pps[:, 2]
        df['pp'] = pps[:, 3]
        df['r1'] = pps[:, 4]
        df['r2'] = pps[:, 5]
        df['r3'] = pps[:, 6]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3'],
                            index=df.index)
    else:
        return pps


def woodie_pivots(df, add_col=False, return_struct='numpy'):
    """ Woodie Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.copy()  # Use today's open and yesterday's HL
    _df['prev_high'] = df['high'].shift(1)
    _df['prev_low'] = df['low'].shift(1)

    pp = (_df['prev_high'] + _df['prev_low'] + 2 * _df['open']) / 4
    r1 = pp * 2 - _df['prev_low']
    s1 = pp * 2 - _df['prev_high']
    r2 = pp + (_df['prev_high'] - _df['prev_low'])
    s2 = pp - (_df['prev_high'] - _df['prev_low'])
    r3 = _df['prev_high'] + 2 * (pp - _df['prev_low'])
    s3 = _df['prev_low'] - 2 * (_df['prev_high'] - pp)

    pps = np.vstack((s3, s2, s1, pp, r1, r2, r3)).transpose()

    if add_col == True:
        df['s3'] = pps[:, 0]
        df['s2'] = pps[:, 1]
        df['s1'] = pps[:, 2]
        df['pp'] = pps[:, 3]
        df['r1'] = pps[:, 4]
        df['r2'] = pps[:, 5]
        df['r3'] = pps[:, 6]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3'],
                            index=df.index)
    else:
        return pps


def demark_pivots(df, add_col=False, return_struct='numpy'):
    """ Demark Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.shift(1, axis=0)  # Use yesterday's HLC

    if (_df['open'] == _df['close']).all():
        num = 2 * _df['close'] + _df['high'] + _df['low']
    elif (_df['close'] > _df['open']).all():
        num = 2 * _df['high'] + _df['low'] + _df['close']
    else:
        num = 2 * _df['low'] + _df['high'] + _df['close']

    pp = num / 4
    r1 = num / 2 - _df['low']
    s1 = num / 2 - _df['high']

    pps = np.vstack((s1, pp, r1)).transpose()

    if add_col == True:
        df['s1'] = pps[:, 0]
        df['pp'] = pps[:, 1]
        df['r1'] = pps[:, 2]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s1', 'pp', 'r1'],
                            index=df.index)
    else:
        return pps


# Camarilla Pivots
def camarilla_pivots(df, add_col=False, return_struct='numpy'):
    """ Camarilla Pivot Points

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    _df = df.shift(1, axis=0)  # Use yesterday's HLC

    pp = (_df['high'] + _df['low'] + _df['close']) / 3
    r1 = _df['close'] + 1.1 * (_df['high'] - _df['low']) / 12
    s1 = _df['close'] - 1.1 * (_df['high'] - _df['low']) / 12
    r2 = _df['close'] + 1.1 * (_df['high'] - _df['low']) / 6
    s2 = _df['close'] - 1.1 * (_df['high'] - _df['low']) / 6
    r3 = _df['close'] + 1.1 * (_df['high'] - _df['low']) / 4
    s3 = _df['close'] - 1.1 * (_df['high'] - _df['low']) / 4

    pps = np.vstack((s3, s2, s1, pp, r1, r2, r3)).transpose()

    if add_col == True:
        df['s3'] = pps[:, 0]
        df['s2'] = pps[:, 1]
        df['s1'] = pps[:, 2]
        df['pp'] = pps[:, 3]
        df['r1'] = pps[:, 4]
        df['r2'] = pps[:, 5]
        df['r3'] = pps[:, 6]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(pps,
                            columns=['s3', 's2', 's1', 'pp', 'r1', 'r2', 'r3'],
                            index=df.index)
    else:
        return pps


# Full Stochastic Oscillator
def stochastic(df, n_k=14, n_d=3, n_slow=1, ma_method='sma',
               add_col=False, return_struct='numpy'):
    """ Stochastic Oscillator

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n_k : Int, optional. The default is 14
        The lookback period over which the highest high and lowest low
        are determined to compute %k.
    n_d : Int, optional. The default is 3
        The number of periods that are used to smooth the full_k.
    n_slow : Int, optional. The default is 1
        The number of periods that are used to smooth %k. If left at the
        default (1) the function returns values that match a "Fast Stochastic".
        If a different value is used, the function return values that match a
        "Slow Stochastic".
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the stochastics.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n_k=n_k, n_d=n_d, n_slow=n_slow, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    low = df['low'].rolling(n_k).min()
    high = df['high'].rolling(n_k).max()
    percent_k = ((df['close'] - low) / (high - low) * 100).to_frame(name='%k')

    _ma_func = moving_average_mapper(ma_method)

    full_k = _ma_func(percent_k, column='%k', n=n_slow,
                      return_struct='pandas')
    full_d = _ma_func(full_k, column=f'{ma_method}({n_slow})', n=n_d,
                      return_struct='pandas')

    full_stoch = np.vstack((full_k[f'{ma_method}({n_slow})'],
                            full_d[f'{ma_method}({n_d})'])).transpose()

    if add_col == True:
        df[f'%k({n_k},{n_slow})'] = full_k
        df[f'%d({n_d})'] = full_d
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(full_stoch,
                            columns=[f'%k({n_k},{n_slow})', f'%d({n_d})'],
                            index=df.index)
    else:
        return full_stoch


# Stochastic-RSI Oscillator
def stochastic_rsi(df, n_k=14, n_d=3, n_slow=1, ma_method='sma',
                   add_col=False, return_struct='numpy'):
    """ Stochastic-RSI Oscillator -- Using a stochastic oscillator on RSI values

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n_k : Int, optional. The default is 14
        The lookback period over which the highest high and lowest low
        are determined to compute %k.
    n_d : Int, optional. The default is 3
        The number of periods that are used to smooth the full_k.
    n_slow : Int, optional. The default is 1
        The number of periods that are used to smooth %k. If left at the
        default (1) the function returns values that match a "Fast Stochastic".
        If a different value is used, the function return values that match a
        "Slow Stochastic".
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the RSI and stochastics.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n_k=n_k, n_d=n_d, n_slow=n_slow, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    rsi_df = rsi(df, n=n_k, ma_method=ma_method, return_struct='pandas')

    low = rsi_df[f'rsi({n_k})'].rolling(n_k).min()
    high = rsi_df[f'rsi({n_k})'].rolling(n_k).max()
    percent_k = ((rsi_df[f'rsi({n_k})'] - low) /
                 (high - low) * 100).to_frame(name='%k')

    _ma_func = moving_average_mapper(ma_method)

    full_k = _ma_func(percent_k, column='%k', n=n_slow,
                      return_struct='pandas')
    full_d = _ma_func(full_k, column=f'{ma_method}({n_slow})', n=n_d,
                      return_struct='pandas')

    stoch_rsi = np.vstack((full_k[f'{ma_method}({n_slow})'],
                           full_d[f'{ma_method}({n_d})'])).transpose()

    if add_col == True:
        df[f'stoch_RSI %k({n_k},{n_slow})'] = full_k
        df[f'stoch_RSI %d({n_d})'] = full_d
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(stoch_rsi,
                            columns=[f'stoch_RSI %k({n_k},{n_slow})',
                                     f'stoch_RSI %d({n_d})'],
                            index=df.index)
    else:
        return stoch_rsi


# RSI-Stochastic Oscillator
def rsi_stochastic(df, n=14, ma_method='sma',
                   add_col=False, return_struct='numpy'):
    """ RSI-Stochastic Oscillator -- Using RSI on stochastic values

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 14
        The lookback period over which the highest high and lowest low
        are determined to compute %k.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the RSI and stochastics.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    low = df['low'].rolling(n).min()
    high = df['high'].rolling(n).max()
    percent_k = ((df['close'] - low) / (high - low) * 100).to_frame(name='%k')

    rsi_stoch = rsi(percent_k, column='%k', n=n, ma_method=ma_method)

    if add_col == True:
        df[f'RSI_stoch({n})'] = rsi_stoch
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(rsi_stoch,
                            columns=[f'RSI_stoch({n})'],
                            index=df.index)
    else:
        return rsi_stoch


# Ultimate Oscillator
def ultimate_oscillator(df, n_fast=7, n_med=14, n_slow=28,
                        add_col=False, return_struct='numpy'):
    """ Ultimate Oscillator

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n_fast : Int, optional. The default is 7
        The lookback period over which to compute the fast rolling ratio
    n_med : Int, optional. The default is 14
        The lookback period over which to compute the intermediate rolling ratio
    n_slow : Int, optional. The default is 28
        The lookback period over which to compute the slow rolling ratio
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n_fast=n_fast, n_med=n_med, n_slow=n_slow,
                 add_col=add_col, return_struct=return_struct)

    df['prev_clo'] = df['close'].shift(1)
    bp = df['close'] - (df[['low', 'prev_clo']].min(axis=1))
    tr = true_range(df, return_struct='pandas')

    first_ma = (bp.rolling(n_fast).sum() /
                tr['true_range'].rolling(n_fast).sum())
    second_ma = (bp.rolling(n_med).sum() /
                 tr['true_range'].rolling(n_med).sum())
    third_ma = (bp.rolling(n_slow).sum() /
                tr['true_range'].rolling(n_slow).sum())

    ult_osc = ((first_ma * 4 + second_ma * 2 + third_ma) / 7 * 100).to_numpy()

    if add_col == True:
        df[f'ultimate_oscillator({n_fast},{n_med},{n_slow})'] = ult_osc
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(ult_osc,
                            columns=[f'ultimate_oscillator({n_fast},{n_med},{n_slow})'],
                            index=df.index)
    else:
        return ult_osc


# Trix
def trix(df, column='close', n=9, sig=3, ma_method='ema',
         add_col=False, return_struct='numpy'):
    """ TRIX - Triple smoothed exponential moving average

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the column that the moving averages will be calculated on.
    n : Int, optional. The default is 9
        The lookback period over which to compute the moving averages.
    sig : Int, optional. The default is 3
        The lookback period over which the signal line is calculated.
    ma_method : String, optional. The default is 'ema'
        Traditionally, TRIX is an exponential moving average smoothed
        3 times. This variable enables you to select other moving average
        types such as Simple, Weighted, Hull, or Wilders.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, sig=sig, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    _ma_func = moving_average_mapper(ma_method)

    ma_1 = _ma_func(df, column=column, n=n,
                    return_struct='pandas')
    ma_2 = _ma_func(ma_1, column=f'{ma_method}({n})', n=n,
                    return_struct='pandas')
    ma_3 = _ma_func(ma_2, column=f'{ma_method}({n})', n=n,
                    return_struct='pandas')

    ma_3['prev'] = ma_3[f'{ma_method}({n})'].shift(1)
    trix = ((ma_3[f'{ma_method}({n})'] /
             ma_3['prev'] - 1) * 10000).to_frame(name='close')
    trix['signal'] = _ma_func(trix, n=sig)

    trix = trix.to_numpy()

    if add_col == True:
        df[f'trix_({n})'] = trix[:, 0]
        df[f'trix_signal({sig})'] = trix[:, 1]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(trix,
                            columns=[f'trix_({n})',
                                     f'trix_signal({sig})'],
                            index=df.index)
    else:
        return trix


# MACD
def macd(df, column='close', n_fast=12, n_slow=26, n_macd=9, ma_method='ema',
         add_col=False, return_struct='numpy'):
    """ MACD - Moving average convergence divergence

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the column that the moving averages will be calculated on.
    n_fast : Int, optional. The default is 12
        The lookback period over which to compute the first (fast) moving average.
    n_slow : Int, optional. The default is 26
        The lookback period over which to compute the second (slow) moving average.
    n_macd : Int, optional. The default is 9
        The lookback period over which to compute the moving average of the MACD to
        generate the signal line.
    ma_method : String, optional. The default is 'ema'
        Traditionally, MACD is calculated using exponential moving averages.
        This variable enables you to select other moving average types
        such as Simple, Weighted, Hull, or Wilders.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n_fast=n_fast, n_slow=n_slow,
                 n_macd=n_macd, ma_method=ma_method, add_col=add_col,
                 return_struct=return_struct)

    _ma_func = moving_average_mapper(ma_method)

    ma_fast = _ma_func(df, column=column, n=n_fast, return_struct='pandas')
    ma_slow = _ma_func(df, column=column, n=n_slow, return_struct='pandas')
    macd = (ma_fast[f'{ma_method}({n_fast})'] -
            ma_slow[f'{ma_method}({n_slow})']).to_frame(name='macd')
    macd['signal'] = _ma_func(macd, column='macd', n=n_macd)

    macd = macd.to_numpy()

    if add_col == True:
        df[f'macd({n_fast},{n_slow})'] = macd[:, 0]
        df[f'signal({n_macd})'] = macd[:, 1]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(macd,
                            columns=[f'macd({n_fast},{n_slow})', f'signal({n_macd})'],
                            index=df.index)
    else:
        return macd


# Triangular RSI
def triangular_rsi(df, column='close', n=5, ma_method='sma',
                   add_col=False, return_struct='numpy'):
    """ Triangular RSI

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        This is the column that the moving averages will be calculated on.
    n : Int, optional. The default is 5
        The lookback period over which to compute all 3 rsi's.
    ma_method : String, optional. The default is 'sma'
        Traditionally, RSI is calculated using Wilder's moving average.
        This variable enables you to select other moving average types
        such as Simple, Exponential, Weighted, or Hull.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    rsi_1 = rsi(df, column=column, n=n, ma_method=ma_method,
                return_struct='pandas')
    rsi_2 = rsi(rsi_1, column=f'rsi({n})', n=n, ma_method=ma_method,
                return_struct='pandas')
    tri_rsi = rsi(rsi_2, column=f'rsi({n})', n=n, ma_method=ma_method)

    if add_col == True:
        df[f'triangular_rsi({n})'] = tri_rsi
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(tri_rsi,
                            columns=[f'triangular_rsi({n})'],
                            index=df.index)
    else:
        return tri_rsi


# Mass Index
def mass_index(df, n=9, n_sum=25, ma_method='ema',
               add_col=False, return_struct='numpy'):
    """ Mass Index
que se utiliza en el análisis técnico para predecir cambios de tendencia. Se basa en la noción de que existe una tendencia a la reversión cuando el rango de precios se amplía y, por lo tanto, compara los rangos comerciales anteriores.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 9
        The lookback period over which the 2 moving averages are calculated.
    n_sum : Int, optional. The default is 25
        The lookback period over which the mass is summed to calculate the index.
    ma_method : String, optional. The default is 'ema'
        Traditionally, Mass Index is calculated using an Exponential moving average.
        This variable enables you to select other moving average types
        such as Simple, Weighted, Hull, or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, n_sum=n_sum, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    _ma_func = moving_average_mapper(ma_method)

    high_low = (df['high'] - df['low']).to_frame(name='close')
    ma_1 = _ma_func(high_low, n=n, return_struct='pandas')
    ma_2 = _ma_func(ma_1, column=f'{ma_method}({n})', n=n, return_struct='pandas')
    mass = ma_1 / ma_2

    mass_idx = (mass.rolling(n_sum).sum()).to_numpy()

    if add_col == True:
        df[f'mass_index({n},{n_sum})'] = mass_idx
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(mass_idx,
                            columns=[f'mass_index({n},{n_sum})'],
                            index=df.index)
    else:
        return mass_idx


# Vortex Indicator
def vortex(df, n=5, add_col=False, return_struct='numpy'):
    """ Vortex Indicator
El Indicador Vortex (VI) está compuesto de 2 líneas que muestran tanto el movimiento de tendencia positivo (VI+) como el negativo (VI-). El indicador se ideó fruto de una fuente de inspiración basada en ciertos movimientos del agua y fue desarrollado por Etienne Botes y Douglas Siepman. El indicador Vortex tiene una aplicación relativamente simple: los traders lo utilizan para identificar el inicio de una tendencia.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 5
        The lookback period over which vm_pos and vm_neg are summed.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, add_col=add_col, return_struct=return_struct)

    vm_pos = abs(df['high'] - df['low'].shift(1))
    vm_neg = abs(df['low'] - df['high'].shift(1))
    tr = true_range(df, return_struct='pandas')

    vm_pos_sum = vm_pos.rolling(n).sum()
    vm_neg_sum = vm_neg.rolling(n).sum()
    tr_sum = tr['true_range'].rolling(n).sum()

    vtx_pos = vm_pos_sum / tr_sum
    vtx_neg = vm_neg_sum / tr_sum

    vtx = np.vstack((vtx_pos, vtx_neg)).transpose()

    if add_col == True:
        df[f'vortex_pos({n})'] = vtx[:, 0]
        df[f'vortex_neg({n})'] = vtx[:, 1]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(vtx,
                            columns=[f'vortex_pos({n})', f'vortex_neg({n})'],
                            index=df.index)
    else:
        return vtx


# KST Oscillator
def kst(df, n_1=10, n_2=15, n_3=20, n_4=30, ma_1=10, ma_2=10, ma_3=10, ma_4=15,
        sig=9, ma_method='sma', add_col=False, return_struct='numpy'):
    """ KST Oscillator -- Martin Pring's "Know Sure Thing"

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n_1 : Int, optional. The default is 10
        The lookback period over which the first ROC is calculated.
    n_2 : Int, optional. The default is 15
        The lookback period over which the second ROC is calculated.
    n_3 : Int, optional. The default is 20
        The lookback period over which the third ROC is calculated.
    n_4 : Int, optional. The default is 30
        The lookback period over which the fourth ROC is calculated.
    ma_1 : Int, optional. The default is 10
        The lookback period over which the first ROC is smoothed.
    ma_2 : Int, optional. The default is 10
        The lookback period over which the second ROC is smoothed.
    ma_3 : Int, optional. The default is 10
        The lookback period over which the third ROC is smoothed.
    ma_4 : Int, optional. The default is 15
        The lookback period over which the fourth ROC is smoothed.
    sig : Int, optional. The default is 9
        The lookback over which the final KST is smoothed to calculated
        the signal.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing for the entire function. Traditionally, this
        oscillator uses a Simple Moving Average. This input variable enables
        the user to use other types of smoothing such as Exponential, Weighted,
        Hull, or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n_1=n_1, n_2=n_2, n_3=n_3, n_4=n_4,
                 ma_1=ma_1, ma_2=ma_2, ma_3=ma_3, ma_4=ma_4,
                 sig=sig, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    roc_1 = rate_of_change(df, n=n_1, return_struct='pandas')
    roc_2 = rate_of_change(df, n=n_2, return_struct='pandas')
    roc_3 = rate_of_change(df, n=n_3, return_struct='pandas')
    roc_4 = rate_of_change(df, n=n_4, return_struct='pandas')

    _ma_func = moving_average_mapper(ma_method)

    roc_ma_1 = _ma_func(roc_1, column=f'roc({n_1})', n=ma_1, return_struct='pandas')
    roc_ma_2 = _ma_func(roc_2, column=f'roc({n_2})', n=ma_2, return_struct='pandas')
    roc_ma_3 = _ma_func(roc_3, column=f'roc({n_3})', n=ma_3, return_struct='pandas')
    roc_ma_4 = _ma_func(roc_4, column=f'roc({n_4})', n=ma_4, return_struct='pandas')

    kst = pd.DataFrame(roc_ma_1[f'{ma_method}({ma_1})'] +
                       roc_ma_2[f'{ma_method}({ma_2})'] * 2 +
                       roc_ma_3[f'{ma_method}({ma_3})'] * 3 +
                       roc_ma_4[f'{ma_method}({ma_4})'] * 4,
                       columns=['close'])
    kst['signal'] = _ma_func(kst, n=sig)

    kst = kst.to_numpy()

    if add_col == True:
        df['kst'] = kst[:, 0]
        df['kst_signal'] = kst[:, 1]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(kst,
                            columns=['kst', 'kst_signal'],
                            index=df.index)
    else:
        return kst


# Commodity Channel Index
def cci(df, n=14, ma_method='sma', constant=0.015,
        add_col=False, return_struct='numpy'):
    """ Commodity Channel Index

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 14
        The lookback period over which the pivot points and MAD are averaged.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing for the entire function. Traditionally, this
        oscillator uses a Simple Moving Average. This input variable enables
        the user to use other types of smoothing such as Exponential, Weighted,
        Hull, or Wilder's.
    constant : Float, optional. The default is 0.015.
        This is the constant used by the creator of this indicator.  It can be
        changed by the user.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    pp = ((df['high'] + df['low'] + df['close']) / 3).to_frame(name='close')
    mad_func = lambda x: np.fabs(x - x.mean()).mean()
    mad = pp.rolling(n).apply(mad_func)

    _ma_func = moving_average_mapper(ma_method)

    pp_avg = _ma_func(pp, n=n)

    cci = ((pp['close'] - pp_avg) / (mad['close'] * constant)).to_numpy()

    if add_col == True:
        df[f'cci({n})'] = cci
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(cci,
                            columns=[f'cci({n})'],
                            index=df.index)
    else:
        return cci


# Chaikin Oscillator
def chaikin_oscillator(df, n_slow=10, n_fast=3, ma_method='ema',
                       add_col=False, return_struct='numpy'):
    """ Chaikin Oscillator
 Cuanto más cerca esté al máximo el nivel de cierre de la acción o del índice, más activa será la acumulación.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n_slow : Int, optional. The default is 10
        The longer lookback period over which the adl is averaged.
    n_fast : Int, optional. The default is 3
        The shorter lookback period over which the adl is averaged.
    ma_method : String, optional. The default is 'ema'
        The method of smoothing for the adl. Traditionally, this
        oscillator uses an Exponential Moving Average. This input variable enables
        the user to use other types of smoothing such as Simple, Weighted,
        Hull, or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n_slow=n_slow, n_fast=n_fast, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    mfm = (((df['close'] - df['low']) -
            (df['high'] - df['close'])) /
           (df['high'] - df['low']))

    mfv = mfm * df['volume']
    adl = (mfv.cumsum()).to_frame(name='close')

    _ma_func = moving_average_mapper(ma_method)
    chaikin = _ma_func(adl, n=n_fast) - _ma_func(adl, n=n_slow)

    if add_col == True:
        df[f'chaikin({n_slow},{n_fast})'] = chaikin
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(chaikin,
                            columns=[f'chaikin({n_slow},{n_fast})'],
                            index=df.index)
    else:
        return chaikin


# Money Flow Index
def money_flow_index(df, n=14, add_col=False, return_struct='numpy'):
    """ Money Flow Index

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 14
        The lookback period over which the up and down raw money flow
        is summed.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, add_col=add_col, return_struct=return_struct)

    pp = (df['high'] + df['low'] + df['close']) / 3
    rmf = pp * df['volume']

    _mask = pp.mask(pp >= pp.shift(1), other=1)
    mask = _mask.where(_mask == 1, other=-1)
    volume = mask * rmf

    up, dn = volume.copy(), volume.copy()
    up[up < 0] = 0
    dn[dn > 0] = 0

    sum_up = up.rolling(n).sum()
    sum_dn = abs(dn).rolling(n).sum()
    mfr = sum_up / sum_dn

    mfi = (100.0 - 100.0 / (1 + mfr)).to_numpy()

    if add_col == True:
        df[f'mfi({n})'] = mfi
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(mfi,
                            columns=[f'mfi({n})'],
                            index=df.index)
    else:
        return mfi


# Force Index
def force_index(df, column='close', n=13, ma_method='ema',
                add_col=False, return_struct='numpy'):
    """ Force Index
El índice de fuerza es un indicador utilizado en el análisis técnico para ilustrar qué tan fuerte es la presión real de compra o venta. Los valores positivos altos significan que hay una fuerte tendencia al alza, y los valores bajos significan una fuerte tendencia a la baja.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        The column to use to calculate the change that is multiplied by
        the volume.
    n : Int, optional. The default is 13
        The lookback period over which the force number is smoothed.
    ma_method : String, optional. The default is 'ema'
        The method of smoothing the force number. Traditionally, this
        indicator uses an Exponential Moving Average. This input variable enables
        the user to use other types of smoothing such as Simple, Weighted,
        Hull, or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    force = (df[column].diff(1).fillna(0) * df['volume']).to_frame(name='close')

    _ma_func = moving_average_mapper(ma_method)

    fi = _ma_func(force, n=n)

    if add_col == True:
        df[f'force_index({n})'] = fi
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(fi,
                            columns=[f'force_index({n})'],
                            index=df.index)
    else:
        return fi


# Ease of Movement
def ease_of_movement(df, n=14, ma_method='sma',
                     add_col=False, return_struct='numpy'):
    """ Ease of Movement - Note that we scale the volume using a
     La intención es usar este valor para discernir si los precios pueden subir o bajar, con poca resistencia en el movimiento direccional.
    max absolute scaling technique. The .abs() is left out of the function
    because volume is always positive. This can be replicated with SKLearn's
    MaxAbsScaler function as well. This adjustment allows values to be compared
    from one security to another.

    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 14
        The lookback period over which the raw ease of movement is smoothed.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the raw ease of movement. Traditionally, this
        indicator uses an Simple Moving Average. This input variable enables
        the user to use other types of smoothing such as Exponential, Weighted,
        Hull, or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    distance = (((df['high'] + df['low']) / 2) -
                ((df['high'].shift(1) + df['low'].shift(1)) / 2))

    scaled_volume = df['volume'] / df['volume'].max()

    box_ratio = scaled_volume / (df['high'] - df['low'])
    eom_raw = (distance / box_ratio).to_frame(name='close')

    _ma_func = moving_average_mapper(ma_method)

    eom = _ma_func(eom_raw, n=n)

    if add_col == True:
        df[f'ease_of_movement({n})'] = eom
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(eom,
                            columns=[f'ease_of_movement({n})'],
                            index=df.index)
    else:
        return eom


# Coppock Curve
def coppock(df, column='close', n_1=14, n_2=11, ma_1=10, ma_method='wma',
            add_col=False, return_struct='numpy'):
    """ Coppock Curve
    Una pregunta recurrente que me hacen es, ¿cómo puedo saber cuándo van a caer «las bolsas»?
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    column : String, optional. The default is 'close'
        The column to use to calculate the ROC's in the function.
    n_1 : Int, optional. The default is 14
        The lookback period over which the first ROC is calculated.
    n_2 : Int, optional. The default is 11
        The lookback period over which the second ROC is calculated.
    ma_1 : Int, optional. The default is 10
        The lookback period over which the summed ROC's are smoothed.
    ma_method : String, optional. The default is 'sma'
        The method of smoothing the raw ease of movement. Traditionally, this
        indicator uses a Weighted Moving Average. This input variable enables
        the user to use other types of smoothing such as Simple, Exponential, Hull,
        or Wilder's.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n_1=n_1, n_2=n_2, ma_1=ma_1,
                 ma_method=ma_method, add_col=add_col, return_struct=return_struct)

    roc_1 = rate_of_change(df, column=column, n=n_1, return_struct='pandas')
    roc_2 = rate_of_change(df, column=column, n=n_2, return_struct='pandas')

    roc_sum = (roc_1[f'roc({n_1})'] + roc_2[f'roc({n_2})']).to_frame(name='close')

    _ma_func = moving_average_mapper(ma_method)

    coppock = _ma_func(roc_sum, n=ma_1)

    if add_col == True:
        df[f'coppock({n_1},{n_2},{ma_1})'] = coppock
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(coppock,
                            columns=[f'coppock({n_1},{n_2},{ma_1})'],
                            index=df.index)
    else:
        return coppock


# Donchian Channels
def donchian_channels(df, n=20, add_col=False, return_struct='numpy'):
    """ Donchian Curve
El canal de Donchian es un indicador útil para ver la volatilidad de un precio de mercado. Si un precio es estable, el canal de Donchian será relativamente estrecho. Si el precio fluctúa mucho, el canal Donchian será más ancho.
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 20
        The lookback period from which the rolling high and low is taken.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, add_col=add_col, return_struct=return_struct)

    upper = df['high'].rolling(n).max()
    lower = df['low'].rolling(n).min()
    center = (upper + lower) / 2

    donchian = np.vstack((lower, center, upper)).transpose()

    if add_col == True:
        df[f'donchian_lower({n})'] = donchian[:, 0]
        df[f'donchian_center({n})'] = donchian[:, 1]
        df[f'donchian_upper({n})'] = donchian[:, 2]
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(donchian,
                            columns=[f'donchian_lower({n})',
                                     f'donchian_center({n})',
                                     f'donchian_upper({n})'],
                            index=df.index)
    else:
        return donchian


# Choppiness Index
def choppiness(df, n=14, add_col=False, return_struct='numpy'):
    """ Choppiness Index
El Índice Choppiness (CHOP) es un indicador diseñado para determinar si el mercado es variable (negociaciones transversales) o no variable (negociaciones dentro de una tendencia en cualquier dirección).
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date. open/high/low/close should all
        be floats. volume should be an int. The date index should be
        a Datetime.
    n : Int, optional. The default is 14
        The lookback period over which the true range is summed.
    add_col : Boolean, optional. The default is False
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'
        Only two values accepted: 'numpy' and 'pandas'. If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in

    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, n=n, add_col=add_col, return_struct=return_struct)

    _atr = atr(df, n=1, ma_method='sma', return_struct='pandas')
    hh = df['high'].rolling(n).max()
    ll = df['low'].rolling(n).min()
    choppiness = (100 *
                  np.log10(_atr['sma_atr(1)'].rolling(n).sum() / (hh - ll)) /
                  np.log10(n)).to_numpy()

    if add_col == True:
        df[f'choppiness({n})'] = choppiness
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(choppiness,
                            columns=[f'choppiness({n})'],
                            index=df.index)
    else:
        return choppiness
