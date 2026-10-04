import numpy as np
import pandas as pd

#FROM https://community.backtrader.com/topic/2671/converting-pinescript-indicators/3
def line2arr(line, size=-1):
    '''
    Creates an numpy array from a backtrader line

    This method wraps the lines array in numpy. This can
    be used for conditions.
    '''
    if size <= 0:
        return np.array(line.array)
    else:
        return np.array(line.get(size=size))

def change(x, lenght = 1):
    '''
    change Difference between current value and previous, x - x[y].
    change(source, length) → series
    change(source) → series
    RETURNS
    Differences series.
    ARGUMENTS
    source (series)
    length (integer) Offset from the current bar to the previous bar. Optional, if not given, length = 1 is used.
    '''
    return x - x.shift(lenght)

def suma(x, lenght = 1):
    '''
    The sum function returns the sliding sum of last y values of x.
    sum(source, length) → series
    RETURNS
    Sum of x for y bars back.
    ARGUMENTS
    source (series) Series of values to process.
    length (integer) Number of bars (length).
    '''
    return x.rolling(lenght, min_periods=1).sum()

def na(val):
    '''
    RETURNS
    true if x is not a valid number (x is NaN), otherwise false.
    '''
    return val != val


def nz(x, y=None):
    '''
    RETURNS
    Two args version: returns x if it's a valid (not NaN) number, otherwise y
    One arg version: returns x if it's a valid (not NaN) number, otherwise 0
    ARGUMENTS
    x (val) Series of values to process.
    y (float) Value that will be inserted instead of all NaN values in x series.
    '''
    return x.fillna(0)
    # if type(x) == pd.core.series.Series:
    #     x = x.to_numpy()
    #
    # if isinstance(x, np.generic):
    #     return x.fillna(y or 0)
    # if x != x:
    #     if y is not None:
    #         return y
    #     return 0
    # return x


def barssince(condition, occurrence=0):
    '''
    Impl of barssince

    RETURNS
    Number of bars since condition was true.
    REMARKS
    If the condition has never been met prior to the current bar, the function returns na.
    '''
    cond_len = len(condition)
    occ = 0
    since = 0
    res = float('nan')
    while cond_len - (since+1) >= 0:
        cond = condition[cond_len-(since+1)]
        # check for nan cond != cond == True when nan
        if cond and not cond != cond:
            if occ == occurrence:
                res = since
                break
            occ += 1
        since += 1
    return res


def valuewhen(condition, source, occurrence=0):
    '''
    Impl of valuewhen
    + added occurrence

    RETURNS
    Source value when condition was true
    '''
    res = float('nan')
    since = barssince(condition, occurrence)
    if since is not None:
        res = source[-(since+1)]
    return res