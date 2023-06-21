import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] =  (40,30)


def get_buy_sell_point_HT_pp(df_l, period=12, rolling=24):
    print("get_HT_pp df.shape:", df_l.shape, " len_right: ", rolling, " len_left: ", period)

    # https://stackoverflow.com/questions/64019553/how-pivothigh-and-pivotlow-function-work-on-tradingview-pinescript
    # se puede usar high and low , pero distorsina los puntos
    pivots_hh = df_l['high'].shift(-period).rolling(rolling).max() # , fill_value=0
    # For 'High' pivots pd.Series 'high_column' and:
    pivots_ll = df_l['low'].shift(-period).rolling(rolling).min() #, fill_value=0
    df_l['pp_high'] = pivots_hh
    df_l['pp_low'] = pivots_ll
    df_l.insert(loc=len(df_l.columns), column="touch_low", value=False)
    df_l.insert(loc=len(df_l.columns), column="touch_high", value=False)
    df_l.loc[(df_l["pp_low"] >= df_l['low']), "touch_low"] = True
    df_l.loc[(df_l["pp_high"] <= df_l['high']), "touch_high"] = True

    df_details = pd.DataFrame()
    df_details['count'] = df_l[["touch_low", "touch_high"]].value_counts().to_frame()
    df_details['per%'] = df_l[["touch_low", "touch_high"]].value_counts(normalize=True).mul(100).round(2)
    # df_details.index = ['Buy_True/Sell_True', 'Buy_True/Sell_False', 'Buy_False/Sell_True', 'Buy_False/Sell_False']
    print("DEBUG Count ht pivot point stadistic balance ")
    print(df_details, "\n")
    df_l[["touch_low", "touch_high"]].count()

    df_l.insert(loc=len(df_l.columns), column="target", value=0)
    df_l.loc[(df_l["pp_low"] >= df_l['low']), "target"] = 1
    df_l.loc[(df_l["pp_high"] <= df_l['high']), "target"] = 2
    # df = pd.get_dummies(df_l['target'])

    return df_l['target'] ,df_l, df_details
