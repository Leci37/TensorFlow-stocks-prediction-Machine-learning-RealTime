import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] =  (40,30)
df_details = pd.DataFrame()


def get_buy_sell_point(df: pd.DataFrame, period=12, rolling=24):
    df = df.copy()

    # Touch High
    df['pivot_hi'] = df['high'].shift(-period).rolling(rolling).max()
    df['pivot_lo'] = df['low'].shift(-period).rolling(rolling).min()

    # Touch high/low of the pivot
    df['target'] = 0
    df.loc[df['pivot_lo'] >= df['low'], 'target'] = 1  # Touch Low
    df.loc[df['pivot_hi'] <= df['high'], 'target'] = 2  # Touch High
    df = pd.get_dummies(df['target'])

    return df.dropna()


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
    #INFO code
    df_l.describe()
    # df_details['count'] = df_l[["touch_low", "touch_high"]].value_counts().to_frame()
    # df_details['per%'] = df_l[["touch_low", "touch_high"]].value_counts(normalize=True).mul(100).round(2)
    # df_details.index = ['Buy_True/Sell_True', 'Buy_True/Sell_False', 'Buy_False/Sell_True', 'Buy_False/Sell_False']
    # print("DEBUG Count Grount True ht pivot point stadistic balance ")
    # print(df_details, "\n")
    df_l[["touch_low", "touch_high"]].count()
    # INFO code

    df_l.insert(loc=len(df_l.columns), column="target", value=0)
    df_l.loc[(df_l["pp_low"] >= df_l['low']), "target"] = 1
    df_l.loc[(df_l["pp_high"] <= df_l['high']), "target"] = 2
    df = pd.get_dummies(df_l['target'])

    return df.dropna() ,df_l, df_details


def print_result_get_buy_sell_point_HT_pp(df_l, path_plot):
    list_position = [0, 400, 800, 1200, 1600, 2000, 2400]
    for i, p in enumerate(list_position):
        ##only to see outputs/plots/target_Y_0x400.png it prety
        df_m = df_l[['close', 'high', 'low', 'pp_high', 'pp_low']][p:p + 300]
        df_m[[ 'target']] = df_l[['target']][p:p + 300].astype(float) -2 + df_l['close'][p:p + 300].mean()
        # df_m['target'] = (df_m['target'] -2 ) # to see it prety
        df_m[['close', 'target']].plot()
        # plt.show()
        path_img = path_plot+ str(p) + "x" + str(p + 300) + ".png"
        print(path_img)
        plt.savefig(path_img)