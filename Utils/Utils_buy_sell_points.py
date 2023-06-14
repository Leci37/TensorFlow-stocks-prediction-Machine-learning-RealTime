import numpy
import pandas as pd
from datetime import datetime

from Utils import Utils_Yfinance
import _KEYS_DICT
from LogRoot.Logging import Logger
from sklearn.metrics import confusion_matrix
Y_TARGET = 'buy_sell_point'

from Utils import  Utils_plotter

#When the TOLERANCE_LOST percentage is exceeded, the trade will be closed, and that would be the hypothetical STOP LOSS selling price.
MARGIN_LOST_WIN = 0.03  # porcentage del 5% 0.95
TOLERANCE_LOST = 1 - MARGIN_LOST_WIN
TOLERANCE_WIN = 1 + MARGIN_LOST_WIN

# Old method repleced by rolling_get_sell_price_POS_next_value
def rolling_get_sell_price_POS(rolling_col_slection):
    rolling_col_slection = [x + 100 for x in rolling_col_slection]
    start_value_buy = rolling_col_slection[0]
    raise ValueError("Old method repleced by rolling_get_sell_price_POS_next_value")

    update_value = start_value_buy
    for i in range(1, len(rolling_col_slection)):
        next_value = rolling_col_slection[i]

        if start_value_buy * (TOLERANCE_LOST + 0.025) > next_value:
            #print("STOP LOSS after n interactions (by start) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return start_value_buy * (
                    TOLERANCE_LOST + 0.025) - 100  # * TALERANCE_OF_LOST -100  #end_point_if -start_value_buy
        if update_value * (TOLERANCE_LOST + i / 220) > next_value:  # cada pasada de i se vuelve mas exigente
            #print("STOP LOSS after n interactions (by update) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return update_value * (
                    TOLERANCE_LOST + i / 220) - 100  # * (TALERANCE_OF_LOST + i/220)  -100   #end_point_if -start_value_buy

        if next_value > update_value:
            update_value = next_value

    #print("TAKE PROFICT after: ", len(rolling_col_slection), "interactions Win: ",update_value - start_value_buy)  # , i , " ".join(str(p) for p in a) )
    return update_value - 100

# Old method repleced by rolling_get_sell_price_NEG_next_value
def rolling_get_sell_price_NEG(rolling_col_slection):
    #si el next_value SUBE se tira fuera
    rolling_col_slection = [x + 100 for x in rolling_col_slection]
    start_value_buy = rolling_col_slection[0]
    raise ValueError("Old method repleced by rolling_get_sell_price_NEG_next_value")
    update_value = start_value_buy
    for i in range(1, len(rolling_col_slection)):
        next_value = rolling_col_slection[i]

        if start_value_buy * (TOLERANCE_WIN - 0.025) < next_value:
            #print("STOP LOSS after n interactions (by start) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return start_value_buy * (TOLERANCE_WIN - 0.025) - 100  # * TALERANCE_OF_LOST -100  #end_point_if -start_value_buy
        if update_value * (TOLERANCE_WIN - i / 220) < next_value:  # cada pasada de i se vuelve mas exigente
            #print("STOP LOSS after n interactions (by update) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return update_value * (TOLERANCE_WIN - i / 220) - 100  # * (TALERANCE_OF_LOST + i/220)  -100   #end_point_if -start_value_buy

        if next_value < update_value:
            update_value = next_value

    #print("TAKE PROFICT after: ", len(rolling_col_slection), "interactions Win: ",update_value - start_value_buy)  # , i , " ".join(str(p) for p in a) )
    return update_value - 100

def rolling_get_sell_price_POS_next_value(rolling_col_slection):
    rolling_col_slection = [x + 100 for x in rolling_col_slection]
    start_value_buy = rolling_col_slection[0]

    update_value = start_value_buy
    for i in range(1, len(rolling_col_slection)):
        next_value = rolling_col_slection[i]

        if start_value_buy * (TOLERANCE_LOST + 0.025) > next_value:
            #print("STOP LOSS after n interactions (by start) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return next_value - 100 #return start_value_buy * (TOLERANCE_LOST + 0.025) - 100  # * TALERANCE_OF_LOST -100  #end_point_if -start_value_buy
        if update_value * (TOLERANCE_LOST + i / 220) > next_value:  # cada pasada de i se vuelve mas exigente
            #print("STOP LOSS after n interactions (by update) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return next_value - 100 #return update_value * (TOLERANCE_LOST + i / 220) - 100  # * (TALERANCE_OF_LOST + i/220)  -100   #end_point_if -start_value_buy

        if next_value > update_value:
            update_value = next_value

    #print("TAKE PROFICT after: ", len(rolling_col_slection), "interactions Win: ",update_value - start_value_buy)  # , i , " ".join(str(p) for p in a) )
    return update_value - 100


def rolling_get_sell_price_NEG_next_value(rolling_col_slection):
    #si el next_value SUBE se tira fuera
    rolling_col_slection = [x + 100 for x in rolling_col_slection]
    start_value_buy = rolling_col_slection[0]
    update_value = start_value_buy
    for i in range(1, len(rolling_col_slection)):
        next_value = rolling_col_slection[i]

        if start_value_buy * (TOLERANCE_WIN - 0.025) < next_value:
            #print("STOP LOSS after n interactions (by start) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return next_value - 100 #start_value_buy * (TOLERANCE_WIN - 0.025) - 100  # * TALERANCE_OF_LOST -100  #end_point_if -start_value_buy
        if update_value * (TOLERANCE_WIN - i / 220) < next_value:  # cada pasada de i se vuelve mas exigente
            #print("STOP LOSS after n interactions (by update) Win: ", next_value - start_value_buy, " n: ",i)  # " ".join(str(p) for p in a) )
            return next_value - 100 #update_value * (TOLERANCE_WIN - i / 220) - 100  # * (TALERANCE_OF_LOST + i/220)  -100   #end_point_if -start_value_buy

        if next_value < update_value:
            update_value = next_value

    #print("TAKE PROFICT after: ", len(rolling_col_slection), "interactions Win: ",update_value - start_value_buy)  # , i , " ".join(str(p) for p in a) )
    return update_value - 100

def select_work_buy_or_sell_point(cleaned_df, opcion : _KEYS_DICT.Op_buy_sell, Y_TARGET = 'buy_sell_point'):

    if type(opcion) is not _KEYS_DICT.Op_buy_sell :
        Logger.logr.error("the variable op_buy_sell is not of type .op_buy_sell or has no valid value (only POS and NEG are valid).")
        raise ValueError("the variable op_buy_sell is not of type .op_buy_sell or has no valid value (only POS and NEG are valid).")


    cleaned_df[Y_TARGET].astype(int).replace([101, -101], [100, -100], inplace=True)

    if opcion == _KEYS_DICT.Op_buy_sell.POS : #or opcion.casefold() == str(_KEYS_DICT.Op_buy_sell.POS.value).casefold():
        cleaned_df[Y_TARGET] = cleaned_df[Y_TARGET].astype(int).replace(-100, 0) # Solo para puntos de compra POS

    elif opcion == _KEYS_DICT.Op_buy_sell.NEG :
        cleaned_df[Y_TARGET] = cleaned_df[Y_TARGET].astype(int).replace(100, 0)
        cleaned_df[Y_TARGET] = cleaned_df[Y_TARGET].astype(int).replace(-100, 100)

    return cleaned_df

# Review the way of obtaining ground true
# Before training the models the intervals (of 15min) are classified in buy point 100 or 101, sell point -100 or .-101 or no trade point 0,
# these values are introduced in the column Y_TARGET = 'buy_sell_point' through the method Utils.Utils_buy_sell_points.get_buy_sell_points_Roll()
# The variation is calculated with respect to the following 12 windows (15min * 12 = 3 hours), and from there the 8% points of highest rise and
# highest fall are obtained, these points are assigned values different from 0.
# To obtain the Y_TARGET there are 2 methods that are responsible for the strategy to follow once you buy and sell, in case of loss you will opt for Stop Loss.
# rolling_get_sell_price_POS() and rolling_get_sell_price_NEG()
# Old method repleced by rolling_get_sell_price_XXX_next_value
def get_buy_sell_points_Roll(df_stock, delete_aux_rows = True):
    df_stock['sell_value_POS'] = df_stock.Close.shift(-12).rolling( min_periods = 1, window=12).apply(rolling_get_sell_price_POS_next_value)
    #df_stock['PROFIT_POS'] = (df_stock['sell_value_POS'] + 100) - (df_stock['Close'] + 100)
    #df_stock['per_PROFIT_POS'] = ((df_stock['sell_value_POS'] * 100) / df_stock['Close'] - 100).astype(float)

    df_stock['sell_value_NEG'] = df_stock.Close.shift(-12).rolling(min_periods=1, window=12).apply(rolling_get_sell_price_NEG_next_value)
    #df_stock['PROFIT_NEG'] = (df_stock['sell_value_NEG'] + 100) - (df_stock['Close'] + 100)
    #df_stock['per_PROFIT_NEG'] = ((df_stock['sell_value_NEG'] * 100) / df_stock['Close'] - 100).astype(float)

    df_stock['profit_POS_units'] = df_stock['sell_value_POS']  -  df_stock['Close']
    df_stock['profit_NEG_units'] = (df_stock['Close']+ 100) - (df_stock['sell_value_NEG']+ 100)

    df_threshold = df_stock['profit_NEG_units'].describe(percentiles=[0.90,0.92,0.94, 0.98]).round(4)
    Threshold_NEG_92 = df_threshold["94%"]
    Threshold_NEG_98 = df_threshold["98%"]
    df_threshold = df_stock['profit_POS_units'].describe(percentiles=[0.90,0.92,0.94, 0.98]).round(4)
    Threshold_POS_92 = df_threshold["94%"]
    Threshold_POS_98 = df_threshold["98%"]
    Logger.logr.info("Parameters of acquisition \"buy_sell_points\" for this stock is set to POSITIVE (units) \t92%: "+ str(Threshold_NEG_92) + " \t98%: "+ str(Threshold_NEG_98) +" NEGATIVE (units) \t93%: "+ str(Threshold_POS_92) +" \t98%: "+ str(Threshold_POS_98) )


    if Y_TARGET not in df_stock.columns:
        df_stock.insert(loc=1, column=Y_TARGET, value=0)
    df_stock.loc[df_stock['profit_NEG_units'] > Threshold_NEG_92, Y_TARGET] = -100 # ] >>>>>> T es distinto al metodo OLD
    df_stock.loc[df_stock['profit_NEG_units'] > Threshold_NEG_98, Y_TARGET] = -101
    df_stock.loc[df_stock['profit_POS_units'] > Threshold_POS_92, Y_TARGET] = 100
    df_stock.loc[df_stock['profit_POS_units'] > Threshold_POS_98, Y_TARGET] = 101

    if delete_aux_rows:
        df_stock = df_stock.drop(columns=['sell_value_POS',  'sell_value_NEG','profit_NEG_units', 'profit_POS_units'], errors='ignore')
    #df_stock[df_stock['Volume'] != 0].groupby(Y_TARGET).count()

    return df_stock

# https://stackoverflow.com/questions/64019553/how-pivothigh-and-pivotlow-function-work-on-tradingview-pinescript
def get_buy_sell_points_HT_pp(df_l, LEN_RIGHT, LEN_LEFT):
    print("get_HT_pp df.shape:",df_l.shape," len_right: ",LEN_RIGHT," len_left: ", LEN_LEFT)


    # se puede usar high and low , pero distorsina los puntos
    pivots_hh = df_l['High'].shift(-LEN_RIGHT, fill_value=0).rolling(LEN_LEFT).max()
    # For 'High' pivots pd.Series 'high_column' and:
    pivots_ll = df_l['Low'].shift(-LEN_RIGHT, fill_value=0).rolling(LEN_LEFT).min()
    df_l['pp_high'] = pivots_hh
    df_l['pp_low'] = pivots_ll
    df_l.insert(loc=len(df_l.columns), column="touch_low", value=False)
    df_l.insert(loc=len(df_l.columns), column="touch_high", value=False)
    df_l.loc[(df_l["pp_low"] >= df_l['Low']), "touch_low"] = True
    df_l.loc[(df_l["pp_high"] <= df_l['High']), "touch_high"] = True
    # df_l.value_counts()
    df_r = pd.DataFrame()
    df_r['count'] = df_l[[ "touch_low", "touch_high"]].value_counts().to_frame()
    df_r['per%'] = df_l[["touch_low", "touch_high"]].value_counts(normalize=True).mul(100).round(2)
    # df_r.index = ['Buy_True/Sell_True', 'Buy_True/Sell_False', 'Buy_False/Sell_True', 'Buy_False/Sell_False']
    print("Count ht pivost point stadistic balance ")
    print(df_r)


    # df_l[["touch_low", "touch_high"]].astype(int)
    if Y_TARGET not in df_l.columns:
        df_l.insert(loc=1, column=Y_TARGET, value=0)
    df_l.loc[df_l["touch_low"] == True , Y_TARGET] = 100
    df_l.loc[df_l["touch_high"] == True , Y_TARGET] = -100
    #Why? can bring with "touch_high" and "touch_low" == True
    df_l.loc[(df_l["touch_high"] == True) & (df_l["touch_low"] == True), Y_TARGET] = 0

    df_l = df_l.drop(columns=['pp_high', 'pp_low', "touch_low", "touch_high"],errors='ignore')

    # df_l[Y_TARGET].value_counts()
    # df_l[Y_TARGET+"2"].value_counts()

    return df_l#, df_r

#obsolete method
# Old method repleced by rolling_get_sell_price_XXX_next_value
def get_buy_sell_points_Arcos(df_stock):
    LEN_DF = len(df_stock)
    pd.options.mode.chained_assignment = None


    PER_NON_NOISE = 1 #0.5  # porcentage por el cual una pequeña bajada o subida se pasa a considerar ruido
    PER_ARCO_DETECT = 7.2 #4     porcentage por el cual un arco es considerado arco valido de testeo
    #el df dado contien fechas menores de antiguedad de 2 meses , se aplican otros coeficientes
    if  pd.to_datetime(df_stock['Date'][0], errors='coerce').year > (datetime.now().year-1):
        PER_NON_NOISE = 0.5
        PER_ARCO_DETECT = 4
        Logger.logr.info("Past buying and selling points are taken MONTHLY  PER_NON_NOISE: " +str(
            PER_NON_NOISE)+ " PER_ARCO_DETECT: " + str(PER_ARCO_DETECT) )
    else:
        Logger.logr.info("Past buying and selling points are taken YEARLY PER_NON_NOISE: " +str(
            PER_NON_NOISE)+ " PER_ARCO_DETECT: " + str(PER_ARCO_DETECT) )


    df_stock.insert(loc=len(df_stock.columns), column='arco_member', value=0.0)
    df_stock.insert(loc=len(df_stock.columns), column='arco_member_per_var', value=0.0)
    df_stock.insert(loc=len(df_stock.columns), column='tendency', value=False)
    arco_menber_letter = 1

    df_stock.loc[df_stock['per_Close'] >= 0, 'tendency'] = True
    df_stock.loc[df_stock['per_Close'] < 0, 'tendency'] = False
    per_change_arco = 0

    #is_tendencia_positiva = None
    for c in range(1, LEN_DF):
        i_next = 1
        if LEN_DF <= c+i_next:
            Logger.logr.debug("END THE LOOP return  c+i_next: " + str( c-i_next ))
            continue
        if numpy.isnan(df_stock['per_Close'][c-1]):
            continue

        # cateto_opuesto = df_stock['Close'][c] - df_stock['Close'][c - i_next]
        # print(c, " ",df_stock['Close'][c], " cateto opuesto: ", cateto_opuesto )
        # cotangente = cateto_opuesto / CATETO_CONTIGUO_MIN_15min_SIZE
        # df_stock['arco_next'][c] = atan(cotangente) * 180 / pi #lo paso a grados

        #si hay un cambio superior al 0,5% es cambio de arco, en caso contrario ruido
        if df_stock['tendency'][c] == True and df_stock['tendency'][c-1] == False and df_stock['per_Close'][c-1] <= (PER_NON_NOISE *-1):
            # if df_stock['arco_member_per_var'][c-1] > 0 and  df_stock['per_Close'][c] > 0:
            #     continue
            # else:
            arco_menber_letter = arco_menber_letter + 1
            per_change_arco = 0
        elif df_stock['tendency'][c] == False and  df_stock['tendency'][c-1] == True and  df_stock['per_Close'][c-1] >= PER_NON_NOISE:
            # if df_stock['arco_member_per_var'][c-1] < 0 and  df_stock['per_Close'][c] < 0:
            #     continue
            # else:
            arco_menber_letter = arco_menber_letter + 1
            per_change_arco = 0
        #Tres tendencias seguidas con la anterior distinta es cambio de arco
        elif LEN_DF >= c+3 and c-3 > 0:
            if df_stock['tendency'][c] != df_stock['tendency'][c-1] and df_stock['tendency'][c] == df_stock['tendency'][c+1] and df_stock['tendency'][c] == df_stock['tendency'][c+2]:
                # si es si si no si si si  , por ese no , no cambiamos la tendencia del arco , salvo cambio superior al 0.5
            # if df_stock['tendencia'][c-1] != df_stock['tendencia'][c-2] and df_stock['tendencia'][c-1] != df_stock['tendencia'][c-3]:
                if (df_stock['arco_member_per_var'][c - 1] < 0 and df_stock['per_Close'][c] < 0)\
                    or (df_stock['arco_member_per_var'][c-1] > 0 and  df_stock['per_Close'][c] > 0):
                    pass
                else:
                    arco_menber_letter = arco_menber_letter + 1
                    per_change_arco = 0

        #print( "angulo: ",df_stock['arco_next'][c] , " ARCO MEMBER: ",arco_menber_letter , "   ", arco_menber_letter )
        df_stock.at[c, 'arco_member'] =  chr(arco_menber_letter+64)
        per_change_arco = per_change_arco + df_stock['per_Close'][c]
        df_stock.at[c,'arco_member_per_var'] = per_change_arco


    # df_stock.groupby(['arco_member']).agg(lambda x: x.value_counts().index[0])
    # df_stock.groupby(['arco_member'])['arco_member_per_var'].agg(lambda x: pd.Series.mode(x)[0])

    df_stock.loc[df_stock['per_Close'] > 0, 'tendency'] = True
    df_stock.loc[df_stock['per_Close'] < 0, 'tendency'] = False
    #añadir a todas las columnas despues de group by
    df_stock['tendency_2']  = df_stock.groupby(['arco_member'])['tendency'].transform(lambda x: x.mode()[0])
    df_stock['arco_member_SUM'] = df_stock.groupby(['arco_member'])['per_Close'].transform('sum')

    #df_stock['arco_member_per_var_max'] = df_stock.groupby(['arco_member'])['arco_member_per_var'].transform(max)
    df_stock.loc[df_stock['arco_member_SUM'] >= 0, 'tendency_2'] = True
    df_stock.loc[df_stock['arco_member_SUM'] < 0, 'tendency_2'] = False

    #Cada vez que cambia de valor el "isStatusChanged" se pone a True
    df_stock["isStatusChanged"] = df_stock["tendency_2"].shift() != df_stock["tendency_2"]

    df_stock['arco_member2'] = 0
    arco_menber_letter = 1
    for t in range(0, LEN_DF):
        if df_stock["isStatusChanged"][t] == True:
            arco_menber_letter = arco_menber_letter + 1
        df_stock.at[t,'arco_member2'] = chr(arco_menber_letter + 64)
    df_stock['arco_member_SUM2'] = df_stock.groupby(['arco_member2'])['per_Close'].transform('sum')
    #df_stock["isStatusChanged"] = df_stock["tendencia2"].shift(1, fill_value=df_stock["tendencia2"].astype(float).head(1)) != df_stock["tendencia2"]
    df_stock["buy_sell_point"] = 0
    df_stock.loc[df_stock['arco_member_SUM2'] > PER_ARCO_DETECT, "buy_sell_point"] = 100
    df_stock.loc[df_stock['arco_member_SUM2'] < -PER_ARCO_DETECT, "buy_sell_point"] = -100

    df_stock['buy_sell_point_count'] = df_stock.groupby(['arco_member2', 'buy_sell_point'])['arco_member2'].transform('count')
    #df_stock.loc[df_stock['buy_sell_point'] == 0, "buy_sell_point"] = 0
    df_stock['buy_sell_point2'] = 0
    df_stock['buy_sell_point_count2'] = 0
    df_stock.loc[(df_stock['buy_sell_point'] != 0) & (df_stock["isStatusChanged"] == True)]
    for t in range(1, LEN_DF):
        if df_stock['buy_sell_point'][t] != 0 and df_stock["isStatusChanged"][t] == True:
            margin_buy_points = int(df_stock['buy_sell_point_count'][t] * 0.3)
            for m in range(0, margin_buy_points):
                df_stock.at[t+m, 'buy_sell_point2'] = df_stock['buy_sell_point'][t]
                df_stock.at[t + m, 'buy_sell_point_count2'] = df_stock['buy_sell_point_count'][t]
            #la anterior y las 20% siguientes son puntos de compras-venta
            df_stock.at[t, 'buy_sell_point2'] = (df_stock['buy_sell_point'][t] *1.01)#se marca como punto maestro de compra
            df_stock.at[t, 'buy_sell_point_count2'] = df_stock['buy_sell_point_count'][t]
            #df_stock.at[t-1, 'buy_sell_point2'] = df_stock['buy_sell_point'][t]

    # df_stock['buy_sell_point2'] = 0
    # df_stock.loc[df_stock['isStatusChanged'] == True, 'buy_sell_point2'] = df_stock['arco_member_SUM2']
    #df_stock.loc[df_stock['isStatusChanged'] == False, 'buy_sell_point2'] = -100

    df_stock['buy_sell_point'] = df_stock['buy_sell_point2']
    df_stock['buy_sell_point_count'] = df_stock['buy_sell_point_count2']

    df_stock = df_stock.drop(['arco_member', 'arco_member_per_var', 'tendency',
     'tendency_2', 'arco_member_SUM', 'isStatusChanged', 'arco_member2',
     'arco_member_SUM2','buy_sell_point_count', 'buy_sell_point_count2', #     parece un dato relevante  'buy_sell_point_count',
     'buy_sell_point2'], 1)

    pd.options.mode.chained_assignment = 'warn'

    return df_stock

def check_buy_points_prediction(df, result_column_name = 'result', path_cm =  "d_price/plot_confusion_matrix_.png", SUM_RESULT_2_VALID = 1.45, generate_CM_for_each_ticker = False):

    LEN_DF = len(df)
    #ESTO ES LA SUMA DE FILA  N + (N-1) SI ESTO DA MAS DE 1.45 SE TOMA COMO PUNTO DE COMPRA VALIDO
    df = df.loc[:, ~df.columns.duplicated()]
    col = df.columns
    #tickers_col = col.startswith('ticker_')

    list_ticker_stocks = [col for col in df if col.startswith('ticker_')]#todas las que empiecen por ticker_ , son variables tontas
    if len(list_ticker_stocks) != 0:
        df['ticker'] = df[list_ticker_stocks].idxmax(axis=1) #undo dummy variable
        df.drop(columns=list_ticker_stocks, inplace=True)

    df = df.sort_values(by=['ticker', "Date"], ascending=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')#time stamp to Date

    #RESET de per_close
    df['Close'] = df['Close'] + 100
    for t in list_ticker_stocks:
        df[df['ticker'] == t] = Utils_Yfinance.add_variation_percentage(df[df['ticker'] == t])

    l2 = ["Date",'ticker', result_column_name, "Close", "per_Close", 'has_preMarket'] #  "buy_sell_point",
    df = df[l2]

    df.index = pd.RangeIndex(len(df.index))
    df["result_sum"] = df[result_column_name] # df[result_column_name].rolling(2).sum()  #para atras
    SUM_RESULT_2_VALID = SUM_RESULT_2_VALID /2

    df["is_result_buy"] = False
    df.loc[df['result_sum'] > SUM_RESULT_2_VALID, "is_result_buy"] = True


    #df['C'] = [window.round(2).to_list() for window in df['per_Close'].shift(-5).rolling(5)]

    # df["per_Close_5_1"] = (df['per_Close'] *-1 ) #+ df['per_Close'].shift(-5).rolling(5).sum() #para alante
    # df["per_Close_12_1"] = (df['per_Close'] *-1 ) #+ df['per_Close'].shift(-12).rolling(12).sum()
    #
    # df["per_Close_5_cast"] = df['per_Close'].shift(-5).rolling(5).mean() #para alante
    # df["per_Close_12_cast"] = df['per_Close'].shift(-12).rolling(12).mean()

    df["per_Close_5"] = df['per_Close'].shift(-5).rolling(5).sum() #para alante
    df["per_Close_12"] = df['per_Close'].shift(-12).rolling(12).sum()

    df["per_Close_5mean"] = df['Close'].shift(-5).rolling(5).mean() #para alante
    df["per_Close_12mean"] = df['Close'].shift(-12).rolling(12).mean()

    df["per_Close_5_2"] = df["per_Close_5mean"] -  df['Close'] #  .shift(-5).rolling(5).mean() #para alante
    df["per_Close_12_2"] = df["per_Close_12mean"] -  df['Close']

    # df["per_Close_last_5"] = df['per_Close'].shift(-5) #para alante
    # df["per_Close_last_12"] = df['per_Close'].shift(-12)
    #
    # df["per_Close_5dif"] =  df['per_Close'].shift(-5).rolling(5).mean() - df['per_Close']  #.shift(-5).rolling(5).sum() #para alante
    # df["per_Close_12dif"] = df['per_Close'].shift(-12).rolling(12).mean() - df['per_Close']  # df['per_Close'].shift(-12) - df['per_Close'].shift(-12).rolling(12).mean()

    df[df['is_result_buy'] == True]["per_Close_5"].mean()
    df[["Date",'ticker', result_column_name,"is_result_buy" , "result_sum", "per_Close_5", "per_Close_12"]]

    df["isValid_to_buy"] = False
    #df.loc[ (df['result_sum'] > SUM_RESULT_2_VALID)   &  (df["per_Close_12"] > 20) & (df["per_Close_5"] > 8), "isValid_to_buy"] = True
    #(df['result_sum'] > SUM_RESULT_2_VALID) no es relevante para si es punto de compra o no
    #df.loc[ (df["per_Close_12"] > 14) & (df["per_Close_5"] > 5), "isValid_to_buy"] = True
    df.loc[ (df["per_Close_12"] > 14) & (df["per_Close_5"] > 5), "isValid_to_buy"] = True

    print(df[['Date', 'ticker', "isValid_to_buy"]].groupby(["ticker","isValid_to_buy"]).count() )
    print(df[['Date', 'ticker', "is_result_buy"]].groupby(["ticker","is_result_buy"]).count() )


    if generate_CM_for_each_ticker:
        for t in list_ticker_stocks:
            dfn = pd.DataFrame()
            dfn = df[df['ticker'] == t]
            cf_matrix = confusion_matrix(  dfn["isValid_to_buy"].astype(int), dfn["is_result_buy"].astype(int))

            print(path_cm+ " MATRIX: "+str(cf_matrix))
            if cf_matrix.shape[0] == 2 and cf_matrix.shape[1] == 2:
                Utils_plotter.plot_confusion_matrix_cm_OUT(cf_matrix, path=path_cm + str(t) + "_.png", title_str ="\n" + str(t))
            else:
                print("WARN la shape de la confusion matrix no es (2,2)",path_cm+ " MATRIX: "+str(cf_matrix))

    cf_matrix = confusion_matrix(df["isValid_to_buy"].astype(int), df["is_result_buy"].astype(int))
    print(path_cm + " MATRIX: " + str(cf_matrix))
    if cf_matrix.shape[0] == 2 and cf_matrix.shape[1] == 2:
        Utils_plotter.plot_confusion_matrix_cm_OUT(cf_matrix, path=path_cm + "full_.png", title_str="\n" + "FULL")
    else:
        print("WARN la shape de la confusion matrix no es (2,2)", path_cm + " MATRIX: " + str(cf_matrix))

    #df = df[l2]
    df["isValid_to_buy"] = df["isValid_to_buy"].astype(int)
    return df["isValid_to_buy"].values

