import pandas as pd
from math import atan, pi
from datetime import datetime

import yhoo_history_stock
from LogRoot.Logging import Logger
from sklearn.metrics import confusion_matrix


import Utils_plotter


def get_buy_sell_points(df_stock):
    LEN_DF = len(df_stock)


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

    df_stock['arco_member'] = 0.0
    df_stock['arco_member_per_var'] = 0.0
    arco_menber_letter = 1

    df_stock['tendency'] = False
    df_stock.loc[df_stock['per_Close'] >= 0, 'tendency'] = True
    df_stock.loc[df_stock['per_Close'] < 0, 'tendency'] = False
    per_change_arco = 0

    #is_tendencia_positiva = None
    for c in range(1, LEN_DF):
        i_next = 1
        if LEN_DF <= c+i_next:
            Logger.logr.debug("END THE LOOP return  c+i_next: " + str( c-i_next ))
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

    return df_stock

def check_buy_points_prediction (df, result_column_name = 'result', path_cm =  "d_price/plot_confusion_matrix_.png", SUM_RESULT_2_VALID = 1.45):
    LEN_DF = len(df)
    #ESTO ES LA SUMA DE FILA  N + (N-1) SI ESTO DA MAS DE 1.45 SE TOMA COMO PUNTO DE COMPRA VALIDO

    col = df.columns
    #tickers_col = col.startswith('ticker_')

    list_ticker_stocks = [col for col in df if col.startswith('ticker_')]#todas las que empiecen por ticker_ , son variables tontas
    df['ticker'] = df[list_ticker_stocks].idxmax(axis=1) #undo dummi variable
    df.drop(columns=list_ticker_stocks, inplace=True)

    df = df.sort_values(by=['ticker', "Date"], ascending=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')#time stamp to Date

    #RESET de per_close
    df['Close'] = df['Close'] + 100
    for t in list_ticker_stocks:
        df[df['ticker'] == t] = yhoo_history_stock.add_variation_percentage(df[df['ticker'] == t])

    l2 = ["Date",'ticker', result_column_name, "Close", "per_Close", 'has_preMarket'] #  "buy_sell_point",
    df = df[l2]

    df.index = pd.RangeIndex(len(df.index))
    df["result_sum"] = df[result_column_name] # df[result_column_name].rolling(2).sum()  #para atras
    SUM_RESULT_2_VALID = SUM_RESULT_2_VALID /2

    df["is_result_buy"] = False
    df.loc[df['result_sum'] > SUM_RESULT_2_VALID, "is_result_buy"] = True


    df['C'] = [window.round(2).to_list() for window in df['per_Close'].shift(-5).rolling(5)]

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
    df.loc[ (df["per_Close_12"] > 14) & (df["per_Close_5"] > 5), "isValid_to_buy"] = True

    print(df[['Date', 'ticker', "isValid_to_buy"]].groupby(["ticker","isValid_to_buy"]).count() )
    print(df[['Date', 'ticker', "is_result_buy"]].groupby(["ticker","is_result_buy"]).count() )

    for t in list_ticker_stocks:
        dfn = pd.DataFrame()
        dfn = df[df['ticker'] == t]
        cf_matrix = confusion_matrix(  dfn["isValid_to_buy"].astype(int), dfn["is_result_buy"].astype(int))

        print(path_cm+ " MATRIX: "+str(cf_matrix))
        if cf_matrix.shape[0] == 2 and cf_matrix.shape[1] == 2:
            Utils_plotter.plot_confusion_matrix_cm_OUT(cf_matrix,path= path_cm + str(t)+"_.png", title_str ="\n" + str(t))
        else:
            print("WARN la shape de la confusion matrix no es (2,2)",path_cm+ " MATRIX: "+str(cf_matrix))


    df = df[l2]






    # TODO hacer los mismos ARCOS con medias: previous_avg = 0
    # from numpy import arctan
    #CATETO_CONTIGUO_MIN_15min_SIZE = 4.81
    # df_stock['arco_next'] = 0.0
    # df_stock['arco_next'] =  df_stock['arco_next'].astype(float)
    # for a in range(c+1,c+30):
    #     actual_avg = df_stock.iloc[c:a]['per_Close'].mean()  #df_stock.iloc[c]['per_Close'].tail(a).mean()
    #     print(c, "roling: ", a, "AVG: ", actual_avg, " previous_avg  ", previous_avg, "   ")
    #     if abs(actual_avg) < abs(previous_avg) * 1.03:
    #         arco_menber_letter = arco_menber_letter + 1
    #         per_change_arco = 0
    #         break
    #     else:
    #         previous_avg = actual_avg
    # si es un cambio de percentage positivo , o el menos es menor del -1% sigue en arco positivo
    # if df_stock['per_Close'][c] > 0 or (is_tendencia_positiva == True and df_stock['per_Close'][c] > -1 ):
    #     if is_tendencia_positiva == False:
    #         print("Cambio de arco crece: ", df_stock['per_Close'][c], "%")
    #         arco_menber_letter  = arco_menber_letter + 1
    #     #print("crece: ",df_stock['per_Close'][c] , "%" )
    #     is_tendencia_positiva = True
    # elif df_stock['per_Close'][c] < 0 or (is_tendencia_positiva == False and df_stock['per_Close'][c] < 1 ):
    #     if is_tendencia_positiva == True:
    #         print("Cambio de arco DEcrece: ", df_stock['per_Close'][c], "%")
    #         arco_menber_letter = arco_menber_letter + 1
    #     is_tendencia_positiva = False

    # df_stock.replace([np.inf, -np.inf], 0, inplace=True)
    # len_df = len(df_stock)
    # list_timeperiod = [5, 10, 20, 50, 100, 200]
    # for time_p in list_timeperiod:
    #     # df_stock['next_min_' + str(time_p)] = df_stock['Low'].rolling(time_p).min()
    #     # df_stock['next_max_' + str(time_p)] = df_stock['High'].rolling(time_p).max()
    #     # df_stock['next_min_' + str(time_p)] = df_stock['Low'].rolling(time_p).min() #recoger para atras df_stock['High'].shift(-1).rolling(time_p).max()
    #     # df_stock['next_max_' + str(time_p)] = df_stock['High'].rolling(time_p).max()
    #     # df_stock["per_next_" + str(time_p)] = (df_stock['next_max_' + str(time_p)] * 100) / df_stock['next_min_' + str(time_p)] - 100
    #     # df_stock['next_min_' + str(time_p)] = df_stock['Close'].rolling(time_p).min() #recoger para atras df_stock['High'].shift(-1).rolling(time_p).max()
    #     # df_stock['next_max_' + str(time_p)] = df_stock['Close'].rolling(time_p).max()
    #     # df_stock["per_next_" + str(time_p)] = (df_stock['next_max_' + str(time_p)] * 100) / df_stock['next_min_' + str(time_p)] - 100
    #     aaa = df_stock.nlargest(n=int(len_df/80), columns="per_next_" + str(time_p))
    #     print(aaa)