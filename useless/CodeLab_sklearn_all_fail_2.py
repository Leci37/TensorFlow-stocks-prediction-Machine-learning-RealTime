from Utils import Utils_plotter
import yhoo_history_stock
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import seaborn as sns # crear los plots
from matplotlib import pyplot as plt
from _KEYS_DICT import Option_Historical


import pandas as pd



stockId = 'MELI'
opion = Option_Historical.MONTH_3

#TODO news news_get_data_NUTS.get_json_news_sentimet(stockId)


def generate_png_correlation(df):
    columns_array = "Date", "Open", "High", "Low", "Close", "Volume", "per_Close", "per_Volume", "per_preMarket", "olap_BBAND_UPPER", "olap_BBAND_MIDDLE", "olap_BBAND_LOWER", "olap_HT_TRENDLINE", "olap_MIDPOINT", "olap_MIDPRICE", "mtum_ADX", "mtum_ADXR", "mtum_APO", "mtum_AROON_down", "mtum_AROON_up", "mtum_AROONOSC", "mtum_BOP", "mtum_CCI", "mtum_CMO", "mtum_DX", "mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list", "mtum_MACD_ext", "mtum_MACD_ext_signal", "mtum_MACD_ext_list", "mtum_MACD_fix", "mtum_MACD_fix_signal", "mtum_MACD_fix_list", "mtum_MFI", "mtum_MINUS_DI", "mtum_MINUS_DM", "mtum_MOM", "mtum_PLUS_DI", "mtum_PLUS_DM", "mtum_PPO", "mtum_ROC", "mtum_ROCP", "mtum_ROCR", "mtum_ROCR100", "mtum_RSI", "mtum_STOCH_k", "mtum_STOCH_d", "mtum_STOCHF_k", "mtum_STOCHF_d", "mtum_STOCHRSI_k", "mtum_STOCHRSI_d", "mtum_TRIX", "mtum_ULTOSC", "mtum_WILLIAMS_R", "volu_Chaikin_AD", "volu_Chaikin_ADOSC", "volu_OBV", "vola_ATR", "vola_NATR", "vola_TRANGE", "cycl_DCPERIOD", "cycl_DCPHASE", "cycl_PHASOR_inph", "cycl_PHASOR_quad", "cycl_SINE_sine", "cycl_SINE_lead", "cycl_HT_TRENDMODE", "cdl_2CROWS", "cdl_3BLACKCROWS", "cdl_3INSIDE", "cdl_3LINESTRIKE", "cdl_3OUTSIDE", "cdl_3STARSINSOUTH", "cdl_3WHITESOLDIERS", "cdl_ABANDONEDBABY", "cdl_ADVANCEBLOCK", "cdl_BELTHOLD", "cdl_BREAKAWAY", "cdl_CLOSINGMARUBOZU", "cdl_CONCEALBABYSWALL", "cdl_COUNTERATTACK", "cdl_DARKCLOUDCOVER", "cdl_DOJI", "cdl_DOJISTAR", "cdl_DRAGONFLYDOJI", "cdl_ENGULFING", "cdl_EVENINGDOJISTAR", "cdl_EVENINGSTAR", "cdl_GAPSIDESIDEWHITE", "cdl_GRAVESTONEDOJI", "cdl_HAMMER", "cdl_HANGINGMAN", "cdl_HARAMI", "cdl_HARAMICROSS", "cdl_HIGHWAVE", "cdl_HIKKAKE", "cdl_HIKKAKEMOD", "cdl_HOMINGPIGEON", "cdl_IDENTICAL3CROWS", "cdl_INNECK", "cdl_INVERTEDHAMMER", "cdl_KICKING", "cdl_KICKINGBYLENGTH", "cdl_LADDERBOTTOM", "cdl_LONGLEGGEDDOJI", "cdl_LONGLINE", "cdl_MARUBOZU", "cdl_MATCHINGLOW", "cdl_MATHOLD", "cdl_MORNINGDOJISTAR", "cdl_MORNINGSTAR", "cdl_ONNECK", "cdl_PIERCING", "cdl_RICKSHAWMAN", "cdl_RISEFALL3METHODS", "cdl_SEPARATINGLINES", "cdl_SHOOTINGSTAR", "cdl_SHORTLINE", "cdl_SPINNINGTOP", "cdl_STALLEDPATTERN", "cdl_STICKSANDWICH", "cdl_TAKURI", "cdl_TASUKIGAP", "cdl_THRUSTING", "cdl_TRISTAR", "cdl_UNIQUE3RIVER", "cdl_UPSIDEGAP2CROWS", "cdl_XSIDEGAP3METHODS", "sti_BETA", "sti_CORREL", "sti_LINEARREG", "sti_LINEARREG_ANGLE", "sti_LINEARREG_INTERCEPT", "sti_LINEARREG_SLOPE", "sti_STDDEV", "sti_TSF", "sti_VAR", "ma_DEMA_5", "ma_EMA_5", "ma_KAMA_5", "ma_SMA_5", "ma_T3_5", "ma_TEMA_5", "ma_TRIMA_5", "ma_WMA_5", "ma_DEMA_10", "ma_EMA_10", "ma_KAMA_10", "ma_SMA_10", "ma_T3_10", "ma_TEMA_10", "ma_TRIMA_10", "ma_WMA_10", "ma_DEMA_20", "ma_EMA_20", "ma_KAMA_20", "ma_SMA_20", "ma_T3_20", "ma_TEMA_20", "ma_TRIMA_20", "ma_WMA_20", "ma_DEMA_50", "ma_EMA_50", "ma_KAMA_50", "ma_SMA_50", "ma_T3_50", "ma_TEMA_50", "ma_TRIMA_50", "ma_WMA_50", "ma_DEMA_100", "ma_EMA_100", "ma_KAMA_100", "ma_SMA_100", "ma_T3_100", "ma_TEMA_100", "ma_TRIMA_100", "ma_WMA_100", "ma_DEMA_200", "ma_EMA_200", "ma_KAMA_200", "ma_SMA_200", "ma_T3_200", "ma_TEMA_200", "ma_TRIMA_200", "ma_WMA_200", "trad_s3", "trad_s2", "trad_s1", "trad_pp", "trad_r1", "trad_r2", "trad_r3", "clas_s3", "clas_s2", "clas_s1", "clas_pp", "clas_r1", "clas_r2", "clas_r3", "fibo_s3", "fibo_s2", "fibo_s1", "fibo_pp", "fibo_r1", "fibo_r2", "fibo_r3", "wood_s3", "wood_s2", "wood_s1", "wood_pp", "wood_r1", "wood_r2", "wood_r3", "demark_s1", "demark_pp", "demark_r1", "cama_s3", "cama_s2", "cama_s1", "cama_pp", "cama_r1", "cama_r2", "cama_r3", "ti_ACC_DIST", "ti_CHAIKIN(10,3)", "ti_CHOPPINESS(14)", "ti_COPPOCK(14,11,10)", "ti_DONCHIAN_LOWER(20)", "ti_DONCHIAN_CENTER(20)", "ti_DONCHIAN_UPPER(20)", "ti_EASE_OF_MOVEMENT(14)", "ti_FORCE_INDEX(13)", "ti_HMA(20)", "ti_KELT(20)_LOWER", "ti_KELT(20)_UPPER", "ti_MASS_INDEX(9,25)", "ti_SUPERTREND(20)", "ti_VORTEX_POS(5)", "ti_VORTEX_NEG(5)"
    columns_array = ["Open", "High", "Low", "Close"]
    for c in range(0, len(columns_array), 3):
        # bolean values are no alone
        a = [columns_array[c], columns_array[c + 1], columns_array[c + 2]]
        print(a)
        sns_plot = sns.pairplot(data=df, vars=a, hue='buy_sell_point',
                                kind="reg", palette="husl")
        name = stockId + "_correlation_" + str(opion.name)
        plt.savefig("d_price/correlations/" + name + "_".join(a) + ".png")
        print("d_price/correlations/" + name + "_".join(a) + ".png")

list_companys_FAV = ["MELI", "TWLO","RIVN","SNOW", "UBER", "U" , "PYPL", "GTLB","MDB", "TSLA", "DDOG" ]


def get_favs_csv_stocks_history(list_companys, csv_name):
    global sc, df_l, stockId
    sc = StandardScaler()
    sc = MinMaxScaler(feature_range=(-100, 100))
    columns_delete_no_enogh_data = ["ma_T3_50", "ma_TEMA_50", "ma_DEMA_100", "ma_T3_100", "ma_TEMA_100"]

    df_all = pd.DataFrame()
    for l in list_companys:
        df_l = yhoo_history_stock.get_stock_history_Tech_download(l, Option_Historical.MONTH_3, get_technical_data=True,
                                                                  prepost=False, interval="15m", add_stock_id_colum=False)
        # df_l = pd.read_csv("d_price/" + l + "_stock_history_" + str(opion.name) + ".csv", index_col=False, sep='\t')
        df_l['buy_sell_point'].replace([101, -101], [100, -100], inplace=True)
        df_l = df_l.drop(columns=columns_delete_no_enogh_data)  # luego hay que borrar los nan y daña mucho el dato
        for c in Utils_col_sele.COLUMNS_CANDLE:  # a pesar de que no se haya dado ningun patron de vela el Scaler tiene que respetar el mas menos
            df_l.at[0,c] = -100
            df_l.at[1,c] = 100

        aux_date_save = df_l['Date'] #despues se añade , hay que pasar el sc.fit_transform
        df_l['Date'] = 0
        array_stock = sc.fit_transform(df_l)
        df_l = pd.DataFrame(array_stock, columns=df_l.columns)
        df_l['ticker'] = l  # int(str(hash(l))[:4])
        df_l['Date'] = aux_date_save #to correct join

        print("d_price/" + l + "_stock_history_" + str(opion.name) + ".csv  Shape: " + str(df_l.shape))
        df_l.to_csv("d_price/" + stockId + "_SCALA_stock_history_" + str(opion.name) + ".csv", sep='\t', index=None)
        df_all = pd.concat([df_all,df_l ])
    #
    df_all = df_all.sort_values(by=['Date', 'ticker'], ascending=True)
    df_all.insert(1, 'ticker', df_all.pop('ticker'))
    df_all.to_csv("d_price/" + csv_name + "_SCALA_stock_history_" + str(opion.name) + ".csv", sep='\t', index=None)
    return df_all


#get_favs_csv_stocks_history(list_companys_FAV, csv_name =)


stockId = "FAV"
df_l = pd.read_csv("d_price/" + stockId + "_SCALA_stock_history_" + str(opion.name) + ".csv", index_col=False, sep='\t')


df_l['Date'] = pd.to_datetime(df_l['Date']).map(pd.Timestamp.timestamp)
#df_l = df_l.sort_values(by=['Date', 'ticker'], ascending=True)
#pd.to_datetime(df_his['Date'], unit='s', origin='unix')


# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
# summarize the shape of the dataset
print(df_l.shape)
# summarize each variable
print(df_l.describe())
# histograms of the variables
print(df_l.info)
pyplot.show()

df_l['buy_sell_point'].replace([101, -101], [100, -100], inplace=True)
Utils_plotter.plot_pie_countvalues(df_l, colum_count ='buy_sell_point', stockid= stockId, opion = str(opion.name))

#RETIRO nan
df_l = df_l.dropna()

pd.set_option('display.expand_frame_repr', False)

dfCopy = df_l
# dfCopy = dfCopy.drop(columns=['TotalCharges'])#'customerID',
dfDummy = pd.get_dummies(dfCopy, columns = ['has_preMarket' , 'ticker'])  #pd.get_dummies(dfCopy, columns = ['buy_sell_point','has_preMarket' ])
#dfDummy = dfDummy.drop(columns=['has_preMarket'])#'has_preMarket_False','has_preMarket_True' 'buy_sell_point_-100','buy_sell_point_0', 'buy_sell_point_100',



Y = dfDummy[['buy_sell_point']]#que quiero predecir
X = dfDummy.drop(columns=['buy_sell_point']) #datos con los que se predice



from sklearn.model_selection import train_test_split
#We'll use a (70%, 20%, 10%) split for the training, validation, and test sets. Note the data is not being randomly shuffled before splitting. This is for two reasons.
#It ensures that chopping the data into windows of consecutive samples is still possible.
#It ensures that the validation/test results are more realistic, being evaluated on data collected after the model was trained.
X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.2 ,shuffle=False)#   ,  random_state=1)


#TENSOR FLOW csv
#https://colab.research.google.com/github/adammichaelwood/tf-docs/blob/csv-feature-columns/site/en/r2/tutorials/load_data/csv.ipynb#scrollTo=iXROZm5f3V4E
LABEL_COLUMN = 'buy_sell_point'
LABELS = [0, 1]



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Classifiers feature x
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

parameters_HistGradientBoostingRegressor = dict(loss="squared_error", max_bins=32, max_iter=50)

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    KNeighborsClassifier(5),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                         max_depth=8, max_features=5, max_leaf_nodes=None,
                         min_impurity_decrease=0.0,# min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                         oob_score=False, random_state=None, verbose=0, warm_start=False),
    RandomForestClassifier(),

    #MORE OTHER WAys
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1, verbose=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1 , max_iter=500),
    AdaBoostClassifier(),
    #NO GaussianNB(),
    LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis(),#En este caso, LDA no puede diferenciar sus influencias en el resto del mundo. No puedo diagnosticar nada específico, ya que no proporcionó el MCVE sugerido.
    ExtraTreesClassifier(verbose=True),
    #NO HistGradientBoostingRegressor(**parameters_HistGradientBoostingRegressor) , # , quantile=[0.95, 0.5, 0.05] https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_1_0.html#sphx-glr-auto-examples-release-highlights-plot-release-highlights-1-1-0-py
    #NO IsotonicRegression(out_of_bounds="clip"), #https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_isotonic_regression.html#sphx-glr-auto-examples-miscellaneous-plot-isotonic-regression-py
    #NO KNeighborsRegressor(5, weights="distance"),
    #NO SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
    #NO DecisionTreeRegressor(max_depth=5),
    #NO AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=np.random.RandomState(1) )#`++++ https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py
]
from sklearn.utils import all_estimators

estimators = all_estimators(type_filter='classifier')
list_valid = ["AdaBoostClassifier","BaggingClassifier","BernoulliNB","CalibratedClassifierCV","CategoricalNB","","ComplementNB",
              "DecisionTreeClassifier","DummyClassifier","ExtraTreeClassifier","ExtraTreesClassifier",#"GaussianNB",#"GaussianProcessClassifier",
                "GradientBoostingClassifier","HistGradientBoostingClassifier","KNeighborsClassifier",
              "LabelPropagation","LabelSpreading","LinearDiscriminantAnalysis","LinearSVC","LogisticRegression","LogisticRegressionCV",
              "MLPClassifier","","MultinomialNB","NearestCentroid","NuSVC","","OneVsRestClassifier","","OutputCodeClassifier",
              "PassiveAggressiveClassifier","Perceptron","QuadraticDiscriminantAnalysis","RadiusNeighborsClassifier","RandomForestClassifier",
              "RidgeClassifier","RidgeClassifierCV","SGDClassifier","SVC"]
all_clfs = []
for name, ClassifierClass in estimators:
    print('Appending', name)
    try:
        if name in list_valid:
            clf = ClassifierClass()
            all_clfs.append(clf)
    except Exception as e:
        print('Unable to import', name)
        print(e)



# iterate over classifiers
for item in all_clfs:# classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    try:
        print()
        # Cree un clasificador, entrenador y probador según algoritmo.
        clf = item
        clf.fit(X_train, y_train.values.ravel())
        pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)

        #VALIDOS
        df_pred =  pd.DataFrame( pred )
        df_pred['counter'] = range(len(df_pred))
        df_yTest  = y_test.copy()
        df_yTest['counter'] = range(len(df_yTest))
        df_pred.groupby(0).count()
        df_yTest.groupby('buy_sell_point').count()
        df_m = pd.merge(df_pred, df_yTest, on=['counter'])#, how='outer'
        df_m = df_m.loc[df_m[0] != 0]
        df_m['r'] = False
        df_m.loc[df_m[0] ==  df_m['buy_sell_point'], 'r'] = True
        # df_m['r'].value_counts()
        countValidTrue = df_m.loc[df_m[0] ==  df_m['buy_sell_point']].count()[0]
        valid_string =  "VALIDOS de los "+str(len(df_m['counter']))+" points , son validos "+ str(countValidTrue)
        df_m.to_csv("./d_price_skl/" + classifier_name + "_" + str(countValidTrue)+"___"+ str(len(df_m['counter'])) +".csv", sep='\t', index=None)
        print(classifier_name + ' Precisión ' + str(round(score, 3) * 100) + '% VALIDOS: '+valid_string )


        # save https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn
        # print("./d_price_skl/" + stockId +"_matr_"+ classifier_name + "_" + str(round(score, 2) * 100) + "per.pkl")
        # joblib.dump(clf, "./d_price_skl/" + stockId +"_S_matr_"+ classifier_name + "_" + str(round(score, 2) * 100) + "per.pkl")

        #Utils_plotter.plot_feature_importances_loans(model=clf, X= X, path="./d_price_skl/" + stockId +"_I_matr_"+ classifier_name + "_RELA_" + str(round(score, 2) * 100) + ".png")
        # load
        # clf2 = joblib.load("model.pkl")
        # clf2.predict(X[0:1])

        print( classifier_name + ' Precisión ' + str(round(score, 3) * 100) + '%')
        # Calcular la matriz de confusión
        # cnf_matrix = confusion_matrix(y_test, pred)
        # np.set_printoptions(precision=2)

        # Trazar una matriz de confusión no normalizada
        class_names = ['b_-100', 'b_0', 'b_100']

        # Utils_plotter.plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Matriz de ' + classifier_name + ' Precisión ' + str(round(score, 3) * 100) + '%', y_test=y_test,
        #                                     pathImg="./d_price_skl/" + stockId +"_matr_"+ classifier_name + "_" + str(round(score, 3) * 100) + "per.png")
    except Exception as e:
        print('Unable to import', name)
        print(e)

# path_bars = "./d_price_skl/" + stockId  +  "_feature_importances_.png"
# Utils_plotter.plot_bars_feature_importances(dfDummy.columns,clf.feature_importances_,num_bars = 20, path_bars =path_bars  )


print("END")