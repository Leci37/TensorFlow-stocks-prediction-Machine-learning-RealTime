'''

https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

'''
from enum import Enum

import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from numpy import array

import a_manage_stocks_dict


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
class Option_Cat(ExtendedEnum):
    BOTH = "both"
    POS = "pos"
    NEG = "neg"

import Utils_col_sele

Y_TARGET = 'buy_sell_point'

def get_best_columns_to_train(cleaned_df, opcion,num_best , CSV_NAME,path = None):
    df_result = pd.DataFrame()

    cleaned_df[Y_TARGET].replace([101, -101], [100, -100], inplace=True)
    if opcion == str(Option_Cat.POS.name):
        cleaned_df['buy_sell_point'].replace([-100], [0], inplace=True)  # Solo para puntos de compra
    if opcion == str(Option_Cat.NEG.name):
        cleaned_df['buy_sell_point'].replace([100], [0], inplace=True)  # Solo para puntos de venta
    df = cleaned_df.dropna()
    X = df.drop(columns=Y_TARGET)
    y = df[Y_TARGET]

    '''SelectKBest chi2'''
    def get_correlation_kbest(x_kb):

        print("SelectKBest: chi2 ")
        print("Feature data dimension: ", x_kb.shape)
        select = SelectKBest(score_func=chi2, k=num_best)
        z = select.fit_transform(x_kb, y)
        print("After selecting best 3 features:", z.shape)
        filter = select.get_support()
        features = array(X.columns)
        print("Selected best ", num_best, ": ")
        print(features[filter])
        df_result['chi2'] = features[filter]


    '''SelectKBest f_regression'''
    def get_correlation_feature(x_kb):
        print("SelectKBest: f_regression ")
        print("Feature data dimension: ", x_kb.shape)
        select = SelectKBest(score_func=f_regression, k=num_best)
        z = select.fit_transform(x_kb, y)
        print("After selecting best 8 features:", z.shape)
        filter = select.get_support()
        features = array(X.columns)
        print("Selected best ", num_best, ": ")
        df_result['f_regression'] = features[filter]


    ''' ExtraTreesClassifier '''
    def get_tree_correlation():
        print(" ExtraTreesClassifier ")
        model = ExtraTreesClassifier()
        model.fit(X, y)
        print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        # feat_importances.nlargest(20).plot(kind='barh')
        # plt.show()
        feat_importances = feat_importances.sort_values(ascending=False)[:num_best]
        feat_importances = feat_importances.reset_index(level=0)
        df_result['ExtraTrees'] = feat_importances['index']
        df_result['ExtraTrees_points'] = feat_importances[0]


    ''' Correlation Matrix with Heatmap '''
    def get_correlation_corrwith():
        global df
        print(" Correlation Matrix with Heatmap ")
        dcf = df.corrwith(df[Y_TARGET])
        dcf = dcf.abs().sort_values(ascending=False)[:num_best]
        df_result['corrwith'] = dcf.index
        df_result['corrwith_points'] = dcf.values

    x_kb = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
    get_correlation_kbest(x_kb)
    get_correlation_feature(x_kb)
    get_tree_correlation()
    get_correlation_corrwith()
    df_result = df_result.round(4)

    if path is not None:
        df_result.to_csv(path,sep='\t', index=None)
        print("END plots_relations/best_selection_" + CSV_NAME + "_" + opcion + "_" + str(num_best) + ".csv")

    return df_result

def get_json_feature_selection(list_all_columns , path):

    df_aux = pd.DataFrame({"ele": list_all_columns, "count": 0})
    df_aux = df_aux.groupby("ele").count().sort_values(["count"], ascending=False)
    df_aux['index'] = df_aux.index
    df_aux["count"] = pd.to_numeric(df_aux["count"])
    df_json = df_aux.groupby('count', as_index=False).agg(list)
    df_json = df_json.sort_values('count', ascending=False)
    df_json.set_index('count', inplace=True)
    dict_json = df_json.to_dict()
    import json
    with open(path, 'w') as fp:
        json.dump(dict_json, fp, allow_nan=True, indent=3)
        print("\tget_json_feature_selection path: ", path)
    print(path)


def generate_json_best_columns(cleaned_df, Option_Cat_op = "POS", list_columns_got = [8, 12, 16, 32, 72], path ="plots_relations/best_selection_sum_up.json"):
    list_all_columns = []
    for n in list_columns_got:
        print("\tget best columns Opcion: ", Option_Cat_op, " Number: ", n)
        df = get_best_columns_to_train(cleaned_df, Option_Cat_op, n, CSV_NAME, path=None)

        for c in ['chi2', 'f_regression', 'ExtraTrees', 'corrwith']:  # , ,
            list_all_columns += df[c].to_list()

    #Remove elements not valid
    list_all_columns = list(filter(lambda a: a != Y_TARGET, list_all_columns))
    list_all_columns = list(filter(lambda a: a != "Date", list_all_columns))
    list_all_columns = list(filter(lambda a: a != "ichi_chikou_span", list_all_columns))
    get_json_feature_selection(list_all_columns,path )



CSV_NAME = "@VOLA"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
# list_stocks = ['@ROLL']
for l in list_stocks:
    CSV_NAME = l


    NUM_BEST_PARAMS_LIST = [8, 12, 16, 32, 72]

    df = pd.read_csv("d_price/" + CSV_NAME + "_SCALA_stock_history_MONTH_3.csv",index_col=False, sep='\t')
    #df = pd.read_csv("d_price/" + CSV_NAME + "_SCALA_stock_history_MONTH_3_sep.csv", index_col=False, sep='\t')
    df = df.drop(columns=Utils_col_sele.DROPS_COLUMNS +['ticker'] )
    cleaned_df = df.copy()
    # You don't want the `Time` column.
    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date']).map(pd.Timestamp.timestamp)


    for option_Cat_op in Option_Cat.list():
        path = "plots_relations/best_selection_" + CSV_NAME + "_" + option_Cat_op + ".json"
        generate_json_best_columns(cleaned_df, option_Cat_op, NUM_BEST_PARAMS_LIST, path)
