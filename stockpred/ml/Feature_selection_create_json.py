'''
"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

'''

import pandas as pd
# import numpy as np
# import sklearn
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, f_regression
# from numpy import array
#
# from Utils import Utils_buy_sell_points, Utils_col_sele, UtilsL, Utils_plotter
# import _KEYS_DICT
# from Utils.Utils_col_sele import DROPS_COLUMNS

Y_TARGET = 'target'

# def get_best_columns_to_train(cleaned_df, op_buy_sell : _KEYS_DICT.Op_buy_sell , num_best , CSV_NAME,path = None):
#     df_result = pd.DataFrame()
#
#     df = Utils_buy_sell_points.select_work_buy_or_sell_point(cleaned_df.copy(), op_buy_sell)
#
#     df = df.dropna()
#     X = df.drop(columns=Y_TARGET)
#     y = df[Y_TARGET]
#
#     '''SelectKBest chi2'''
#     def get_correlation_kbest(x_kb):
#
#         print("SelectKBest: chi2 ")
#         print("Feature data dimension: ", x_kb.shape)
#         select = SelectKBest(score_func=chi2, k=num_best)
#         z = select.fit_transform(x_kb, y)
#         print("After selecting best 3 features_W3:", z.shape)
#         filter = select.get_support()
#         features_W3 = array(X.columns)
#         print("Selected best ", num_best, ": ")
#         print(features_W3[filter])
#         df_result['chi2'] = features_W3[filter]
#
#
#     '''SelectKBest f_regression'''
#     def get_correlation_feature(x_kb):
#         print("SelectKBest: f_regression ")
#         print("Feature data dimension: ", x_kb.shape)
#         select = SelectKBest(score_func=f_regression, k=num_best)
#         z = select.fit_transform(x_kb, y)
#         print("After selecting best 8 features_W3:", z.shape)
#         filter = select.get_support()
#         features_W3 = array(X.columns)
#         print("Selected best ", num_best, ": ")
#         df_result['f_regression'] = features_W3[filter]
#
#
#     ''' ExtraTreesClassifier '''
#     def get_tree_correlation():
#         print(" ExtraTreesClassifier ")
#         model = ExtraTreesClassifier()
#         model.fit(X, y)
#         print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
#         # plot graph of feature importances for better visualization
#         feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#         # feat_importances.nlargest(20).plot(kind='barh')
#         # plt.show()
#         feat_importances = feat_importances.sort_values(ascending=False)[:num_best]
#         feat_importances = feat_importances.reset_index(level=0)
#         df_result['ExtraTrees'] = feat_importances['index']
#         df_result['ExtraTrees_points'] = feat_importances[0]
#
#
#     ''' Correlation Matrix with Heatmap '''
#
#     def get_correlation_corrwith():
#         global df
#         print(" Correlation Matrix with Heatmap ")
#         try:
#             dcf = df.corrwith(df[Y_TARGET])
#         except (ValueError, UnboundLocalError, TypeError) as e:
#             logging.info(f"Exception occurred: {str(e)}")
#             df[Y_TARGET] = pd.to_numeric(df[Y_TARGET], errors='coerce')
#             dcf = df.select_dtypes(include=[np.number]).corrwith(df[Y_TARGET])
#
#         dcf = dcf.abs().sort_values(ascending=False)[:num_best]
#         df_result['corrwith'] = dcf.index
#         df_result['corrwith_points'] = dcf.values
#
#     x_kb = sklearn.preprocessing.MinMaxScaler(feature_range=(_KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER)).fit_transform(X)
#     get_correlation_kbest(x_kb)
#     get_correlation_feature(x_kb)
#     get_tree_correlation()
#     get_correlation_corrwith()
#     df_result = df_result.round(4)
#
#     if path is not None:
#         df_result.to_csv(path,sep='\t', index=None)
#         print("END plots_relations/best_selection_" + CSV_NAME + "_" + opcion.value + "_" + str(num_best) + ".csv")
#
#     return df_result

def created_json_feature_selection(list_all_columns, path_json): #get_json_feature_selection

    df_aux = pd.DataFrame({"ele": list_all_columns, "count": 0})
    df_aux = df_aux.groupby("ele").count().sort_values(["count"], ascending=False)
    #For wildcard files that combine multiple actions
    if "@" in path_json:
        df_aux["count"] = df_aux["count"] - 10
        df_aux = df_aux[(df_aux["count"] > 0) ]

    df_aux['index'] = df_aux.index
    df_aux["count"] = pd.to_numeric(df_aux["count"])
    df_json = df_aux.groupby('count', as_index=False).agg(list)
    df_json = df_json.sort_values('count', ascending=False)
    df_json.set_index('count', inplace=True)
    dict_json = df_json.to_dict()
    import json
    with open(path_json, 'w') as fp:
        json.dump(dict_json, fp, allow_nan=True, indent=3)
        print("\tcreated_json_feature_selection path: ", path_json)
    print(path_json)



# def generate_json_best_columns(cleaned_df, Op_buy_sell: _KEYS_DICT.Op_buy_sell,
#                                list_columns_got=[8, 12, 16, 32, 72], path_json="plots_relations/best_selection_sum_up.json",path_imgs = None,
#                                NUM_MAX_PLOT_RELATION_IMAGE_PER_STOCK=3):
#     list_all_columns = []
#     list_cols_plot = []
#     for n in list_columns_got:
#         print("\tget best columns Opcion: ", Op_buy_sell.value, " Number: ", n)
#         df = get_best_columns_to_train(cleaned_df, Op_buy_sell, n, CSV_NAME, path=None)
#
#         for c in ['chi2', 'f_regression', 'ExtraTrees', 'corrwith']:  # , ,
#             list_all_columns += df[c].to_list()
#         list_cols_plot += df['corrwith'].to_list()
#
#     #Remove elements not valid
#     list_all_columns = list(filter(lambda a: a != Y_TARGET, list_all_columns))
#     list_all_columns = list(filter(lambda a: a != "Date", list_all_columns))
#     list_all_columns = list(filter(lambda a: a != "ichi_chikou_span", list_all_columns))
#     get_json_feature_selection(list_all_columns, path_json)
#
#     if path_imgs is not None:
#         df_aux = pd.DataFrame({"ele": list_cols_plot, "count": 0})
#         df_aux = df_aux.groupby("ele").count().sort_values(["count"], ascending=False)
#         list_most_relation_cols = df_aux.index
#         list_most_relation_cols = list_most_relation_cols[:NUM_MAX_PLOT_RELATION_IMAGE_PER_STOCK+1].tolist()
#         print("Generate plots Path: "+ path_imgs + " Best relations columns: "+ "".join(list_most_relation_cols) )
#         Utils_plotter.plot_relationdist_main_val_and_all_rest_val(cleaned_df[list_most_relation_cols], main_label = Y_TARGET, path =path_imgs)
#     return list_all_columns

#**DOCU**
# #2 Filtering indicators
# It is necessary to separate the technical indicators that are related to buy or sell points and those that are noise. 20 seconds per action
# Run Model_creation_scoring.py
# Three files are generated for each action in the folder: plots_relations , relations for buy "pos", relations for sell "neg" and relations for both "both".
# plots_relations/best_selection_AMD_both.json
# These files contain a ranking of which technical indicator is best for each stock.
# Check that three .json have been generated for each stock.
import logging
numba_logger = logging.getLogger('numba').setLevel(logging.WARNING)
mat_logger = logging.getLogger('matplotlib').setLevel(logging.WARNING)

# CSV_NAME = "@CHILL"
# list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# NUM_BEST_PARAMS_LIST = [8, 12, 16, 32, 68]
# opion = _KEYS_DICT.Option_Historical.MONTH_3
# for l in list_stocks:
#     path_csv_price = "d_price/" + l + "_PLAIN_stock_history_" + str(opion.name) + ".csv"
#     created_json_relations(l, path_csv_price, opion )
#     CSV_NAME = l

# def created_json_relations(S , path_csv_price):
#     global df,CSV_NAME
#     CSV_NAME = S
#     # df = pd.read_csv("d_price/" + CSV_NAME + "_SCALA_stock_history_" + str(opion.name) + ".csv",index_col=False, sep='\t')
#     df = pd.read_csv(path_csv_price, index_col=False,sep='\t')
#     print("created_json_relations: "+path_csv_price)
#     df = df.drop(columns=Utils_col_sele.DROPS_COLUMNS)  # +['ticker']
#     if 'ticker' in df.columns:
#         df = df.drop(columns=['ticker'])  # opcional
#
#     cleaned_df = df.copy()
#     cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date']).map(pd.Timestamp.timestamp)
#
#     for option_Cat_op in _KEYS_DICT.Op_buy_sell.list():  # both pos neg
#         path_json = "plots_relations/best_selection_" + S + "_" + option_Cat_op.value + ".json"
#         path_img = "plots_relations/plot/" + S + "_" + option_Cat_op.value + "_"
#         path_img = None #remove it if you want grafical
#
#         generate_json_best_columns(cleaned_df, option_Cat_op, list_columns_got=NUM_BEST_PARAMS_LIST,
#                                    path_json=path_json, path_imgs=path_img)



