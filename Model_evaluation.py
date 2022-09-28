import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import Utils_buy_sell_points
import Utils_col_sele
import Utils_model_predict
import Utils_plotter

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
from Utils_col_sele import MUY_BUENOS_COLUMNAS_TRAINS

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
#TO CONFIGURE

#raw_df = raw_df[ Columns ]

MODELS_EVAL_RESULT = "Models/all_results/"
# model_h5_name = 'RandomForest.h5'

# raw_df = Utils_model_predict.load_and_clean_DF_Train("d_price/FAV_SCALA_stock_history_MONTH_3.csv")
# Utils_plotter.plot_pie_countvalues(raw_df, Y_TARGET, stockid="", opion="", path=MODELS_EVAL_RESULT)
# df = raw_df
#
#
#
# X = df.drop(columns=Y_TARGET) #iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
# y = df[Y_TARGET] #df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
# print("Shapes  X: ", X.shape, "  Y: ", y.shape)

from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.28, shuffle=False)
# print('df', df.shape,'\t',#'train',train.shape,'\n',
#       'X_train',X_train.shape,'\t',
#       'y_train',y_train.shape,'\n',
#       'X_test',X_test.shape,'\t',
#       'y_test',y_test.shape,'\t'     )
#
# print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
# print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
#
# # smote = SMOTE(random_state=2)
# # X_train, y_train = smote.fit_resample(X_train, y_train)
#
# print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
# print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
#
# print("After OverSampling, counts of label '1': {}".format(sum(y_train==1)))
# print("After OverSampling, counts of label '0': {}".format(sum(y_train==0)))



'''GradientBoostingRegressor'''
def predict_GradientBoostingRegressor(X_test, SAV_files_surname ):
    # Make predictions
    print('Classification of SMOTE-resampled dataset with GradientBoostingRegressor')
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = "Models/Sklearn_smote/" + 'GradientBoost_' + SAV_files_surname + '.sav'
    print("Load: ", save_model_path)
    # load the model from disk
    model_gbr = pickle.load(open(save_model_path, 'rb'))
    # try:
    #     scores = model_gbr.decision_function(X_test)
    # except:
    #     scores = model_gbr.predict_proba(X_test)[:, 1]
    # Make plots
    y_pred = model_gbr.predict(X_test) #para este caso predit y puntuaciones es lo mismo
    p_tolerance = 0.56 #0.47 without SMOTE  # for GradientBoostingRegressor, con esto se consiguen 0 falsos positivos y 74 % de aciertos
    # for to in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:#[0.45,0.47,0.5,0.53,0.56,0.6]:
    #     p_tolerance = to
    # Utils_plotter.plot_confusion_matrix_cm_IN(y_test, y_pred,
    #                                               path=MODELS_EVAL_RESULT + "eval_Gradient_CM_" + str(
    #                                                   p_tolerance) + SAV_files_surname +".png", p=p_tolerance)
    return y_pred #> p_tolerance



'''XGBClassifier'''
def predict_XGBClassifier(X_test, SAV_files_surname  ):
    save_model_path = "Models/Sklearn_smote/" + 'XGboost_' + SAV_files_surname + '.sav'
    print('Classification of SMOTE-resampled dataset with XGboost')
    print("Load: ", save_model_path)
    model_xgb = pickle.load(open(save_model_path, 'rb'))
    # result = model_xgb.score(X_test, y_test)
    # print(result)
    y_pred = model_xgb.predict(X_test)
    try:
        scores = model_xgb.decision_function(X_test)
    except:
        scores = model_xgb.predict_proba(X_test)[:, 1]
    # Make plots
    y_pred = model_xgb.predict(X_test)
    p_tolerance = 0.7 #0.27 without SMOTE
    # for to in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] : # [0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85]:
    #     p_tolerance = to
    # Utils_plotter.plot_confusion_matrix_cm_IN(y_test, scores,
    #                                           path=MODELS_EVAL_RESULT + "eval_XGboost_CM_" + str(
    #                                               p_tolerance) + SAV_files_surname +".png", p=p_tolerance)
    return scores #> p_tolerance

'''RandomForestClassifier'''
def predict_Random_Forest(X_test, SAV_files_surname  ):
    print('Classification of SMOTE-resampled dataset with optimized RF')
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = "Models/Sklearn_smote/" + 'RandomForest_' + SAV_files_surname + '.sav'
    print("Load: ", save_model_path)
    # load the model from disk
    model_rfc = pickle.load(open(save_model_path, 'rb'))
    y_pred = model_rfc.predict(X_test)
    try:
        scores = model_rfc.decision_function(X_test)
    except:
        scores = model_rfc.predict_proba(X_test)[:, 1]
    # Make plots
    p_tolerance = 0.7
    # for to in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: #[0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85]:
    #     p_tolerance = to
    # Utils_plotter.plot_confusion_matrix_cm_IN(y_test, scores,
    #                                           path=MODELS_EVAL_RESULT + "eval_Random_Forest_CM_" + str(p_tolerance) + SAV_files_surname + ".png",
    #                                           p=p_tolerance)
    return scores #> p_tolerance

from tensorflow import keras
def predict_TF_onBalance(X_test,  model_folder, model_h5_name):
    print(" \n", model_folder + model_h5_name)
    resampled_model_2 = keras.models.load_model(model_folder + model_h5_name)
    """### Re-check training history"""
    # plot_metrics(resampled_history)
    """### Evaluate metrics"""
    BATCH_SIZE = 2048
    test_predictions_resampled = resampled_model_2.predict(X_test, batch_size=BATCH_SIZE)
    # resampled_results = resampled_model_2.evaluate(X_test, test_labels,
    #                                                batch_size=BATCH_SIZE, verbose=0)
    # for name, value in zip(resampled_model_2.metrics_names, resampled_results):
    #     print(name, ': ', value)
    print()
    # p_tolerance = 0.7
    # for to in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  # [0.45,0.47,0.5,0.53,0.56,0.6]:
    #     p_tolerance = to
    # Utils_plotter.plot_cm_TF_imbalance(y_test, test_predictions_resampled,
    #                                        path=MODELS_EVAL_RESULT +  "eval_TFbalance_CM_" + model_h5_name.replace(".h5", "") +"_"+ str(p_tolerance)+".png", p=p_tolerance)
    # Utils_plotter.plot_confusion_matrix(cf_matrix, model_folder + "plot_confusion_matrix.png")
    return  test_predictions_resampled

#Make predictions



#df_result = pd.DataFrame()



#model_folder_TF = "Models/TF_in_balance/"
MODEL_FOLDER_TF = "Models/TF_balance/"
path= "d_price/FAV_SCALA_stock_history_MONTH_3.csv"
columns_aux_to_evaluate = ["Close", "per_Close", 'has_preMarket', 'Volume'] #necesario para Utils_buy_sell_points.check_buy_points_prediction
df_result_all = None
df_result_all_isValid_to_buy = None


for k , v in Utils_col_sele.DICT_COLUMNS_TYPES.items():
    columns_selection = ['Date', Y_TARGET, 'ticker'] + v + columns_aux_to_evaluate
    k_aux = k + '_2'
    model_h5_name_k = "TF_" + k_aux + '.h5'
    print("DICT_COLUMNS_TYPES: " +k_aux+" Columns Selected:" + ', '.join(columns_selection))

    df = Utils_model_predict.load_and_clean_DF_Train(path, columns_selection)
    X = df.drop(columns=Y_TARGET)  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, shuffle=False)
    df_result = X_test.copy()
    #PARA organizar la salida en el df_result de los resultados
    if df_result_all is None:
        df_result_all = X_test[['Date', "Close", "per_Close", 'has_preMarket', 'Volume']].copy()
        list_ticker_stocks = [col for col in X_test.columns if col.startswith('ticker_')]  # todas las que empiecen por ticker_ , son variables tontas
        df_result_all['ticker'] = X_test[list_ticker_stocks].idxmax(axis=1).copy()  # undo dummy variable
        df_result_all[Y_TARGET] = y_test.copy()
        df_result_all = df_result_all[['Date',Y_TARGET, 'ticker', "Close", "per_Close", 'has_preMarket', 'Volume']]

    model_h5_name_TF = 'TF_'+k_aux+'.h5'
    p_tolerance = 0.45*2
    # df_TF = Utils_model_predict.cast_Y_label_binary(raw_df.copy(),  label_name = Y_TARGET)
    # df_TF = Utils_model_predict.clean_redifine_df(df_TF)
    df_TF = Utils_model_predict.load_and_clean_DF_Train("d_price/FAV_SCALA_stock_history_MONTH_3.csv", columns_selection)
    df_TF = df_TF.drop(columns=columns_aux_to_evaluate)
    train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(df_TF, label_name = Y_TARGET)
    df_result['result_TF_'+k] = predict_TF_onBalance(test_features, MODEL_FOLDER_TF, model_h5_name_k)
    Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_TF_'+k,
                                                      path_cm=MODELS_EVAL_RESULT + "2_TF_balance_CM_" + k_aux +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)

    p_tolerance = 0.5*2
    df_result['result_gbr_' + k] = predict_GradientBoostingRegressor(X_test.drop(columns=columns_aux_to_evaluate), k_aux)
    Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_gbr_' + k, path_cm=MODELS_EVAL_RESULT + "2_Gradient_CM_" + k_aux +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)


    p_tolerance = 0.24*2
    df_result['result_xgb_' + k] =  predict_XGBClassifier(X_test.drop(columns=columns_aux_to_evaluate), k_aux)
    Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_xgb_' + k, path_cm=MODELS_EVAL_RESULT + "2_XGB_CM_" + k_aux +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)

    p_tolerance = 0.249*2
    df_result['result_rf_' + k] = predict_Random_Forest(X_test.drop(columns=columns_aux_to_evaluate), k_aux)
    df_result['isValid_' + k] = Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_rf_' + k, path_cm=MODELS_EVAL_RESULT + "2_RamdonFo_CM_" + k_aux +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)

    # if Serie_isValid_to_buy is None:
    #     Serie_isValid_to_buy = df_isValid_to_buy_serie["isValid_to_buy"].copy()
    # if "isValid_to_buy" not in df_result_all.columns:
    #df_isValid_to_buy = Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_rf_' + k, path_cm=MODELS_EVAL_RESULT + "2_RamdonFo_CM_" + k +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)
    # df_isValid_to_buy['Date'] = pd.to_datetime(df_isValid_to_buy['Date']).map(pd.Timestamp.timestamp)
    # df_result['ticker'] = df_result[list_ticker_stocks].idxmax(axis=1).copy()  # undo dummy variable
    # df_result_all_isValid_to_buy = pd.merge(df_result, df_isValid_to_buy, on=[ 'Date','ticker'],  how='left')
        #df_result_all["isValid_to_buy"] = df_result_all
    df_result_all[[ 'isValid_' + k,  'result_TF_' + k, 'result_gbr_' + k, 'result_xgb_' + k, 'result_rf_' + k]] = df_result[['isValid_' + k, 'result_TF_' + k, 'result_gbr_' + k, 'result_xgb_' + k, 'result_rf_' + k]]
    df_result_all.groupby('isValid_' + k).mean().T.to_csv(MODELS_EVAL_RESULT + "_RESULTS_modles_isValid_" + k_aux +".csv", sep='\t')
    print(MODELS_EVAL_RESULT + "_RESULTS_modles_isValid_" + k_aux +".csv")

# 'Date', 'buy_sell_point', 'ticker', 'Close', 'per_Close',
#        'has_preMarket', 'Volume', 'isValid_vgood18', 'result_TF_vgood18',
#        'result_gbr_vgood18', 'result_xgb_vgood18', 'result_rf_vgood18',
#        'isValid_vgood28', 'result_TF_vgood28', 'result_gbr_vgood28',
#        'result_xgb_vgood28', 'result_rf_vgood28', 'isValid_vgood27',
#        'result_TF_vgood27', 'result_gbr_vgood27', 'result_xgb_vgood27',
#        'result_rf_vgood27', 'isValid_vgood37', 'result_TF_vgood37',
#        'result_gbr_vgood37', 'result_xgb_vgood37', 'result_rf_vgood37',
#        'isValid_good9', 'result_TF_good9', 'result_gbr_good9',
#        'result_xgb_good9', 'result_rf_good9', 'isValid_good19',
#        'result_TF_good19', 'result_gbr_good19', 'result_xgb_good19',
#        'result_rf_good19', 'isValid_reg37', 'result_TF_reg37',
#        'result_gbr_reg37', 'result_xgb_reg37', 'result_rf_reg37',
#        'isValid_reg52', 'result_TF_reg52', 'result_gbr_reg52',
#        'result_xgb_reg52', 'result_rf_reg52', 'isValid_reg89',
#        'result_TF_reg89', 'result_gbr_reg89', 'result_xgb_reg89',
#        'result_rf_reg89'
df_result_all.to_csv(MODELS_EVAL_RESULT+ "_RESULTS.csv", sep='\t', index=None)
#
# df_result_all[df_result_all["result_TF_vgood18"] > 1]
# df_result_all[df_result_all["result_TF_vgood28"] > 0.8]
#
# df_result_all[df_result_all["isValid_vgood37"] > 0.8]
df_result_all.groupby("isValid_vgood37").mean().T.to_csv(MODELS_EVAL_RESULT+ "_RESULTS_is_valid.csv", sep='\t')
df_result_all.groupby(Y_TARGET).mean().T.to_csv(MODELS_EVAL_RESULT+ "_RESULTS_modles_ground_true.csv", sep='\t')
print(MODELS_EVAL_RESULT+ "_RESULTS_modles_ground_true.csv")


#Utils_buy_sell_points.check_buy_points_prediction(df_result.copy(), result_column_name= 'result_rf_' + k, path_cm=MODELS_EVAL_RESULT + "2_RamdonFo_CM_" + k_aux +"_"+ str(p_tolerance)+".png", SUM_RESULT_2_VALID =p_tolerance)
# df_result_all.groupby("isValid_to_buy").mean().T.to_csv(MODELS_EVAL_RESULT+ "_RESULTS_modles_valid_to_buy.csv", sep='\t')

print("FIN")





