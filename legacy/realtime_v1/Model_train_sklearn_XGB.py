import pickle

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from Utils import Utils_model_predict, Utils_plotter
import _KEYS_DICT

Y_TARGET = 'buy_sell_point'
model_folder = "Models/Sklearn_smote/"


def get_x_y_train_test_sklearn_XGB( columns_selection, path ,op_buy_sell : _KEYS_DICT.Op_buy_sell):
    df = Utils_model_predict.load_and_clean_DF_Train_from_csv(path, op_buy_sell, columns_selection)
    Utils_plotter.plot_pie_countvalues(df, Y_TARGET, stockid="", opion="", path=model_folder)
    # print(df.isnull().sum())
    # Splitting the dataset into the Training set and Test set
    X = df.drop(columns=Y_TARGET)  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    # print("Shapes  X: ", X.shape, "  Y: ", y.shape)
    # smote = SMOTE(random_state=42)
    # X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, shuffle=False)
    print('df', df.shape, '\t',  # 'train',train.shape,'\n',
          'X_train', X_train.shape, '\t',
          'y_train', y_train.shape, '\n',
          'X_test', X_test.shape, '\t',
          'y_test', y_test.shape, '\t')
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
    # smote = SMOTE(random_state=2)
    # https://medium.com/@itbodhi/handling-imbalanced-data-sets-in-machine-learning-5e5f33c70163
    smote_tomek = SMOTETomek(sampling_strategy='all', random_state=2) # minority  majority
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0)))
    return X_train, X_test, y_train, y_test





def train_GradientBoost(X_train, X_test, y_train, y_test, SAV_files_surname = '' ):
    '''GradientBoostingRegressor'''
    model_gbr = GradientBoostingRegressor(n_estimators=500,
                                          max_depth=5,
                                          learning_rate=0.001,
                                          subsample=0.6,
                                          min_samples_split=3)
    model_gbr.fit(X_train, y_train)
    # Make predictions
    print('Classification of SMOTE-resampled dataset with GradientBoostingRegressor')
    y_pred = model_gbr.predict(X_test)
    # try:
    #     scores = gbr.decision_function(X_test)
    # except:
    #     scores = gbr.predict_proba(X_test)[:, 1]
    # Make plots
    p_tolerance = 0.75
    Utils_plotter.plot_confusion_matrix_cm_IN(y_test, y_pred,
                                              path=model_folder + "GradientBoost_"+SAV_files_surname+"_CM_" + str(
                                                  p_tolerance) + ".png", p=p_tolerance)
    Utils_plotter.plot_average_precision_score(y_test, y_pred, path=model_folder + "GradientBoost_" + SAV_files_surname + "_aucprc_" + str(
        p_tolerance) + ".png")
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = model_folder + 'GradientBoost_'+SAV_files_surname+'.sav'
    pickle.dump(model_gbr, open(save_model_path, 'wb'))
    print(" Save model : ", save_model_path)
    # some time later...
    # load the model from disk
    loaded_model = pickle.load(open(save_model_path, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)

def train_XGBClassifier(X_train, X_test, y_train, y_test,SAV_files_surname = '' ):
    '''XGBClassifier'''
    #TUNE 09/ 2022 {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 500, 'scale_pos_weight': 0.01}
    model_xgb = XGBClassifier(n_jobs=-1,n_estimators=600,max_depth=5,min_child_weight=1,criterion='entropy' # bootstrap = True
                              ,scale_pos_weight=0.01)
    # fit the best models so far
    model_xgb.fit(X_train, y_train)
    # Make predictions
    print('Classification of SMOTE-resampled dataset with XGboost')
    y_pred = model_xgb.predict(X_test)
    try:
        scores = model_xgb.decision_function(X_test)
    except:
        scores = model_xgb.predict_proba(X_test)[:, 1]
    # Make plots
    y_pred = model_xgb.predict(X_test)
    p_tolerance = 0.5
    Utils_plotter.plot_confusion_matrix_cm_IN(y_test, scores,
                                              path=model_folder + "XGboost_"+SAV_files_surname+"_CM_" + str(
                                                  p_tolerance) + ".png", p=p_tolerance)
    Utils_plotter.plot_average_precision_score(y_test, scores,
                                               path=model_folder + "XGboost_"+SAV_files_surname+"_aucprc_" + str(p_tolerance) + ".png")
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = model_folder + 'XGboost_'+SAV_files_surname+'.sav'
    pickle.dump(model_xgb, open(save_model_path, 'wb'))
    print(" Save model : ", save_model_path)
    # some time later...
    # load the model from disk
    loaded_model = pickle.load(open(save_model_path, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)





def train_RandomForestClassifier(X_train, X_test, y_train, y_test,SAV_files_surname = '' ):
    '''RandomForestClassifier'''
    model_rfc = RandomForestClassifier(n_jobs=-1,  # random_state = 42,
                                       n_estimators=100,
                                       max_features=2,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       bootstrap=True,
                                       max_depth=85,
                                       criterion='entropy')
    # {'bootstrap': True, 'max_depth': 85, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    # fit the best models so far
    model_rfc.fit(X_train, y_train)
    # Make predictions
    print('Classification of SMOTE-resampled dataset with optimized RF')
    y_pred = model_rfc.predict(X_test)
    try:
        scores = model_rfc.decision_function(X_test)
    except:
        scores = model_rfc.predict_proba(X_test)[:, 1]
    print(y_pred.shape, "  ", y_test.shape)
    # Make plots
    p_tolerance = 0.75
    Utils_plotter.plot_confusion_matrix_cm_IN(y_test, scores,
                                              path=model_folder + "RandomForest_"+SAV_files_surname+"_CM_" + str(
                                                  p_tolerance) + ".png", p=p_tolerance)
    Utils_plotter.plot_average_precision_score(y_test, scores,
                                               path=model_folder + "RandomForest_"+SAV_files_surname+"_aucprc_" + str(
                                                   p_tolerance) + ".png")
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = model_folder + 'RandomForest_'+SAV_files_surname+'.sav'
    pickle.dump(model_rfc, open(save_model_path, 'wb'))
    print(" Save model : ", save_model_path)
    # some time later...
    # load the model from disk
    loaded_model = pickle.load(open(save_model_path, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)







