import pickle

from Utils import Utils_plotter


def __generate_CM_prediction(save_model_path, SAV_files_surname, scores, y_test, list_CM_tolerance):
    model_file = SAV_files_surname.replace('.h5', '').replace('.sav', '')

    if not (y_test is None or (not list_CM_tolerance)):  # list null or empyt
        for p_tolerance in list_CM_tolerance:  # [0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85]:
            path_cm = save_model_path + "_eval_CM_" + model_file + "_" + str(p_tolerance) + ".png"
            Utils_plotter.plot_confusion_matrix_cm_IN(y_test, scores,
                                                      path=path_cm,
                                                      p=p_tolerance)

'''GradientBoostingRegressor'''
def predict_GradientBoostingRegressor(X_test, SAV_files_surname,y_test = None, list_CM_tolerance = [0.45,0.47,0.5,0.53,0.56,0.6] ):
    # Make predictions
    print('Classification of SMOTE-resampled dataset with GradientBoostingRegressor')
    # save the model to disk https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    save_model_path = "Models/Sklearn_smote/" + 'GradientBoost_' + SAV_files_surname + '.sav'
    print("Load: ", save_model_path)
    # load the model from disk
    model_gbr = pickle.load(open(save_model_path, 'rb'))
    y_pred = model_gbr.predict(X_test)  # para este caso predit y puntuaciones es lo mismo

    if not (y_test is None or (not list_CM_tolerance)): #list null or empyt
        __generate_CM_prediction(save_model_path, SAV_files_surname, y_pred, y_test, list_CM_tolerance)

    return y_pred #> p_tolerance



'''XGBClassifier'''
def predict_XGBClassifier(X_test, SAV_files_surname ,y_test = None, list_CM_tolerance = [0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85] ):
    save_model_path = "Models/Sklearn_smote/" + 'XGboost_' + SAV_files_surname + '.sav'
    print('Classification of SMOTE-resampled dataset with XGboost')
    print("Load: ", save_model_path)
    model_xgb = pickle.load(open(save_model_path, 'rb'))
    # result = model_xgb.score(X_test, y_test)
    # print(result)
    scores = None
    y_pred = model_xgb.predict(X_test)
    try:
        scores = model_xgb.decision_function(X_test)
    except:
        scores = model_xgb.predict_proba(X_test)[:, 1]

    if not (y_test is None or (not list_CM_tolerance)): #list null or empyt
        __generate_CM_prediction(save_model_path, SAV_files_surname, scores, y_test, list_CM_tolerance)

    return scores #> p_tolerance

'''RandomForestClassifier'''
def predict_Random_Forest(X_test, SAV_files_surname ,y_test = None, list_CM_tolerance = [0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85]):
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

    if not (y_test is None or (not list_CM_tolerance)):  # list null or empyt
        __generate_CM_prediction(save_model_path, SAV_files_surname, scores, y_test, list_CM_tolerance)
    return scores #> p_tolerance


#DATOS desequilibrados https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
from tensorflow import keras
def predict_TF_onBalance(X_test,  save_model_path, model_h5_name,y_test = None, list_CM_tolerance = [0.45,0.47,0.5,0.53,0.56,0.6] ):
    print(" \n", save_model_path + model_h5_name)
    resampled_model_2 = keras.models.load_model(save_model_path + model_h5_name)
    """### Re-check training history"""
    # plot_metrics(resampled_history)
    """### Evaluate metrics"""
    BATCH_SIZE = 2048
    test_predictions_resampled = resampled_model_2.predict(X_test, batch_size=BATCH_SIZE)

    if not (y_test is None or (not list_CM_tolerance)):  # list null or empyt
        __generate_CM_prediction(save_model_path, model_h5_name, test_predictions_resampled, y_test, list_CM_tolerance)

    return  test_predictions_resampled




