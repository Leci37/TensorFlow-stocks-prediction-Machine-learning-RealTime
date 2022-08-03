import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import Utils_model_predict
import Utils_plotter

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
raw_df = pd.read_csv("d_price/FAV_SCALA_stock_history_E_MONTH_3.csv",index_col=False, sep='\t')
#raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')


Y_TARGET = 'buy_sell_point'
model_folder = "Models/RandomForest/"
model_h5_name = 'RandomForest.h5'

raw_df = Utils_model_predict.cast_Y_label_binary(raw_df,  label_name = Y_TARGET)
df = Utils_model_predict.clean_redifine_df(raw_df)
# neg, pos = np.bincount(df[Y_TARGET])

Utils_plotter.plot_pie_countvalues(df,Y_TARGET , stockid= "", opion = "", path=model_folder )
# print(df.isnull().sum())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df.drop(columns=Y_TARGET) #iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
y = df[Y_TARGET] #df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print("Shapes  X: ", X.shape, "  Y: ", y.shape)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25, random_state = 42)
print('df', df.shape,'\t',#'train',train.shape,'\n',
      'X_train',X_train.shape,'\t',
      'y_train',y_train.shape,'\n',
      'X_test',X_test.shape,'\t',
      'y_test',y_test.shape,'\t'     )

from xgboost import XGBClassifier
xgb = XGBClassifier(n_jobs=-1,
                     n_estimators=500,
                     max_depth=1,
                     min_child_weight=1,
                     criterion = 'entropy',
                     scale_pos_weight=1)

rfc = RandomForestClassifier(n_jobs=-1,# random_state = 42,
                             n_estimators=500,
                             max_features='sqrt',
                             min_samples_leaf=1,
                             criterion = 'entropy')

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=600,
                                max_depth=5,
                                learning_rate=0.01,
                                min_samples_split=3)

gbr.fit(X_train, y_train)

#Make predictions
print('Classification of SMOTE-resampled dataset with GradientBoostingRegressor')
y_pred = gbr.predict(X_test)
try:
    scores = gbr.decision_function(X_test)
except:
    scores = gbr.predict_proba(X_test)[:, 1]
#Make plots
y_pred = gbr.predict(X_test)
p_tolerance = 0.5
Utils_plotter.plot_confusion_matrix_cm_IN( y_test, scores, path= model_folder + "GradientBoostingRegre_SMOTE_confusion_matrix_"+str(p_tolerance)+".png", p=p_tolerance)
Utils_plotter.plot_average_precision_score(y_test, scores, path= model_folder + "GradientBoostingRegre_SMOTE_aucprc_"+str(p_tolerance)+".png")





#fit the best models so far
xgb.fit(X_train, y_train)
rfc.fit(X_train, y_train)

#Make predictions
print('Classification of SMOTE-resampled dataset with XGboost')
y_pred = xgb.predict(X_test)
try:
    scores = xgb.decision_function(X_test)
except:
    scores = xgb.predict_proba(X_test)[:,1]
#Make plots
y_pred = xgb.predict(X_test)
p_tolerance = 0.85
Utils_plotter.plot_confusion_matrix_cm_IN( y_test, scores, path= model_folder + "XGboost_SMOTE_confusion_matrix_"+str(p_tolerance)+".png", p=p_tolerance)
Utils_plotter.plot_average_precision_score(y_test, scores, path= model_folder + "XGboost_SMOTE_aucprc_"+str(p_tolerance)+".png")


#Make predictions
print('Classification of SMOTE-resampled dataset with optimized RF')
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]


print(y_pred.shape, "  ", y_test.shape)
#Make plots
p_tolerance = 0.2
Utils_plotter.plot_confusion_matrix_cm_IN( y_test, scores, path= model_folder + "RandomForestClassifier_SMOTE_confusion_matrix_"+str(p_tolerance)+".png", p=p_tolerance)
Utils_plotter.plot_average_precision_score(y_test, scores, path= model_folder + "RandomForestClassifier_SMOTE_aucprc_"+str(p_tolerance)+".png")