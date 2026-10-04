import Model_train_sklearn_XGB
from Utils import Utils_col_sele
import xgboost as xgb


Y_TARGET = 'buy_sell_point'
k = "reg89"
v = Utils_col_sele.DICT_COLUMNS_TYPES[k]

# for k , v in Utils_col_sele.DICT_COLUMNS_TYPES:
columns_selection = ['Date', Y_TARGET, 'ticker'] + v
print("DICT_COLUMNS_TYPES: " +k+" Columns Selected:" + ', '.join(columns_selection))
X_train, X_test, y_train, y_test = Model_train_sklearn_XGB.get_x_y_train_test_sklearn_XGB(columns_selection, path="d_price/FAV_SCALA_stock_history_MONTH_3.csv")

"""Ok, now we're talking. Any chances of getting better with optimization?"""
xgb_model = xgb.XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)


fraud_ratio =y_train.value_counts()[1 ] /y_train.value_counts()[0]
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [1 ,3 ,5], #'max_depth': range(2, 10, 2),
              'min_child_weight': [1 ,3 ,5],
              'n_estimators': [100 ,200 ,500 ,1000],#'n_estimators': range(100, 1000, 40),
              'learning_rate': [0.1, 0.01, 0.05],
              'scale_pos_weight': [1, 0.1, 0.01, fraud_ratio]
              }
print("TEST PARAM: ",param_grid)
# {'max_depth': 1, 'min_child_weight': 1, 'n_estimators': 500, 'scale_pos_weight': 1}
# [05:20:01] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.6.0/src/learner.cc:627:
# Parameters: { "criterion" } might not be used.
CV_GBM = GridSearchCV(estimator = xgb_model,
                      param_grid = param_grid,
                      scoring = 'f1',
                      cv = 10,
                      n_jobs = -1,
                      refit = True)
print("START CV_GBM.fit  ")
CV_GBM.fit(X_train, y_train, verbose=10)

CV_GBM.best_params_
print(CV_GBM.best_params_)

# TEST PARAM:  {'max_depth': [1, 3, 5], 'min_child_weight': [1, 3, 5], 'n_estimators': [100, 200, 500, 1000], 'learning_rate': [0.1, 0.01, 0.05], 'scale_pos_weight': [1, 0.1, 0.01, 1.0]}
# START CV_GBM.fit
# {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 500, 'scale_pos_weight': 0.01}