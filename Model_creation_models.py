import Model_train_TF_onBalance
import Model_train_sklearn_XGB
import Utils_col_sele

Y_TARGET = 'buy_sell_point'
model_folder = "Models/Sklearn_smote/"
csv_file_SCALA = "d_price/FAV_SCALA_stock_history_MONTH_3.csv" #"FAV_SCALA_stock_history_L_MONTH_3_sep.csv"#

#TO CONFIGURE
#Columns =['Date', Y_TARGET, 'ticker'] +  MUY_BUENOS_COLUMNAS_TRAINS
#SAV_files_surname = "veryGood_16"
#TO CONFIGURE
'''Para ENTRENAR los distintos tipos de configuracion TF GradientBoost XGBClassifier RandomForestClassifier '''
for k , v in Utils_col_sele.DICT_COLUMNS_TYPES.items():
    columns_selection = ['Date', Y_TARGET, 'ticker'] + v
    k_aux = k + '_2'
    print("GradientBoost XGBClassifier RandomForestClassifier \n DICT_COLUMNS_TYPES: " +k+" Columns Selected:" + ', '.join(columns_selection))
    X_train, X_test, y_train, y_test = Model_train_sklearn_XGB.get_x_y_train_test_sklearn_XGB(columns_selection, path= csv_file_SCALA)

    print("\nTF_onBalance")
    model_h5_name_k = "TF_" + k_aux + '.h5'
    Model_train_TF_onBalance.train_TF_onBalance(columns_selection, model_h5_name_k,
                                                path_csv=csv_file_SCALA)

    SAV_surname = k_aux
    print("\nGradientBoost")
    Model_train_sklearn_XGB.train_GradientBoost(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nXGBClassifier")
    Model_train_sklearn_XGB.train_XGBClassifier(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nRandomForestClassifier")
    Model_train_sklearn_XGB.train_RandomForestClassifier(X_train, X_test, y_train, y_test, SAV_surname)


