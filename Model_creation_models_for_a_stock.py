import Feature_selection_get_columns_json
import Model_train_TF_onBalance
import Model_train_sklearn_XGB
import Utils_col_sele
import a_manage_stocks_dict

Y_TARGET = 'buy_sell_point'
model_folder = "Models/Sklearn_smote/"


# csv_file_SCALA = "d_price/FAV_SCALA_stock_history_MONTH_3.csv" #"FAV_SCALA_stock_history_L_MONTH_3_sep.csv"#
#TO CONFIGURE
#Columns =['Date', Y_TARGET, 'ticker'] +  MUY_BUENOS_COLUMNAS_TRAINS
#SAV_files_surname = "veryGood_16"
#TO CONFIGURE
'''Para ENTRENAR los distintos tipos de configuracion TF GradientBoost XGBClassifier RandomForestClassifier '''
def train_model_with_custom_columns(name_model, columns_list, csv_file_SCALA):
    columns_selection = ['Date', Y_TARGET, 'ticker'] + columns_list
    print(
        "GradientBoost XGBClassifier RandomForestClassifier \n DICT_COLUMNS_TYPES: " + name_model + " Columns Selected:" + ', '.join(
            columns_selection))
    X_train, X_test, y_train, y_test = Model_train_sklearn_XGB.get_x_y_train_test_sklearn_XGB(columns_selection,
                                                                                              path=csv_file_SCALA)
    print("\nTF_onBalance")
    model_h5_name_k = "TF_" + name_model + '.h5'
    Model_train_TF_onBalance.train_TF_onBalance(columns_selection, model_h5_name_k,
                                                path_csv=csv_file_SCALA)
    SAV_surname = name_model
    print("\nGradientBoost")
    Model_train_sklearn_XGB.train_GradientBoost(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nXGBClassifier")
    Model_train_sklearn_XGB.train_XGBClassifier(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nRandomForestClassifier")
    Model_train_sklearn_XGB.train_RandomForestClassifier(X_train, X_test, y_train, y_test, SAV_surname)



CSV_NAME = "@VOLA"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
type_detect = 'pos'

for S in  list_stocks:
    path_csv_file_SCALA = "d_price/" + S + "_SCALA_stock_history_MONTH_3.csv"
    print(" START STOCK: ", S ,  " type: ", type_detect, " \t path: ", path_csv_file_SCALA)
    columns_json = Feature_selection_get_columns_json.JsonColumns(S, type_detect)

    train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.VGOOD.value, columns_json.vgood16, path_csv_file_SCALA)
    train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.GOOD.value, columns_json.get_vGood_and_Good(), path_csv_file_SCALA)
    train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.REG.value, columns_json.get_All(), path_csv_file_SCALA)


