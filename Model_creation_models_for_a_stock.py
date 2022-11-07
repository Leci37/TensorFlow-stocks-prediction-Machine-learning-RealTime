import Feature_selection_get_columns_json
import Model_train_TF_onBalance
import Model_train_sklearn_XGB
import a_manage_stocks_dict

from Utils.UtilsL import bcolors


Y_TARGET = 'buy_sell_point'
model_folder = "Models/Sklearn_smote/"


# csv_file_SCALA = "d_price/FAV_SCALA_stock_history_MONTH_3.csv" #"FAV_SCALA_stock_history_L_MONTH_3_sep.csv"#
#TO CONFIGURE
#Columns =['Date', Y_TARGET, 'ticker'] +  MUY_BUENOS_COLUMNAS_TRAINS
#SAV_files_surname = "veryGood_16"
#TO CONFIGURE
#TODO ananalizar en más detalle https://github.com/huseinzol05/Stock-Prediction-Models

'''Para ENTRENAR los distintos tipos de configuracion TF GradientBoost XGBClassifier RandomForestClassifier '''
def train_model_with_custom_columns(name_model, columns_list, csv_file_SCALA, op_buy_sell : a_manage_stocks_dict.Op_buy_sell):
    columns_selection = ['Date', Y_TARGET, 'ticker'] + columns_list
    print(
        "GradientBoost XGBClassifier RandomForestClassifier \n DICT_COLUMNS_TYPES: " + name_model + " Columns Selected:" + ', '.join(
            columns_selection))
    X_train, X_test, y_train, y_test = Model_train_sklearn_XGB.get_x_y_train_test_sklearn_XGB(columns_selection,path=csv_file_SCALA, op_buy_sell=op_buy_sell)

    for type_mo in a_manage_stocks_dict.MODEL_TF_DENSE_TYPE_ONE_DIMENSI.list():
        model_h5_name_k = "TF_" + name_model + type_mo.value+'.h5'
        print(bcolors.OKBLUE + "\t\t "+model_h5_name_k + bcolors.ENDC)
        print("START")
        Model_train_TF_onBalance.train_TF_onBalance_One_dimension(columns_selection, model_h5_name_k,csv_file_SCALA, op_buy_sell=op_buy_sell, type_model_dime=type_mo)

    SAV_surname = name_model
    print("\nGradientBoost")
    Model_train_sklearn_XGB.train_GradientBoost(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nXGBClassifier")
    Model_train_sklearn_XGB.train_XGBClassifier(X_train, X_test, y_train, y_test, SAV_surname)
    print("\nRandomForestClassifier")
    Model_train_sklearn_XGB.train_RandomForestClassifier(X_train, X_test, y_train, y_test, SAV_surname)


CSV_NAME = "@CHIC"
list_stocks_chic = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@FOLO3"
list_stocks = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]

opion = a_manage_stocks_dict.Option_Historical.MONTH_3_AD
# a_manage_stocks_dict.Op_buy_sell.POS
# list_stocks = ['@ROLL']
for S in   list_stocks +list_stocks_chic :  #list_stocks:
    path_csv_file_SCALA = "d_price/" + S + "_SCALA_stock_history_" + str(opion.name) + ".csv"

    for type_buy_sell in [ a_manage_stocks_dict.Op_buy_sell.NEG , a_manage_stocks_dict.Op_buy_sell.POS  ]:
        print(" START STOCK: ", S,  " type: ", type_buy_sell, " \t path: ", path_csv_file_SCALA)
        columns_json = Feature_selection_get_columns_json.JsonColumns(S, type_buy_sell)


        for type_cols, list_cols in columns_json.get_Dict_JsonColumns().items():
            print("START")
            print(S + "_" + type_buy_sell.value + type_cols)
            print("START")
            train_model_with_custom_columns(S + "_" + type_buy_sell.value + type_cols,  list_cols, path_csv_file_SCALA, type_buy_sell )

    # train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.VGOOD.value, columns_json.vgood16, path_csv_file_SCALA)
    # train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.GOOD.value, columns_json.get_vGood_and_Good(), path_csv_file_SCALA)
    # train_model_with_custom_columns(S + "_" + type_detect + a_manage_stocks_dict.MODEL_TYPE_COLM.REG.value, columns_json.get_Regulars(), path_csv_file_SCALA)


