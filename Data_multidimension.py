import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

import a_manage_stocks_dict
from LogRoot.Logging import Logger
from Model_TF_definitions import ModelDefinition
from Utils.UtilsL import bcolors
from a_manage_stocks_dict import Y_TARGET
from Utils import Utils_model_predict


class Data_multidimension:
    BACHT_SIZE_LOOKBACK = 10
    will_shuffle = True
    path_csv = None
    op_buy_sell = None
    columns_selection = None
    name_models_stock = None

    array_aux_np = None
    train_labels = None
    val_labels = None
    test_labels = None
    train_features = None
    val_features = None
    test_features = None
    bool_train_labels = None

    cols_df = None
    imput_shape = None

    dict_Model_Definition = None

    def __init__(self,columns_selection_a: [], op_buy_sell_a : a_manage_stocks_dict.Op_buy_sell, path_csv_a, name_models_stock):
        self.columns_selection = columns_selection_a
        self.op_buy_sell = op_buy_sell_a
        self.path_csv = path_csv_a
        self.name_models_stock = name_models_stock

        self.load_split_data_multidimension()

        self.dict_Model_Definition = ModelDefinition(shape_inputs_m=self.imput_shape, num_features_m=len(self.cols_df)).get_dicts_models_multi_dimension()
        Logger.logr.debug(bcolors.HEADER +'Created object TF_multidimension Path from: ' + self.path_csv+ bcolors.ENDC)

    def load_split_data_multidimension(self):

        if not "_PLAIN_" in  self.path_csv :
            Logger.logr.error(bcolors.HEADER + 'The input data must not have any scaling on the input to be correctly scaled. Path: ' + self.path_csv + bcolors.ENDC)
            raise ValueError('The input data must not have any scaling on the input to be correctly scaled. Path: ' + self.path_csv )
        df = Utils_model_predict.load_and_clean_DF_Train_from_csv(self.path_csv, self.op_buy_sell, self.columns_selection)
        self.cols_df = df.columns
        if 'ticker' in self.cols_df:
            Logger.logr.error(bcolors.HEADER + '\"ticker\" column detected, development required for multi-stock predictions. Path: ' + self.path_csv + bcolors.ENDC)
            raise ValueError('\"ticker\" column detected, development required for multi-stock predictions. Path: ' + self.path_csv )

        # SMOTE and Tomek links
        # The SMOTE oversampling approach could generate noisy samples since it creates synthetic data. To solve this problem, after SMOTE, we could use undersampling techniques to clean up. We’ll use the Tomek links undersampling technique in this example.
        # Utils_plotter.plot_2d_space(df.drop(columns=[Y_TARGET]).iloc[:,4:5] , df[Y_TARGET], path = "SMOTE_antes.png")
        array_aux_np = df[Y_TARGET]  # TODO antes o despues del balance??
        self.array_aux_np = array_aux_np
        # En caso de que las predicciones den numeros identicos
        # https://datascience.stackexchange.com/questions/21955/tensorflow-regression-model-giving-same-prediction-every-time

        #1. Obtener array 2D , con BACHT_SIZE_LOOKBACK de "miradas atras"
        arr_mul_labels, arr_mul_features = Utils_model_predict.df_to_df_multidimension_array_2D(df.reset_index(drop=True), BACHT_SIZE_LOOKBACK = self.BACHT_SIZE_LOOKBACK)
        shape_imput_3d = (-1,self.BACHT_SIZE_LOOKBACK, len(df.columns)-1)
        #1.1 validar la estructructura de los datos
        arr_vali = arr_mul_features.reshape(shape_imput_3d)
        for i in range(1, arr_vali.shape[0], self.BACHT_SIZE_LOOKBACK * 3):
            list_fails_dates = [x for x in arr_vali[i][:, 0] if not (2018 <= datetime.fromtimestamp(x).year <= 2024)]
            if list_fails_dates:
                Logger.logr.error("The dates of the new 2D array do not appear in the first column. ")
                raise ValueError("The dates of the new 2D array do not appear in the first column. ")

        #2. Hacer el smote con 2D , con 3D no se puede
        X_smt, y_smt = Utils_model_predict.prepare_to_split_SMOTETomek_01(arr_mul_features, arr_mul_labels)
        #3. Hacer el scaler para entrar a la TF
        x_feature =  Utils_model_predict.scaler_min_max_array(X_smt,path_to_save= a_manage_stocks_dict.PATH_SCALERS_FOLDER+self.name_models_stock+".scal")
        y_label = Utils_model_predict.scaler_min_max_array(y_smt.reshape(-1,1))

        #4. partir los datos en train ,validation y  test
        cleaned_df = pd.DataFrame(x_feature)
        cleaned_df[Y_TARGET] = y_label.reshape(-1,)
        # Use a utility from sklearn to split and shuffle your dataset.
        train_df, test_df = train_test_split(cleaned_df, test_size=0.16, shuffle=self.will_shuffle)
        train_df, val_df = train_test_split(train_df, test_size=0.32, shuffle=self.will_shuffle)

        #5.1 pasar los labeles Y_TARGET a array 2D requerido para TF
        train_labels = np.asarray(train_df[Y_TARGET]).astype('float32').reshape((-1, 1))
        bool_train_labels = (train_labels != 0).reshape((-1))
        val_labels = np.asarray(val_df[Y_TARGET]).astype('float32').reshape((-1, 1))
        test_labels = np.asarray(test_df[Y_TARGET]).astype('float32').reshape((-1, 1))
        #5.2 pasar los arrays 2D a 3D
        train_features = np.array(train_df.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d)
        test_features = np.array(test_df.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d )
        val_features = np.array(val_df.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d )

        Utils_model_predict.log_shapes_trains_val_data(test_features, test_labels, train_features, train_labels, val_features, val_labels)
        # for N_row in [return_feature.shape[0] // 2, return_feature.shape[0] // 3, return_feature.shape[0] // 5,
        #               return_feature.shape[0] - 2]:
        # N_row = 88
        # arr_check = np.array(dataframe.loc[(N_row - self.BACHT_SIZE_LOOKBACK + 1):N_row, dataframe.columns.drop(Y_TARGET)])[::-1].reshape( 1, -1)
        # # return_feature[N_row-BACHT_SIZE_LOOKBACK+1]
        # if not (arr_check == return_feature[N_row - self.BACHT_SIZE_LOOKBACK + 1]).all():
        #     Logger.logr.error("data has not been reshaped 2D correctly ")
        #     raise ValueError("df_to_df_multidimension_array_2D() - data has not been reshaped 2D correctly ")

        self.imput_shape = (train_features.shape[1], train_features.shape[2])

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features
        self.bool_train_labels = bool_train_labels

    def get_all_data(self):
        return self.array_aux_np, self.train_labels, self.val_labels, self.test_labels, self.train_features, self.val_features, self.test_features, self.bool_train_labels

    def get_dicts_models_multi_dimension(self, model_type : a_manage_stocks_dict.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI):
        return self.dict_Model_Definition[model_type]
    # ModelDefinition(shape_inputs_m=imput_shape, num_features_m=len(columns_df)).get_dicts_models_multi_dimension()


