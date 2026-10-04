import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import _KEYS_DICT
from LogRoot.Logging import Logger
from Model_TF_definitions import ModelDefinition
from Utils.UtilsL import bcolors
from _KEYS_DICT import Y_TARGET
from Utils import Utils_model_predict


class Data_multidimension:
    BACHT_SIZE_LOOKBACK = _KEYS_DICT.BACHT_SIZE_LOOKBACK
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

    def __init__(self,columns_selection_a: [], op_buy_sell_a : _KEYS_DICT.Op_buy_sell, path_csv_a, name_models_stock):
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
        df = Utils_model_predict.load_and_clean_DF_Train_from_csv(self.path_csv, self.op_buy_sell, self.columns_selection) # shape is (5086, 13)
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

        # 1.0 ADD MULTIDIMENSION  Get 2D array , with BACHT_SIZE_LOOKBACK from "backward glances".
        # Values go from (10000 rows, 10 columns ) to (10000 rows, ( 10-1[groundTrue] * 10 dimensions ) columns ) but for the moment it does not go to 3d array remains 2d.
        # df.shape: (1000, 10) to (1000, 90)
        arr_mul_labels, arr_mul_features = Utils_model_predict.df_to_df_multidimension_array_2D(df.reset_index(drop=True), BACHT_SIZE_LOOKBACK = self.BACHT_SIZE_LOOKBACK)
        shape_imput_3d = (-1,self.BACHT_SIZE_LOOKBACK, len(df.columns)-1) # (-1, 10, 12)
        # 1.1 validate the structure of the data, this can be improved by
        arr_vali = arr_mul_features.reshape(shape_imput_3d) # 5077, 10, 12
        for i in range(1, arr_vali.shape[0], self.BACHT_SIZE_LOOKBACK * 3):
            list_fails_dates = [x for x in arr_vali[i][:, 0] if not (2018 <= datetime.fromtimestamp(x).year <= 2024)]
            if list_fails_dates:
                Logger.logr.error("The dates of the new 2D array do not appear in the first column. ")
                raise ValueError("The dates of the new 2D array do not appear in the first column. ")


        # 2.0 SCALER  scaling the data before, save a .scal file (it will be used to know how to scale the model for future predictions )
        # Do I have to scale now or can I wait until after I split
        # You can scale between the following values _KEYS_DICT.MIN_SCALER, _KEYS_DICT.MAX_SCALER
        # " that you learn for your scaling so that doing scaling before or after may give you the same results (but this depends on the actual scaling function)."  https://datascience.stackexchange.com/questions/71515/should-i-scale-data-before-or-after-balancing-dataset
        # TODO verify the correct order to "scaler split and SMOTE" order SMOTE.  sure: SMOTE only aplay on train_df
        arr_mul_features =  Utils_model_predict.scaler_min_max_array(arr_mul_features,path_to_save= _KEYS_DICT.PATH_SCALERS_FOLDER+self.name_models_stock+".scal")
        arr_mul_labels = Utils_model_predict.scaler_min_max_array(arr_mul_labels.reshape(-1,1))
        # 2.1 Let's put real groound True Y_TARGET  in a copy of scaled dataset
        df_with_target = pd.DataFrame(arr_mul_features)
        df_with_target[Y_TARGET] = arr_mul_labels.reshape(-1,)

        # 3.0 SPLIT Ok we should split in 3 train val and test
        # "you divide your data first and then apply synthetic sampling SMOTE on the training data only" https://datascience.stackexchange.com/questions/15630/train-test-split-after-performing-smote
        # CAUTION SMOTE generates twice as many rows
        train_df, test_df = train_test_split(df_with_target, test_size=0.18, shuffle=self.will_shuffle) # Shuffle in a time series? hmmm
        train_df, val_df = train_test_split(train_df, test_size=0.35, shuffle=self.will_shuffle)  # Shuffle in a time series? hmmm
        # Be carefull not to touch test_df, val_df
        # Apply smote only to train_df but first remove Y_TARGET from train_df
        # 3.1 Create a array 2d form dfs . Remove Y_target from train_df, because that's we want to predict and that would be cheating
        train_df_x = np.asarray(train_df.drop(columns=[Y_TARGET] ) )
        # In train_df_y We drop everything except Y_TARGET
        train_df_y = np.asarray(train_df[Y_TARGET] )

        # 4.0 SMOTE train_df to balance the data since there are few positive inputs, you have to generate "neighbors" of positive inputs. only in the df_train. according to the documentation of the imblearn pipeline:
        # Now we can smote only train_df . Doing the smote with 2D, with 3D is not possible.
        X_smt, y_smt = Utils_model_predict.prepare_to_split_SMOTETomek_01(train_df_x, train_df_y)
        # 4.1 Let's put real groound True Y_TARGET  in a copy of scaled dataset
        train_cleaned_df_target = pd.DataFrame(X_smt)
        train_cleaned_df_target[Y_TARGET] = y_smt.reshape(-1,)
        #the SMOTE leaves the positives very close together
        train_cleaned_df_target = shuffle(train_cleaned_df_target)

        # 5 PREPARE the data to be entered in TF with the correct dimensions
        # 5.1 pass Y_TARGET labels to 2D array required for TF
        train_labels = np.asarray(train_cleaned_df_target[Y_TARGET]).astype('float32').reshape((-1, 1)) # no need already 2d
        bool_train_labels = (train_labels != 0).reshape((-1))
        val_labels = np.asarray(val_df[Y_TARGET]).astype('float32').reshape((-1, 1)) # no need already 2d
        test_labels = np.asarray(test_df[Y_TARGET]).astype('float32').reshape((-1, 1)) # no need already 2d
        # 5.2 all array windows that were in 2D format (to overcome the SCALER and SMOTE methods),
        # must be displayed in 3D for TF by format of varible shape_imput_3d
        train_features = np.array(train_cleaned_df_target.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d)
        test_features = np.array(test_df.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d )
        val_features = np.array(val_df.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d )

        # 6 DISPLAY show the df format before accessing TF
        Utils_model_predict.log_shapes_trains_val_data(test_features, test_labels, train_features, train_labels, val_features, val_labels)
        # for N_row in [return_feature.shape[0] // 2, return_feature.shape[0] // 3, return_feature.shape[0] // 5,
        #               return_feature.shape[0] - 2]:
        # N_row = 88
        # arr_check = np.array(dataframe.loc[(N_row - self.BACHT_SIZE_LOOKBACK + 1):N_row, dataframe.columns.drop(Y_TARGET)])[::-1].reshape( 1, -1)
        # # return_feature[N_row-BACHT_SIZE_LOOKBACK+1]
        # if not (arr_check == return_feature[N_row - self.BACHT_SIZE_LOOKBACK + 1]).all():
        #     Logger.logr.error("data has not been reshaped 2D correctly ")
        #     raise ValueError("df_to_df_multidimension_array_2D() - data has not been reshaped 2D correctly ")
        # TODO validate the correct tranformation 2d to 3d by raise

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

    def get_dicts_models_multi_dimension(self, model_type : _KEYS_DICT.MODEL_TF_DENSE_TYPE_MULTI_DIMENSI):
        return self.dict_Model_Definition[model_type]
    # ModelDefinition(shape_inputs_m=imput_shape, num_features_m=len(columns_df)).get_dicts_models_multi_dimension()


