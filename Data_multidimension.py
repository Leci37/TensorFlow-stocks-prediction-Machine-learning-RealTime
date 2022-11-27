import a_manage_stocks_dict
from LogRoot.Logging import Logger
from a_manage_stocks_dict import Y_TARGET
from Utils import Utils_model_predict


class Data_multidimension:
    BACHT_SIZE_LOOKBACK = 8
    will_shuffle = True
    path_csv = None
    op_buy_sell = None
    columns_selection = None

    array_aux_np = None
    train_labels = None
    val_labels = None
    test_labels = None
    train_features = None
    val_features = None
    test_features = None
    bool_train_labels = None

    cols_df = None

    def __init__(self,columns_selection_a: [], op_buy_sell_a : a_manage_stocks_dict.Op_buy_sell, path_csv_a):
        self.columns_selection = columns_selection_a
        self.op_buy_sell = op_buy_sell_a
        self.path_csv = path_csv_a
        self.load_split_data_multidimension()

        Logger.logr.debug('Created object TF_multidimension Path from: ' + self.path_csv)

    def load_split_data_multidimension(self):

        df = Utils_model_predict.load_and_clean_DF_Train_from_csv(self.path_csv, self.op_buy_sell, self.columns_selection)

        # SMOTE and Tomek links
        # The SMOTE oversampling approach could generate noisy samples since it creates synthetic data. To solve this problem, after SMOTE, we could use undersampling techniques to clean up. We’ll use the Tomek links undersampling technique in this example.
        # Utils_plotter.plot_2d_space(df.drop(columns=[Y_TARGET]).iloc[:,4:5] , df[Y_TARGET], path = "SMOTE_antes.png")
        array_aux_np = df[Y_TARGET]  # TODO antes o despues del balance??
        self.array_aux_np = array_aux_np
        # En caso de que las predicciones den numeros identicos
        # https://datascience.stackexchange.com/questions/21955/tensorflow-regression-model-giving-same-prediction-every-time

        df = Utils_model_predict.prepare_to_split_SMOTETomek_and_scaler01(df)
        self.cols_df = df.columns

        train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(
            df, label_name=Y_TARGET, BACHT_SIZE_LOOKBACK=self.BACHT_SIZE_LOOKBACK, will_shuffle=self.will_shuffle)

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features
        self.bool_train_labels = bool_train_labels

    def get_all_data(self):
        return self.array_aux_np, self.train_labels, self.val_labels, self.test_labels, self.train_features, self.val_features, self.test_features, self.bool_train_labels, self.cols_df
