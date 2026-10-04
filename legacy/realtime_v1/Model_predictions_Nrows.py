import pandas as pd

import Model_predictions_handle_Nrows
from Utils import Utils_send_message

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
from _KEYS_DICT import Op_buy_sell, DICT_COMPANYS
import yhoo_history_stock
from LogRoot.Logging import Logger

Y_TARGET = 'buy_sell_point'

COL_GANAN = ["Date", "stock", "type_buy_sell","value_start", "message" ]
df_registre = pd.DataFrame(columns=COL_GANAN)

def resgister_predictions(S, df_b_s, type_b_s, path_regis):
    global df_registre
    modles_evaluated = [col for col in df_b_s.columns if col.startswith('br_')]
    modles_evaluated_TF = [col for col in df_b_s.columns if col.startswith('br_TF')]
    dict_predictions = df_b_s.T.to_dict()

    #register data each time UNLAST
    for i in range(0,df_b_s.shape[0] -1):
        Utils_send_message.register_in_zTelegram_Registers(S, dict_predictions[list(dict_predictions.keys())[i]], modles_evaluated, type_b_s, path = path_regis)  #unlast row




import logging
numba_logger = logging.getLogger('numba').setLevel(logging.WARNING)
mat_logger = logging.getLogger('matplotlib').setLevel(logging.WARNING)

#**DOCU**
# 5.0 make predictions for the last week Optional Test
# Run Model_predictions_Nrows.py
# This run generates the log file d_result/prediction_results_N_rows.csv
# It generates a sample file with predictions for the last week, data obtained with yfinance.
# Check that the logs exist

CSV_NAME = "@FOLO3"
list_stocks = DICT_COMPANYS[CSV_NAME]
PATH_EVAL_REGIS_BUY_SELL = "d_result/prediction_results_N_rows.csv"


df_buy = pd.DataFrame()
df_sell = pd.DataFrame()
for S in list_stocks:
    df_buy, df_sell = Model_predictions_handle_Nrows.get_RealTime_buy_seel_points(S, yhoo_history_stock.Option_Historical.MONTH_3_AD, NUM_LAST_REGISTERS_PER_STOCK=140)
    if df_buy is not None:
        resgister_predictions(S, df_buy, Op_buy_sell.POS, path_regis = PATH_EVAL_REGIS_BUY_SELL)
    if df_sell is not None:
        resgister_predictions(S, df_sell, Op_buy_sell.NEG, path_regis = PATH_EVAL_REGIS_BUY_SELL)

    print("Ended "+S)




