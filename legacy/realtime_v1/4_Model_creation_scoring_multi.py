import pandas as pd
from Utils import Utils_model_predict, Utils_scoring

#https://www.kaggle.com/code/andreanuzzo/balance-the-imbalanced-rf-and-xgboost-with-smote/notebook
import _KEYS_DICT
from LogRoot.Logging import Logger
from Utils.UtilsL import bcolors

Y_TARGET = 'buy_sell_point'

#TO CONFIGURE
Columns =['Date', Y_TARGET, 'ticker']
MODELS_EVAL_RESULT = "Models/all_results/"

MODEL_FOLDER_TF = "Models/TF_balance/"
 #necesario para Utils_buy_sell_points.check_buy_points_prediction
COLS_TO_EVAL_R = ["Close", "per_Close", 'has_preMarket', 'Volume']
columns_order_entre_to_model = []
df_result_all = None
df_result_all_isValid_to_buy = None





def get_best_moldels(df_moldes_S_NP, NUMBERS_BEST_MODELS = 2):
    std_highs = df_moldes_S_NP.loc["std"].nlargest(NUMBERS_BEST_MODELS)
    std_mean = df_moldes_S_NP.loc["89%":"96%"].mean().nlargest(NUMBERS_BEST_MODELS + 1)
    list_models = list(pd.concat([std_highs, std_mean]).index)
    list_models = [i.replace("_per", "") for i in list_models]
    valid_moldel = [item for item in set(list_models) if list_models.count(item) > 1]
    return valid_moldel


def get_the_2_best_models(df_moldes_S_NP, NUMBERS_BEST_MODELS):
    valid_moldel = get_best_moldels(df_moldes_S_NP)
    for i in range(4, 8):
        if 2 > len(valid_moldel):
            valid_moldel = get_best_moldels(df_moldes_S_NP, NUMBERS_BEST_MODELS=i)
        else:
            return valid_moldel[:2]
    print("Models Selections NONE stock: " + S + "_" + type_buy_sell.value + " Count: " + str(len(valid_moldel)) + "  Try with others parameters ")
    return valid_moldel

    # return list_valid_params_down , scoring_sum
#**DOCU**
#4 Evaluate the quality of the predictive models
# From the 36 models created for each OHLCV history of each action, only the best ones will be run in real time, in order to select and evaluate the best ones.
# Run Model_creation_scoring.py
#
# In the Models/Scoring folder
# AMD_yyy__groupby_buy_sell_point_000.json
# AMD_yyy__when_model_ok_threshold.csv
# Check that two have been generated for each action.

df_final_list_stocks = pd.DataFrame()
df_valids = pd.DataFrame()
df_ALL_score = pd.DataFrame()


CSV_NAME = "@CHILL"
list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
opion = _KEYS_DICT.Option_Historical.MONTH_3_AD

for S in list_stocks:
    for type_buy_sell in [ _KEYS_DICT.Op_buy_sell.NEG , _KEYS_DICT.Op_buy_sell.POS  ]:

        print(bcolors.OKBLUE + "\t\t " + S +"_"+type_buy_sell.value + bcolors.ENDC)

        df_moldes_S_NP, df_ALL_score = Utils_scoring.get_models_Multi_not_bads( S ,type_buy_sell, df_ALL_score)
        if df_moldes_S_NP.empty:
            print("EMPTY NONE Models Valid stock: " + S + "_" + type_buy_sell.value )
            continue
        valid_moldel = get_the_2_best_models(df_moldes_S_NP, NUMBERS_BEST_MODELS =2)
        print("Models Selections stock: "+ S + "_" + type_buy_sell.value +" Count: "+str(len(valid_moldel))+"  Names: " + "\t".join(valid_moldel))
        valid_moldel = valid_moldel + [i+"_per" for i in valid_moldel]
        df_valids[valid_moldel] = df_moldes_S_NP[valid_moldel]


df_valids.to_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", sep='\t')
print("\n\nModels/TF_multi/_RESULTS_profit_multi_all.csv" )
df_ALL_score.to_csv("Models/TF_multi/_SCORE_ALL_multi_all.csv", sep='\t')
df_ALL_score.T.to_csv("Models/TF_multi/_SCORE_ALL_T_multi_all.csv", sep='\t')
print("\n\nModels/TF_multi/_SCORE_ALL_multi_all.csv" )

