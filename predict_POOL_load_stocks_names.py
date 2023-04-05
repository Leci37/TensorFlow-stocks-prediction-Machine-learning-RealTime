import Model_predictions_handle
from _KEYS_DICT import Op_buy_sell, DICT_COMPANYS
from LogRoot.Logging import Logger


import pandas as pd
NUM_MIN_MODLES  = 3
NUM_MIN_MODLES_TF = 1



CSV_NAME = "@CHIC"
list_stocks_chic = DICT_COMPANYS[CSV_NAME]
CSV_NAME = "@FOLO3"
list_stocks = DICT_COMPANYS[CSV_NAME]
list_stocks = list_stocks + list_stocks_chic

list_models_pos_neg = {}

def get_a_model_to_use(stock_id):
    list_models_pos_neg = {}
    list_score_POS = Model_predictions_handle.get_dict_scoring_evaluation(stock_id, Op_buy_sell.POS)['list_good_params_down']
    list_score_POS_TF = [col for col in list_score_POS if col.startswith('r_TF')]
    if len(list_score_POS) <= NUM_MIN_MODLES and len(list_score_POS_TF) < NUM_MIN_MODLES_TF:
        Logger.logr.info("Los modelos POS presentes son menores de:" + str(NUM_MIN_MODLES) + ", or las TF son menores de " + str(NUM_MIN_MODLES_TF) + " no se realiza prediccion STOCK: " + stock_id)
    else:
        list_models_pos_neg[stock_id +"_"+Op_buy_sell.POS.name] = list_score_POS

    list_score_NEG = Model_predictions_handle.get_dict_scoring_evaluation(stock_id, Op_buy_sell.NEG)['list_good_params_down']
    list_score_NEG_TF = [col for col in list_score_NEG if col.startswith('r_TF')]
    if len(list_score_NEG) <= NUM_MIN_MODLES and len(list_score_NEG_TF) < NUM_MIN_MODLES_TF :
        Logger.logr.info("Los modelos NEG presentes son menores de:" + str(NUM_MIN_MODLES) + ", or las TF son menores de " + str(NUM_MIN_MODLES_TF) + " no se realiza prediccion STOCK: " + stock_id)
    else:
        list_models_pos_neg[stock_id +"_"+Op_buy_sell.NEG.name] = list_score_NEG

    return list_models_pos_neg

def get_list_models_to_use():
    list_models_l = {}
    for S in list_stocks:
        Logger.logr.debug("Buscar lista de modelos validos STOCK: " + S)
        list_models_l = {**list_models_l, **get_a_model_to_use(S)}
    Logger.logr.info("Combinaciones validas para ser \"predict\" len: " + str(len(list_models_l)) + " Keys: " + ", ".join(list_models_l.keys()))

    return dict(sorted(list_models_l.items()))

#LIST GOOD PARAMS
list_models_pos_neg = get_list_models_to_use()
list_pos = [x.replace("_"+Op_buy_sell.POS.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.POS.name)]
list_neg = [x.replace("_"+Op_buy_sell.NEG.name, '') for x in list_models_pos_neg.keys() if x.endswith("_" + Op_buy_sell.NEG.name)]
list_stocks =  set(list_pos +list_neg)

print(list_models_pos_neg)





