import pandas as pd
import os.path

import Feature_selection_json_columns
from _KEYS_DICT import Op_buy_sell, Option_Historical, DICT_COMPANYS, MODEL_TF_DENSE_TYPE_MULTI_DIMENSI
from Utils.UtilsL import bcolors





MIN_DIF_REQUIRED = 0.18
MIN_SCORE_PER_DFTEST = 22
MIN_SCORE_STD = 0.18      #por debajo de esta standar dev , no se aceptara el modelos, los modelos son mejores cuando 25% 0.25 y 90% 0.9 , cuanta mas diferencia hay entre puntuaciones
def get_models_Multi_not_bads( S ,type_buy_sell: Op_buy_sell, df_ALL_score = None):
    df_moldes_S_NP = pd.DataFrame()
    columns_json = Feature_selection_json_columns.JsonColumns(S, type_buy_sell)
    for type_cols, list_cols in columns_json.get_Dict_JsonColumns().items():

        name_model = S + "_" + type_buy_sell.value + type_cols
        print("\n\nPath: Models/TF_multi/" + name_model + "_per_score.csv")
        if not os.path.isfile("Models/TF_multi/" + name_model + "_per_score.csv"):
            print("NOT EXIST Path: Models/TF_multi/" + name_model + "_per_score.csv")
            continue
        df_result = pd.read_csv("Models/TF_multi/" + name_model + "_per_score.csv", index_col=0, sep='\t')
        # cols_per =  [x for x in df_result.columns if x.endswith("_per")]
        for c in [x for x in df_result.columns if x.endswith("_per")]:
            col_r = c.replace("_per", '')
            if MIN_DIF_REQUIRED > (df_result[col_r].loc["95%"] - df_result[col_r].loc["50%"]):
                print("Modelo no calidad, poca DIFerencia en el score, revisar. Modelo: " + col_r + " Score50: " + str(
                    df_result[col_r].loc["50%"]) + " Score93: " + str(df_result[col_r].loc["93%"]))
            elif MIN_SCORE_PER_DFTEST > df_result[c].loc["93%"] and MIN_SCORE_PER_DFTEST > df_result[c].loc["94%"] and MIN_SCORE_PER_DFTEST > df_result[c].loc["95%"]:
                print("Modelo no calidad, poco PERcentage de acierto, revisar. Modelo: " + col_r + " Score93: " + str(df_result[c].loc["93%"]))
            elif MIN_SCORE_STD > df_result[col_r].loc["std"]:
                print("Modelo no calidad, poco STDandar de acierto, revisar. Modelo: " + col_r + " Score_std: " + str(
                    df_result[col_r].loc["std"]))
            else:
                print("Modelo VALIDO Modelo: " + col_r + " Score93: " + str(df_result[c].loc["93%"]))
                df_moldes_S_NP[col_r] = df_result[col_r]
                df_moldes_S_NP[c] = df_result[c]

            # TODO muchas columnas +-8000,se deberian insertar como filas .T
            if df_ALL_score is not None:
                df_ALL_score[col_r] = df_result[col_r]
                df_ALL_score[c] = df_result[c]

        print("Type: " + name_model)
    return df_moldes_S_NP, df_ALL_score