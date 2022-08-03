# import tensorflow_decision_forests as tfdf
# graficos preciosos
# https://www.kaggle.com/code/usharengaraju/tensorflow-decision-forests-w-b
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.metrics import accuracy_score, confusion_matrix

import Utils_buy_sell_points
import Utils_plotter


#task: Task to solve (e.g. CLASSIFICATION, REGRESSION, RANKING).
    # if self._task == Task.CLASSIFICATION:
    # elif self._task == Task.REGRESSION:
    #   label_guide.type = data_spec_pb2.NUMERICAL
    # elif self._task == Task.RANKING:
    #   label_guide.type = data_spec_pb2.NUMERICAL
taskss = ["tfdf.keras.Task.CLASSIFICATION", "tfdf.keras.Task.REGRESSION" ]
  # categorical_algorithm	How to learn splits on categorical attributes.
# CART: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.
# ONE_HOT: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.
# RANDOM: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001. Default: "CART".
categorical_algorithms = ["CART", "ONE_HOT", "RANDOM"]
  # split_axis	What structure of split to consider for numerical features.
# AXIS_ALIGNED: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.
# SPARSE_OBLIQUE: Sparse oblique splits (i.e. splits one a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. Default: "AXIS_ALIGNED".
split_axiss = ["AXIS_ALIGNED", "SPARSE_OBLIQUE"]
  # growing_strategy	How to grow the tree.
# LOCAL: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.
# BEST_FIRST_GLOBAL: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details. Default: "LOCAL".
growing_strategy = ["LOCAL","BEST_FIRST_GLOBAL" ]

num_trees = [300]#, 500]
max_depth = [10]#[8, 13, 18]


for t in taskss:
    for cat in categorical_algorithms:
        for spli in split_axiss:
            for grow in growing_strategy:
                for ntre in num_trees:
                    for maxd in max_depth:
                        options_str = "Task: "+ str(t) + " Categorial: "+str(cat) + " Splits: "+ str(spli)+ " Grows: "+ str(grow) + " Trees: "+str(ntre) + " MaxDeep: "+str(maxd)
                        print(options_str )



path_cm = "d_price/result_2.csv"
path_csv = "C:/Users/Luis/Downloads/REsult_SCALA_rfc_stock_history_E_.csv"
#path = "C:/Users/Luis/Downloads/result_train.csv"
df = pd.read_csv(path_csv, index_col=False, sep='\t')

path_cm = "Models/RandomForest/cm_rfc_stock_history.png"

Utils_buy_sell_points.check_buy_points_prediction(df, path_cm= path_cm)


df = df.loc[df["buy_sell_point"] == 1]
df.loc[df["result"] > 0.5]

aa = accuracy_score(df["result"].round(), df["buy_sell_point"], normalize = False)
#Generate the confusion matrix
df['result'] = np.round(np.where(df['result'] > 0.1, 1,0))
cf_matrix = confusion_matrix( df["buy_sell_point"] , df["result"].round())
Utils_plotter.plot_confusion_matrix_cm_OUT(cf_matrix, path_cm)

# Variable importances (VI) describe the impact of each feature to the model.
#
# VIs generally indicates how much a variable contributes to the model predictions or quality. Different VIs have different semantics and are generally not comparable.
#
# The VIs returned by variable_importances() depends on the learning algorithm and its hyper-parameters. For example, the hyperparameter compute_oob_variable_importances=True of the Random Forest learner enables the computation of permutation out-of-bag variable importances.
#
# Variable importances can be obtained with tfdf.inspector.make_inspector(path).variable_importances().
#
# The available variable importances are:
#
# Model agnostic
#
# MEAN_{INCREASE,DECREASE}IN{metric}: Estimated metric change from removing a feature using permutation importance . Depending on the learning algorithm and hyper-parameters, the VIs can be computed with validation, cross-validation or out-of-bag. For example, the MEAN_DECREASE_IN_ACCURACY of a feature is the drop in accuracy (the larger, the most important the feature) caused by shuffling the values of a features. For example, MEAN_DECREASE_IN_AUC_3_VS_OTHERS is the expected drop in AUC when comparing the label class "3" to the others.
#
# Decision Forests specific
#
# SUM_SCORE: Sum of the split scores using a specific feature. The larger, the most important.
#
# NUM_AS_ROOT: Number of root nodes using a specific feature. The larger, the most important.
#
# NUM_NODES: Number of nodes using a specific feature. The larger, the most important.
#
# MEAN_MIN_DEPTH: Average minimum depth of the first occurence of a feature across all the tree paths. The smaller, the most important.
# #
# Variable Importance: NUM_NODES:
#     1.                  "ticker" 651.000000 ################
#     2.                    "Date" 598.000000 ##############
#     3.          "cycl_SINE_lead" 507.000000 ############
#     4.           "cycl_DCPERIOD" 506.000000 ############
#     5.      "mtum_MACD_ext_list" 504.000000 ############
#     6.           "mtum_MINUS_DM" 476.000000 ###########
#     7.               "vola_NATR" 468.000000 ###########
#     8.                  "Volume" 460.000000 ###########
#     9.                "volu_OBV" 457.000000 ###########
#    10.                "mtum_ADX" 452.000000 ###########
#    11.               "mtum_ADXR" 450.000000 ###########
#    12.    "mtum_MACD_ext_signal" 450.000000 ###########
#    13.            "mtum_PLUS_DM" 444.000000 ##########
#    14.      "volu_Chaikin_ADOSC" 442.000000 ##########
#    15.                "vola_ATR" 438.000000 ##########
#    16.       "ti_CHOPPINESS(14)" 437.000000 ##########
#    17.                "mtum_MFI" 434.000000 ##########
#    18.             "vola_TRANGE" 433.000000 ##########
#    19.             "ti_ACC_DIST" 430.000000 ##########
#    20.            "cycl_DCPHASE" 429.000000 ##########
#    21.           "mtum_MINUS_DI" 418.000000 ##########
#    22.          "cycl_SINE_sine" 414.000000 ##########
#    23.     "ti_MASS_INDEX(9_25)" 411.000000 ##########
#    24.         "volu_Chaikin_AD" 401.000000 #########
#    25.        "ti_CHAIKIN(10_3)" 400.000000 #########
#    26.              "sti_CORREL" 399.000000 #########
#    27.                "mtum_PPO" 396.000000 #########
#    28.               "mtum_TRIX" 395.000000 #########
#    29.        "mtum_MACD_signal" 391.000000 #########
#    30.            "mtum_STOCH_d" 387.000000 #########
#    31.           "mtum_MACD_fix" 382.000000 #########
#    32.    "mtum_MACD_fix_signal" 376.000000 #########
#    33.               "per_Close" 376.000000 #########
#    34.           "mtum_MACD_ext" 373.000000 #########
#    35.                "mtum_BOP" 367.000000 #########
#    36. "ti_EASE_OF_MOVEMENT(14)" 366.000000 ########
#    37.                "sti_BETA" 364.000000 ########
#    38.             "mtum_ULTOSC" 361.000000 ########
#    39.            "mtum_PLUS_DI" 352.000000 ########
#    40.           "mtum_STOCHF_d" 350.000000 ########
#    41.      "mtum_MACD_fix_list" 349.000000 ########
#    42.          "mtum_MACD_list" 347.000000 ########
#    43.         "mtum_STOCHRSI_d" 347.000000 ########
#    44.              "ma_SMA_100" 345.000000 ########
#    45.                 "mtum_DX" 344.000000 ########
#    46.              "sti_STDDEV" 337.000000 ########
#    47.        "ti_VORTEX_POS(5)" 335.000000 ########
#    48.      "ti_FORCE_INDEX(13)" 334.000000 ########
#    49.                 "sti_VAR" 333.000000 ########
#    50.            "mtum_STOCH_k" 331.000000 ########
#    51.        "cycl_PHASOR_quad" 330.000000 ########
#    52.     "sti_LINEARREG_ANGLE" 327.000000 ########
#    53.               "mtum_MACD" 326.000000 ########
#    54.              "ma_EMA_100" 321.000000 #######
#    55.              "per_Volume" 321.000000 #######
#    56.             "olap_SAREXT" 313.000000 #######
#    57.           "mtum_STOCHF_k" 312.000000 #######
#    58.                "mtum_APO" 311.000000 #######
#    59.     "sti_LINEARREG_SLOPE" 310.000000 #######
#    60.            "ma_TRIMA_100" 309.000000 #######
#    61.              "ma_WMA_100" 309.000000 #######
#    62.        "cycl_PHASOR_inph" 306.000000 #######
#    63.       "ti_SUPERTREND(20)" 303.000000 #######
#    64.                "mtum_CCI" 301.000000 #######
#    65.    "ti_COPPOCK(14_11_10)" 295.000000 #######
#    66.           "mtum_AROONOSC" 292.000000 #######
#    67.        "ti_VORTEX_NEG(5)" 292.000000 #######
#    68.                "mtum_CMO" 289.000000 #######
#    69.             "ma_TRIMA_50" 286.000000 #######
#    70.                "mtum_MOM" 286.000000 #######
#    71.             "ma_KAMA_100" 281.000000 ######
#    72.               "mtum_ROCP" 274.000000 ######
#    73.              "ma_KAMA_50" 272.000000 ######
#    74.               "ma_EMA_50" 270.000000 ######
#    75.               "ma_SMA_50" 268.000000 ######
#    76.         "mtum_WILLIAMS_R" 261.000000 ######
#    77.                "mtum_ROC" 259.000000 ######
#    78.               "mtum_ROCR" 259.000000 ######
#    79.                "mtum_RSI" 253.000000 ######
#    80.         "mtum_AROON_down" 246.000000 ######
#    81.              "ma_DEMA_50" 245.000000 ######
#    82.                "olap_SAR" 243.000000 #####
#    83.               "ma_WMA_50" 240.000000 #####
#    84.            "mtum_ROCR100" 232.000000 #####
#    85.                "ma_T3_20" 230.000000 #####
#    86.  "ti_DONCHIAN_CENTER(20)" 227.000000 #####
#    87. "sti_LINEARREG_INTERCEPT" 226.000000 #####
#    88.                 "clas_s3" 225.000000 #####
#    89.       "olap_HT_TRENDLINE" 224.000000 #####
#    90.                "ma_T3_10" 217.000000 #####
#    91.           "mtum_AROON_up" 215.000000 #####
#    92.   "ti_DONCHIAN_LOWER(20)" 215.000000 #####
#    93.       "ti_KELT(20)_LOWER" 213.000000 #####
#    94.   "ti_DONCHIAN_UPPER(20)" 212.000000 #####
#    95.                 "trad_s3" 208.000000 #####
#    96.              "ma_TEMA_20" 203.000000 ####
#    97.              "ma_KAMA_20" 202.000000 ####
#    98.               "ma_EMA_20" 201.000000 ####
#    99.        "olap_BBAND_LOWER" 198.000000 ####
#   100.              "ma_DEMA_20" 195.000000 ####
#   101.                 "wood_s3" 195.000000 ####
#   102.                 "sti_TSF" 192.000000 ####
#   103.               "ma_SMA_10" 190.000000 ####
#   104.             "ma_TRIMA_20" 190.000000 ####
#   105.       "ti_KELT(20)_UPPER" 190.000000 ####
#   106.           "sti_LINEARREG" 189.000000 ####
#   107.                    "High" 187.000000 ####
#   108.               "ma_SMA_20" 184.000000 ####
#   109.        "olap_BBAND_UPPER" 183.000000 ####
#   110.               "ma_WMA_20" 182.000000 ####
#   111.                 "clas_r3" 179.000000 ####
#   112.             "ma_TRIMA_10" 178.000000 ####
#   113.                 "wood_r3" 177.000000 ####
#   114.                 "wood_s1" 177.000000 ####
#   115.           "olap_MIDPRICE" 176.000000 ####
#   116.               "ma_TEMA_5" 175.000000 ####
#   117.              "ti_HMA(20)" 174.000000 ####
#   118.               "ma_KAMA_5" 167.000000 ####
#   119.                   "Close" 166.000000 ####
#   120.                    "Open" 166.000000 ####
#   121.                 "trad_s2" 166.000000 ####
#   122.               "ma_EMA_10" 165.000000 ####
#   123.              "ma_TEMA_10" 164.000000 ####
#   124.              "ma_KAMA_10" 160.000000 ###
#   125.                 "ma_T3_5" 160.000000 ###
#   126.               "demark_s1" 159.000000 ###
#   127.                     "Low" 157.000000 ###
#   128.                 "wood_s2" 155.000000 ###
#   129.                 "trad_r3" 151.000000 ###
#   130.                 "trad_r2" 150.000000 ###
#   131.                 "clas_s1" 149.000000 ###
#   132.                 "fibo_s3" 146.000000 ###
#   133.         "mtum_STOCHRSI_k" 146.000000 ###
#   134.                 "trad_s1" 146.000000 ###
#   135.                 "clas_s2" 144.000000 ###
#   136.                 "clas_r2" 143.000000 ###
#   137.                "ma_WMA_5" 142.000000 ###
#   138.              "ma_TRIMA_5" 141.000000 ###
#   139.       "olap_BBAND_MIDDLE" 141.000000 ###
#   140.               "ma_WMA_10" 140.000000 ###
#   141.               "ma_DEMA_5" 139.000000 ###
#   142.              "ma_DEMA_10" 134.000000 ###
#   143.           "olap_MIDPOINT" 134.000000 ###
#   144.                 "wood_r2" 134.000000 ###
#   145.                "ma_EMA_5" 131.000000 ###
#   146.                 "wood_r1" 130.000000 ###
#   147.                 "cama_s3" 129.000000 ###
#   148.                "ma_SMA_5" 128.000000 ###
#   149.                 "fibo_r3" 125.000000 ###
#   150.                 "cama_r3" 124.000000 ###
#   151.                 "cama_s1" 124.000000 ###
#   152.                 "fibo_s1" 123.000000 ###
#   153.                 "trad_r1" 123.000000 ###
#   154.                 "cama_s2" 121.000000 ##
#   155.                 "fibo_s2" 117.000000 ##
#   156.                 "fibo_r2" 116.000000 ##
#   157.               "demark_pp" 114.000000 ##
#   158.                 "clas_r1" 113.000000 ##
#   159.                 "cama_r1" 110.000000 ##
#   160.           "per_preMarket" 109.000000 ##
#   161.                 "cama_r2" 104.000000 ##
#   162.                 "fibo_pp" 104.000000 ##
#   163.                 "cama_pp" 98.000000 ##
#   164.                 "clas_pp" 98.000000 ##
#   165.                 "wood_pp" 96.000000 ##
#   166.               "demark_r1" 91.000000 ##
#   167.                 "trad_pp" 91.000000 ##
#   168.                 "fibo_r1" 88.000000 ##
#   169.            "cdl_BELTHOLD" 83.000000 ##
#   170.             "cdl_HIKKAKE" 64.000000 #
#   171.       "cycl_HT_TRENDMODE" 39.000000
#   172.          "cdl_HIKKAKEMOD" 32.000000
#   173.         "cdl_SPINNINGTOP" 26.000000
#   174.            "cdl_LONGLINE" 25.000000
#   175.           "cdl_ENGULFING" 23.000000
#   176.           "has_preMarket" 20.000000
#   177.           "cdl_SHORTLINE" 19.000000
#   178.            "cdl_HIGHWAVE" 17.000000
#   179.                "cdl_DOJI" 15.000000
#   180.         "cdl_RICKSHAWMAN" 15.000000
#   181.      "cdl_LONGLEGGEDDOJI" 14.000000
#   182.              "cdl_HARAMI" 11.000000
#   183.     "cdl_CLOSINGMARUBOZU" 10.000000
#   184.            "cdl_3OUTSIDE"  7.000000
#   185.            "cdl_MARUBOZU"  6.000000
#   186.        "cdl_ADVANCEBLOCK"  5.000000
#   187.      "cdl_GRAVESTONEDOJI"  5.000000
#   188.     "cdl_SEPARATINGLINES"  5.000000
#   189.    "cdl_XSIDEGAP3METHODS"  5.000000
#   190.       "cdl_DRAGONFLYDOJI"  3.000000
#   191.         "cdl_MATCHINGLOW"  3.000000
#   192.      "cdl_STALLEDPATTERN"  3.000000
#   193.              "cdl_TAKURI"  3.000000
#   194.              "cdl_HAMMER"  2.000000
#   195.         "cdl_HARAMICROSS"  2.000000
#   196.             "cdl_3INSIDE"  1.000000
#   197.      "cdl_DARKCLOUDCOVER"  1.000000
#   198.    "cdl_GAPSIDESIDEWHITE"  1.000000
#   199.          "cdl_HANGINGMAN"  1.000000
#   200.        "cdl_HOMINGPIGEON"  1.000000
#   201.         "cdl_MORNINGSTAR"  1.000000
#   202.        "cdl_SHOOTINGSTAR"  1.000000
#
#
# {'MEAN_MIN_DEPTH': [("cdl_2CROWS" (1; #13), 10.056818713553245),
#   ("cdl_3BLACKCROWS" (1; #14), 10.056818713553245),
#   ("cdl_3LINESTRIKE" (1; #16), 10.056818713553245),
#   ("cdl_3STARSINSOUTH" (1; #18), 10.056818713553245),
#   ("cdl_3WHITESOLDIERS" (1; #19), 10.056818713553245),
#   ("cdl_ABANDONEDBABY" (1; #20), 10.056818713553245),
#   ("cdl_BREAKAWAY" (1; #23), 10.056818713553245),
#   ("cdl_CONCEALBABYSWALL" (1; #25), 10.056818713553245),
#   ("cdl_COUNTERATTACK" (1; #26), 10.056818713553245),
#   ("cdl_DOJISTAR" (1; #29), 10.056818713553245),
#   ("cdl_EVENINGDOJISTAR" (1; #32), 10.056818713553245),
#   ("cdl_EVENINGSTAR" (1; #33), 10.056818713553245),
#   ("cdl_IDENTICAL3CROWS" (1; #44), 10.056818713553245),
#   ("cdl_INNECK" (1; #45), 10.056818713553245),
#   ("cdl_INVERTEDHAMMER" (1; #46), 10.056818713553245),
#   ("cdl_KICKING" (1; #47), 10.056818713553245),
#   ("cdl_KICKINGBYLENGTH" (1; #48), 10.056818713553245),
#   ("cdl_LADDERBOTTOM" (1; #49), 10.056818713553245),
#   ("cdl_MATHOLD" (1; #54), 10.056818713553245),
#   ("cdl_MORNINGDOJISTAR" (1; #55), 10.056818713553245),
#   ("cdl_ONNECK" (1; #57), 10.056818713553245),
#   ("cdl_PIERCING" (1; #58), 10.056818713553245),
#   ("cdl_RISEFALL3METHODS" (1; #60), 10.056818713553245),
#   ("cdl_STICKSANDWICH" (1; #66), 10.056818713553245),
#   ("cdl_TASUKIGAP" (1; #68), 10.056818713553245),
#   ("cdl_THRUSTING" (1; #69), 10.056818713553245),
#   ("cdl_TRISTAR" (1; #70), 10.056818713553245),
#   ("cdl_UNIQUE3RIVER" (1; #71), 10.056818713553245),
#   ("cdl_UPSIDEGAP2CROWS" (1; #72), 10.056818713553245),
#   ("__LABEL" (4; #231), 10.056818713553245),
#   ("cdl_HAMMER" (1; #36), 10.0566270708175),
#   ("cdl_SHOOTINGSTAR" (1; #62), 10.056193713553245),
#   ("cdl_HANGINGMAN" (1; #37), 10.056043519754795),
#   ("cdl_3INSIDE" (1; #15), 10.055866332600864),
#   ("cdl_TAKURI" (1; #67), 10.055676266429895),
#   ("cdl_DRAGONFLYDOJI" (1; #30), 10.05530203381582),
#   ("cdl_GRAVESTONEDOJI" (1; #35), 10.055267786315735),
#   ("cdl_MORNINGSTAR" (1; #56), 10.05526620213772),
#   ("cdl_MATCHINGLOW" (1; #53), 10.055220523099242),
#   ("cdl_DARKCLOUDCOVER" (1; #27), 10.054805726540257),
#   ("cdl_HIGHWAVE" (1; #40), 10.054754678937881),
#   ("cdl_HARAMICROSS" (1; #39), 10.054156778301632),
#   ("cdl_HOMINGPIGEON" (1; #43), 10.05405933165479),
#   ("cdl_STALLEDPATTERN" (1; #65), 10.05343127529372),
#   ("cdl_3OUTSIDE" (1; #17), 10.053315183832076),
#   ("cdl_CLOSINGMARUBOZU" (1; #24), 10.052917982516089),
#   ("cdl_DOJI" (1; #28), 10.052199404669272),
#   ("cdl_GAPSIDESIDEWHITE" (1; #34), 10.052105337757066),
#   ("cdl_HARAMI" (1; #38), 10.051556254703392),
#   ("cdl_SPINNINGTOP" (1; #64), 10.051422257281898),
#   ("cdl_ADVANCEBLOCK" (1; #21), 10.051147064999672),
#   ("cdl_MARUBOZU" (1; #52), 10.049988175596123),
#   ("cdl_ENGULFING" (1; #31), 10.049450253579169),
#   ("cdl_RICKSHAWMAN" (1; #59), 10.047833315413747),
#   ("cdl_LONGLEGGEDDOJI" (1; #50), 10.046007573671089),
#   ("cdl_XSIDEGAP3METHODS" (1; #73), 10.044693824788364),
#   ("has_preMarket" (1; #98), 10.038825083411144),
#   ("cdl_SEPARATINGLINES" (1; #61), 10.035264733160572),
#   ("cycl_HT_TRENDMODE" (1; #83), 10.0351505201152),
#   ("cdl_SHORTLINE" (1; #63), 10.03029809966178),
#   ("cdl_LONGLINE" (1; #51), 10.027274736220619),
#   ("mtum_STOCHRSI_k" (1; #168), 10.013150299289647),
#   ("cdl_HIKKAKEMOD" (1; #42), 10.011967445465157),
#   ("cdl_HIKKAKE" (1; #41), 10.007799132369822),
#   ("demark_r1" (1; #89), 10.001142679130172),
#   ("fibo_r1" (1; #92), 9.999742129060749),
#   ("trad_r1" (1; #212), 9.985383021287824),
#   ("per_preMarket" (1; #184), 9.98244978357927),
#   ("trad_r2" (1; #213), 9.981182857256242),
#   ("clas_r2" (1; #76), 9.971037777535711),
#   ("trad_r3" (1; #214), 9.96830362794715),
#   ("fibo_r3" (1; #94), 9.964047958053156),
#   ("olap_BBAND_MIDDLE" (1; #175), 9.960272524875323),
#   ("olap_MIDPOINT" (1; #178), 9.958883177186571),
#   ("ti_HMA(20)" (1; #203), 9.950027756862083),
#   ("mtum_AROON_up" (1; #139), 9.94738417996917),
#   ("ma_SMA_20" (1; #115), 9.946706088986735),
#   ("ma_WMA_20" (1; #131), 9.946688376352698),
#   ("olap_BBAND_UPPER" (1; #176), 9.945471863746523),
#   ("ti_KELT(20)_UPPER" (1; #205), 9.940242765597805),
#   ("fibo_r2" (1; #93), 9.937593375991502),
#   ("ma_SMA_5" (1; #116), 9.935465459479904),
#   ("mtum_CCI" (1; #141), 9.933305049645176),
#   ("ma_KAMA_10" (1; #108), 9.92874160543867),
#   ("ma_KAMA_5" (1; #111), 9.924155390461916),
#   ("cama_r1" (1; #7), 9.920367831380585),
#   ("ma_TRIMA_5" (1; #127), 9.919362427566384),
#   ("ma_TRIMA_20" (1; #126), 9.915638518134562),
#   ("olap_MIDPRICE" (1; #179), 9.915381181847538),
#   ("ma_WMA_50" (1; #133), 9.914344052292188),
#   ("ma_EMA_20" (1; #105), 9.913929921433201),
#   ("ma_SMA_10" (1; #113), 9.908725875214845),
#   ("sti_LINEARREG_INTERCEPT" (1; #189), 9.906942981272385),
#   ("mtum_STOCHF_k" (1; #166), 9.906933735220205),
#   ("clas_r3" (1; #77), 9.906325429150879),
#   ("cycl_PHASOR_quad" (1; #85), 9.904221251626323),
#   ("ma_DEMA_20" (1; #100), 9.899390830567732),
#   ("ma_WMA_10" (1; #129), 9.895702361127292),
#   ("demark_pp" (1; #88), 9.891539090894963),
#   ("cama_r2" (1; #8), 9.890720336704844),
#   ("wood_r2" (1; #226), 9.889968288593225),
#   ("clas_r1" (1; #75), 9.889575714899678),
#   ("ma_KAMA_20" (1; #110), 9.889281235028779),
#   ("ma_T3_5" (1; #120), 9.888250915007884),
#   ("ma_T3_10" (1; #118), 9.887693243800014),
#   ("ma_EMA_10" (1; #103), 9.88720417453673),
#   ("per_Volume" (1; #183), 9.883922442409784),
#   ("ti_VORTEX_NEG(5)" (1; #208), 9.875946447017819),
#   ("mtum_MOM" (1; #156), 9.875072449847972),
#   ("mtum_WILLIAMS_R" (1; #173), 9.870583183034686),
#   ("mtum_DX" (1; #143), 9.86620185840939),
#   ("cdl_BELTHOLD" (1; #22), 9.856570894611837),
#   ("ti_KELT(20)_LOWER" (1; #204), 9.855235633026604),
#   ("olap_SAR" (1; #180), 9.852779742182804),
#   ("sti_BETA" (1; #185), 9.850464349464506),
#   ("cycl_PHASOR_inph" (1; #84), 9.849035834071266),
#   ("ma_WMA_5" (1; #132), 9.83722753409747),
#   ("ma_TRIMA_10" (1; #124), 9.833114902873406),
#   ("ti_DONCHIAN_CENTER(20)" (1; #198), 9.83089363356806),
#   ("Close" (1; #0), 9.8290216191753),
#   ("ti_DONCHIAN_LOWER(20)" (1; #199), 9.823351990249927),
#   ("ma_T3_20" (1; #119), 9.822721659012013),
#   ("cama_s1" (1; #10), 9.819938078269235),
#   ("fibo_pp" (1; #91), 9.818636296900122),
#   ("trad_pp" (1; #211), 9.81766695690896),
#   ("ma_EMA_5" (1; #106), 9.81617299245947),
#   ("mtum_ROCR100" (1; #163), 9.8153095448452),
#   ("ma_DEMA_50" (1; #102), 9.810293718352519),
#   ("mtum_ADX" (1; #134), 9.810018718298949),
#   ("ma_EMA_100" (1; #104), 9.809162669871297),
#   ("sti_LINEARREG_SLOPE" (1; #190), 9.8070876089782),
#   ("sti_CORREL" (1; #186), 9.80344849650179),
#   ("sti_LINEARREG" (1; #187), 9.80302216219833),
#   ("ti_MASS_INDEX(9_25)" (1; #206), 9.801245802364809),
#   ("wood_r3" (1; #227), 9.792798317208307),
#   ("mtum_STOCHRSI_d" (1; #167), 9.790800278092028),
#   ("Low" (1; #3), 9.79061428560308),
#   ("ma_TRIMA_50" (1; #128), 9.788226468550395),
#   ("ma_SMA_50" (1; #117), 9.780145590039904),
#   ("mtum_RSI" (1; #164), 9.779313404536754),
#   ("ma_DEMA_10" (1; #99), 9.777853427181562),
#   ("ti_DONCHIAN_UPPER(20)" (1; #200), 9.773340468461274),
#   ("mtum_ROCP" (1; #161), 9.77256680374739),
#   ("ma_TEMA_20" (1; #122), 9.769834039356917),
#   ("mtum_CMO" (1; #142), 9.766452069981293),
#   ("mtum_PLUS_DI" (1; #157), 9.764912217304378),
#   ("cama_pp" (1; #6), 9.762673893292352),
#   ("ma_KAMA_50" (1; #112), 9.762062054448599),
#   ("mtum_ROC" (1; #160), 9.753438659924972),
#   ("wood_r1" (1; #225), 9.753006708106566),
#   ("mtum_ADXR" (1; #135), 9.752992786583619),
#   ("wood_s2" (1; #229), 9.749410255388119),
#   ("ti_COPPOCK(14_11_10)" (1; #197), 9.744031189171093),
#   ("sti_LINEARREG_ANGLE" (1; #188), 9.741663414907162),
#   ("sti_VAR" (1; #193), 9.738797229401817),
#   ("fibo_s1" (1; #95), 9.734416488399665),
#   ("ma_TEMA_10" (1; #121), 9.733606993574409),
#   ("ma_KAMA_100" (1; #109), 9.731364319390483),
#   ("mtum_STOCHF_d" (1; #165), 9.722489885666322),
#   ("cama_r3" (1; #9), 9.720591934981146),
#   ("ma_TRIMA_100" (1; #125), 9.718441944856655),
#   ("sti_STDDEV" (1; #191), 9.71786271810214),
#   ("olap_HT_TRENDLINE" (1; #177), 9.717648923882866),
#   ("ti_FORCE_INDEX(13)" (1; #202), 9.715050076966097),
#   ("ti_CHOPPINESS(14)" (1; #196), 9.708153450902168),
#   ("mtum_STOCH_k" (1; #170), 9.70791369400681),
#   ("clas_pp" (1; #74), 9.696015977937343),
#   ("mtum_MACD_fix_list" (1; #149), 9.695062325470674),
#   ("olap_SAREXT" (1; #181), 9.694636729538038),
#   ("Open" (1; #4), 9.691844403466797),
#   ("wood_pp" (1; #224), 9.689293384053062),
#   ("fibo_s3" (1; #97), 9.684606506582652),
#   ("mtum_ROCR" (1; #162), 9.677371900541768),
#   ("mtum_AROONOSC" (1; #137), 9.670267195840712),
#   ("cycl_SINE_sine" (1; #87), 9.659146237484281),
#   ("fibo_s2" (1; #96), 9.65774007035956),
#   ("mtum_MACD_list" (1; #151), 9.648475299942971),
#   ("olap_BBAND_LOWER" (1; #174), 9.64355462524335),
#   ("ti_EASE_OF_MOVEMENT(14)" (1; #201), 9.641471730731523),
#   ("mtum_AROON_down" (1; #138), 9.63774769202298),
#   ("clas_s1" (1; #78), 9.636962584711128),
#   ("ma_WMA_100" (1; #130), 9.636545860142315),
#   ("cama_s3" (1; #12), 9.636167206417138),
#   ("ma_TEMA_5" (1; #123), 9.634747409979285),
#   ("demark_s1" (1; #90), 9.632412626796103),
#   ("ti_VORTEX_POS(5)" (1; #209), 9.63216795395226),
#   ("ma_EMA_50" (1; #107), 9.631182013010008),
#   ("per_Close" (1; #182), 9.629489318584135),
#   ("ma_SMA_100" (1; #114), 9.627211546115724),
#   ("ma_DEMA_5" (1; #101), 9.610642831062156),
#   ("clas_s2" (1; #79), 9.6011859971081),
#   ("cycl_DCPERIOD" (1; #81), 9.592979710597122),
#   ("sti_TSF" (1; #192), 9.578222434881972),
#   ("wood_s1" (1; #228), 9.577508730766015),
#   ("High" (1; #2), 9.577002796199588),
#   ("mtum_ULTOSC" (1; #172), 9.569055964440372),
#   ("cama_s2" (1; #11), 9.553946165316992),
#   ("mtum_MACD_ext" (1; #145), 9.551583104986959),
#   ("mtum_PLUS_DM" (1; #158), 9.537255027378444),
#   ("trad_s2" (1; #216), 9.531390587775721),
#   ("mtum_MACD_ext_list" (1; #146), 9.520580652417202),
#   ("mtum_MACD_signal" (1; #152), 9.514553190051636),
#   ("mtum_MINUS_DI" (1; #154), 9.512230404225967),
#   ("volu_OBV" (1; #223), 9.509765289723312),
#   ("clas_s3" (1; #80), 9.505453029419924),
#   ("mtum_MACD_fix" (1; #148), 9.485147825857569),
#   ("mtum_APO" (1; #136), 9.483788150668126),
#   ("trad_s1" (1; #215), 9.478287293788732),
#   ("vola_ATR" (1; #218), 9.472200269213527),
#   ("ti_SUPERTREND(20)" (1; #207), 9.467826188550212),
#   ("mtum_MFI" (1; #153), 9.445977411698488),
#   ("mtum_MACD" (1; #144), 9.444376717471497),
#   ("mtum_TRIX" (1; #171), 9.4408696598303),
#   ("mtum_BOP" (1; #140), 9.429098569474952),
#   ("Volume" (1; #5), 9.418447184894186),
#   ("trad_s3" (1; #217), 9.397451855318986),
#   ("ti_CHAIKIN(10_3)" (1; #195), 9.38909496373947),
#   ("volu_Chaikin_AD" (1; #221), 9.360495224790542),
#   ("vola_TRANGE" (1; #220), 9.35588900438718),
#   ("mtum_MACD_fix_signal" (1; #150), 9.354223396565725),
#   ("mtum_PPO" (1; #159), 9.33889318751425),
#   ("mtum_MINUS_DM" (1; #155), 9.334628587668496),
#   ("wood_s3" (1; #230), 9.302183367640692),
#   ("vola_NATR" (1; #219), 9.29269355120498),
#   ("mtum_MACD_ext_signal" (1; #147), 9.257106918259034),
#   ("mtum_STOCH_d" (1; #169), 9.187951755514456),
#   ("ticker" (4; #210), 9.18559578052634),
#   ("Date" (1; #1), 9.152407083883382),
#   ("cycl_DCPHASE" (1; #82), 9.073452316851474),
#   ("ti_ACC_DIST" (1; #194), 9.023966047568058),
#   ("volu_Chaikin_ADOSC" (1; #222), 8.98488097740124),
#   ("cycl_SINE_lead" (1; #86), 8.96219100972475)],
#  'NUM_AS_ROOT': [("wood_s3" (1; #230), 16.0),
#   ("trad_s3" (1; #217), 14.0),
#   ("cama_s2" (1; #11), 10.0),
#   ("trad_s1" (1; #215), 10.0),
#   ("clas_s1" (1; #78), 9.0),
#   ("clas_s3" (1; #80), 9.0),
#   ("trad_s2" (1; #216), 9.0),
#   ("clas_s2" (1; #79), 8.0),
#   ("ma_DEMA_5" (1; #101), 8.0),
#   ("volu_Chaikin_ADOSC" (1; #222), 8.0),
#   ("wood_s1" (1; #228), 8.0),
#   ("cama_r3" (1; #9), 7.0),
#   ("demark_s1" (1; #90), 7.0),
#   ("fibo_s1" (1; #95), 7.0),
#   ("olap_BBAND_LOWER" (1; #174), 7.0),
#   ("sti_TSF" (1; #192), 7.0),
#   ("ti_ACC_DIST" (1; #194), 7.0),
#   ("wood_pp" (1; #224), 7.0),
#   ("cama_s3" (1; #12), 6.0),
#   ("clas_pp" (1; #74), 6.0),
#   ("fibo_s2" (1; #96), 6.0),
#   ("ma_TEMA_5" (1; #123), 6.0),
#   ("High" (1; #2), 5.0),
#   ("cycl_DCPHASE" (1; #82), 5.0),
#   ("fibo_s3" (1; #97), 5.0),
#   ("ma_DEMA_10" (1; #99), 5.0),
#   ("wood_s2" (1; #229), 5.0),
#   ("Open" (1; #4), 4.0),
#   ("cama_pp" (1; #6), 4.0),
#   ("cama_s1" (1; #10), 4.0),
#   ("ma_TEMA_10" (1; #121), 4.0),
#   ("mtum_AROON_down" (1; #138), 4.0),
#   ("ti_SUPERTREND(20)" (1; #207), 4.0),
#   ("trad_pp" (1; #211), 4.0),
#   ("wood_r1" (1; #225), 4.0),
#   ("Low" (1; #3), 3.0),
#   ("fibo_pp" (1; #91), 3.0),
#   ("ma_TEMA_20" (1; #122), 3.0),
#   ("mtum_MACD_fix_signal" (1; #150), 3.0),
#   ("mtum_STOCH_d" (1; #169), 3.0),
#   ("sti_LINEARREG" (1; #187), 3.0),
#   ("Close" (1; #0), 2.0),
#   ("cama_r2" (1; #8), 2.0),
#   ("clas_r1" (1; #75), 2.0),
#   ("ma_EMA_50" (1; #107), 2.0),
#   ("ma_TRIMA_10" (1; #124), 2.0),
#   ("ma_WMA_10" (1; #129), 2.0),
#   ("ma_WMA_5" (1; #132), 2.0),
#   ("mtum_MINUS_DI" (1; #154), 2.0),
#   ("mtum_RSI" (1; #164), 2.0),
#   ("volu_Chaikin_AD" (1; #221), 2.0),
#   ("wood_r2" (1; #226), 2.0),
#   ("wood_r3" (1; #227), 2.0),
#   ("cama_r1" (1; #7), 1.0),
#   ("cycl_SINE_lead" (1; #86), 1.0),
#   ("demark_pp" (1; #88), 1.0),
#   ("fibo_r2" (1; #93), 1.0),
#   ("ma_DEMA_20" (1; #100), 1.0),
#   ("ma_EMA_5" (1; #106), 1.0),
#   ("ma_SMA_5" (1; #116), 1.0),
#   ("mtum_APO" (1; #136), 1.0),
#   ("mtum_AROONOSC" (1; #137), 1.0),
#   ("mtum_MACD" (1; #144), 1.0),
#   ("mtum_MACD_ext" (1; #145), 1.0),
#   ("mtum_MACD_ext_signal" (1; #147), 1.0),
#   ("mtum_MACD_fix" (1; #148), 1.0),
#   ("mtum_MFI" (1; #153), 1.0),
#   ("mtum_MINUS_DM" (1; #155), 1.0),
#   ("mtum_TRIX" (1; #171), 1.0),
#   ("mtum_ULTOSC" (1; #172), 1.0),
#   ("ti_CHAIKIN(10_3)" (1; #195), 1.0),
#   ("ti_VORTEX_POS(5)" (1; #209), 1.0)],
#  'NUM_NODES': [("ticker" (4; #210), 651.0),
#   ("Date" (1; #1), 598.0),
#   ("cycl_SINE_lead" (1; #86), 507.0),
#   ("cycl_DCPERIOD" (1; #81), 506.0),
#   ("mtum_MACD_ext_list" (1; #146), 504.0),
#   ("mtum_MINUS_DM" (1; #155), 476.0),
#   ("vola_NATR" (1; #219), 468.0),
#   ("Volume" (1; #5), 460.0),
#   ("volu_OBV" (1; #223), 457.0),
#   ("mtum_ADX" (1; #134), 452.0),
#   ("mtum_ADXR" (1; #135), 450.0),
#   ("mtum_MACD_ext_signal" (1; #147), 450.0),
#   ("mtum_PLUS_DM" (1; #158), 444.0),
#   ("volu_Chaikin_ADOSC" (1; #222), 442.0),
#   ("vola_ATR" (1; #218), 438.0),
#   ("ti_CHOPPINESS(14)" (1; #196), 437.0),
#   ("mtum_MFI" (1; #153), 434.0),
#   ("vola_TRANGE" (1; #220), 433.0),
#   ("ti_ACC_DIST" (1; #194), 430.0),
#   ("cycl_DCPHASE" (1; #82), 429.0),
#   ("mtum_MINUS_DI" (1; #154), 418.0),
#   ("cycl_SINE_sine" (1; #87), 414.0),
#   ("ti_MASS_INDEX(9_25)" (1; #206), 411.0),
#   ("volu_Chaikin_AD" (1; #221), 401.0),
#   ("ti_CHAIKIN(10_3)" (1; #195), 400.0),
#   ("sti_CORREL" (1; #186), 399.0),
#   ("mtum_PPO" (1; #159), 396.0),
#   ("mtum_TRIX" (1; #171), 395.0),
#   ("mtum_MACD_signal" (1; #152), 391.0),
#   ("mtum_STOCH_d" (1; #169), 387.0),
#   ("mtum_MACD_fix" (1; #148), 382.0),
#   ("mtum_MACD_fix_signal" (1; #150), 376.0),
#   ("per_Close" (1; #182), 376.0),
#   ("mtum_MACD_ext" (1; #145), 373.0),
#   ("mtum_BOP" (1; #140), 367.0),
#   ("ti_EASE_OF_MOVEMENT(14)" (1; #201), 366.0),
#   ("sti_BETA" (1; #185), 364.0),
#   ("mtum_ULTOSC" (1; #172), 361.0),
#   ("mtum_PLUS_DI" (1; #157), 352.0),
#   ("mtum_STOCHF_d" (1; #165), 350.0),
#   ("mtum_MACD_fix_list" (1; #149), 349.0),
#   ("mtum_MACD_list" (1; #151), 347.0),
#   ("mtum_STOCHRSI_d" (1; #167), 347.0),
#   ("ma_SMA_100" (1; #114), 345.0),
#   ("mtum_DX" (1; #143), 344.0),
#   ("sti_STDDEV" (1; #191), 337.0),
#   ("ti_VORTEX_POS(5)" (1; #209), 335.0),
#   ("ti_FORCE_INDEX(13)" (1; #202), 334.0),
#   ("sti_VAR" (1; #193), 333.0),
#   ("mtum_STOCH_k" (1; #170), 331.0),
#   ("cycl_PHASOR_quad" (1; #85), 330.0),
#   ("sti_LINEARREG_ANGLE" (1; #188), 327.0),
#   ("mtum_MACD" (1; #144), 326.0),
#   ("ma_EMA_100" (1; #104), 321.0),
#   ("per_Volume" (1; #183), 321.0),
#   ("olap_SAREXT" (1; #181), 313.0),
#   ("mtum_STOCHF_k" (1; #166), 312.0),
#   ("mtum_APO" (1; #136), 311.0),
#   ("sti_LINEARREG_SLOPE" (1; #190), 310.0),
#   ("ma_TRIMA_100" (1; #125), 309.0),
#   ("ma_WMA_100" (1; #130), 309.0),
#   ("cycl_PHASOR_inph" (1; #84), 306.0),
#   ("ti_SUPERTREND(20)" (1; #207), 303.0),
#   ("mtum_CCI" (1; #141), 301.0),
#   ("ti_COPPOCK(14_11_10)" (1; #197), 295.0),
#   ("mtum_AROONOSC" (1; #137), 292.0),
#   ("ti_VORTEX_NEG(5)" (1; #208), 292.0),
#   ("mtum_CMO" (1; #142), 289.0),
#   ("ma_TRIMA_50" (1; #128), 286.0),
#   ("mtum_MOM" (1; #156), 286.0),
#   ("ma_KAMA_100" (1; #109), 281.0),
#   ("mtum_ROCP" (1; #161), 274.0),
#   ("ma_KAMA_50" (1; #112), 272.0),
#   ("ma_EMA_50" (1; #107), 270.0),
#   ("ma_SMA_50" (1; #117), 268.0),
#   ("mtum_WILLIAMS_R" (1; #173), 261.0),
#   ("mtum_ROC" (1; #160), 259.0),
#   ("mtum_ROCR" (1; #162), 259.0),
#   ("mtum_RSI" (1; #164), 253.0),
#   ("mtum_AROON_down" (1; #138), 246.0),
#   ("ma_DEMA_50" (1; #102), 245.0),
#   ("olap_SAR" (1; #180), 243.0),
#   ("ma_WMA_50" (1; #133), 240.0),
#   ("mtum_ROCR100" (1; #163), 232.0),
#   ("ma_T3_20" (1; #119), 230.0),
#   ("ti_DONCHIAN_CENTER(20)" (1; #198), 227.0),
#   ("sti_LINEARREG_INTERCEPT" (1; #189), 226.0),
#   ("clas_s3" (1; #80), 225.0),
#   ("olap_HT_TRENDLINE" (1; #177), 224.0),
#   ("ma_T3_10" (1; #118), 217.0),
#   ("mtum_AROON_up" (1; #139), 215.0),
#   ("ti_DONCHIAN_LOWER(20)" (1; #199), 215.0),
#   ("ti_KELT(20)_LOWER" (1; #204), 213.0),
#   ("ti_DONCHIAN_UPPER(20)" (1; #200), 212.0),
#   ("trad_s3" (1; #217), 208.0),
#   ("ma_TEMA_20" (1; #122), 203.0),
#   ("ma_KAMA_20" (1; #110), 202.0),
#   ("ma_EMA_20" (1; #105), 201.0),
#   ("olap_BBAND_LOWER" (1; #174), 198.0),
#   ("ma_DEMA_20" (1; #100), 195.0),
#   ("wood_s3" (1; #230), 195.0),
#   ("sti_TSF" (1; #192), 192.0),
#   ("ma_SMA_10" (1; #113), 190.0),
#   ("ma_TRIMA_20" (1; #126), 190.0),
#   ("ti_KELT(20)_UPPER" (1; #205), 190.0),
#   ("sti_LINEARREG" (1; #187), 189.0),
#   ("High" (1; #2), 187.0),
#   ("ma_SMA_20" (1; #115), 184.0),
#   ("olap_BBAND_UPPER" (1; #176), 183.0),
#   ("ma_WMA_20" (1; #131), 182.0),
#   ("clas_r3" (1; #77), 179.0),
#   ("ma_TRIMA_10" (1; #124), 178.0),
#   ("wood_r3" (1; #227), 177.0),
#   ("wood_s1" (1; #228), 177.0),
#   ("olap_MIDPRICE" (1; #179), 176.0),
#   ("ma_TEMA_5" (1; #123), 175.0),
#   ("ti_HMA(20)" (1; #203), 174.0),
#   ("ma_KAMA_5" (1; #111), 167.0),
#   ("Close" (1; #0), 166.0),
#   ("Open" (1; #4), 166.0),
#   ("trad_s2" (1; #216), 166.0),
#   ("ma_EMA_10" (1; #103), 165.0),
#   ("ma_TEMA_10" (1; #121), 164.0),
#   ("ma_KAMA_10" (1; #108), 160.0),
#   ("ma_T3_5" (1; #120), 160.0),
#   ("demark_s1" (1; #90), 159.0),
#   ("Low" (1; #3), 157.0),
#   ("wood_s2" (1; #229), 155.0),
#   ("trad_r3" (1; #214), 151.0),
#   ("trad_r2" (1; #213), 150.0),
#   ("clas_s1" (1; #78), 149.0),
#   ("fibo_s3" (1; #97), 146.0),
#   ("mtum_STOCHRSI_k" (1; #168), 146.0),
#   ("trad_s1" (1; #215), 146.0),
#   ("clas_s2" (1; #79), 144.0),
#   ("clas_r2" (1; #76), 143.0),
#   ("ma_WMA_5" (1; #132), 142.0),
#   ("ma_TRIMA_5" (1; #127), 141.0),
#   ("olap_BBAND_MIDDLE" (1; #175), 141.0),
#   ("ma_WMA_10" (1; #129), 140.0),
#   ("ma_DEMA_5" (1; #101), 139.0),
#   ("ma_DEMA_10" (1; #99), 134.0),
#   ("olap_MIDPOINT" (1; #178), 134.0),
#   ("wood_r2" (1; #226), 134.0),
#   ("ma_EMA_5" (1; #106), 131.0),
#   ("wood_r1" (1; #225), 130.0),
#   ("cama_s3" (1; #12), 129.0),
#   ("ma_SMA_5" (1; #116), 128.0),
#   ("fibo_r3" (1; #94), 125.0),
#   ("cama_r3" (1; #9), 124.0),
#   ("cama_s1" (1; #10), 124.0),
#   ("fibo_s1" (1; #95), 123.0),
#   ("trad_r1" (1; #212), 123.0),
#   ("cama_s2" (1; #11), 121.0),
#   ("fibo_s2" (1; #96), 117.0),
#   ("fibo_r2" (1; #93), 116.0),
#   ("demark_pp" (1; #88), 114.0),
#   ("clas_r1" (1; #75), 113.0),
#   ("cama_r1" (1; #7), 110.0),
#   ("per_preMarket" (1; #184), 109.0),
#   ("cama_r2" (1; #8), 104.0),
#   ("fibo_pp" (1; #91), 104.0),
#   ("cama_pp" (1; #6), 98.0),
#   ("clas_pp" (1; #74), 98.0),
#   ("wood_pp" (1; #224), 96.0),
#   ("demark_r1" (1; #89), 91.0),
#   ("trad_pp" (1; #211), 91.0),
#   ("fibo_r1" (1; #92), 88.0),
#   ("cdl_BELTHOLD" (1; #22), 83.0),
#   ("cdl_HIKKAKE" (1; #41), 64.0),
#   ("cycl_HT_TRENDMODE" (1; #83), 39.0),
#   ("cdl_HIKKAKEMOD" (1; #42), 32.0),
#   ("cdl_SPINNINGTOP" (1; #64), 26.0),
#   ("cdl_LONGLINE" (1; #51), 25.0),
#   ("cdl_ENGULFING" (1; #31), 23.0),
#   ("has_preMarket" (1; #98), 20.0),
#   ("cdl_SHORTLINE" (1; #63), 19.0),
#   ("cdl_HIGHWAVE" (1; #40), 17.0),
#   ("cdl_DOJI" (1; #28), 15.0),
#   ("cdl_RICKSHAWMAN" (1; #59), 15.0),
#   ("cdl_LONGLEGGEDDOJI" (1; #50), 14.0),
#   ("cdl_HARAMI" (1; #38), 11.0),
#   ("cdl_CLOSINGMARUBOZU" (1; #24), 10.0),
#   ("cdl_3OUTSIDE" (1; #17), 7.0),
#   ("cdl_MARUBOZU" (1; #52), 6.0),
#   ("cdl_ADVANCEBLOCK" (1; #21), 5.0),
#   ("cdl_GRAVESTONEDOJI" (1; #35), 5.0),
#   ("cdl_SEPARATINGLINES" (1; #61), 5.0),
#   ("cdl_XSIDEGAP3METHODS" (1; #73), 5.0),
#   ("cdl_DRAGONFLYDOJI" (1; #30), 3.0),
#   ("cdl_MATCHINGLOW" (1; #53), 3.0),
#   ("cdl_STALLEDPATTERN" (1; #65), 3.0),
#   ("cdl_TAKURI" (1; #67), 3.0),
#   ("cdl_HAMMER" (1; #36), 2.0),
#   ("cdl_HARAMICROSS" (1; #39), 2.0),
#   ("cdl_3INSIDE" (1; #15), 1.0),
#   ("cdl_DARKCLOUDCOVER" (1; #27), 1.0),
#   ("cdl_GAPSIDESIDEWHITE" (1; #34), 1.0),
#   ("cdl_HANGINGMAN" (1; #37), 1.0),
#   ("cdl_HOMINGPIGEON" (1; #43), 1.0),
#   ("cdl_MORNINGSTAR" (1; #56), 1.0),
#   ("cdl_SHOOTINGSTAR" (1; #62), 1.0)],
#  'SUM_SCORE': [("Date" (1; #1), 5675.7604386177845),
#   ("ticker" (4; #210), 5658.446488027577),
#   ("cycl_SINE_lead" (1; #86), 4500.95762444986),
#   ("mtum_MACD_ext_list" (1; #146), 4007.798186265165),
#   ("vola_NATR" (1; #219), 3949.8760705799796),
#   ("mtum_MINUS_DM" (1; #155), 3901.2249186115805),
#   ("cycl_DCPHASE" (1; #82), 3823.048266883241),
#   ("cycl_DCPERIOD" (1; #81), 3790.7871085291263),
#   ("ti_ACC_DIST" (1; #194), 3665.3691992759705),
#   ("volu_Chaikin_ADOSC" (1; #222), 3637.230332563049),
#   ("volu_OBV" (1; #223), 3571.403255097568),
#   ("mtum_MFI" (1; #153), 3497.364759203512),
#   ("Volume" (1; #5), 3481.2183875543997),
#   ("mtum_MACD_ext_signal" (1; #147), 3417.862042220542),
#   ("vola_TRANGE" (1; #220), 3402.375161490403),
#   ("volu_Chaikin_AD" (1; #221), 3355.884704746073),
#   ("vola_ATR" (1; #218), 3309.467480246909),
#   ("mtum_PLUS_DM" (1; #158), 3270.415049384581),
#   ("mtum_ADXR" (1; #135), 3269.407475147629),
#   ("mtum_MINUS_DI" (1; #154), 3205.4519880560692),
#   ("mtum_PPO" (1; #159), 3164.822621941101),
#   ("ti_CHOPPINESS(14)" (1; #196), 3121.358527815668),
#   ("mtum_STOCH_d" (1; #169), 3120.23851559544),
#   ("mtum_TRIX" (1; #171), 3088.7882192069665),
#   ("ti_CHAIKIN(10_3)" (1; #195), 3073.8961076757405),
#   ("mtum_MACD_signal" (1; #152), 3040.9152887894306),
#   ("mtum_ADX" (1; #134), 3005.6544976562727),
#   ("mtum_MACD_fix_signal" (1; #150), 2969.8609392447397),
#   ("ti_MASS_INDEX(9_25)" (1; #206), 2930.3839368252084),
#   ("cycl_SINE_sine" (1; #87), 2907.057753758272),
#   ("mtum_MACD_fix" (1; #148), 2843.1138502312824),
#   ("mtum_MACD_ext" (1; #145), 2766.18948431802),
#   ("mtum_BOP" (1; #140), 2746.8571826482657),
#   ("ti_EASE_OF_MOVEMENT(14)" (1; #201), 2734.1191980510484),
#   ("ma_SMA_100" (1; #114), 2684.3243264371995),
#   ("sti_CORREL" (1; #186), 2652.2224815494847),
#   ("mtum_ULTOSC" (1; #172), 2641.3844424793497),
#   ("per_Close" (1; #182), 2632.9543417638633),
#   ("mtum_MACD" (1; #144), 2581.080485533923),
#   ("mtum_APO" (1; #136), 2499.8057228892576),
#   ("mtum_MACD_fix_list" (1; #149), 2483.1155128036626),
#   ("mtum_MACD_list" (1; #151), 2478.8397285030223),
#   ("ti_SUPERTREND(20)" (1; #207), 2474.4336176500656),
#   ("ma_WMA_100" (1; #130), 2403.053989228094),
#   ("mtum_PLUS_DI" (1; #157), 2382.7436499593314),
#   ("ma_EMA_100" (1; #104), 2326.967464795802),
#   ("olap_SAREXT" (1; #181), 2320.1064081673976),
#   ("sti_VAR" (1; #193), 2315.4900267524645),
#   ("ma_EMA_50" (1; #107), 2303.362470244756),
#   ("ti_FORCE_INDEX(13)" (1; #202), 2288.1710556722246),
#   ("ti_VORTEX_POS(5)" (1; #209), 2286.223458743654),
#   ("sti_STDDEV" (1; #191), 2275.6127756494097),
#   ("mtum_STOCHF_d" (1; #165), 2269.4439655705355),
#   ("mtum_STOCHRSI_d" (1; #167), 2265.693410494365),
#   ("sti_LINEARREG_ANGLE" (1; #188), 2261.503583154641),
#   ("ma_TRIMA_100" (1; #125), 2260.905761442613),
#   ("sti_BETA" (1; #185), 2252.4626615939196),
#   ("mtum_STOCH_k" (1; #170), 2216.042813512031),
#   ("mtum_AROONOSC" (1; #137), 2206.2500336077064),
#   ("sti_LINEARREG_SLOPE" (1; #190), 2201.3114842826035),
#   ("mtum_DX" (1; #143), 2200.2269476319198),
#   ("ma_SMA_50" (1; #117), 2125.8578490088694),
#   ("ma_TRIMA_50" (1; #128), 2100.8202671532054),
#   ("ti_COPPOCK(14_11_10)" (1; #197), 2086.788117384305),
#   ("ma_KAMA_100" (1; #109), 2073.9870650102384),
#   ("cycl_PHASOR_inph" (1; #84), 2055.3799238412175),
#   ("ma_KAMA_50" (1; #112), 2052.8863624006044),
#   ("mtum_AROON_down" (1; #138), 2006.7571135917678),
#   ("wood_s3" (1; #230), 1984.0169706449378),
#   ("cycl_PHASOR_quad" (1; #85), 1953.1982820259873),
#   ("clas_s3" (1; #80), 1875.5841983295977),
#   ("mtum_CMO" (1; #142), 1851.9361845613457),
#   ("trad_s3" (1; #217), 1820.763198295841),
#   ("mtum_STOCHF_k" (1; #166), 1808.1076870865654),
#   ("olap_SAR" (1; #180), 1794.1377222933806),
#   ("ma_DEMA_50" (1; #102), 1783.0552058035973),
#   ("ti_DONCHIAN_LOWER(20)" (1; #199), 1780.3727348928805),
#   ("mtum_RSI" (1; #164), 1778.460384065751),
#   ("ti_VORTEX_NEG(5)" (1; #208), 1775.7743654614314),
#   ("ma_T3_20" (1; #119), 1773.357443881454),
#   ("mtum_MOM" (1; #156), 1767.7596929899883),
#   ("per_Volume" (1; #183), 1762.693216394633),
#   ("ma_WMA_50" (1; #133), 1742.364407547284),
#   ("mtum_ROCR" (1; #162), 1738.0146334201563),
#   ("mtum_ROC" (1; #160), 1721.2674231491983),
#   ("mtum_CCI" (1; #141), 1708.7706580299418),
#   ("ti_DONCHIAN_CENTER(20)" (1; #198), 1700.357850230299),
#   ("mtum_ROCP" (1; #161), 1690.9112683953717),
#   ("sti_TSF" (1; #192), 1684.3043928635307),
#   ("ti_KELT(20)_LOWER" (1; #204), 1681.4194886656478),
#   ("mtum_WILLIAMS_R" (1; #173), 1664.913660444785),
#   ("olap_HT_TRENDLINE" (1; #177), 1654.699427477317),
#   ("High" (1; #2), 1624.8632079674862),
#   ("wood_s1" (1; #228), 1613.8671532375738),
#   ("olap_BBAND_LOWER" (1; #174), 1613.3847000964452),
#   ("ma_TEMA_5" (1; #123), 1593.555122755235),
#   ("ti_DONCHIAN_UPPER(20)" (1; #200), 1590.9926119938027),
#   ("ma_TEMA_20" (1; #122), 1557.5819467515685),
#   ("trad_s1" (1; #215), 1556.942165436456),
#   ("sti_LINEARREG_INTERCEPT" (1; #189), 1556.1289966460317),
#   ("ma_T3_10" (1; #118), 1555.1198408843484),
#   ("ma_EMA_20" (1; #105), 1544.657863367349),
#   ("trad_s2" (1; #216), 1493.065091153141),
#   ("mtum_ROCR100" (1; #163), 1457.3033520667814),
#   ("ma_DEMA_20" (1; #100), 1438.083428335376),
#   ("ma_KAMA_20" (1; #110), 1433.710974108195),
#   ("sti_LINEARREG" (1; #187), 1357.3826568133663),
#   ("demark_s1" (1; #90), 1353.7440595589578),
#   ("ti_KELT(20)_UPPER" (1; #205), 1339.06329685729),
#   ("mtum_AROON_up" (1; #139), 1338.8180273247417),
#   ("ma_TRIMA_10" (1; #124), 1338.7829630684573),
#   ("Open" (1; #4), 1335.6564048053697),
#   ("ma_SMA_10" (1; #113), 1328.8244849895127),
#   ("ma_SMA_20" (1; #115), 1320.5839177831076),
#   ("olap_MIDPRICE" (1; #179), 1312.1051891627721),
#   ("ma_TEMA_10" (1; #121), 1306.6756870711688),
#   ("wood_r3" (1; #227), 1304.1849101958796),
#   ("ma_KAMA_5" (1; #111), 1293.204384311568),
#   ("ma_WMA_20" (1; #131), 1284.9300091527402),
#   ("ma_TRIMA_20" (1; #126), 1278.38838886749),
#   ("clas_s1" (1; #78), 1265.469847629778),
#   ("ma_DEMA_5" (1; #101), 1262.5970046082512),
#   ("olap_BBAND_UPPER" (1; #176), 1258.6273374601733),
#   ("clas_r3" (1; #77), 1245.1070946627297),
#   ("fibo_s3" (1; #97), 1242.1461772909388),
#   ("wood_s2" (1; #229), 1240.5231050569564),
#   ("Close" (1; #0), 1235.2903135679662),
#   ("clas_s2" (1; #79), 1233.903036833275),
#   ("ma_EMA_10" (1; #103), 1222.1757627949119),
#   ("ti_HMA(20)" (1; #203), 1217.9745613592677),
#   ("cama_s3" (1; #12), 1217.4170852294192),
#   ("ma_KAMA_10" (1; #108), 1205.0215339660645),
#   ("cama_s2" (1; #11), 1177.8711303411983),
#   ("Low" (1; #3), 1149.6092290920205),
#   ("ma_T3_5" (1; #120), 1139.1478124528658),
#   ("ma_WMA_5" (1; #132), 1104.0201214351691),
#   ("fibo_s2" (1; #96), 1061.3014832767658),
#   ("ma_WMA_10" (1; #129), 1057.177856287919),
#   ("ma_DEMA_10" (1; #99), 1052.9028366622515),
#   ("ma_TRIMA_5" (1; #127), 1029.38805654319),
#   ("wood_r1" (1; #225), 1014.8110927862581),
#   ("ma_EMA_5" (1; #106), 977.6376244928688),
#   ("olap_MIDPOINT" (1; #178), 975.1967904390767),
#   ("fibo_s1" (1; #95), 965.9921284732409),
#   ("ma_SMA_5" (1; #116), 963.3567199176177),
#   ("cama_r3" (1; #9), 946.083716403693),
#   ("wood_r2" (1; #226), 936.5765162326861),
#   ("trad_r3" (1; #214), 932.2609510235488),
#   ("cama_s1" (1; #10), 929.1827370892279),
#   ("trad_r2" (1; #213), 916.4719577739015),
#   ("demark_pp" (1; #88), 908.5740719777532),
#   ("olap_BBAND_MIDDLE" (1; #175), 905.5282357339747),
#   ("clas_pp" (1; #74), 889.1516397902742),
#   ("clas_r2" (1; #76), 883.461267685052),
#   ("cama_r1" (1; #7), 838.6118671265431),
#   ("cama_pp" (1; #6), 816.7398435431533),
#   ("wood_pp" (1; #224), 804.9477973135654),
#   ("mtum_STOCHRSI_k" (1; #168), 797.8100565690547),
#   ("fibo_pp" (1; #91), 779.6397007042542),
#   ("cama_r2" (1; #8), 772.0980045143515),
#   ("fibo_r2" (1; #93), 769.4688386269845),
#   ("trad_pp" (1; #211), 755.4097204795107),
#   ("clas_r1" (1; #75), 749.0971500272863),
#   ("cdl_BELTHOLD" (1; #22), 722.9604555040132),
#   ("fibo_r3" (1; #94), 716.9538499009795),
#   ("trad_r1" (1; #212), 665.537115668878),
#   ("per_preMarket" (1; #184), 649.190015403321),
#   ("demark_r1" (1; #89), 507.4871474811807),
#   ("cdl_HIKKAKE" (1; #41), 505.97568327048793),
#   ("fibo_r1" (1; #92), 497.96562323439866),
#   ("cycl_HT_TRENDMODE" (1; #83), 261.5225486308336),
#   ("cdl_HIKKAKEMOD" (1; #42), 216.95708867209032),
#   ("has_preMarket" (1; #98), 147.32496676500887),
#   ("cdl_SPINNINGTOP" (1; #64), 145.61264508776367),
#   ("cdl_LONGLINE" (1; #51), 133.06689357757568),
#   ("cdl_ENGULFING" (1; #31), 132.171355700586),
#   ("cdl_SHORTLINE" (1; #63), 100.37469867197797),
#   ("cdl_RICKSHAWMAN" (1; #59), 81.068463713862),
#   ("cdl_LONGLEGGEDDOJI" (1; #50), 76.5796693759039),
#   ("cdl_HIGHWAVE" (1; #40), 71.57211948023178),
#   ("cdl_DOJI" (1; #28), 69.91886213514954),
#   ("cdl_CLOSINGMARUBOZU" (1; #24), 53.7321413224563),
#   ("cdl_HARAMI" (1; #38), 52.08180280216038),
#   ("cdl_SEPARATINGLINES" (1; #61), 47.78811174072325),
#   ("cdl_3OUTSIDE" (1; #17), 41.87768318876624),
#   ("cdl_XSIDEGAP3METHODS" (1; #73), 40.77702275104821),
#   ("cdl_MARUBOZU" (1; #52), 38.89893465396017),
#   ("cdl_ADVANCEBLOCK" (1; #21), 29.853826835751534),
#   ("cdl_DRAGONFLYDOJI" (1; #30), 22.620155725628138),
#   ("cdl_GRAVESTONEDOJI" (1; #35), 21.545906896702945),
#   ("cdl_DARKCLOUDCOVER" (1; #27), 18.3607691898942),
#   ("cdl_TAKURI" (1; #67), 17.677047036588192),
#   ("cdl_HARAMICROSS" (1; #39), 15.68235603813082),
#   ("cdl_STALLEDPATTERN" (1; #65), 15.639400023035705),
#   ("cdl_3INSIDE" (1; #15), 15.160921663045883),
#   ("cdl_MATCHINGLOW" (1; #53), 14.7969416892156),
#   ("cdl_MORNINGSTAR" (1; #56), 11.925127413123846),
#   ("cdl_GAPSIDESIDEWHITE" (1; #34), 10.406118504703045),
#   ("cdl_HOMINGPIGEON" (1; #43), 10.011952053755522),
#   ("cdl_HANGINGMAN" (1; #37), 9.79833785444498),
#   ("cdl_SHOOTINGSTAR" (1; #62), 6.361247068271041),
#   ("cdl_HAMMER" (1; #36), 4.105077577754855)]}