

TIME_WINDOW_GT  = 36 #Roling windows for GT
INDI_TIMEFRAME = ['5min', '15min', '60min']

DICT_COMPANYS = {
    "@FAV":
        ['U' ] # ['U' , 'UPST']
}


INDicator_label_3W51560_ = "_3W51560_"
INDicator_label_3W5_ = "_3W5_"
INDicator_label_3W515_ = "_3W515_"
INDicator_label_3W560_ = "_3W560_"
INDI_TIMEFRAME_OPTIONS = {
    INDicator_label_3W51560_:['5min', '15min', '60min'],
    INDicator_label_3W5_ :   ['5min'],
    INDicator_label_3W515_ : ['5min', '15min'],
    INDicator_label_3W560_ : ['5min', '60min']
}



# Training Settings
SPLIT_RATIO_X_Y = 0.80
val_ratio=0.2
WINDOWS_SIZE = 48 # 48 cuatro horas
predict_period = 1

indicator_timeframe = "H1"
timeframe = 'M15'

# Symbol Information
symbols = {
    'EURUSD': { 'point': 10000, 'spread': 0.4 },
    'GBPUSD': { 'point': 10000, 'spread': 0.6 },
    'USDCAD': { 'point': 10000, 'spread': 1.4 },
    'USDCHF': { 'point': 10000, 'spread': 1.4 },
    'USDJPY': { 'point': 100,   'spread': 1.4 },
    'GBPJPY': { 'point': 100,   'spread': 2.6 },
    'AUDUSD': { 'point': 10000, 'spread': 1.4 },
    'XAUUSD': { 'point': 10,    'spread': 17.0},
}

use_percentage = False
NUMS_FEATURES = 35

# Multi Environment settings
train_limit = 53000

# Max processes for tech df parallel processing
MAX_PROCESSES = 10




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#should be removed from the training because they use global averages, i.e. if you remove the last column, they change the value of all previous columns, useless in realtime.
LIST_TECH_REMOVE_NOT_EQUAL_IF_REMOVE_THE_FIRSH = ["mtum_APO", "mtum_CCI", "mtum_MACD_ext", "mtum_MACD_ext_signal", "mtum_MACD_ext_list", "mtum_PPO",
                                                  "mtum_STOCH_RSI_d", "mtum_STOCH_RSI_kd", "ma_TRIMA_5", "ma_TRIMA_20", "ma_WMA_20", "ma_TRIMA_50",
                                                  "ma_TRIMA_100", "mtum_BIAS_SMA_26", "mtum_BR_26", "olap_VMAP", "perf_CUMLOGRET_1", "perf_CUMPCTRET_1", "sti_ENTP_10"]
#should be removed because they generate a lot of None, in the last columns, useless in RealTime.
LIST_TECH_REMOVE_GENERATED_NONE_LAST = ['ti_acc_dist']
LIST_TECH_REMOVE_GENERATED_INFINITE = ['olap_MCGD_10']

LIST_TECH_REMOVE = LIST_TECH_REMOVE_NOT_EQUAL_IF_REMOVE_THE_FIRSH + LIST_TECH_REMOVE_GENERATED_NONE_LAST +LIST_TECH_REMOVE_GENERATED_INFINITE

dict_correlations = {
    "v1" : ['chi2'],
    "v2" : ['ExtraTrees'],
    "v3" : ['corrwith'],
    "v4" : ['f_regression'],
    "v5" : ['chi2','ExtraTrees'],
    "v6" : ['chi2','corrwith'],
    "v7" : ['chi2','f_regression'],
    "v8" : ['ExtraTrees','corrwith'],
    "v9" : ['ExtraTrees','f_regression'],
    "v10" : ['corrwith','f_regression'],
    "v11" : ['chi2','ExtraTrees','corrwith'],
    "v12" : ['chi2','ExtraTrees','f_regression'],
    "v13" : ['chi2','corrwith','f_regression'],
    "v14" : ['ExtraTrees','corrwith','f_regression'],
    "v15" : ['chi2','ExtraTrees','corrwith','f_regression']
}