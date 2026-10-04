import pandas as pd
import glob, os
import features_W3

# from features_W3 import extract_features_v3, uncorrelate_selection
# from features_W3 import extract_features_v3, uncorrelate_selection
# from features_W3.check import check_leakage
# from features_X import extract_features_v3, uncorrelate_selection
# from features_X.check import check_leakage
#
# stocks_list = [ 'UPST']
stocks_list = [  "U", "DDOG","UPST", "RIVN", "SNOW", "LYFT",  "GTLB"] + [ "MDB", "HUBS", "TTD", "ASAN",    "APPS" , "DOCN", "SHOP", "RBLX", "NIO"]
# stocks_list =  [ "MDB", "HUBS", "TTD", "ASAN",    "APPS" , "DOCN", "SHOP", "RBLX", "NIO"]
stocks_types_times = ['5min']#, '15min', '60min']


# def get_time_market(df_raw):
#     df_raw.insert(loc=len(df_raw.columns), column="time_market", value=0)
#     df_raw['Date'] = pd.to_datetime(df_raw.index, errors="coerce")
#     TIME_ALPHA_OPEN = "13:30:00";
#     TIME_ALPHA_CLOSE = "20:00:00";
#     df_raw.loc[(df_raw['Date'].dt.time <= pd.Timestamp(TIME_ALPHA_OPEN).time()), "time_market"] = 1
#     df_raw.loc[(df_raw['Date'].dt.time >= pd.Timestamp(TIME_ALPHA_CLOSE).time()), "time_market"] = 2
#     return df_raw

time = "5min"
for s in stocks_list:
    # for time in stocks_types_times:
    #     print(time, " ", s)
    #     #file format name E:\GitHub\d_price\RAW_alpha\GTLB_5min_20230616__20211014.csv
    #     #E:\TradeDL\data\alpa_APPS_5min.csv
    #     files = glob.glob('E:\TradeDL\data' + f'alpa_pre_{s}_{time}' + "*.csv")
    # for file in glob.glob('E:\GitHub' + f'\d_price\{s}_{time}' + "*.csv"):
    #     print(file)
    # print("File: ", files)
    file_rwa_alpha =  f'data/alpa_pre_{s}_{time}' + ".csv"
    # file_rwa_alpha =   files[0]

    df_raw = pd.read_csv(file_rwa_alpha, parse_dates=['Date'], index_col='Date', sep='\t')
    for c in df_raw.columns:
        df_raw = df_raw.rename(columns={c: c.lower()})
    print("read csv: ", file_rwa_alpha, ' with Shape: ', df_raw.shape)
    #
    #
    # df_features_X_ALL = extract_features_v3(df_raw)
    df_features_X_ALL = features_W3.ta.extract_features(df_raw)

    print("df_features_X_ALL: ", f'data/alpa_pre_{s}_{time}.csv with Shape: ', df_features_X_ALL.shape)
    df_features_X_ALL['date'] = df_features_X_ALL.index
    #
    print('Generated: ',  f'data/alpa_pre_{s}_{time}_tech_all.csv')
    df_features_X_ALL.to_csv(f'data/alpa_pre_{s}_{time}_tech_all.csv',sep="\t")
    print("\tSTART: ", str(df_features_X_ALL.index.min()),  "  END: ", str(df_features_X_ALL.index.max()) , " shape: ", df_features_X_ALL.shape, "\n")

# Get All Featrures tech params
# print("DEBUG All extract_features_v3")
# features_X = bars
# features_X_ALL = extract_features_v3(bars)


