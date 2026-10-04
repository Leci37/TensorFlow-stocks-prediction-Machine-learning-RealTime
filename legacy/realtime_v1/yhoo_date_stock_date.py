from Utils import Utils_Yfinance
import yfinance as yf
import pandas as pd

from LogRoot.Logging import Logger

#TODO
# print(s.get_info) no date
# print(s.get_institutional_holders) no date
# print(s.get_isin) no date
# print(s.get_major_holders) no date
# print(s.get_mutualfund_holders) no date
#print(s.get_sustainability()) no date
#print(s.history())
# print(s.info()) no date
# print(s.institutional_holders())no date
# print(s.isin)no date
# print(s.major_holders())no date
#print(s.sustainability)# no dete

BEGIN_DATE = '2019-01-01'
END_DATE = '2025-01-01'


def get_all_df_dates_from_yhoo(sto_yhoo):
    list_df = []
    # Todo refactor in more litte methods
    try:
        df_actions = sto_yhoo.actions
        if df_actions is not None:# or len(df_actions) != 0:
            df_actions = df_actions.loc[BEGIN_DATE:END_DATE]
            df_actions['Date'] = df_actions.index
            df_actions.reset_index(drop=True, inplace=True)
            list_df.append(df_actions)
    except Exception as e:
        print("")

    try:
        df_analysis = sto_yhoo.analysis
        if df_analysis is not None:# or len(df_analysis) != 0:
            df_analysis.rename(columns={'End Date': 'Date'}, inplace=True)
            list_df.append(df_analysis)
    except Exception as e:
        print("")

    try:
        df_balancesheet = sto_yhoo.get_balancesheet(freq='yearly').T
        if df_balancesheet is not None:# or len(df_balancesheet) != 0:
            df_balancesheet['Date'] = df_balancesheet.index
            df_balancesheet.reset_index(drop=True, inplace=True)
            df_balancesheet_q = sto_yhoo.get_balancesheet(freq='quarterly').T
            if df_balancesheet_q is not None:# or len(df_balancesheet_q) != 0:
                df_balancesheet_q['Date'] = df_balancesheet_q.index
                df_balancesheet_q.reset_index(drop=True, inplace=True)
                df_balancesheet = pd.merge(df_balancesheet, df_balancesheet_q, how='outer')
                list_df.append(df_balancesheet)
    except Exception as e:
        print("")

    try:
        # print(s.calendar)
        df_calendar = sto_yhoo.calendar.T
        if df_calendar is not None:# or len(df_calendar) != 0:
            df_calendar.rename(columns={'Earnings Date': 'Date'}, inplace=True)
            list_df.append(df_calendar)
    except Exception as e:
        print("")

    try:
        df_cashflow = sto_yhoo.get_cashflow(freq="yearly").T
        if df_cashflow is not None :#or len(df_cashflow) != 0:
            df_cashflow['Date'] = df_cashflow.index
            df_cashflow.reset_index(drop=True, inplace=True)
            df_cashflow_q = sto_yhoo.get_cashflow(freq="quarterly").T
            if df_cashflow_q is not None:# or len(df_cashflow_q) != 0:
                df_cashflow_q['Date'] = df_cashflow_q.index
                df_cashflow_q.reset_index(drop=True, inplace=True)
                df_cashflow = pd.merge(df_cashflow, df_cashflow_q, how='outer')
                list_df.append(df_cashflow)
    except Exception as e:
        print("")

    try:
        # print(s.dividends)
        df_dividends = sto_yhoo.dividends.loc[BEGIN_DATE:END_DATE]
        if df_dividends is not None:# or len(df_dividends) != 0:
            df_dividends = pd.DataFrame({'Dividends': df_dividends})
            df_dividends['Date'] = df_dividends.index
            df_dividends.reset_index(drop=True, inplace=True)
            list_df.append(df_dividends)
    except Exception as e:
        print("")

    try:
        # print(s.earnings)
        df_earnings = sto_yhoo.get_earnings(freq="yearly")
        if df_earnings is not None:# or len(df_earnings) != 0:
            df_earnings['Date'] = df_earnings.index
            df_earnings['Date'] = pd.to_datetime(df_earnings.Date, format='%Y')
            df_earnings.reset_index(drop=True, inplace=True)
            df_earnings_q = sto_yhoo.get_earnings(freq="quarterly")
            if df_earnings_q is not None:# or len(df_earnings_q) != 0:
                df_earnings_q['Date'] = df_earnings_q.index
                df_earnings_q['Date'] = pd.to_datetime(df_earnings_q['Date'].str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors='coerce')
                df_earnings_q.reset_index(drop=True, inplace=True)
                df_earnings = pd.merge(df_earnings, df_earnings_q, how='outer')
                list_df.append(df_earnings)
    except Exception as e:
        print("")

    try:
        # print(s.financials)
        df_financials = sto_yhoo.get_financials(freq="yearly").T
        if df_financials is not None:# or len(df_financials) != 0:
            df_financials['Date'] = df_financials.index
            df_financials.reset_index(drop=True, inplace=True)
            df_financials_q = sto_yhoo.get_financials(freq="quarterly").T
            if df_financials_q is not None:# or len(df_financials_q) != 0:
                df_financials_q['Date'] = df_financials_q.index
                df_financials_q.reset_index(drop=True, inplace=True)
                df_financials = pd.merge(df_financials, df_financials_q, how='outer')
                list_df.append(df_financials)
    except Exception as e:
        print("")

    try:
        # print(s.get_news)
        df_recommendations = sto_yhoo.get_recommendations()
        if df_recommendations is not None:# or len(df_recommendations) != 0:
            df_recommendations = df_recommendations.loc[BEGIN_DATE:END_DATE]
            df_recommendations['Date'] = df_recommendations.index
            df_recommendations.reset_index(drop=True, inplace=True)
            list_df.append(df_recommendations)
    except Exception as e:
        print("")

    try:
        # print(s.shares)
        df_shares = sto_yhoo.shares
        if df_shares is not None:# or len(df_shares) != 0:
            df_shares['Date'] = df_shares.index
            df_shares.reset_index(drop=True, inplace=True)
            list_df.append(df_shares)
    except Exception as e:
        print("")

    try:
        # print(s.splits)
        df_split = sto_yhoo.splits
        if df_split is not None:# or len(df_split) != 0:
            df_split = df_split.loc[BEGIN_DATE:END_DATE]
            list_df.append(df_split)

            if sto_yhoo.option_chain() is not None or sto_yhoo.option_chain()[1] is not None or len(sto_yhoo.option_chain()[1]) != 0:
                df_option = pd.merge(sto_yhoo.option_chain()[1], sto_yhoo.option_chain()[0], how='outer')
                df_option.rename(columns={'lastTradeDate': 'Date'}, inplace=True)
                list_df.append(df_option)
    except Exception as e:
        print("")

    try:
        df_sus = sto_yhoo.get_sustainability()
        if df_sus is not None:
            df_sus[df_sus.index.name] = df_sus.index
            date_sus = pd.to_datetime(df_sus.index.name).strftime('%Y-%m-%d')
            df_sus.rename(columns={df_sus.index.name:'Date'  }, inplace=True)
            df_sus.rename(columns={'Value': date_sus}, inplace=True)
            df_sus.reset_index(drop=True, inplace=True)
            #df_sus = df_sus.pivot_table(date_sus , 'Values' )#, ['Values'], 'medal')
            df_sus = df_sus.T
            df_sus['Date'] = df_sus.index
            df_sus = df_sus.rename(columns=df_sus.loc['Date'] )
            df_sus = df_sus.drop(df_sus.index[1])
            df_sus.reset_index(drop=True, inplace=True)
            list_df.append(df_sus)
    except Exception as e:
        print("")

    try:
        # print(s.shares)
        df_inst_fold = sto_yhoo.get_institutional_holders()
        df_inst_fold_2 = sto_yhoo.get_mutualfund_holders()

        if (df_inst_fold is not None) and (df_inst_fold_2 is not None):# or len(df_shares) != 0:
            df_inst_fold.rename(columns={'Date Reported': 'Date'}, inplace=True)
            df_inst_fold_2.rename(columns={'Date Reported': 'Date'}, inplace=True)

            df_inst_fold = pd.merge(df_inst_fold, df_inst_fold_2, how='outer')
            list_rename = ['Holder', 'Shares','% Out', 'Value']
            for l in list_rename:
                df_inst_fold.rename(columns={l: l+'_holder'}, inplace=True)
            #df_inst_fold['Date'] = df_inst_fold.index
            df_inst_fold.drop_duplicates(inplace=True)
            df_inst_fold.reset_index(drop=True, inplace=True)
            list_df.append(df_inst_fold)
    except Exception as e:
        print("")

    try:
        dic_info = sto_yhoo.get_info()
        if (dic_info is not None):
            df_info = pd.DataFrame([dic_info], columns=dic_info.keys())
            df_info = df_info.drop('longBusinessSummary', 1)

            for c in df_info.columns:
                df_info[c] = df_info[c].astype(str)
                df_info.rename(columns={c: c + '_info'}, inplace=True)

            df_info['Date'] = pd.Timestamp("today").strftime("%Y-%m-%d")#.astype(str)
            list_df.append(df_info)
    except Exception as e:
        print("")




    #Index(['Holder', 'Shares', 'Date Reported', '% Out', 'Value'], dtype='object')

    return  list_df


def get_all_date_info_yhoo(stockID):

    stock_yhoo = yf.Ticker(stockID)
    list_df = get_all_df_dates_from_yhoo(stock_yhoo)
    df_full = Utils_Yfinance.merge_all_df_of_the_list(list_df, stockID)
    if df_full is None or len(df_full) == 0:
        Logger.logr.error(" NOT data fount Generated YAHOO info : " + stockID)
        pass

    df_full.to_csv("d_info_profile/" + stockID + "_yahoo_full_data.csv", sep="\t")
    Logger.logr.info(
        "Full merge of all Dataframes of yahoo completed shape: " + str(df_full.shape) + " Stock: " + stockID)
    dict_j = Utils_Yfinance.prepare_df_to_json_by_date(df_full)

    dict_json = {}
    # dict_json['Date'] = dict_j
    dict_json = dict_j

    import json
    with open("d_info_profile/" + stockID + "_yahoo_full_data.json", 'w') as fp:
        json.dump(dict_json, fp, allow_nan=True)
    Logger.logr.info("d_info_profile/" + stockID + "_yahoo_full_data.json   Numbres of Keys: "+ str(len(dict_json)))



list_stocks = ["RIVN", "VWDRY", "TWLO",          "GOOG","ASML","SNOW","ADBE","LYFT","UBER","ZI", "BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L",]
list_stocks = ["NATGAS","GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC","BBVA.MC","OB"]

#get_all_date_info_yhoo("BA")
#list_stocks = ["MSFT"]
# for s in list_stocks:
#     try:
#         get_all_date_info_yhoo(s)
#     except Exception as e:
#          Logger.logr.debug(str(e)+ " Exception ALL stockID: " + s)



