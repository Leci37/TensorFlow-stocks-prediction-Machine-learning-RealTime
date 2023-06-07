from enum import Enum

import investpy
import pandas as pd
from datetime import datetime

from Utils import UtilsL
from LogRoot.Logging import Logger
'''
df = investpy.get_stock_historical_data(stock='AAPL',
                                        country='United States',
                                        from_date='01/01/2010',
                                        to_date='01/01/2020')
#print(df.head())

search_result = investpy.search_quotes(text='apple', products=['stocks'],
                                       countries=['united states'], n_results=1)
#print(search_result)

recent_data = search_result.retrieve_recent_data()
#print(recent_data)
historical_data = search_result.retrieve_historical_data(from_date='01/01/2019', to_date='01/01/2020')
#print(historical_data)
information = search_result.retrieve_information()
#print(information)
default_currency = search_result.retrieve_currency()
#print(default_currency)
technical_indicators = search_result.retrieve_technical_indicators(interval='daily')

#print(technical_indicators)
'''
PRODUCT_TYPE_FILES = {
    #"certificate": "certificates.csv", only works with 5 countrys
    #"commodity": "commodities.csv",
    #"currency_cross": "currency_crosses.csv",
    #"etf": "etfs.csv",
    #"fund": "funds.csv",
    #"index": "indices.csv",
    "stock": "stocks.csv",
    #"bond": "bonds.csv",
}
INTERVAL_FILTERS = {
    #"1min": 60,
    #"5mins": 60 * 5,
    #"15mins": 60 * 15,
    #"30mins": 60 * 30,
    "1hour": 60 * 60,
    "5hours": 60 * 60 * 5,
    "daily": 60 * 60 * 24,
    "weekly": "week",
    "monthly": "month",
}

class TABLES_TYPE(Enum):
    TECIN = "investpy.technical_indicators"
    MOVAVE = "investpy.technical.moving_averages"
    ST_DIV = "stock_dividends"
    STCSUM = "get_stock_financial_summary.quarterly"

LIST_YEARS_ALLOW = ['2019','2021','2022','2023','2024']

#NAME_STOCK =  'AAPL'#'VWDRY'
#COUNTRY = "united states"
PRODUCT_TYPE = str("stock")

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def GET_df_technical_indicators(stockId, Country= "united states"):
    # for info_stock in PRODUCT_TYPE_FILES.keys():
    # `5hours`, `daily`, `weekly` and `monthly`.
    df_technical_indicators = pd.DataFrame(columns=['DATE', 'TYPE_DATA'])
    # df_technical_indicators['Date'] = pd.Timestamp("today").strftime("%Y-%m-%d")
    for interval in INTERVAL_FILTERS:

        df_Aux_tech = get_indicators_move_avg(Country, interval, stockId)
        # LUIS
        df_Aux_tech['Date'] = pd.Timestamp("today").strftime("%Y-%m-%d")

        # if df_technical_indicators is None:
        #    df_technical_indicators = df_Aux_tech[["Date", "technical_indicator"]].copy()

        # technical_indicators
        colums_df_tech = [column for column in df_Aux_tech if (column != "technical_indicator" and column != "DATE")]
        for c in colums_df_tech:
            Logger.logr.debug(c+ "  "+ str(c))
            df_Aux_tech = df_Aux_tech.rename({c: str(c)}, axis='columns')
        df_Aux_tech['TYPE_DATA'] = TABLES_TYPE.TECIN.name + "_" + interval
        df_technical_indicators = df_technical_indicators._append(
            df_Aux_tech)  # pd.merge(df_technical_indicators, df_Aux_tech, on=['technical_indicator', 'Date'])

        # moving_averages
        colums_df_move = [column for column in df_Aux_move if (column != "DATE")]
        for c in colums_df_move:
            Logger.logr.debug(c+ "  "+ str(c))
            df_Aux_move = df_Aux_move.rename({c: str(c)}, axis='columns')
        df_Aux_move['TYPE_DATA'] = TABLES_TYPE.MOVAVE.name + "_" + interval
        df_technical_indicators = df_technical_indicators._append(df_Aux_move)
        # print(df_Aux_tech.head())
        # print(df_Aux_move.head())
        # print(df_technical_indicators.head())

        # df_technical_indicators = pd.merge(df_technical_indicators, df_Aux_move, on=['Date'], how = 'outer')#, indicator = True)

        # df_technical_indicators.set_index('technical_indicator').join(df_Aux.set_index('technical_indicator'))

        # print(df_Aux2.head())
    Logger.logr.info("")
    '''
    df_stock_div = investpy.get_stock_dividends(stock='AAPL', country="united states")
    # print(df_get_stock_dividends.head())
    df_stock_div = pd.DataFrame(
        df_stock_div)  # pd.DataFrame(df_get_stock_dividends.items(), columns=['DATE', 'Dividend','Type', 'Payment', 'date_dividends', 'Yield'])
    df_stock_div['TYPE_DATA'] = TABLES_TYPE.ST_DIV.name
    df_stock_div = df_stock_div.rename({'Payment Date': str('Payment')}, axis='columns')
    df_stock_div = df_stock_div.rename({'Date': str('DATE')}, axis='columns')
    # df_stock_div = df_stock_div['DATE'].str.replace([' 00:00:00'], [''])
    df_stock_div = df_stock_div[
        df_stock_div['DATE'].astype(str).str.contains("|".join(LIST_YEARS_ALLOW), na=False)]  # REGEX
    df_stock_div["DATE"] = df_stock_div["DATE"].astype(str).str.replace(' 00:00:00', '')
    df_technical_indicators = df_technical_indicators._append(df_stock_div)'''
    df_technical_indicators['DATE'] = df_technical_indicators['DATE'].astype(str)
    return df_technical_indicators


def get_indicators_move_avg(stockId ,country, interval):
    df_Aux_tech = investpy.technical_indicators(name=stockId, country=country, product_type=PRODUCT_TYPE,
                                                interval=interval)  # interval='daily')
    df_Aux_move = investpy.technical.moving_averages(name=stockId, country=country, product_type=PRODUCT_TYPE,
                                                     interval=interval)
    # LUIS
    df_Aux_move = df_Aux_move.rename(columns={'period': 'technical_indicator'})
    df_Aux_move_sma = df_Aux_move[['technical_indicator', 'sma_value', 'sma_signal']]
    df_Aux_move_ema = df_Aux_move[['technical_indicator', 'ema_value', 'ema_signal']]
    df_Aux_move_sma['technical_indicator'] = "mov_sma_" + df_Aux_move_sma['technical_indicator']
    df_Aux_move_ema['technical_indicator'] = "mov_ema_" + df_Aux_move_ema['technical_indicator']
    df_Aux_move_sma = df_Aux_move_sma.rename(columns={'sma_value': 'value', 'sma_signal': 'signal'})
    df_Aux_move_ema = df_Aux_move_ema.rename(columns={'ema_value': 'value', 'ema_signal': 'signal'})
    df_m = pd.concat([df_Aux_move_sma, df_Aux_move_ema])
    df_m = pd.concat([df_Aux_tech, df_m])
    df_m = UtilsL.remove_chars_in_columns(df_m, 'technical_indicator')
    return df_m


STOCK_SUMMARY_TYPE = {
    "balance_sheet": "balance_sheet",
    "income_statement": "income_statement",
    "cash_flow_statement": "cash_flow_statement"
}

def GET_df_stock_finance_summary(stockId, Country ="united states"):
    Logger.logr.info("")
    df_get_stock_summary = pd.DataFrame(columns=['Date', 'TYPE_DATA'])
    for stock_summary in STOCK_SUMMARY_TYPE.keys():
        df_aux_a = investpy.get_stock_financial_summary(stock=stockId, country=Country, summary_type=stock_summary,
                                                        period='annual')
        df_aux_p = investpy.get_stock_financial_summary(stock=stockId, country=Country, summary_type=stock_summary,
                                                        period='quarterly')
        df_aux = df_aux_a._append(df_aux_p)
        # df_get_stock_summary_2 = pd.merge(df_get_stock_summary_2, df_aux, on=[df_aux.columns],how='outer')  # , indicator = True)
        Logger.logr.info("   "+ stock_summary+ "   "+ df_aux.columns)
        df_get_stock_summary = pd.merge(df_get_stock_summary, df_aux, on=['Date'], how='outer')
    df_get_stock_summary = df_get_stock_summary.rename({'Date': str('DATE')}, axis='columns')
    df_get_stock_summary['TYPE_DATA'] = TABLES_TYPE.ST_DIV.name
    df_get_stock_summary['DATE'] = df_get_stock_summary['DATE'].astype(str)
    df_get_stock_summary = df_get_stock_summary.drop_duplicates()
    df_get_stock_summary = df_get_stock_summary.drop_duplicates(subset=["DATE"], keep="first")
    df_get_stock_summary = df_get_stock_summary.sort_values('DATE', ascending=False)
    return df_get_stock_summary

def GET_df_stock_historical(stockId,Country = "united states"):
    TODAY = datetime.today().strftime('%d/%m/%Y')  # .strftime('%Y-%m-%d %H:%M:%S')'2021-01-26 16:50:03'
    TODAY_2AGO = datetime.now().replace(year=datetime.now().year - 2).strftime('%d/%m/%Y')
    Logger.logr.info("----------------"+ "get_stock_historical_data"+ "----------------")
    df_stock_historical = pd.DataFrame(columns=['Date', 'TYPE_DATA'])
    df_aux = investpy.get_stock_historical_data(stock=stockId, country=Country, order="descending",
                                                from_date=TODAY_2AGO, to_date=TODAY, interval="Daily")
    df_stock_historical = pd.merge(df_stock_historical, df_aux, on=['Date'], how='outer')
    df_stock_historical['TYPE_DATA'] = TABLES_TYPE.ST_DIV.name
    df_stock_historical = df_stock_historical.rename({'Date': str('DATE')}, axis='columns')
    df_stock_historical['DATE'] = df_stock_historical['DATE'].astype(str)
    df_stock_historical.drop('Currency', axis=1, inplace=True)
    #df_stock_historical = df_stock_historical.drop('Currency', 1)
    return df_stock_historical


NAME_STOCK =  'AAPL'#'VWDRY'
COUNTRY = "united states"

# df_T = GET_df_technical_indicators(NAME_STOCK,COUNTRY )
df_S = GET_df_stock_finance_summary(NAME_STOCK, COUNTRY)
#df_H = GET_df_stock_historical(NAME_STOCK,COUNTRY )


df_S.to_csv( NAME_STOCK + "_GET_df_stock_finance_summary.csv", sep="\t")
Logger.logr.debug(" Generated File info: " + NAME_STOCK + "_technical_invAPI.csv")
'''
df_S.to_csv( NAME_STOCK + "_financial_invAPI_summary.csv", sep="\t")
Logger.logr.debug(" Generated File info: " + NAME_STOCK + "_financial_invAPI_summary.csv")

df_H.to_csv( NAME_STOCK + "_stock_invAPI_historical.csv", sep="\t")
Logger.logr.debug(" Generated File info: " + NAME_STOCK + "_stock_financial_invAPI_historical.csv")


#print(df_S)
#df = pd.merge(df_T, df_S, on=['DATE','TYPE_DATA'], how='outer')
#df = pd.merge(df, df_H, on=['DATE','TYPE_DATA'], how='outer')
#df_T.to_csv("test1.csv", sep='|')

'''



'''
print("----------------","get_stock_information", "----------------" )
df_get_stock_information = investpy.get_stock_information(stock=NAME_STOCK, country=COUNTRY)
print(df_get_stock_information.head())

print("----------------","get_stock_recent_data", "----------------" )
df_get_stock_recent_data = investpy.get_stock_recent_data(stock='AAPL', country="united states")
print(df_get_stock_recent_data.head())

#df = investpy.get_stocks_dict(country="united states") #<country, name, full_name, isin, currency, symbol>
#print(df)
#df = investpy.get_stocks_list(country="united states")
#print(df)
print("----------------","get_stocks_overview", "----------------" )
df_get_stocks_overview = investpy.get_stocks_overview(country="united states")
print(df_get_stocks_overview.sample())
#investpy.search_stocks


print(vars(investpy.currency_crosses))
print(vars(investpy.indices))
print(vars(investpy.stocks))
print(vars(investpy.technical))
print(vars(investpy.technical_indicators()))
'''



'''
print("----------------","moving_averages", "----------------" )
df_moving_averages = investpy.technical.moving_averages(name='AAPL', country="united states", product_type=str("stock"), interval='daily')
print(df_moving_averages.head())

print("----------------","get_stock_company_profile", "----------------" )
df_get_stock_company_profile = investpy.get_stock_company_profile(stock='AAPL', country="united states", language="english")
print(df_get_stock_company_profile)
#'url': 'https://www.investing.com/equities/apple-computer-inc-company-profile', 'desc':
'''


