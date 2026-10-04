# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from LogRoot.Logging import Logger

from statistics import *

# https://towardsdatascience.com/stock-news-sentiment-analysis-with-python-193d4b4378d4
import news_sentiment_va_and_txtBlod
from news_sentiment_flair import get_sentiment_flair
from news_sentiment_t5 import get_sentiment_t5

pd.set_option('display.max_columns', None)
# Parameters
n = 3  # the # of article headlines displayed per ticker
tickers = ["AMD", "GE", "BA", "UBER", "AMZN", "AAPL", "PYPL",
           "MELI"]  # ["PGS.OL", "NRGU", "ATHX", "GDYN", "UONE", "UNM", "GRF.MC", "NRG", "UDMY", "CERS", "HES", "EOG", "CVET", "LABD", "RGA", "POST", "CAN", "APA", "OUT1V.HE", "ATGE", "TWOU", "TXRH", "WW", "HEAR", "NOG.L", "ZH", "CEMI", "DYDX", "AXSM", "CLNE", "QRTEA", "NWSA", "RVMD", "ILMN", "MQ", "LABU.US", "FNTN.DE", "UI", "BE", "SCATC.OL", "GPRO", "ADJ.DE", "DISH", "NET", "CFLT", "BILL", "UAA", "UA", "GH"]
LIST_YEARS_ALLOW = ['-20', '-21', '-22', '-23', '-24']

# Get Data

news_tables = {}

STOCK_NAME = "TSLA"


def get_sentiment_news_finviz(stock_id: str):
    '''
    va a la web de noticias y recoge las noticias segun la Stock_ID
    lo devuelve en forma de DataSet
    :param stock_id:
    :return:
    '''
    FINWIZ_URL = 'https://finviz.com/quote.ashx?t='
    date = ""
    try:
        url = FINWIZ_URL + stock_id
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
        resp = urlopen(req)
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[stock_id] = news_table

        df = news_tables[stock_id]
        df_tr = df.findAll('tr')

        Logger.logr.info('\n')
        Logger.logr.info('Get recent News Headlines for {}: '.format(stock_id))

        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            # print(a_text, '(', td_text, ')')
            if i == n - 1:
                break
    except KeyError as e:
        Logger.logr.warn(str(e)+ " KeyError url: "+FINWIZ_URL + stock_id)
        return None
    except Exception as e:
        Logger.logr.warn(str(e)+ " Exception url: "+ FINWIZ_URL + stock_id)
        return None
    # Iterate through the news
    parsed_news = []
    del parsed_news[:]
    text = ""
    # for file_name, news_table in news_tables[stock_id]:# news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]

        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = stock_id  # file_name.split('_')[0]

        parsed_news.append([ticker, date, time, text])
    # Sentiment Analysis

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    df_news = None
    df_news = pd.DataFrame(parsed_news, columns=columns)
    df_news['Date'] = pd.to_datetime(df_news['Date'].astype(str), format="%b-%d-%y")  # May-05-22
    df_news['Date'] = df_news["Date"].dt.strftime("%Y-%m-%d")
    Logger.logr.info(" Stock: "+ stock_id+ " Numbers of news: "+
                     str(len(df_news.index)))
    return df_news


'''
#def get_sentiment_full(stock_id:str, is_print = False):
    df_news = None
    df_news = get_sentiment_news_finviz(stock_id)
    df_news = news_sentiment_va.get_sentiment_predictorS(df_news)

    return get_avg_newsOp(df_news, stock_id, is_print)
'''


def get_avg_newsOp(df_news, stock_id="", is_print=False):
    if is_print:
        df_news.to_csv("sentiment/stock_news_DATE_" + stock_id + ".csv", sep="\t")
    df_news = df_news.drop(columns=['Headline', 'Time'],
                           errors='ignore')  # If 'ignore', suppress error and only existing labels are dropped.
    # Count number of news
    count_news = df_news.groupby(by=['Ticker', 'Date'],
                                 dropna=False).count().reset_index()  # para que no mezcle las columnas del by
    count_news.columns = count_news.columns.str.replace('news_va',
                                                        'news_count')  # TODO cuidado con el el nombre coincida con otro cacho de columna , genera 2
    count_news = count_news.drop(columns=['news_t5', 'news_t5Be', 'news_fl','news_txtBlod'], errors='ignore')
    # df_news3 = df_news.groupby(by=['Ticker','Date'], dropna=False).mean().Time.transform('count')
    df_news = df_news.groupby(by=['Ticker', 'Date'], as_index=True,
                              dropna=False).mean().reset_index()  # para que no mezcle las columnas del by
    # df_news = df_news.drop_duplicates()

    df_news['news_va'] = df_news['news_va'].astype(float).map('{:,.3f}'.format).astype(
        float)  # df_news['news_va'].map(lambda x: '${:,.3f}'.format)
    df_news['news_fl'] = df_news['news_fl'].astype(float).map('{:,.3f}'.format).astype(float)
    df_news['news_t5'] = df_news['news_t5'].astype(float).map('{:,.3f}'.format).astype(float)
    df_news['news_t5Be'] = df_news['news_t5Be'].astype(float).map('{:,.3f}'.format).astype(float)
    df_news['news_txtBlod'] = df_news['news_txtBlod'].astype(float).map('{:,.3f}'.format).astype(float)


    df_news = pd.merge(df_news, count_news, on=['Ticker', 'Date'])
    df_news = df_news.sort_values('Date', ascending=False)
    # .map('${:,.3f}'.format)
    if is_print:
        df_news.to_csv("sentiment/stock_news_AVG_DATE_" + stock_id + ".csv", sep="\t")
    return df_news


# df_news = df_news[df_news['Date'].astype(str).str.contains("|".join(LIST_YEARS_ALLOW), na=False)]#TODO error si el dia es -20 -21 ..., es recogida la noticia
# df_news.to_csv("stock_news_DATE", sep="\t")
# TEST class
'''
df = None
for s in tickers:
    df1 = get_sentiment_full(s , True)
    if df is None:
        df = df1
    else:
        df = pd.merge(df, df1)

df.to_csv("sentiment/stock_news_FULL_.csv", sep="\t")
# View Data
#df_news['Date'] = pd.to_datetime(df_news.Date).dt.date
'''


def aux_foreach_stock(df_news):  # TODO Borrar
    global name, scores_flair, scores_t5, df
    unique_ticker = df_news['Ticker'].unique().tolist()
    news_dict = {name: df_news.loc[df_news['Ticker'] == name] for name in unique_ticker}
    # news.to_csv("sentiment_FULL_per.csv",sep="\t")
    # news_dict.to_csv("sentiment_FULL_per.csv",sep="\t")
    v_mean = []
    v_mean_flir = []
    v_mean_t5 = []
    for ticker in news_dict.keys():
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        # dataframe = dataframe.drop(columns=['Headline'])
        Logger.logr.info('\n')
        Logger.logr.info(dataframe.head())

        listDF = dataframe['compound']
        # print(listDF[3])
        listDF_F = dataframe['scores_flair']
        # print(listDF_F[3])
        scores_vader_comp_list = [x for x in dataframe['compound'] if -0.45 >= x or x >= 0.45]
        scores_flair = [x for x in dataframe['scores_flair'] if -97 >= x or x >= 97]
        scores_t5 = [x for x in dataframe['scores_t5'] if -97 >= x or x >= 97]
        # scores_flair = filter(lambda x: -0.97 >= (float(x)) >= 0.97, listDF_F)
        try:
            v_mean.append(round(mean(scores_vader_comp_list), 3))  # Media aritmética («promedio») de los datos.
            v_mean_flir.append(round(mean(scores_flair), 3))  # Media aritmética («promedio») de los datos.
            v_mean_t5.append(round(mean(scores_t5), 3))  # Media aritmética («promedio») de los datos.


        except Exception as e:
            Logger.logr.warn(str(e))
            pass
        # dataframe.to_csv("sentiment_"+ticker+".csv",sep="\t")
    Logger.logr.info("")
    lists = [news_dict.keys(), v_mean, v_mean_flir, v_mean_t5]
    df = pd.concat([pd.Series(x) for x in lists], axis=1)
    df.columns = ['Ticker', "v_mean", "v_mean_flir", "v_mean_t5"]
    # df = pd.DataFrame( zip(tickers,v_mean,v_median,v_median_grouped,v_mode,v_multimode,v_quantiles,v_pstdev,v_pvariance,v_stdev,v_variance,v_mean_flir,v_median_flir,v_median_grouped_flir,v_mode_flir,v_multimode_flir,v_quantiles_flir,v_pstdev_flir,v_pvariance_flir,v_stdev_flir,v_variance_flir) ,
    #                  columns=['Ticker','v_mean','v_median','v_median_grouped','v_mode','v_multimode','v_quantiles','v_pstdev','v_pvariance','v_stdev','v_variance','v_mean_flir','v_median_flir','v_median_grouped_flir','v_mode_flir','v_multimode_flir','v_quantiles_flir','v_pstdev_flir','v_pvariance_flir','v_stdev_flir','v_variance_flir'])
    df = df.set_index('Ticker')
    df = df.sort_values('v_mean', ascending=False)
    Logger.logr.info('\n')
    Logger.logr.info(df.head())
    df.to_csv("sentiment_FULL.csv", sep="\t")
