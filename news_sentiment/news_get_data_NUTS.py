import pandas as pd

from Utils import UtilsL, Utils_Yfinance
import news_investing_analy_opi_sentiment
import news_sentiment_va_and_txtBlod
from news_sentiment import get_sentiment_news_finviz, get_avg_newsOp
from LogRoot.Logging import Logger




def get_news_sentiment_data(stockid):
    df_news = pd.DataFrame(columns=['Ticker', 'Date', 'Headline']).astype(float)
    Logger.logr.info("get_sentiment_data: "+ stockid)
    # TODO API investpy  suspended https://github.com/alvarobartt/investpy
    # u = UtilsL.Url_stocks_pd.get_url(stockid)
    # if u is None:
    #     Logger.logr.info("The company stock dont have linked url.  stock: "+ stockid)
    #     return None
    #
    # # INVESTING.com
    # Logger.logr.info("get_sentiment_data: "+ stockid+ " url_investing: "+ u)
    # df_investing_news_opinion = news_investing_analy_opi_sentiment.get_sentiment_investing_news_opinion(u)
    # if df_investing_news_opinion is not None:
    #     df_investing_news_opinion['Ticker'] = stockid
    #     df_news = pd.merge(df_news, df_investing_news_opinion, on=['Ticker', 'Date', 'Headline'],
    #                        how='outer')  # TODO error in case of be None in the next merge
    # else:
    #     Logger.logr.warn("the dataframe News is None. DataFrame: INVESTING.com Stockid:"+ stockid)

    # YAHOO.com
    df_yahoo_news = Utils_Yfinance.get_news_opinion_yahoo(stockid)
    if df_yahoo_news is not None:
        df_yahoo_news['Ticker'] = stockid
        df_news = pd.merge(df_news, df_yahoo_news, on=['Ticker', 'Date', 'Headline'], how='outer')
    else:
        Logger.logr.warn("the dataframe News is None. DataFrame: YAHOO.com Stockid: " + stockid)

    # FINVIZ.COM
    df_finviz_news_opinion = get_sentiment_news_finviz(stockid)
    if df_finviz_news_opinion is not None:
        df_finviz_news_opinion = df_finviz_news_opinion.drop(columns=['Time'])
        df_news = pd.merge(df_news, df_finviz_news_opinion, on=['Ticker', 'Date', 'Headline'],
                           how='outer')  # df_investing_news_opinion.append(df_finviz_news_opinion)
    else:
        Logger.logr.warning(" The dataframe News is None. DataFrame: FINVIZ.com Stockid: "+ stockid)

    df_news = UtilsL.change_date_in_weekend_monday(df_news)

    # Get sentiment
    df_news = news_sentiment_va_and_txtBlod.get_sentiment_predictorS(df_news)
    df_news.to_csv("d_sentiment/stock_news_DATE_" + stockid + ".csv", sep="\t")

    df_avg = get_avg_newsOp(df_news, stockid)

    df = df_avg.sort_values('Date', ascending=False)

    firstRowDate = (df.iloc[[0]])['Date'].values[0]  # obtener la primera y la ultima fecha del df
    lastRowDate = (df.iloc[[-1]])['Date'].values[0]
    get_stocks_values_with_news = False
    if get_stocks_values_with_news:
        df_val_stocks = Utils_Yfinance.get_stock_value(stockid, lastRowDate, firstRowDate)
        df = pd.merge(df, df_val_stocks, on=['Ticker', 'Date'], how='outer')
    df.to_csv("d_sentiment/stock_news_DATE_AVG_" + stockid + ".csv", sep="\t")

    return df


def get_json_news_sentimet(stockId):
    global df
    df_aux = get_news_sentiment_data(stockId)
    if df_aux is None:  # si hay fallo en la empres a pasa a la siguiente
        Logger.logr.error(" with stock search, pass to the next. stock:  " + stockId)
        return
    elif df is None:
        df = df_aux
    else:
        # df2 = pd.merge(df2, df_aux, on=['Ticker', 'Date'])
        df = df.append(df_aux)
    df_news = df_aux
    df_news = df_news.drop('Ticker', 1)
    df_news.drop_duplicates(inplace=True)
    df_news = df_news.sort_values('Date', ascending=False)
    # df_full['Date'] = df_full['Date'].dt.strftime('%Y-%m-%d')
    df_news.to_csv("d_sentiment/" + stockId + "_stock_news.csv", sep="\t")

    #TO JSON
    df_news = df_news.groupby('Date', as_index=True).agg(list)
    dict_j = df_news.to_dict('index')  # df.to_dict('records') .to_dict('index'
    dict_j = {k: {a: b for a, b in v.items() if (not pd.isna(b).all() and not pd.isnull(b).all())} for k, v in
              dict_j.items()}
    dict_j = UtilsL.replace_list_in_sub_keys_dicts(dict_j)
    dict_json = {}
    #dict_json['Date'] = dict_j
    dict_json =dict_j
    import json
    with open("d_sentiment/" + stockId + "_stock_news.json", 'w') as fp:
        json.dump(dict_json, fp, allow_nan=True)
    Logger.logr.info("d_sentiment/" + stockId + "_stock_news.json  Numbres of Keys: " + str(len(dict_json)))




tickers = ["AMD", "GE", "BA", "AMZN", "AAPL", "PYPL", "MELI"]
tickers = ['MELI']#["MELI" ,"VWDRY"]

# def ggggg():
Logger.logr.info("start")
df = None

for s in tickers:
    Logger.logr.info("DO: "+ s)
    get_json_news_sentimet(s)

df.to_csv("d_sentiment/stock_news_DATE_FULL_.csv", sep="\t")
Logger.logr.info("Sentiment news filr  d_sentiment/stock_news_DATE_FULL_.csv")
