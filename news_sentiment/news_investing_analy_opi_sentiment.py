# Import libraries
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

from Utils import Utils_Yfinance
from LogRoot.Logging import Logger

# https://towardsdatascience.com/stock-news-sentiment-analysis-with-python-193d4b4378d4

ANALYZAR_VA = SentimentIntensityAnalyzer()


# url_stock = "https://www.investing.com/equities/amazon-com-inc-"

def extrat_opinion_headers(root):
    '''
    Obtiene los tituales de investing de una empresa
    :param root:
    :return:
    '''
    list_header = []
    list_date = []
    for a in range(1, 10):
        try:
            header_text = root.xpath('//*[@id="leftColumn"]/div[8]/article[' + str(a) + ']/div[1]/a')[0].text
            date_text = root.xpath('//*[@id="leftColumn"]/div[8]/article[' + str(a) + ']/div[1]/span/span[2]')[0].text
            p_text = root.xpath('//*[@id="leftColumn"]/div[8]/article[' + str(a) + ']/div[1]/p')[0].text
            # print(a, "Page: ", "p", "\t\t", header_text, "\t", date_text, "\t", p_text)
            list_header.append((header_text + "     " + p_text).replace("\n", " "))
            try:
                datetime_object = datetime.strptime(
                    date_text.replace("-", "").replace(",", "").replace(" ", "").replace(u'\xa0', u''),
                    "%b%d%Y")  # May-05-2022
                date_text = datetime_object.strftime("%Y-%m-%d")
            except TypeError:  # in case thay -15 hours ago
                date_text = datetime.today().strftime("%Y-%m-%d")
                pass
            except ValueError:  # in case thay -15 hours ago
                date_text = datetime.today().strftime("%Y-%m-%d")
                pass
            list_date.append(date_text)
        except IndexError:
            pass
    return list_header, list_date


def __get_finance_tabs_Analysis_Opinion(url, option, numbers_pages=10):
    # u = "opinion"
    url = url + option
    Logger.logr.info(url)
    # get_Params = {'q': postalCode, 'country': country}
    list_header = []
    list_date = []
    index_web = ""

    print("PAGEs: ", end="", flush=True)
    for p in range(1, numbers_pages):
        if p != 1:  # si es 1 no hay que poner el /2 en la url
            index_web = ("/" + str(p))

        root = Utils_Yfinance.get_GET_root_from_url(url + index_web)
        if root is None:
            Logger.logr.warn("ERROR company_profile in get_root_from_url")
            pass
        # for p in range(1, 8):
        h, d = extrat_opinion_headers(root)
        list_header = list_header + h  # lista mas lista
        list_date = list_date + d
        # after de add lists, because some of the data of the list could be not duplicated
        if len(list_header) > len(set(list_header)) + 9:  # si ahya mas de 9 duplicadas
            Logger.logr.info("INFO there are starting to be duplicates in the news query. no continued search  Page: "+
                             str(p)+ "url: "+ url)
            break

        print(" ", str(p), end=" ", flush=True)

    df_header = pd.DataFrame(columns=['Date', 'Headline'])
    df_header['Date'] = list_date
    df_header['Headline'] = list_header
    Logger.logr.info("exit")
    return df_header


def get_sentiment_investing_news_opinion(urlInvestingStock, getOpinion=True, getAnalysis=True):
    # url_stock = urlInvestingStock
    df_return = None
    if getOpinion:
        df_opinion = __get_finance_tabs_Analysis_Opinion(urlInvestingStock, "-opinion")
        if df_return is None:
            df_return = df_opinion
        else:
            df_return.append(df_opinion)
    if getAnalysis:
        df_news_inv = __get_finance_tabs_Analysis_Opinion(urlInvestingStock, "-news")
        if df_return is None:
            df_return = df_news_inv
        else:
            df_return.append(df_news_inv)

    df_return = df_return.drop_duplicates()
    df_return = df_return.sort_values('Date', ascending=False)
    Logger.logr.info(" StockUrl: "+ urlInvestingStock+ "  Numbers of news: "+ str(len(df_return.index)) )
    return df_return


'''
df = df_opinion.append(df_news_inv)
df = df.sort_values('Date', ascending=False)
print(df_news_inv.head())

df_pred = news_sentiment_va.get_sentiment_predictorS(df_news_inv)
df_pred.to_csv("sentiment/stock_newsInv_"+"AMZN"+".csv", sep="\t")
print(df_pred.head())
'''
# //*[@id="leftColumn"]/div[8]/article[8]/div[1]/a
# //*[@id="leftColumn"]/div[8]/article[2]/div[1]/a


# //*[@id="leftColumn"]/div[8]/article[1]/div[1]/a
# //*[@id="leftColumn"]/div[8]/article[9]/div[1]/a

Logger.logr.info("")
