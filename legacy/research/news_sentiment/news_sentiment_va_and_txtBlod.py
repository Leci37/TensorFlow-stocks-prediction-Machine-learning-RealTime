from nltk.sentiment.vader import SentimentIntensityAnalyzer

import news_sentiment_flair
import news_sentiment_t5
from LogRoot.Logging import Logger

ANALYZAR_VA = SentimentIntensityAnalyzer()


def get_sentiment_va(df_score):
    scores_vader = df_score['Headline'].apply(ANALYZAR_VA.polarity_scores).tolist()
    return scores_vader

# https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524
from textblob import TextBlob
def get_sentiment_textBlod(text):
    text_object = TextBlob(text)

    sentiment_polarity = text_object.sentiment.polarity #Return the polarity score as a float within the range [-1.0, 1.0]
    subjectivity_polarity = text_object.sentiment.subjectivity #Return the subjectivity score as a float within the range [0.0, 1.0]
    # where 0.0 is very objective and 1.0 is very subjective.
    subjectivity_polarity = (subjectivity_polarity -1) * -1 #objetivo 1 subjetivo 0

    #return (sentiment_polarity * 10) + (subjectivity_polarity * sentiment_polarity*2) * 10
    return  sentiment_polarity *130 + (subjectivity_polarity * sentiment_polarity*10)



def get_sentiment_predictorS(df_score):
    '''
    A partir de los titulares dados , los analiza y obtiene una puntuacion de sentimiento para cada uno con 3 modelos:
    Vader SentimentIntensityAnalyzer Vader
    Flair sentiment_model = flair.Models.TextClassifier.load('en-sentiment')
    t5 #https://huggingface.co/cometrain/stocks-news-t5?text=Texas+Roadhouse+posts+strong+revenue+growth+despite+commodity+price+pressures

    :param df_score:
    :return:
    '''
    scores_vader = get_sentiment_va(df_score)
    scores_flair = df_score['Headline'].apply(news_sentiment_flair.get_sentiment_flair).tolist()
    # TODO si df_scores no tiene valores ??
    df_score['news_va'] = [s['compound'] * 140 for s in
                           scores_vader]  # scores_vader[7]['compound'] # 140 para tratar de evitar el esceso de valores entre 20 y 40  , es decir un 95 pasa a ser un 70
    df_score['news_fl'] = scores_flair

    # remove the non-concise results
    # df_score = df_score[
    #    ((-60 >= df_score.news_va) | (df_score.news_va >= 60)) | ((-96 >= df_score.news_fl) | (df_score.news_fl >= 96))]

    # t5 es más pesado solo se aplica a los que han dado bien en fl y va
    scores_t5 = df_score['Headline'].apply(news_sentiment_t5.get_sentiment_t5).tolist()
    df_score['news_t5'] = scores_t5

    scores_t5be = df_score['Headline'].apply(news_sentiment_t5.get_sentiment_t5Be).tolist()
    df_score['news_t5Be'] = scores_t5be

    scores_textBlod = df_score['Headline'].apply(get_sentiment_textBlod).tolist()
    df_score['news_txtBlod'] = scores_textBlod

    df_score['news_va'] = df_score['news_va'].astype(float).apply(lambda x: round(x,
                                                                                  3))  # df_score['news_va'] = df_score['news_va'].astype(float).map( '{:,.3f}'.format).astype(float)
    df_score['news_fl'] = df_score['news_fl'].astype(float).apply(lambda x: round(x, 3))
    df_score['news_t5'] = df_score['news_t5'].astype(float).apply(lambda x: round(x, 3))
    df_score['news_t5Be'] = df_score['news_t5Be'].astype(float).apply(lambda x: round(x, 3))
    df_score['news_txtBlod'] = df_score['news_txtBlod'].astype(float).apply(lambda x: round(x, 3))

    # df_score['news_fl'] = df_score['news_fl'].astype(float).map('{:,.3f}'.format).astype(float)
    # df_score['news_t5'] = df_score['news_t5'].astype(float).map('{:,.3f}'.format).astype(float)
    # df_score['news_t5Be'] = df_score['news_t5Be'].astype(float).map('{:,.3f}'.format).astype(float)

    cols = list(df_score.columns.values)
    cols.append(cols.pop(cols.index('Headline')))  # put at the end of the list
    df_score = df_score.reindex(columns=cols)
    Logger.logr.info(" Numbers of rows: "+ str(len(df_score.index)) )
    return df_score
