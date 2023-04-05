import flair
from LogRoot.Logging import Logger

sentiment_model = flair.models.TextClassifier.load('en-sentiment')


# https://towardsdatascience.com/sentiment-analysis-for-stock-price-prediction-in-python-bed40c65d178

def get_sentiment_flair(text):
    sentence = flair.data.Sentence(text)
    # sentiment_model.predict_istilroberta_finetuned_financial(sentence)
    sentiment_model.predict(sentence)
    score = float("%.6f" % sentence.score)
    score = score - ((
                                 1 - score) * 2.5)  # para tratar de evitar el esceso de valores entre 95 y 99  , es decir un 95 pasa a ser un 70
    if sentence.tag == "POSITIVE":
        score = score * 100
    elif sentence.tag == "NEGATIVE":
        score = score * -100
    else:
        Logger.logr.error(" is niether \"POSITIVE\" or  \"NEGATIVE\". It is: "+ sentence.tag)

    return score
