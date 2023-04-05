import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, InputFeatures, \
    AutoModel, AutoModelForMaskedLM, AutoConfig, RobertaModel, RobertaTokenizer, RobertaConfig, \
    RobertaForSequenceClassification
import nltk
from LogRoot.Logging import Logger
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# MODEL T5 https://huggingface.co/cometrain/stocks-news-t5?text=Texas+Roadhouse+posts+strong+revenue+growth+despite+commodity+price+pressures
tokenizer_t5 = AutoTokenizer.from_pretrained("cometrain/stocks-news-t5")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("cometrain/stocks-news-t5")

# MODEL BER https://huggingface.co/ProsusAI/finbert
tokenizer_ber = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_ber = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
# from finbert.utils import *
import numpy as np
import logging

from transformers import pipeline

RESULT_SENTIMENT = {
    "positive": 100,
    "negative": -100,  # compensar el prejuicio a favor de positive TODO
    "neutral": 0
}


class InputExample(object):
    """
    https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/finbert/utils.py#L27
    A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


def chunks(l, n):
    """
    https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/finbert/utils.py#L27
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


class InputFeatures(object):
    """
    A single set of features for the data.
    https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/finbert/utils.py#L27
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree


def get_sentiment_Rob(text):  # TODO pendiente de la respuesta del twet
    # MODEL ROB https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
    tokenizer_rob = RobertaTokenizer.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    # tokenizer_rob = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    # model_rob = AutoModel.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    encoder_config_rob = RobertaConfig.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    encoder_config_rob.is_decoder = True
    model_rob = RobertaModel.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                                             config=encoder_config_rob)
    model = RobertaForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", config=encoder_config_rob,
        problem_type="multi_label_classification", num_labels=encoder_config_rob.num_labels)
    # num_labels = len(model_rob.config.id2label)
    # model_rob = RobertaForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", num_labels=num_labels)
    # tokenizer_Rob = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    # model_Rob_ForMaskedLM = AutoModelForMaskedLM.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    inputs = tokenizer_rob(text, return_tensors="pt")
    outputs = model_rob(**inputs)
    # https: // github.com / mrm8488 / shared_colab_notebooks / blob / master / T5_wikiSQL_with_HF_transformers.ipynb
    Logger.logr.debug(encoder_config_rob)

    inputs = tokenizer_rob("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]

    predicted_class_id = logits.argmax().item()
    aa = model_rob.config.id2label[predicted_class_id]

    # model_rob.config.id2label
    # prediction_logits = outputs.logits

    inputs = tokenizer_rob(text, padding='longest', max_length=64, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model_rob.generate(input_ids, attention_mask=attention_mask, max_length=64)

    return tokenizer_rob.decode(output[0], skip_special_tokens=True)
    prediction = None
    sentiment_score = None
    # classifier = pipeline("sentiment-analysis", model=model_rob, tokenizer=tokenizer_rob)
    # r = classifier(text)
    # labels = torch.tensor([1]).unsqueeze(0)
    # inputs = tokenizer_rob(text, return_tensors="pt")
    # outputs = model_rob(**inputs, labels=labels)

    # unmasker = pipeline('fill-mask', model=model_Rob_ForMaskedLM, tokenizer=tokenizer_rob)
    # unmasker("the patient is a 55 year old [MASK] admitted with pneumonia")

    # inputs = tokenizer_rob(text, padding='longest', max_length=64, return_tensors='pt')
    # input_ids = inputs.input_ids
    # attention_mask = inputs.attention_mask
    # output = model_rob.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)

    # aa =  tokenizer_rob.decode(output[0], skip_special_tokens=True)
    # print([last_hidden_state, pooled_output])


def get_sentiment_t5Be(text):
    '''
    https://github.com/ProsusAI/finBERT/tree/44995e0c5870c4ab37a189d756550654ae87cdf0
    :param text:
    :return:
    '''

    prediction, sentiment_score = predict_ber(text.replace(".", " "), model_ber)  # if put "." are two sentences
    return sentiment_score[0] * 100


def get_sentiment_t5(text):
    tokenized_text = tokenizer_t5(text.replace(".", " "), return_tensors="pt", truncation=True)  # , padding="longest")
    # print(tokenized_text)

    summaries = model_t5.generate(
        # https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py
        input_ids=tokenized_text.input_ids,
        attention_mask=tokenized_text.attention_mask,
        # output_scores=True,
        # return_dict_in_generate=True
    )
    # summaries['sequences'],
    dec = tokenizer_t5.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print("@@@@@", dec[0], "\t000 ", "".join(summaries['scores'][0]), ' 111',"\t", "".join(summaries['scores'][1]))
    # print("t5: ", dec[0],"\t",  summaries, text)
    return RESULT_SENTIMENT[dec[0]]


def predict_ber(text, model, write_to_csv=False, path=None, use_gpu=False,
                gpu_name='cuda:0', batch_size=5):
    """
    https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/finbert/finbert.py
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.
    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    use_gpu: (optional): bool
        enables inference on GPU
    gpu_name: (optional): string
        multi-gpu support: allows specifying which gpu to use
    batch_size: (optional): int
        size of batching chunks
    """
    model.eval()

    sentences = sent_tokenize(text)

    device = gpu_name if use_gpu and torch.cuda.is_available() else "cpu"
    # logging.info("Using device: %s " % device)
    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])
    for batch in chunks(sentences, batch_size):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer_ber)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        with torch.no_grad():
            model = model.to(device)

            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            #logging.info(logits)
            logits = softmax(np.array(logits.cpu()))
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score': sentiment_score}

            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    # if write_to_csv:
    #    result.to_csv(path, sep=',', index=False)

    return result['prediction'], result['sentiment_score']


def convert_examples_to_features(examples, label_list, max_seq_length, tokeni_be, mode='classification'):
    """
    https://github.com/ProsusAI/finBERT/blob/44995e0c5870c4ab37a189d756550654ae87cdf0/finbert/utils.py#L118
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.
    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokeni_be: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'
    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokeni_be.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                          len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokeni_be.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding

        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        # if ex_index < 1:
        # print("*** Example ***")
        # print("guid: %s" % (example.guid))
        # print("tokens: %s" % " ".join(
        #    [str(x) for x in tokens]))
        # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        # print(
        #    "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        # print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          agree=agree))
    return features


# TEST class
get_sentiment_t5Be("I am happy and positive")
get_sentiment_t5Be("the company grew and earned more than expected on the stock exchange ")
get_sentiment_t5Be("the company declined and lost money  ")
get_sentiment_t5Be(
    "	Spotify falls 11% on first quarter earnings despite beat on top and bottom   Spotify, which has heavily invested in its podcasting business and is trying to grow ads in the space, said ad-supported revenue came in at 282 million euros.")
get_sentiment_t5Be(
    "S&P 500 Advances as Twitter Leads Tech Higher   By Yasin Ebrahim Investing.com – The S&P 500 rose Monday as a Twitter-fueled jump in d_tech helped the broader market start the week on the front foot  The S&P 500 rose 0.7%, the Dow Jones Industrial..")
