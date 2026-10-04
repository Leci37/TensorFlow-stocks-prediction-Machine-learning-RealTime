import pandas as pd
import os.path
import re

from Utils import  UtilsL
import _KEYS_DICT

#https://www.iemoji.com/#?category=objects&version=36&theme=appl&skintone=default
BAR_UP = "📈"
BAR_DOWN = "📉"
BAR_SIMPLE = "📊"
FLECHA_UP = "⬆"
FLECHA_DOWN = "⬇"
def get_text_alert(type_b_s:_KEYS_DICT.Op_buy_sell):
    text_alert = " ERROR "
    flecha_simbol_stock = ""
    if type_b_s == _KEYS_DICT.Op_buy_sell.POS:
        text_alert = BAR_UP+" BUY "+BAR_UP
        #flecha_simbol = FLECHA_UP
    elif type_b_s == _KEYS_DICT.Op_buy_sell.NEG:
        text_alert = BAR_DOWN+" SELL "+BAR_DOWN
        #flecha_simbol = FLECHA_DOWN
    return text_alert, flecha_simbol_stock,





URL_INVESTING = "https://www.investing.com/search/?q=" #https://www.investing.com/search/?q=RIVN
URL_WE_BULL = "https://www.webull.com/quote/nasdaq-" # "https://www.webull.com/quote/nasdaq-meli/news"
def get_string_alert_message(S, dict_pred, modles_evaluated,  type_b_s, date_detect, value_detect):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves = URL_INVESTING + S
    url_info_webull = URL_WE_BULL + S.lower()

    s88, s93, s95, stf, names_models_r = get_fraciones_afirmativos_results(S, dict_pred, modles_evaluated, type_b_s)

    # **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>"+dict_pred['Date'] + "</em>" + (flecha_simbol*1) +"\nValue:<pre> "+ "{:.2f}".format(value_detect)+"</pre>\n"
    s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n  "
    s4 = "88%: " + s88 + "\t  93%: " + s93 + "\n  95%: " +  s95 + "\t  TF: " + stf + "\n\n"
    s5 = "📊Model names: <pre>" + names_models_r + "</pre>\n"
    s6 = "<em>Date alert: "+date_detect + "</em>"

    alert_message_html =  s1 + s2 + s3 + s4 + s5 + s6
    alert_message_without_tags = UtilsL.clean_html_tags(alert_message_html).replace('WeBull.com', '').replace("Confidence of models:", '').replace('Investing.com', '').replace('\t', ' ') #.replace('\n',' ')

    return  alert_message_html , alert_message_without_tags

DICT_SCORE_RATE = {
    "0" : 90,
    "1" : 95,
    "2" : 185,
    "3" : 196,
    "4" : 400
}
DF_INVESTING = pd.read_csv("Utils/URL_investing_Stocks_lins_news.csv", index_col=0, sep='|')
def get_invesing_url(S):
    s3_aux_1 = ""
    if len(DF_INVESTING[DF_INVESTING['STOCK'] == S]['URL']) > 0:
        url_info_inves = DF_INVESTING[DF_INVESTING['STOCK'] == S]['URL'].values[0]
    else:
        url_info_inves = ""
    return url_info_inves, s3_aux_1


df_RESULT = None

# URL_INVESTING = "https://www.investing.com/search/?q=" #https://www.investing.com/search/?q=RIVN
URL_WE_BULL = "https://www.webull.com/quote/nasdaq-" # "https://www.webull.com/quote/nasdaq-meli/news"
def get_MULTI_string_alert_message(S, dictR, type_b_s:_KEYS_DICT.Op_buy_sell, list_model:list, list_png:list, url_trader_view ):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves, s3_aux_1 = get_invesing_url(S)
    url_info_webull = URL_WE_BULL + S.lower()
    url_info_trader_view = url_trader_view

    # s88, s93, s95, stf, names_models_r = get_fraciones_afirmativos_results(S, dict_pred, modles_evaluated, type_b_s)
    global df_RESULT
    if df_RESULT is None:
        df_RESULT = pd.read_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", index_col=0, sep='\t')
    # df_RESULT[ [x for x in df_RESULT.columns if not x.endswith("_per") and x.startswith("TFm_"+S +"_"+type_b_s.value) ]].loc["df_test_%"]
    list_model_per_result = [(x.replace("Acert_TFm_", '') + ": " + str(dictR[x])) + "% |"
                             + str(int(df_RESULT[x.replace("Acert_TFm_", 'TFm_')].loc["df_test_%"]))  for x in list_model]

    # **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    # list_model_per_result = [(x.replace("Acert_TFm_", '')+": " +str(dictR[x])) +"% " for x in list_model ]
    percentage_visible_POS = (dictR['POS_score'][0] * 90) / DICT_SCORE_RATE[dictR['POS_num'][0].replace(']', '')]
    percentage_visible_NEG = (dictR['NEG_score'][0] * 90) / DICT_SCORE_RATE[dictR['NEG_num'][0].replace(']', '')]

    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>" + str(dictR['Date']) + "</em>" + (flecha_simbol * 1) + "-𝘓𝘰𝘯𝘥𝘰𝘯\n☝Value:<pre> " + str(dictR['Close']) + "</pre>\n"
    s2 = "<em>" + str(dictR['Date']) + "</em>" + (flecha_simbol * 1) + "-𝘓𝘰𝘯𝘥𝘰𝘯\n☝Value:<pre> " + str(dictR['Close']) + "</pre>\n"
    # s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n"
    if url_info_inves != "":
        s3_aux_1 = "News: <a href=\"" + url_info_inves + "\">Investing.com</a>\n"
    s3 = s3_aux_1 + "<a href=\"" + url_info_trader_view + "\">TraderView.com</a>\n\n<b><i>Confidence of models:</i></b>\n"
    # s4 = "\t   POS_score: " + str(dictR['POS_score']) + "%/"+str(dictR['POS_num'])+"\n\t   NEG_score: " + str(dictR['NEG_score']) + "%/"+str(dictR['NEG_num'])+"\n"
    s4 = "\t\t⬆POS_score: " + "{:.1f}".format(percentage_visible_POS) + "%/" + str(dictR['POS_num']) + "\n\t\t⬇NEG_score: " + "{:.1f}".format(percentage_visible_NEG) + "%/" + str(dictR['NEG_num']) + "\n"
    s5 = "<b><i>📊⚙Model names:</i></b><pre>\n\t\t⦁" + "\n\t\t⦁".join(list_model_per_result).replace(S+"_"+type_b_s.value+"_",'') + "</pre>"
    # s6 = "📊Percentage: <pre>\n\t" + "POS: "+"{:.1f}".format(percentage_visible_POS)+ "%\tNEG: "+"{:.1f}".format(percentage_visible_NEG)+  "%</pre>"

    alert_message_html =  (s1 + s2 + s3 + s4 + s5).replace('[', '').replace(']', '').replace('\'', '').replace('__', '_')
    alert_message_without_tags = UtilsL.clean_html_tags(alert_message_html).replace('WeBull.com', '').replace('Investing.com', url_info_inves).replace('TraderView.com', url_info_trader_view)#.replace(": ", ":")
    # alert_message_without_tags = alert_message_without_tags#.replace('\t', ' ').replace('\n', ' ')

    return  alert_message_html , alert_message_without_tags

# URL_INVESTING = "https://www.investing.com/search/?q=" #https://www.investing.com/search/?q=RIVN
URL_WE_BULL = "https://www.webull.com/quote/nasdaq-" # "https://www.webull.com/quote/nasdaq-meli/news"
def get_MULTI_string_alert_message(S, dictR, type_b_s:_KEYS_DICT.Op_buy_sell, list_model:list, list_png:list, url_trader_view ):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves, s3_aux_1 = get_invesing_url(S)
    url_info_webull = URL_WE_BULL + S.lower()
    url_info_trader_view = url_trader_view
    # **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    # list_model_per_result = [(x.replace("Acert_TFm_", '')+": " +str(dictR[x])) +"% " for x in list_model ]
    percentage_visible_POS = (dictR['POS_score'][0] * 90) / DICT_SCORE_RATE[dictR['POS_num'][0].replace(']', '')]
    percentage_visible_NEG = (dictR['NEG_score'][0] * 90) / DICT_SCORE_RATE[dictR['NEG_num'][0].replace(']', '')]

    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>" + str(dictR['Date']) + "</em>" + (flecha_simbol * 1) + "-𝘓𝘰𝘯𝘥𝘰𝘯\n☝Value:<pre> " + str(dictR['Close']) + "</pre>\n"
    # s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n"
    if url_info_inves != "":
        s3_aux_1 = "News: <a href=\"" + url_info_inves + "\">Investing.com</a>\n"
    s3 = s3_aux_1 + "<a href=\"" + url_info_trader_view + "\">TraderView.com</a>\n\n<b><i>Confidence of models:</i></b>\n"
    # s4 = "\t   POS_score: " + str(dictR['POS_score']) + "%/"+str(dictR['POS_num'])+"\n\t   NEG_score: " + str(dictR['NEG_score']) + "%/"+str(dictR['NEG_num'])+"\n"
    s4 = "\t\t⬆POS_score: " + "{:.1f}".format(percentage_visible_POS) + "%/" + str(dictR['POS_num']) + "\n\t\t⬇NEG_score: " + "{:.1f}".format(percentage_visible_NEG) + "%/" + str(dictR['NEG_num']) + "\n"
    s5 = "<b><i>📊⚙Model names:</i></b><pre>\n\t\t⦁" + "\n\t\t⦁".join(list_model_per_result).replace(S+"_"+type_b_s.value+"_",'') + "</pre>"
    # s6 = "📊Percentage: <pre>\n\t" + "POS: "+"{:.1f}".format(percentage_visible_POS)+ "%\tNEG: "+"{:.1f}".format(percentage_visible_NEG)+  "%</pre>"

    alert_message_html =  (s1 + s2 + s3 + s4 + s5).replace('[', '').replace(']', '').replace('\'', '').replace('__', '_')
    alert_message_without_tags = UtilsL.clean_html_tags(alert_message_html).replace('WeBull.com', '').replace('Investing.com', url_info_inves).replace('TraderView.com', url_info_trader_view)#.replace(": ", ":")
    # alert_message_without_tags = alert_message_without_tags#.replace('\t', ' ').replace('\n', ' ')

    return  alert_message_html , alert_message_without_tags



from datetime import datetime, timedelta
DATE_FORMAT_STR = '%Y-%m-%d %H:%M:%S'
def add_two_hours(date_str):
    origin_date = datetime.strptime(str(date_str).replace("[Timestamp(\'", "").replace("\')]", ""), DATE_FORMAT_STR)
    three_hour_later = origin_date + timedelta(hours=2)
    return datetime.strftime(three_hour_later, DATE_FORMAT_STR)

# URL_INVESTING = "https://www.investing.com/search/?q=" #https://www.investing.com/search/?q=RIVN
URL_WE_BULL = "https://www.webull.com/quote/nasdaq-" # "https://www.webull.com/quote/nasdaq-meli/news"
def get_MULTI_W3_string_alert_message(S, dictR, type_b_s:_KEYS_DICT.Op_buy_sell, list_model:list, list_png:list, url_trader_view ):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves, s3_aux_1 = get_invesing_url(S)
    url_info_webull = URL_WE_BULL + S.lower()
    url_info_trader_view = url_trader_view
    # **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    # list_model_per_result = [(x.replace("Acert_TFm_", '')+": " +str(dictR[x])) +"% " for x in list_model ]
    dictR['score_0'][0] = round( float(dictR['score_0'][0]), 3)
    dictR['score_1'][0] = round( float(dictR['score_1'][0]), 3)
    dictR['score_2'][0] = round( float(dictR['score_2'][0]), 3)
    percentage_visible_POS = (dictR['score_1'][0] * 100) # / DICT_SCORE_RATE[dictR['POS_num'][0].replace(']', '')]
    percentage_visible_NEG = (dictR['score_2'][0] * 100) #/ DICT_SCORE_RATE[dictR['NEG_num'][0].replace(']', '')]

    # per_sta_accura, per_sta_neg, per_sta_pos = extrac_stadistics_info(S)

    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>" + add_two_hours( dictR['Date'] ) + "</em>" + (flecha_simbol * 1) + "-𝘗𝘢𝘳𝘪𝘴\n☝Value:<pre> " + str(dictR['close']) + "</pre>\n"
    # s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n"
    if url_info_inves != "":
        s3_aux_1 = "News: <a href=\"" + url_info_inves + "\">Investing.com</a>\n"
    s3 = s3_aux_1 + "<a href=\"" + url_info_trader_view + "\">TraderView.com</a>\n\n<b><i>Confidence of models:</i></b>\n"
    # s4 = "\t   POS_score: " + str(dictR['POS_score']) + "%/"+str(dictR['POS_num'])+"\n\t   NEG_score: " + str(dictR['NEG_score']) + "%/"+str(dictR['NEG_num'])+"\n"
    # s4 = "\t\t⬆POS_score: " + "{:.2f}".format(percentage_visible_POS) + "%"  + "\n\t\t⬇NEG_score: " + "{:.2f}".format(percentage_visible_NEG) + "%\n"
    # s4 = "\t\t⬆POS_score: " + "{:.2f}".format(percentage_visible_POS) + "%|" +per_sta_pos.group(1)+  "\n\t\t⬇NEG_score: " + "{:.2f}".format( percentage_visible_NEG) + "%|" +per_sta_neg.group(1)+ "\n"
    s4 = "\t\t⬆POS_score: " + "{:.2f}".format(percentage_visible_POS) + "%\t|f1: " +str(dictR['f1_score_1'][0])+  "\n\t\t⬇NEG_score: " + "{:.2f}".format( percentage_visible_NEG) + "%\t|f1: " +str(dictR['f1_score_2'][0])+ "\n"

    s5 = "" # "<b><i>📊⚙Model names:</i></b>" #<pre>\n\t\t⦁"   + "\n\t\t⦁".join(list_model_per_result).replace(S+"_"+type_b_s.value+"_",'') + "</pre>"
    # s6 = "📊Percentage: <pre>\n\t" + "POS: "+"{:.1f}".format(percentage_visible_POS)+ "%\tNEG: "+"{:.1f}".format(percentage_visible_NEG)+  "%</pre>"
    # s6 = "\t\t📊⚙Model quality: "+str(float(per_sta_accura.group(1))*100)+"%"
    s6 = "\t\t📊⚙Model quality: " + str(round( float(dictR['f1_buy_sell_score'][0] * 100), 2)) + "%\n"  + r'<pre><code class="language-python">' + str(dictR ).replace(", \'", "\n").replace("'", "").replace(": ", ":\t\t\t") + "</code></pre>"  #Code block

    alert_message_html =  (s1 + s2 + s3 + s4 + s5+s6).replace('[', '').replace(']', '').replace('\'', '').replace('__', '_').replace("(", '').replace(")", '').replace("Timestamp", '')
    alert_message_without_tags = UtilsL.clean_html_tags(alert_message_html).replace('WeBull.com', '').replace('Investing.com', url_info_inves).replace('TraderView.com', url_info_trader_view)#.replace(": ", ":")
    # alert_message_without_tags = alert_message_without_tags#.replace('\t', ' ').replace('\n', ' ')

    return  alert_message_html , alert_message_without_tags
#TODO move to CONFIG file
PATH_MODEL = r"E:\TradeDL/"
REF_MODEL = "_3W5_"+"_tut_A1_"
def extrac_stadistics_info(S):

    print(PATH_MODEL + f'outputs/model_info/{S}_{REF_MODEL}_.info')
    text_info = open(PATH_MODEL + f'outputs/model_info/{S}_{REF_MODEL}_.info', "r").read()
    per_sta_pos = re.search(r" {4,}1 {4,}(0.\d\d) {4,}(0.\d\d)", text_info)
    per_sta_neg = re.search(r" {4,}2 {4,}(0.\d\d) {4,}(0.\d\d)", text_info)  #
    per_sta_accura = re.search(r" {3,}accuracy {14,}(0.\d\d)", text_info)
    return per_sta_accura, per_sta_neg, per_sta_pos


def get_fraciones_afirmativos_results(S, dict_pred, modles_evaluated, type_b_s):
    names_models_r = ""
    count_TF_models = 0
    modles_evaluated.sort()
    for k in modles_evaluated:
        if dict_pred[k] == 1:  # dict_predict_1[k] == 1:
            names_models_r = names_models_r + k.replace('br_', '').replace("_" + type_b_s.sub_dict, '').replace('__', '_').replace("_" + S, '') + '%, '
        if k.startswith('br_TF') and k.endswith('_93'):
            count_TF_models = count_TF_models + 1
    s88 = str(dict_pred['sum_r_88']) + "/" + str(dict_pred['num_models'])
    s93 = str(dict_pred['sum_r_93']) + "/" + str(dict_pred['num_models'])
    s95 = str(dict_pred['sum_r_95']) + "/" + str(dict_pred['num_models'])
    stf = str(dict_pred['sum_r_TF']) + "/" + str(int(count_TF_models))
    return  s88, s93, s95, stf , names_models_r

def register_in_zTelegram_Registers(S, dict_predict, modles_evaluated, type_b_s , path = _KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME):
    s88, s93, s95, stf, names_models_r = get_fraciones_afirmativos_results(S, dict_predict, modles_evaluated, type_b_s)
    df_res = pd.DataFrame([[dict_predict['Date'], S, type_b_s.name, "{:.1f}".format(dict_predict['Close']),"{:.2f}".format(dict_predict['Volume']), s88, s93, s95, stf, names_models_r]],
                          columns=['Date', 'Stock', 'buy_sell','Close', 'Volume','88%', '93%', '95%', 'TF%', "Models_names"])
    if os.path.isfile(path):
        df_res.to_csv(path, sep="\t", index=None, mode='a', header=False)
    else:
        df_res.to_csv(path, sep="\t", index=None)
        print("Created : " + path)


def register_MULTI_in_zTelegram_Registers(S, df_r, path = _KEYS_DICT.PATH_REGISTER_RESULT_REAL_TIME):
    if os.path.isfile(path):
        df_r.to_csv(path, sep="\t", mode='a', header=False)
    else:
        df_r.to_csv(path, sep="\t")
        print("Created MULTI : " + path)

USER_ID_RE_SU_MADC_KONDORDE ="1q24vjd7"
USER_ID_QUOKA ="x0nIfdrH"
LIST_NYSE_STOCKS = ['U','NVST','TWLO','UBER','SNOW','SPOT','HUBS', 'GME', 'EPAM', 'ASAN', 'DOCN','RBLX', 'SHOP', 'NIO','SPG','STAG' , 'PINS']
def get_traderview_url(S):
    if S.endswith("-USD"):
        url = "https://es.tradingview.com/chart/"+USER_ID_RE_SU_MADC_KONDORDE+"/?symbol=BITSTAMP%3A" + S.replace("-USD", "") + "USD"
    elif S in LIST_NYSE_STOCKS:
        url = "https://www.tradingview.com/chart/"+USER_ID_RE_SU_MADC_KONDORDE+"/?symbol=NYSE%3A" + S
    else:
        url = "https://www.tradingview.com/chart/"+USER_ID_RE_SU_MADC_KONDORDE+"/?symbol=NASDAQ%3A" + S

    return url


html_example = '''<b>bold</b>
<strong>bold</strong>
<i>italic</i>
<em>italic</em>
<a href="http://www.example.com/">inline URL</a>
<code>inline fixed-width code</code>
<pre>pre-formatted fixed-width code block</pre>'''