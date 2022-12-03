import pandas as pd
import os.path

from Utils import  UtilsL
import a_manage_stocks_dict

#https://www.iemoji.com/#?category=objects&version=36&theme=appl&skintone=default
BAR_UP = "📈"
BAR_DOWN = "📉"
BAR_SIMPLE = "📊"
FLECHA_UP = "⬆"
FLECHA_DOWN = "⬇"
def get_text_alert(type_b_s:a_manage_stocks_dict.Op_buy_sell):
    text_alert = " ERROR "
    flecha_simbol = ""
    if type_b_s == a_manage_stocks_dict.Op_buy_sell.POS:
        text_alert = BAR_UP+" BUY "+BAR_UP
        #flecha_simbol = FLECHA_UP
    elif type_b_s == a_manage_stocks_dict.Op_buy_sell.NEG:
        text_alert = BAR_DOWN+" SELL "+BAR_DOWN
        #flecha_simbol = FLECHA_DOWN
    return text_alert, flecha_simbol





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

URL_INVESTING = "https://www.investing.com/search/?q=" #https://www.investing.com/search/?q=RIVN
URL_WE_BULL = "https://www.webull.com/quote/nasdaq-" # "https://www.webull.com/quote/nasdaq-meli/news"
def get_MULTI_string_alert_message(S, dictR, type_b_s:a_manage_stocks_dict.Op_buy_sell, list_model):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves = URL_INVESTING + S
    url_info_webull = URL_WE_BULL + S.lower()

    # s88, s93, s95, stf, names_models_r = get_fraciones_afirmativos_results(S, dict_pred, modles_evaluated, type_b_s)

    # **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>" + str(dictR['Date']) + "</em>" + (flecha_simbol * 1) + "\nValue:<pre> " + str(dictR['Close']) + "</pre>\n"
    s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n"
    s4 = "\t   POS_score: " + str(dictR['POS_score']) + "%/"+str(dictR['POS_num'])+"\n\t   NEG_score: " + str(dictR['NEG_score']) + "%/"+str(dictR['NEG_num'])+"\n"
    s5 = "📊Model names: <pre>\n\t" + "\n\t".join(list_model).replace("Acert_TFm_",'') + "</pre>"

    alert_message_html =  (s1 + s2 + s3 + s4 + s5).replace('[', '').replace(']', '').replace('\'', '')
    alert_message_without_tags = UtilsL.clean_html_tags(alert_message_html).replace('WeBull.com', '').replace("Confidence of models:", '').replace('Investing.com', '')
    alert_message_without_tags = alert_message_without_tags.replace('\t', ' ').replace('\n', ' ')

    return  alert_message_html , alert_message_without_tags


def get_fraciones_afirmativos_results(S, dict_pred, modles_evaluated, type_b_s):
    names_models_r = ""
    count_TF_models = 0
    modles_evaluated.sort()
    for k in modles_evaluated:
        if dict_pred[k] == 1:  # dict_predict_1[k] == 1:
            names_models_r = names_models_r + k.replace('br_', '').replace("_" + type_b_s.value, '').replace('__','_').replace("_" + S, '') + '%, '
        if k.startswith('br_TF') and k.endswith('_93'):
            count_TF_models = count_TF_models + 1
    s88 = str(dict_pred['sum_r_88']) + "/" + str(dict_pred['num_models'])
    s93 = str(dict_pred['sum_r_93']) + "/" + str(dict_pred['num_models'])
    s95 = str(dict_pred['sum_r_95']) + "/" + str(dict_pred['num_models'])
    stf = str(dict_pred['sum_r_TF']) + "/" + str(int(count_TF_models))
    return  s88, s93, s95, stf , names_models_r

def register_in_zTelegram_Registers(S, dict_predict, modles_evaluated, type_b_s , path = a_manage_stocks_dict.PATH_REGISTER_RESULT_REAL_TIME):
    s88, s93, s95, stf, names_models_r = get_fraciones_afirmativos_results(S, dict_predict, modles_evaluated, type_b_s)
    df_res = pd.DataFrame([[dict_predict['Date'], S, type_b_s.name, "{:.1f}".format(dict_predict['Close']),"{:.2f}".format(dict_predict['Volume']), s88, s93, s95, stf, names_models_r]],
                          columns=['Date', 'Stock', 'buy_sell','Close', 'Volume','88%', '93%', '95%', 'TF%', "Models_names"])
    if os.path.isfile(path):
        df_res.to_csv(path, sep="\t", index=None, mode='a', header=False)
    else:
        df_res.to_csv(path, sep="\t", index=None)
        print("Created : " + path)


def register_MULTI_in_zTelegram_Registers(S, df_r, path = a_manage_stocks_dict.PATH_REGISTER_RESULT_REAL_TIME):
    if os.path.isfile(path):
        df_r.to_csv(path, sep="\t", index=None, mode='a', header=False)
    else:
        df_r.to_csv(path, sep="\t", index=None)
        print("Created MULTI : " + path)




html_example = '''<b>bold</b>
<strong>bold</strong>
<i>italic</i>
<em>italic</em>
<a href="http://www.example.com/">inline URL</a>
<code>inline fixed-width code</code>
<pre>pre-formatted fixed-width code block</pre>'''