import a_manage_stocks_dict

#https://www.iemoji.com/#?category=objects&version=36&theme=appl&skintone=default
BAR_UP = "📈"
BAR_DOWN = "📉"
BAR_SIMPLE = "📊"
FLECHA_UP = "⬆"
FLECHA_DOWN = "⬇"
def get_text_alert(type_b_s):
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
def get_string_alert_message(S, dict_pred, modles_evaluated, modles_evaluated_TF, type_b_s, date_detect, value_detect):
    text_alert_main_b_s, flecha_simbol = get_text_alert(type_b_s)
    url_info_inves = URL_INVESTING + S
    url_info_webull = URL_WE_BULL + S.lower()

    names_models_r = ""
    count_TF_models = 0
    modles_evaluated.sort()
    for k in modles_evaluated:
        if dict_pred[k] == 1:  # dict_predict_1[k] == 1:
            names_models_r = names_models_r + k.replace('br_', '').replace("_" + type_b_s.value, '').replace('__','_').replace("_" + S, '') + '%, '
            if k.startswith( 'br_TF'):
                count_TF_models = count_TF_models + 1

    s88 = "88%: " + str(dict_pred['sum_r_88']) + "/" + str(dict_pred['num_models'])
    s93 = "93%: " + str(dict_pred['sum_r_93']) + "/" + str(dict_pred['num_models'])
    s95 = "95%: " + str(dict_pred['sum_r_95']) + "/" + str(dict_pred['num_models'])
    stf = "TF: " + str(dict_pred['sum_r_TF']) + "/" + str(int(count_TF_models))

# **negrita** escribe el texto en negrita
# __cursiva__ escribe el texto en cursiva
# ```monospace``` escribe el texto en monospace
# ~~tachado~~ escribe el texto tachado
    #https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#message-formatting-bold-italic-code-
    #Message Formatting (bold, italic, code, ...)
    s1 = "<strong>"+text_alert_main_b_s + ": " + S + "</strong> "
    s2 = "<em>"+dict_pred['Date'] + "</em>" + (flecha_simbol*1) +"\nValue:<pre> "+ "{:.2f}".format(value_detect)+"</pre>\n"
    s3 = "<a href=\"" + url_info_inves + "\">Investing.com</a>\n<a href=\"" + url_info_webull + "\">WeBull.com</a>\n\nConfidence of models:\n  "
    s4 = s88 + "\t  " + s93 + "\n  " + s95 + "\t  " + stf + "\n\n"
    s5 = "📊Model names: <pre>" + names_models_r + "</pre>\n"
    s6 = "<em>Date alert: "+date_detect + "</em>"
    return  s1 + s2 + s3 + s4 + s5 + s6


html_example = '''<b>bold</b>
<strong>bold</strong>
<i>italic</i>
<em>italic</em>
<a href="http://www.example.com/">inline URL</a>
<code>inline fixed-width code</code>
<pre>pre-formatted fixed-width code block</pre>'''