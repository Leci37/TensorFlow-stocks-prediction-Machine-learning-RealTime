import datetime
import logging
import traceback
from threading import Lock
lock = Lock()

# https://pypi.org/project/XTBApi/
from useless.XTBApi.api import Client
#
# logging.root.manager.loggerDict['root'].disabled = False


DICT_CRTPO_XTB = {
    "BTC-USD" : "BITCOIN",
    "ETH-USD" : "ETHEREUM",
    "DASH-USD" : None,#Dash Dólar
    "LTC-USD" : "LITECOIN",
    "XLM-USD" : "STELLAR",
    "ZEC-USD" : "ZCASH",
    "LINK-USD": "CHAINLINK",
    "DOGE-USD": "DOGECOIN",
    "BNB-USD": "BINACECOIN",
    "ADA-USD": "CARDANO",
    "DOT-USD": "POLKADOT",
    "DYDX-USD": "DYDX",
    "BCH-USD": None, #Bitcoin Cash Dólar
}

# FIRST INIT THE CLIENT
import _KEYS_DICT


# df_all_symbol =  pd.DataFrame(client.get_all_symbols())
Meli = 'MELI.US_4'
S = 'EURUSD' # 'ETHEREUM'
XTB_CLIENT = None

XTB_USER_ID = ""
XTB_PASS = "L.PugaEV!7y+@+T"

# OPEN BUY
# client.open_trade('buy', 'ETHEREUM', dolars=100, custom_Messege="buy", tp_per=0.07, sl_per=0.02)
# OPEN SELL
# client.open_trade('sell', 'ETHEREUM', dolars=100, custom_Messege="sell", tp_per=0.07, sl_per=0.02)
# ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
# client.open_trade('buy', 'ETHEREUM', dolars=100, custom_Messege="ORDER BUY", tp_per=0.07, sl_per=0.02, order_margin_per=-0.004, expiration_stamp=expitarion_timestamp)
# ORDER SELL when price increase  0.01percentage order_margin_per=0.01
# client.open_trade('sell', 'ETHEREUM', dolars=100, custom_Messege="ORDER SELL", tp_per=0.07, sl_per=0.02, order_margin_per=+0.004, expiration_stamp=expitarion_timestamp)

def get_expiration_timeStamp(minutes):
    expitarion_timestamp = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(minutes=minutes)
    expitarion_timestamp = int(datetime.datetime.timestamp(expitarion_timestamp)) * 1000
    return expitarion_timestamp

def recive_and_manage_data_to_xtb(S, type_b_s):
    if S in DICT_CRTPO_XTB:  # CRIPTO
        S = DICT_CRTPO_XTB[S]
    else:
        S_ele = [x for x in LIST_STOCKS_XTB if x.startswith(S + ".US")]
        if len(S_ele) > 1:
            raise ValueError("hay mas de un stock XTB para ese S: " + S)
        elif len(S_ele) == 1:
            S = S_ele[0]
        elif len(S_ele) == 0:
            print("no se encuetra el stock XTB para ese S: " + S)
    type_string_pos_neg = None
    if type_b_s == _KEYS_DICT.Op_buy_sell.POS:
        type_string_pos_neg = 'buy'
    elif type_b_s == _KEYS_DICT.Op_buy_sell.NEG:
        type_string_pos_neg = 'sell'
    return S, type_string_pos_neg



def login_client():
    global XTB_CLIENT
    if XTB_CLIENT is None:
        XTB_CLIENT = Client()
        # THEN LOGIN
        XTB_CLIENT.login(XTB_USER_ID, XTB_PASS)  # mode={demo,real})
    return XTB_CLIENT




CSV_NAME = "@CRT"
list_stocks_crt = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks_crt = [DICT_CRTPO_XTB[l] for l in list_stocks_crt]         #[v for k, v in DICT_CRTPO_XTB.items()]
list_stocks_crt.remove(None)
# CSV_NAME = "@CHIC"
# list_stocks_chic = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks_chic = [   "TER.US_4", "KLAC.US_4", "ALGN.US_4",  "SPG.US_4", "STAG.US_4" ] #"UONE.US_4",
# CSV_NAME = "@FOLO3"
# list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks = [ "GOOG.US_4","MSFT.US_4", "TSLA.US_4","UPST.US_4", "MELI.US_4", "TWLO.US_4", "RIVN.US_4", "SNOW.US_4", "LYFT.US_4", "ADBE.US_4", "UBER.US_4",
                "ZI.US_4", "QCOM.US_4", "PYPL.US_4", "SPOT.US_4", "GTLB.US", "MDB.US_4", "NVDA.US_4", "AMD.US_4" , "ADSK.US_4", "AMZN.US_4", "CRWD.US",
                "NVST.US_4", "HUBS.US_4", "EPAM.US_4", "PINS.US_4", "TTD.US_4", "SNAP.US_4", "APPS.US_4", "ASAN.US_4", "AFRM.US_4", "DOCN.US", "ETSY.US_4",
                "DDOG.US", "SHOP.US_4", "NIO.US_4", "U.US_4", "GME.US_4", "RBLX.US_4", "SOFI.US_4", "STNE.US_4","PDD.US_4", "INMD.US_4" ,  "CRSR.US_4"]
list_stocks =    list_stocks +list_stocks_crt + list_stocks_chic  #
LIST_STOCKS_XTB = list_stocks + list_stocks_chic


#https://stackoverflow.com/questions/39145796/locking-a-method-in-python
def xtb_operate_Lock_thread(S, dictR, type_b_s:_KEYS_DICT.Op_buy_sell, list_model):
    lock.acquire()
    try:
        client_xtb = login_client()
        __xtb_operate(client_xtb, S, dictR, type_b_s, list_model)
        logging.root.manager.loggerDict['root'].disabled = False
    finally:
        lock.release() #release lock

def __xtb_operate(client_xtb, S , dictR, type_b_s:_KEYS_DICT.Op_buy_sell, list_model):

    S_xtb, type_string_pos_neg = recive_and_manage_data_to_xtb(S , type_b_s)
    if S_xtb is None:
        print("WARN __xtb_operate No se encuentra en el xtb dict Stock:"+S)
        return

    print("Stock: " + str(S_xtb) + " Type_buy_sell: " + type_string_pos_neg)
    messege_custom = (" " + str(dictR['Date']) + " "  + "Type:  "+type_b_s.name +" Value:  " + str(dictR['Close']) + " Models: " + ", ".join(list_model) ).replace('[', '').replace(']', '').replace('\'', '')
    print(messege_custom)

    try:
        if client_xtb.check_if_market_open([S_xtb]):
            expitarion_timestamp = get_expiration_timeStamp(minutes=15)
            print("mercado abierto Stock: " + S_xtb + " Type: " + type_string_pos_neg)
            #ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
            resp = client_xtb.open_trade(type_string_pos_neg, S_xtb, dolars=100, custom_Messege=messege_custom, tp_per=0.09, sl_per=0.045, order_margin_per=-0.001,
                                         expiration_stamp=expitarion_timestamp, comment=messege_custom) #  order_margin_per=-0.004,
            #ORDER SELL when price increase  0.01percentage order_margin_per=0.01
            # client_xtb.open_trade('sell', S, dolars=100, custom_Messege=messege_custom, tp_per=0.07, sl_per=0.03,order_margin_per=-0.003, expiration_stamp=expitarion_timestamp) # order_margin_per=+0.004,
        else:
            print("mercado cerrado Stock: " + S_xtb)
    except Exception as ex:
        print("Exception OPEN Stock" + S_xtb + " ", traceback.format_exc())
        # Logger.logr.warning(" Exception Stock: " + S + "  Exception: " + traceback.format_exc())




#GTLB.US
# CHECK IF MARKET IS OPEN FOR EURUSD
# for S in list_stocks:
#     print(S)
#     if client.check_if_market_open([S]):
#         expitarion_timestamp = get_expiration_timeStamp(minutes=10)
#         print("mercado abierto Stock: " + S)
#         client.open_trade('buy', S, 1)
#
#         client.open_trade('buy', S, dolars=100)
#         #ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
#         client.open_trade('buy',S, dolars=100, custom_Messege="ORDER BUY: "+S , tp_per=0.07, sl_per=0.02, expiration_stamp=expitarion_timestamp) #  order_margin_per=-0.004,
#         #ORDER SELL when price increase  0.01percentage order_margin_per=0.01
#         client.open_trade('sell', S, dolars=100, custom_Messege="ORDER SELL: "+S, tp_per=0.07, sl_per=0.02, expiration_stamp=expitarion_timestamp) # order_margin_per=+0.004,
#     else:
#         print("mercado cerrado Stock: " + S)

    # BUY ONE VOLUME (FOR EURUSD THAT CORRESPONDS TO 100000 units)
    # client.open_trade('sell', 'ETHEREUM', dolars=100,custom_Messege="Hola" , tp_per=0.07, sl_per=0.02)

    # open_trade(self, mode, symbol, dolars, custom_Messege="", tp_per=0.05, sl_per=0.05, order_margin_per=0)


# SEE IF ACTUAL GAIN IS ABOVE 100 THEN CLOSE THE TRADE
# trades = client.update_trades() # GET CURRENT TRADES
# trade_ids = [trade_id for trade_id in trades.keys()]
# for trade in trade_ids:
#     actual_profit = client.get_trade_profit(trade) # CHECK PROFIT
#     if actual_profit >= 100:
#         client.close_trade(trade) # CLOSE TRADE
# # CLOSE ALL OPEN TRADES
# # client.close_all_trades()
# # THEN LOGOUT
# client.logout()


