import datetime


import pandas as pd
import numpy as np
# from LogRoot.Logging import Logger

# https://pypi.org/project/XTBApi/
from useless.XTBApi.api import Client, TRANS_TYPES, DICT_MODES_XTB, MODES

DICT_CRTPO_XTB = {
    "BTC-USD" : "BITCOIN",
    "ETH-USD" : "ETHEREUM",
    "DASH-USD" : None,
    "LTC-USD" : "LITECOIN",
    "XLM-USD" : "STELLAR",
    "ZEC-USD" : "ZCASH",
    "LINK-USD": "CHAINLINK",
    "DOGE-USD": "DOGECOIN"}

# FIRST INIT THE CLIENT
import _KEYS_DICT

XTB_CLIENT = Client()
# THEN LOGIN
userId = 14114573  # 14114573
password = "aNLVLUfUR3JZ+Yg"  # aNLVLUfUR3JZ+Yg
XTB_CLIENT.login(userId, password) # mode={demo,real})

# df_all_symbol =  pd.DataFrame(client.get_all_symbols())
Meli = 'MELI.US_4'
S = 'EURUSD' # 'ETHEREUM'

# OPEN BUY
# client.open_trade('buy', 'ETHEREUM', dolars=100, custom_Messege="buy", tp_per=0.07, sl_per=0.02)
# OPEN SELL
# client.open_trade('sell', 'ETHEREUM', dolars=100, custom_Messege="sell", tp_per=0.07, sl_per=0.02)
# ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
# client.open_trade('buy', 'ETHEREUM', dolars=100, custom_Messege="ORDER BUY", tp_per=0.07, sl_per=0.02, order_margin_per=-0.004, expiration_stamp=expitarion_timestamp)
# ORDER SELL when price increase  0.01percentage order_margin_per=0.01
# client.open_trade('sell', 'ETHEREUM', dolars=100, custom_Messege="ORDER SELL", tp_per=0.07, sl_per=0.02, order_margin_per=+0.004, expiration_stamp=expitarion_timestamp)

import datetime
def get_expiration_timeStamp(minutes):
    expitarion_timestamp = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(minutes=minutes)
    expitarion_timestamp = int(datetime.datetime.timestamp(expitarion_timestamp)) * 1000
    return expitarion_timestamp

CSV_NAME = "@CRT"
list_stocks_crt = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks_crt = [DICT_CRTPO_XTB[l] for l in list_stocks_crt]         #[v for k, v in DICT_CRTPO_XTB.items()]
list_stocks_crt.remove(None)
CSV_NAME = "@CHIC"
list_stocks_chic = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
list_stocks_chic = [   "TER.US_4", "KLAC.US_4", "ALGN.US_4",  "SPG.US_4", "STAG.US_4" ] #"UONE.US_4",
CSV_NAME = "@FOLO3"
list_stocks = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
#"GOOG.US_4"
list_stocks = ["GOOGC.US_4", "MSFT.US_4", "TSLA.US_4","UPST.US_4", "MELI.US_4", "TWLO.US_4", "RIVN.US_4", "SNOW.US_4", "LYFT.US_4", "ADBE.US_4", "UBER.US_4", "ZI.US_4", "QCOM.US_4", "PYPL.US_4", "SPOT.US_4", "GTLB.US", "MDB.US_4", "NVDA.US_4", "AMD.US_4" , "ADSK.US_4", "AMZN.US_4", "CRWD.US", "NVST.US_4", "HUBS.US_4", "EPAM.US_4", "PINS.US_4", "TTD.US_4", "SNAP.US_4", "APPS.US_4", "ASAN.US_4", "AFRM.US_4", "DOCN.US", "ETSY.US_4", "DDOG.US", "SHOP.US_4", "NIO.US_4", "U.US_4", "GME.US_4", "RBLX.US_4"]
list_stocks =    list_stocks +list_stocks_crt + list_stocks_chic  #
list_stocks = list_stocks_crt



all_trades = XTB_CLIENT.update_trades() # GET CURRENT TRADES
# trade_ids = [trade_id for trade_id in trades.keys()]
for trade_id, data in all_trades.items():
    try:
        actual_profit = XTB_CLIENT.get_trade_profit(trade_id) # CHECK PROFIT
        resp = XTB_CLIENT.commandExecute('tradeTransaction', dict(cmd=0, price=12, symbol='EURUSD', volume=1.0))
        print("cerrar stock: " + all_trades[trade_id].symbol + " profit: " + str(actual_profit) + " mode: " + all_trades[trade_id].mode)
        # if actual_profit >= 100:
        # tp = client.open_trade()
        MODES[data.mode.upper]
        XTB_CLIENT.trade_transaction(mode=MODES[data.mode.upper()].sub_dict, symbol = data.symbol, order =trade_id, volume=data.volume, trans_type=TRANS_TYPES.MODIFY.value)

        XTB_CLIENT.close_trade(trade_id)
    except Exception as ex:
        print("Exception CLOSE Stock"+ str(trade_id) +" ", ex)


#GTLB.US
# CHECK IF MARKET IS OPEN FOR EURUSD
for S in list_stocks:
    print(S)
    XTB_CLIENT.check_if_market_open([S])
    # BUY ONE VOLUME (FOR EURUSD THAT CORRESPONDS TO 100000 units)
    try:
        if XTB_CLIENT.check_if_market_open([S]):
            expitarion_timestamp = get_expiration_timeStamp(minutes=10)
            print("mercado abierto Stock: " + S)
            #ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
            XTB_CLIENT.open_trade('buy', S, dolars=100, custom_Messege="ORDER BUY: " + S, tp_per=0.07, sl_per=0.03, order_margin_per=-0.003, expiration_stamp=expitarion_timestamp) #  order_margin_per=-0.004,
            #ORDER SELL when price increase  0.01percentage order_margin_per=0.01
            XTB_CLIENT.open_trade('sell', S, dolars=100, custom_Messege="ORDER SELL: " + S, tp_per=0.07, sl_per=0.03, order_margin_per=-0.003, expiration_stamp=expitarion_timestamp) # order_margin_per=+0.004,
        else:
            print("mercado cerrado Stock: " + S)
    except Exception as ex:
        print("Exception OPEN Stock"+ S +" ", ex)

    # if client.check_if_market_open([S]):
    #     expitarion_timestamp = get_expiration_timeStamp(minutes=10)
    #     print("mercado abierto Stock: " + S)
    #     client.open_trade('buy', S, 1)
    #
    #     client.open_trade('buy', S, dolars=100)
    #     #ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
    #     client.open_trade('buy',S, dolars=100, custom_Messege="ORDER BUY: "+S , tp_per=0.07, sl_per=0.02, expiration_stamp=expitarion_timestamp) #  order_margin_per=-0.004,
    #     #ORDER SELL when price increase  0.01percentage order_margin_per=0.01
    #     client.open_trade('sell', S, dolars=100, custom_Messege="ORDER SELL: "+S, tp_per=0.07, sl_per=0.02, expiration_stamp=expitarion_timestamp) # order_margin_per=+0.004,
    # else:
    #     print("mercado cerrado Stock: " + S)

    # BUY ONE VOLUME (FOR EURUSD THAT CORRESPONDS TO 100000 units)
    # client.open_trade('sell', 'ETHEREUM', dolars=100,custom_Messege="Hola" , tp_per=0.07, sl_per=0.02)

    # open_trade(self, mode, symbol, dolars, custom_Messege="", tp_per=0.05, sl_per=0.05, order_margin_per=0)


# SEE IF ACTUAL GAIN IS ABOVE 100 THEN CLOSE THE TRADE
all_trades = XTB_CLIENT.update_trades() # GET CURRENT TRADES
trade_ids = [trade_id for trade_id in all_trades.keys()]
for trade in trade_ids:
    actual_profit = XTB_CLIENT.get_trade_profit(trade) # CHECK PROFIT
    if actual_profit >= 100:
        XTB_CLIENT.close_trade(trade) # CLOSE TRADE
# CLOSE ALL OPEN TRADES
XTB_CLIENT.close_all_trades()
# THEN LOGOUT
XTB_CLIENT.logout()


