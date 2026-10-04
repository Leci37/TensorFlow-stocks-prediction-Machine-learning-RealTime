import enum
import json
import socket
import logging
import time
import ssl
import traceback
from datetime import datetime
from threading import Thread

# set to true on debug environment only
from useless import XTBApi

DEBUG = True

#default connection properites
DEFAULT_XAPI_ADDRESS        = 'xapi.xtb.com'
DEFAULT_XAPI_PORT           = 5124
DEFUALT_XAPI_STREAMING_PORT = 5125

# wrapper name and version
WRAPPER_NAME    = 'python'
WRAPPER_VERSION = '2.5.0'

# API inter-command timeout (in ms)
API_SEND_TIMEOUT = 100

# max connection tries
API_MAX_CONN_TRIES = 3

# logger properties
logger = logging.getLogger("jsonSocket")
FORMAT = '[%(asctime)-15s][%(funcName)s:%(lineno)d] %(message)s'
logging.basicConfig(format=FORMAT)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.CRITICAL)


class TransactionSide(object):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    
class TransactionType(object):
    ORDER_OPEN = 0
    ORDER_CLOSE = 2
    ORDER_MODIFY = 3
    ORDER_DELETE = 4

class JsonSocket(object):
    def __init__(self, address, port, encrypt = False):
        self._ssl = encrypt 
        if self._ssl != True:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket = ssl.wrap_socket(sock)
        self.conn = self.socket
        self._timeout = None
        self._address = address
        self._port = port
        self._decoder = json.JSONDecoder()
        self._receivedData = ''

    def connect(self):
        for i in range(API_MAX_CONN_TRIES):
            try:
                self.socket.connect( (self.address, self.port) )
            except socket.error as msg:
                logger.error("SockThread Error: %s" % msg)
                time.sleep(0.25);
                continue
            logger.info("Socket connected")
            return True
        return False

    def _sendObj(self, obj):
        msg = json.dumps(obj)
        self._waitingSend(msg)

    def _waitingSend(self, msg):
        if self.socket:
            sent = 0
            msg = msg.encode('utf-8')
            while sent < len(msg):
                sent += self.conn.send(msg[sent:])
                logger.info('Sent: ' + str(msg))
                time.sleep(API_SEND_TIMEOUT/1000)

    def _read(self, bytesSize=4096):
        if not self.socket:
            raise RuntimeError("socket connection broken")
        while True:
            char = self.conn.recv(bytesSize).decode()
            self._receivedData += char
            try:
                (resp, size) = self._decoder.raw_decode(self._receivedData)
                if size == len(self._receivedData):
                    self._receivedData = ''
                    break
                elif size < len(self._receivedData):
                    self._receivedData = self._receivedData[size:].strip()
                    break
            except ValueError as e:
                continue
        logger.info('Received: ' + str(resp))
        return resp

    def _readObj(self):
        msg = self._read()
        return msg

    def close(self):
        logger.debug("Closing socket")
        self._closeSocket()
        if self.socket is not self.conn:
            logger.debug("Closing connection socket")
            self._closeConnection()

    def _closeSocket(self):
        self.socket.close()

    def _closeConnection(self):
        self.conn.close()

    def _get_timeout(self):
        return self._timeout

    def _set_timeout(self, timeout):
        self._timeout = timeout
        self.socket.settimeout(timeout)

    def _get_address(self):
        return self._address

    def _set_address(self, address):
        pass

    def _get_port(self):
        return self._port

    def _set_port(self, port):
        pass

    def _get_encrypt(self):
        return self._ssl

    def _set_encrypt(self, encrypt):
        pass

    timeout = property(_get_timeout, _set_timeout, doc='Get/set the socket timeout')
    address = property(_get_address, _set_address, doc='read only property socket address')
    port = property(_get_port, _set_port, doc='read only property socket port')
    encrypt = property(_get_encrypt, _set_encrypt, doc='read only property socket port')
    
    
class APIClient(JsonSocket):
    def __init__(self, address=DEFAULT_XAPI_ADDRESS, port=DEFAULT_XAPI_PORT, encrypt=True):
        super(APIClient, self).__init__(address, port, encrypt)
        if(not self.connect()):
            raise Exception("Cannot connect to " + address + ":" + str(port) + " after " + str(API_MAX_CONN_TRIES) + " retries")

    def execute(self, dictionary):
        self._sendObj(dictionary)
        return self._readObj()    

    def disconnect(self):
        self.close()
        
    def commandExecute(self,commandName, arguments=None):
        return self.execute(baseCommand(commandName, arguments))

class APIStreamClient(JsonSocket):
    def __init__(self, address=DEFAULT_XAPI_ADDRESS, port=DEFUALT_XAPI_STREAMING_PORT, encrypt=True, ssId=None, 
                 tickFun=None, tradeFun=None, balanceFun=None, tradeStatusFun=None, profitFun=None, newsFun=None):
        super(APIStreamClient, self).__init__(address, port, encrypt)
        self._ssId = ssId

        self._tickFun = tickFun
        self._tradeFun = tradeFun
        self._balanceFun = balanceFun
        self._tradeStatusFun = tradeStatusFun
        self._profitFun = profitFun
        self._newsFun = newsFun
        
        if(not self.connect()):
            raise Exception("Cannot connect to streaming on " + address + ":" + str(port) + " after " + str(API_MAX_CONN_TRIES) + " retries")

        self._running = True
        self._t = Thread(target=self._readStream, args=())
        self._t.setDaemon(True)
        self._t.start()

    def _readStream(self):
        while (self._running):
                msg = self._readObj()
                logger.info("Stream received: " + str(msg))
                if (msg["command"]=='tickPrices'):
                    self._tickFun(msg)
                elif (msg["command"]=='trade'):
                    self._tradeFun(msg)
                elif (msg["command"]=="balance"):
                    self._balanceFun(msg)
                elif (msg["command"]=="tradeStatus"):
                    self._tradeStatusFun(msg)
                elif (msg["command"]=="profit"):
                    self._profitFun(msg)
                elif (msg["command"]=="news"):
                    self._newsFun(msg)
    
    def disconnect(self):
        self._running = False
        self._t.join()
        self.close()

    def execute(self, dictionary):
        self._sendObj(dictionary)

    def subscribePrice(self, symbol):
        self.execute(dict(command='getTickPrices', symbol=symbol, streamSessionId=self._ssId))
        
    def subscribePrices(self, symbols):
        for symbolX in symbols:
            self.subscribePrice(symbolX)
    
    def subscribeTrades(self):
        self.execute(dict(command='getTrades', streamSessionId=self._ssId))
        
    def subscribeBalance(self):
        self.execute(dict(command='getBalance', streamSessionId=self._ssId))

    def subscribeTradeStatus(self):
        self.execute(dict(command='getTradeStatus', streamSessionId=self._ssId))

    def subscribeProfits(self):
        self.execute(dict(command='getProfits', streamSessionId=self._ssId))

    def subscribeNews(self):
        self.execute(dict(command='getNews', streamSessionId=self._ssId))


    def unsubscribePrice(self, symbol):
        self.execute(dict(command='stopTickPrices', symbol=symbol, streamSessionId=self._ssId))
        
    def unsubscribePrices(self, symbols):
        for symbolX in symbols:
            self.unsubscribePrice(symbolX)
    
    def unsubscribeTrades(self):
        self.execute(dict(command='stopTrades', streamSessionId=self._ssId))
        
    def unsubscribeBalance(self):
        self.execute(dict(command='stopBalance', streamSessionId=self._ssId))

    def unsubscribeTradeStatus(self):
        self.execute(dict(command='stopTradeStatus', streamSessionId=self._ssId))

    def unsubscribeProfits(self):
        self.execute(dict(command='stopProfits', streamSessionId=self._ssId))

    def unsubscribeNews(self):
        self.execute(dict(command='stopNews', streamSessionId=self._ssId))


# Command templates
def baseCommand(commandName, arguments=None):
    if arguments==None:
        arguments = dict()
    return dict([('command', commandName), ('arguments', arguments)])

def loginCommand(userId, password, appName=''):
    return baseCommand('login', dict(userId=userId, password=password, appName=appName))



# example function for processing ticks from Streaming socket
def procTickExample(msg): 
    print("TICK: ", msg)

# example function for processing trades from Streaming socket
def procTradeExample(msg): 
    print("TRADE: ", msg)

# example function for processing trades from Streaming socket
def procBalanceExample(msg): 
    print("BALANCE: ", msg)

# example function for processing trades from Streaming socket
def procTradeStatusExample(msg): 
    print("TRADE STATUS: ", msg)

# example function for processing trades from Streaming socket
def procProfitExample(msg): 
    print("PROFIT: ", msg)

class MODES(enum.Enum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    BALANCE = 6
    CREDIT = 7
class TRANS_TYPES(enum.Enum):
    OPEN = 0
    PENDING = 1
    CLOSE = 2
    MODIFY = 3
    DELETE = 4
# example function for processing news from Streaming socket
def procNewsExample(msg): 
    print("NEWS: ", msg)

DICT_CRTPO_XTB = {
    "BTC-USD" : "BITCOIN",
    "ETH-USD" : "ETHEREUM",
    "DASH-USD" : None,
    "LTC-USD" : "LITECOIN",
    "XLM-USD" : "STELLAR",
    "ZEC-USD" : "ZCASH",
    "LINK-USD": "CHAINLINK",
    "DOGE-USD": "DOGECOIN"}

def get_prices_operate(mode, symbol):
    conversion_mode = {MODES.BUY.value: 'ask', MODES.SELL.value: 'bid'}
    symbol_info = useless.XTBApi.TW_API_CREDENT.BaseClient.get_symbol(symbol)
    price = symbol_info[conversion_mode[mode.sub_dict]]
    conversion_mode_2 = {MODES.BUY.value: 'low', MODES.SELL.value: 'high'}
    price_2 = symbol_info[conversion_mode_2[mode.sub_dict]]

    FACTOR_PRICE_2 = 0.008
    if mode == MODES.BUY or mode == MODES.BUY_LIMIT:
        price_2 = round(price_2 * (1 - FACTOR_PRICE_2), 2)
    elif mode == MODES.SELL or mode == MODES.SELL_LIMIT:
        price_2 = round(price_2 * (1 + FACTOR_PRICE_2), 2)

    return price, price_2


# https://pypi.org/project/XTBApi/
from useless.XTBApi.api import Client
XTB_CLIENT = None
def login_client():
    global XTB_CLIENT
    if client is None:
        client = Client()
        # THEN LOGIN
        client.login(userId, password)  # mode={demo,real})
    return client

def close_operation_2(client_api, trade_id):
    try:
        client_api.close_trade(trade_id)
    except Exception as ex:
        print("Exception close_operation "+traceback.format_exc())

def update_operation(client_api, trade_Id, data_xtb: useless.XTBApi.TW_API_CREDENT.Transaction):
    try:
        client_api.update_trade(trade_Id, data_xtb, tp_new_value=data_xtb._trans_dict['tp'], sl_new_value=data_xtb._trans_dict['sl'])
        return "update"
    except Exception as ex:
        print("Exception close_operation  "+traceback.format_exc())
        return "CommandFailed"


def main():
    # S = 'EURUSD'
    # client_xtb = login_client()
    # if client_xtb.check_if_market_open([S]):
    #     # expitarion_timestamp = get_expiration_timeStamp(minutes=15)
    #     print("mercado abierto Stock: " + S + " Type: " + 'buy')
    #     # ORDER BUY when price reduce 0.01percentage order_margin_per=-0.01
    #     resp = client_xtb.open_trade('buy', S, dolars=100, custom_Messege='buy', tp_per=0.07,
    #                                  sl_per=0.03, order_margin_per=-0.002, expiration_stamp=0)

    # enter your login credentials here
    # enter your login credentials here

    # create & connect to RR socket
    client = APIClient()
    
    # connect to RR socket, login
    loginResponse = client.execute(loginCommand(userId=userId, password=password))
    logger.info(str(loginResponse)) 

    # check if user logged in correctly
    if(loginResponse['status'] == False):
        print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))
        return

    # get ssId from login response
    ssid = loginResponse['streamSessionId']
    # second method of invoking commands
    updates_tp_sp_1_all_stocks_xtb()

    # S ='EURUSD' #'MELI.US_4' # 'EURUSD'
    # resp = client.commandExecute('getSymbol', dict( symbol=S))
    # resp['returnData']['bid']
    # resp['returnData']['ask']
    # # create & connect to Streaming socket with given ssID
    # # and functions for processing ticks, trades, profit and tradeStatus
    # # client.commandExecute("tradeTransaction", DICT_TRade)
    # # {'command': 'tradeTransaction', 'arguments': {
    # #     'tradeTransInfo': {'cmd': 2, 'symbol': 'EURUSD', 'type': 0, 'volume': 95.0, 'price': 1.05,
    # #                        'customComment': 'buy', 'tp': 1.12, 'sl': 1.02, 'expiration': 0}}}
    # DICT_TRade ={
    #     'tradeTransInfo':
    #     {
    #     "cmd": 0,
    #     "customComment": "Some text",
    #     "comment": "Some text",
    #     "expiration": 0,
    #     "offset": 0,
    #     "order": 0,
    #     "price": resp['returnData']['ask'],
    #     "symbol": S,
    #     # "sl": 0.0,
    #     # "tp": 0.0,
    #     "type": 0,
    #     "volume": 0.1
    #     }
    # }
    # # resp = client.commandExecute("tradeTransaction", DICT_TRade )
    #
    #
    #
    # sclient = APIStreamClient(ssId=ssid, tickFun=procTickExample, tradeFun=procTradeExample, profitFun=procProfitExample, tradeStatusFun=procTradeStatusExample)
    #
    # # subscribe for trades
    # sclient.subscribeTrades()
    #
    # # subscribe for prices
    # # sclient.subscribePrices(['EURUSD', 'EURGBP', 'EURJPY'])
    #
    # # subscribe for profits
    # sclient.subscribeProfits()
    #
    # # this is an example, make it run for 5 seconds
    # time.sleep(5)
    #
    # # gracefully close streaming socket
    # sclient.disconnect()
    #
    # # gracefully close RR socket
    # client.disconnect()
    #

def close_operation_1(client_api, data_xtb, trade_Id):
    print("New value Symbol: " + data_xtb.symbol + " Mode: " + data_xtb.mode.upper() + " TP: " + str(
        data_xtb._trans_dict['tp']) + " SL: " + str(data_xtb._trans_dict['sl']))
    print("Symbol: " + data_xtb.symbol + "  operation MODES.BUY , sp higher than tp Close: " + str(
        data_xtb.price) + " Profit: " + str(data_xtb.actual_profit) + " Trader_id: " + str(trade_Id))
    close_operation_2(client_api, trade_Id)

def get_TP_SL_in_data_xtb(data_xtb):
    sl_b, tp_b = useless.XTBApi.TW_API_CREDENT.Client.get_tp_sl(mode=MODES.BUY.value, price=data_xtb._trans_dict['open_price'],
                                                                tp_per=0.08, sl_per=0.03)
    sl_s, tp_s = useless.XTBApi.TW_API_CREDENT.Client.get_tp_sl(mode=MODES.SELL.value, price=data_xtb._trans_dict['open_price'],
                                                                tp_per=0.08, sl_per=0.03)
    if data_xtb._trans_dict['tp'] == 0 and data_xtb.mode.upper() == MODES.BUY.name.upper():
        data_xtb._trans_dict['tp'] = tp_b
    if data_xtb._trans_dict['sl'] == 0 and data_xtb.mode.upper() == MODES.BUY.name.upper():
        data_xtb._trans_dict['sl'] = sl_b
    if data_xtb._trans_dict['tp'] == 0 and data_xtb.mode.upper() == MODES.SELL.name.upper():
        data_xtb._trans_dict['tp'] = tp_s
    if data_xtb._trans_dict['sl'] == 0 and data_xtb.mode.upper() == MODES.SELL.name.upper():
        data_xtb._trans_dict['sl'] = sl_s
    return data_xtb


import pandas as pd

df_h = pd.DataFrame(columns=['key', 'previus_profit'])

# pd.DataFrame(data_xtb._trans_dict)
# df = df.append(data_xtb._trans_dict, ignore_index=True)
# df[(df["symbol"] == data_xtb._trans_dict["symbol"]) & (df["open_time"] == data_xtb._trans_dict["open_time"]) & (df["order"] == data_xtb._trans_dict["order"]) & (df["cmd"] == data_xtb._trans_dict["cmd"])]
# df['open_time_S'] = pd.to_datetime(df['open_time']/1000, unit='s', errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S")
# df['timestamp_S'] = pd.to_datetime(df['timestamp']/1000, unit='s', errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S")
# df = df.sort_values(["symbol", "open_time","order", "cmd" ], ascending=True)

# import json
# d = {"humbral": 0,}
# json.dump(d, open('../'+file_path_humbral+".json", 'w'))
DICT_MODES_XTB = {0: 'buy', 1: 'sell', 2: 'buy_limit', 3: 'sell_limit'}
class Transaction(object):
    def __init__(self, trans_dict):
        self._trans_dict = trans_dict
        self.mode = DICT_MODES_XTB[trans_dict['cmd']]
        self.order_id = trans_dict['order']
        self.symbol = trans_dict['symbol']
        self.volume = trans_dict['volume']
        self.price = trans_dict['close_price']
        self.actual_profit = trans_dict['profit']
        self.timestamp = trans_dict['open_time'] / 1000
        # LOGGER.debug(f"Transaction {self.order_id} inited")
XTB_DATA_COLUMNS_DITC = ["symbol", "open_time", "order", "cmd", "close_price", "open_price", "profit", "sl", "tp",
                       "volume", "closed", "offset", "digits", "open_timeString", "order2", "position", "comment",
                       "customComment", "commission", "storage", "margin_rate", "nominalValue", "timestamp",
                       "spread", "taxes", "close_time", "close_timeString", "expiration", "expirationString"]
def create_data_xtb_from_yahoo(S, df_y, cmd, order ):

    df_r = pd.DataFrame(columns=XTB_DATA_COLUMNS_DITC)
    df_y["Date"] = df_y.index
    df_y.reset_index(drop=True, inplace=True)
    df_y['profit'] = df_y['Close'] - df_y['Close'].shift(-2)


    df_r["open_price"] = df_y["Open"]
    df_r["open_time"] = pd.to_datetime(df_y["Date"]).map(pd.Timestamp.timestamp)
    df_r["order"] = order #"o"+str(pd.to_datetime(df_y["Date"]).map(pd.Timestamp.timestamp))
    df_r["cmd"] = cmd
    df_r["symbol"] = S

    list_ditc = df_r.to_dict('records')
    return list_ditc

    # data_xtb{}

def updates_tp_sp_2_A_stocks_xtb(client_api, trade_Id,data_xtb ):
    global df_h
    print(data_xtb.symbol + " traderid: " + str(trade_Id))
    if not client_api.check_if_market_open([data_xtb.symbol])[data_xtb.symbol]:
        print("MErcado cerrrado " + data_xtb.symbol)
        return "close_market"
    key_humbral = data_xtb._trans_dict["symbol"] + "__" + str(data_xtb._trans_dict["open_time"]) + "__" + str(
        data_xtb._trans_dict["order"]) + "__" + str(data_xtb._trans_dict["cmd"])
    print("key_humbral:  " + key_humbral + " actual_profit: " + str(data_xtb.actual_profit))

    previus_profit = round(data_xtb.actual_profit * 1.001, 4)
    if df_h is not None and df_h[(df_h["key"] == key_humbral)].shape[0] == 1:
        previus_profit = df_h.loc[(df_h["key"] == key_humbral), "previus_profit"].values[0]
    elif df_h is None or df_h.shape[0] == 0 or df_h[(df_h["key"] == key_humbral)].shape[0] == 0:
        df_h = df_h.append({"key": key_humbral, "previus_profit": previus_profit}, ignore_index=True)
    else:
        raise ValueError("más de una df_h[(df_hkey] == key_humbral)]")

    epchs_ago = (datetime.now() - datetime.fromtimestamp(data_xtb._trans_dict['open_time'] / 1000)).seconds / (
                60 * 15)  # 10 minutos
    # cuanto mas tiempo pase mas se aprieta
    coefi_reduction = round((0.025 * epchs_ago) / 16, 3)  # 4horas en gurpos de 15 min
    num_increase_umbral = round(abs(data_xtb.actual_profit) * (0.65 + coefi_reduction), 2)

    print("BEFORE: Symbol: " + data_xtb.symbol + " Mode: " + data_xtb.mode.upper() + " TP: " + str(
        data_xtb._trans_dict['tp']) + " SL: " + str(data_xtb._trans_dict['sl']))

    if data_xtb._trans_dict['tp'] == 0 or data_xtb._trans_dict['sl'] == 0:
        data_xtb = get_TP_SL_in_data_xtb(data_xtb)

    if data_xtb.mode.upper() == MODES.BUY.name.upper():
        if data_xtb.actual_profit > previus_profit:
            print("Update operation MODES.SELL , Uptates TP: " + str(data_xtb._trans_dict['tp']) + "_to_: " + str(
                data_xtb._trans_dict['tp'] + num_increase_umbral) + "\n\t SL: " + str(
                data_xtb._trans_dict['sl']) + "_to_: " + str(data_xtb._trans_dict['sl'] + num_increase_umbral))
            print("Update actual_profit: replace ", previus_profit, " for ", data_xtb.actual_profit)
            data_xtb._trans_dict['sl'] = round(data_xtb._trans_dict['sl'] + num_increase_umbral, 2)
            data_xtb._trans_dict['tp'] = round(data_xtb._trans_dict['tp'] + num_increase_umbral, 2)
            df_h.loc[(df_h["key"] == key_humbral), "previus_profit"] = data_xtb.actual_profit
        else:
            data_xtb._trans_dict['sl'] = round(data_xtb._trans_dict['sl'] * (1 + coefi_reduction), 2)
            data_xtb._trans_dict['tp'] = round(data_xtb._trans_dict['tp'] * (1 - coefi_reduction), 2)
            print("Ncoefi_reduction: " + str(
                coefi_reduction) + " Symbol: " + data_xtb.symbol + " Mode: " + data_xtb.mode.upper())
        if data_xtb._trans_dict['tp'] <= data_xtb._trans_dict['sl']:
            close_operation_1(client_api, data_xtb, trade_Id)
            return "close_operation_BUY"
    if data_xtb.mode.upper() == MODES.SELL.name.upper():
        if data_xtb.actual_profit > previus_profit:
            print("Update operation MODES.SELL , Uptates TP: " + str(data_xtb._trans_dict['tp']) + "_to_: " + str(
                data_xtb._trans_dict['tp'] + num_increase_umbral) + "\n\t SL: " + str(
                data_xtb._trans_dict['sl']) + "_to_: " + str(data_xtb._trans_dict['sl'] + num_increase_umbral))
            print("Update actual_profit: replace ", previus_profit, " for ", data_xtb.actual_profit)
            data_xtb._trans_dict['sl'] = round(data_xtb._trans_dict['sl'] - num_increase_umbral, 2)
            data_xtb._trans_dict['tp'] = round(data_xtb._trans_dict['tp'] - num_increase_umbral, 2)
            df_h.loc[(df_h["key"] == key_humbral), "previus_profit"] = data_xtb.actual_profit
        else:
            data_xtb._trans_dict['sl'] = round(data_xtb._trans_dict['sl'] * (1 - coefi_reduction), 2)
            data_xtb._trans_dict['tp'] = round(data_xtb._trans_dict['tp'] * (1 + coefi_reduction), 2)
            print("Ncoefi_reduction: " + str(
                coefi_reduction) + " Symbol: " + data_xtb.symbol + " Mode: " + data_xtb.mode.upper())
        if data_xtb._trans_dict['sl'] <= data_xtb._trans_dict['tp']:
            close_operation_1(client_api, data_xtb, trade_Id)
            return "close_operation_SELL"

    print("UPDATE: Symbol: " + data_xtb.symbol + " Mode: " + data_xtb.mode.upper() + " TP: " + str(
        data_xtb._trans_dict['tp']) + " SL: " + str(data_xtb._trans_dict['sl']))
    # data_xtb._trans_dict['comment'] = datetime.now().strftime("%H:%M:%S") + " Update TP ST_"+data_xtb._trans_dict['customComment']
    return_value = update_operation(client_api, trade_Id, data_xtb)
    print("update")
    return return_value


PATH_FILE_HUMBRAL_XTB = "XTB_api/humbral/"
def updates_tp_sp_1_all_stocks_xtb():
    global df_h
    # resp = client.commandExecute('getAllSymbols', dict(cmd=0, price=12, symbol='EURUSD', volume=1.0))
    client_api = login_client()
    import yfinance as yf

    trades = client_api.update_trades()  # GET CURRENT TRADES
    # trade_ids = [trade_id for trade_id in trades.keys()]
    aux_trade = trades.copy()
    for trade_Id, data_xtb in aux_trade.items():
        df_15min = yf.download(tickers=[k for k, v in DICT_CRTPO_XTB.items() if v == data_xtb.symbol],period='6d', interval="15m", prepos=False)
        list_dict_trans = create_data_xtb_from_yahoo(data_xtb.symbol, df_15min, 0, order = 123)

        for di in list_dict_trans:
            di_t = Transaction(di)





if __name__ == "__main__":
    while True:

        updates_tp_sp_1_all_stocks_xtb()
        time.sleep(80*1)