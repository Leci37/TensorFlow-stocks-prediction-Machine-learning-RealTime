import json
import socket
import logging
import time
import ssl
from datetime import datetime
from threading import Thread

# set to true on debug environment only
from XTB_api import Trader_class

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

# example function for processing news from Streaming socket
def procNewsExample(msg): 
    print("NEWS: ", msg)
    
# http://developers.xstore.pro/documentation/current#logi
# http://developers.xstore.pro/documentation/current#login
#Objeto json to python
#https://jsonformatter.org/json-to-python
def main():

    # enter your login credentials here
    userId = 14114573 # 14114573
    password = "aNLVLUfUR3JZ+Yg"   #aNLVLUfUR3JZ+Yg

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
    # resp = client.commandExecute('getAllSymbols')

    operation = Trader_class.Trade(
        # custom_comment: str, expiration: int, offset: int, order: int, price: float, sl: float, symbol: str, tp: float, type: Transaction_Type, volume: float
        cmd = Trader_class.Transaction_CMD.BUY,
        custom_comment= "primer buy test",
        expiration = datetime.timestamp(datetime(2023,8,1)),
        offset = 10,
        order = 0,
        price= 12, #Precio al que se debe ejecutar
        sl = 10.0,
        symbol ='EURUSD', #"ETHEREUM",
        tp = 10.0,
        type = Trader_class.Transaction_Type.OPEN,
        volume = 1.0)

    price, price_2 = get_prices_operate(mode, symbol)

    resp = client.commandExecute('tradeTransaction', dict(cmd=0, price= 12, symbol ='EURUSD',volume = 1.0 ))

    
    # create & connect to Streaming socket with given ssID
    # and functions for processing ticks, trades, profit and tradeStatus
    sclient = APIStreamClient(ssId=ssid, tickFun=procTickExample, tradeFun=procTradeExample, profitFun=procProfitExample, tradeStatusFun=procTradeStatusExample)

    # SymbolResponse
    # symbolResponse = APICommandFactory.ExecuteSymbolCommand(connector, "EURUSD");

    # subscribe for trades
    sclient.subscribeTrades()
    
    # subscribe for prices
    sclient.subscribePrices(['EURUSD', 'EURGBP', 'EURJPY'])

    # subscribe for profits
    sclient.subscribeProfits()

    # this is an example, make it run for 5 seconds
    time.sleep(5)
    
    # gracefully close streaming socket
    sclient.disconnect()
    
    # gracefully close RR socket
    client.disconnect()
    
    
if __name__ == "__main__":
    main()	
