# -*- coding utf-8 -*-

"""
XTBApi.api
~~~~~~~

Main module
"""

import enum
import inspect
import json
import logging
import time
from datetime import datetime

from websocket import create_connection
from websocket._exceptions import WebSocketConnectionClosedException

from useless.XTBApi.exceptions import *

LOGGER = logging.getLogger('XTBApi.api')
LOGIN_TIMEOUT = 120
MAX_TIME_INTERVAL = 0.200


class STATUS(enum.Enum):
    LOGGED = enum.auto()
    NOT_LOGGED = enum.auto()


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


class PERIOD(enum.Enum):
    ONE_MINUTE = 1
    FIVE_MINUTES = 5
    FIFTEEN_MINUTES = 15
    THIRTY_MINUTES = 30
    ONE_HOUR = 60
    FOUR_HOURS = 240
    ONE_DAY = 1440
    ONE_WEEK = 10080
    ONE_MONTH = 43200


def _get_data(command, **parameters):
    data = {
        "command": command,
    }
    if parameters:
        data['arguments'] = {}
        for (key, value) in parameters.items():
            data['arguments'][key] = value
    return data


def _check_mode(mode):
    """check if mode acceptable"""
    modes = [x.sub_dict for x in MODES]
    if mode not in modes:
        raise ValueError("mode must be in {}".format(modes))


def _check_period(period):
    """check if period is acceptable"""
    if period not in [x.sub_dict for x in PERIOD]:
        raise ValueError("Period: {} not acceptable".format(period))


def _check_volume(volume):
    """normalize volume"""
    if not isinstance(volume, float):
        try:
            return float(volume)
        except Exception:
            raise ValueError("vol must be float")
    else:
        return volume


class BaseClient(object):
    """main client class"""

    def __init__(self):
        self.ws = None
        self._login_data = None
        self._time_last_request = time.time() - MAX_TIME_INTERVAL
        self.status = STATUS.NOT_LOGGED
        LOGGER.debug("BaseClient inited")
        self.LOGGER = logging.getLogger('XTBApi.api.BaseClient')

    def _login_decorator(self, func, *args, **kwargs):
        if self.status == STATUS.NOT_LOGGED:
            raise NotLogged()
        try:
            return func(*args, **kwargs)
        except SocketError as e:
            LOGGER.info("re-logging in due to LOGIN_TIMEOUT gone")
            self.login(self._login_data[0], self._login_data[1])
            return func(*args, **kwargs)
        except Exception as e:
            LOGGER.warning(e)
            self.login(self._login_data[0], self._login_data[1])
            return func(*args, **kwargs)

    def _send_command(self, dict_data):
        """send command to api"""
        time_interval = time.time() - self._time_last_request
        self.LOGGER.debug("took {} s.".format(time_interval))
        if time_interval < MAX_TIME_INTERVAL:
            time.sleep(MAX_TIME_INTERVAL - time_interval)
        try:
            self.ws.send(json.dumps(dict_data))
            response = self.ws.recv()
        except WebSocketConnectionClosedException:
            raise SocketError()
        self._time_last_request = time.time()
        res = json.loads(response)
        if res['status'] is False:
            raise CommandFailed(res)
        if 'returnData' in res.keys():
            self.LOGGER.info("CMD: done")
            self.LOGGER.debug(res['returnData'])
            return res['returnData']

    def _send_command_with_check(self, dict_data):
        """with check login"""
        return self._login_decorator(self._send_command, dict_data)

    def login(self, user_id, password, mode='demo'):
        """login command"""
        data = _get_data("login", userId=user_id, password=password)
        self.ws = create_connection(f"wss://ws.xtb.com/{mode}")
        response = self._send_command(data)
        self._login_data = (user_id, password)
        self.status = STATUS.LOGGED
        self.LOGGER.info("CMD: login...")
        return response

    def logout(self):
        """logout command"""
        data = _get_data("logout")
        response = self._send_command(data)
        self.status = STATUS.LOGGED
        self.LOGGER.info("CMD: logout...")
        return response

    def get_all_symbols(self):
        """getAllSymbols command"""
        data = _get_data("getAllSymbols")
        self.LOGGER.info("CMD: get all symbols...")
        return self._send_command_with_check(data)

    def get_calendar(self):
        """getCalendar command"""
        data = _get_data("getCalendar")
        self.LOGGER.info("CMD: get calendar...")
        return self._send_command_with_check(data)

    def get_chart_last_request(self, symbol, period, start):
        """getChartLastRequest command"""
        _check_period(period)
        args = {
            "period": period,
            "start": start * 1000,
            "symbol": symbol
        }
        data = _get_data("getChartLastRequest", info=args)
        self.LOGGER.info(f"CMD: get chart last request for {symbol} of period"
                         f" {period} from {start}...")

        return self._send_command_with_check(data)

    def get_chart_range_request(self, symbol, period, start, end, ticks):
        """getChartRangeRequest command"""
        if not isinstance(ticks, int):
            raise ValueError(f"ticks value {ticks} must be int")
        self._check_login()
        args = {
            "end": end * 1000,
            "period": period,
            "start": start * 1000,
            "symbol": symbol,
            "ticks": ticks
        }
        data = _get_data("getChartRangeRequest", info=args)
        self.LOGGER.info(f"CMD: get chart range request for {symbol} of "
                         f"{period} from {start} to {end} with ticks of "
                         f"{ticks}...")
        return self._send_command_with_check(data)

    def get_commission(self, symbol, volume):
        """getCommissionDef command"""
        volume = _check_volume(volume)
        data = _get_data("getCommissionDef", symbol=symbol, volume=volume)
        self.LOGGER.info(f"CMD: get commission for {symbol} of {volume}...")
        return self._send_command_with_check(data)

    def get_margin_level(self):
        """getMarginLevel command
        get margin information"""
        data = _get_data("getMarginLevel")
        self.LOGGER.info("CMD: get margin level...")
        return self._send_command_with_check(data)

    def get_margin_trade(self, symbol, volume):
        """getMarginTrade command
        get expected margin for volumes used symbol"""
        volume = _check_volume(volume)
        data = _get_data("getMarginTrade", symbol=symbol, volume=volume)
        self.LOGGER.info(f"CMD: get margin trade for {symbol} of {volume}...")
        return self._send_command_with_check(data)

    def get_profit_calculation(self, symbol, mode, volume, op_price, cl_price):
        """getProfitCalculation command
        get profit calculation for symbol with vol, mode and op, cl prices"""
        _check_mode(mode)
        volume = _check_volume(volume)
        data = _get_data("getProfitCalculation", closePrice=cl_price,
                         cmd=mode, openPrice=op_price, symbol=symbol,
                         volume=volume)
        self.LOGGER.info(f"CMD: get profit calculation for {symbol} of "
                         f"{volume} from {op_price} to {cl_price} in mode "
                         f"{mode}...")
        return self._send_command_with_check(data)

    def get_server_time(self):
        """getServerTime command"""
        data = _get_data("getServerTime")
        self.LOGGER.info("CMD: get server time...")
        return self._send_command_with_check(data)

    def get_symbol(self, symbol):
        """getSymbol command"""
        data = _get_data("getSymbol", symbol=symbol)
        self.LOGGER.info(f"CMD: get symbol {symbol}...")
        return self._send_command_with_check(data)

    def get_tick_prices(self, symbols, start, level=0):
        """getTickPrices command"""
        data = _get_data("getTickPrices", level=level, symbols=symbols,
                         timestamp=start)
        self.LOGGER.info(f"CMD: get tick prices of {symbols} from {start} "
                         f"with level {level}...")
        return self._send_command_with_check(data)

    def get_trade_records(self, trade_position_list):
        """getTradeRecords command
        takes a list of position id"""
        data = _get_data("getTradeRecords", orders=trade_position_list)
        self.LOGGER.info(f"CMD: get trade records of len "
                         f"{len(trade_position_list)}...")
        return self._send_command_with_check(data)

    def get_trades(self, opened_only=True):
        """getTrades command"""
        data = _get_data("getTrades", openedOnly=opened_only)
        self.LOGGER.info("CMD: get trades...")
        return self._send_command_with_check(data)

    def get_trades_history(self, start, end):
        """getTradesHistory command
        can take 0 as actual time"""
        data = _get_data("getTradesHistory", end=end, start=start)
        self.LOGGER.info(f"CMD: get trades history from {start} to {end}...")
        return self._send_command_with_check(data)

    def get_trading_hours(self, trade_position_list):
        """getTradingHours command"""
        # EDITED IN ALPHA2
        data = _get_data("getTradingHours", symbols=trade_position_list)
        self.LOGGER.info(f"CMD: get trading hours of len "
                         f"{len(trade_position_list)}...")
        response = self._send_command_with_check(data)
        for symbol in response:
            for day in symbol['trading']:
                day['fromT'] = int(day['fromT'] / 1000)
                day['toT'] = int(day['toT'] / 1000)
            for day in symbol['quotes']:
                day['fromT'] = int(day['fromT'] / 1000)
                day['toT'] = int(day['toT'] / 1000)
        return response

    def get_version(self):
        """getVersion command"""
        data = _get_data("getVersion")
        self.LOGGER.info("CMD: get version...")
        return self._send_command_with_check(data)

    def ping(self):
        """ping command"""
        data = _get_data("ping")
        self.LOGGER.info("CMD: get ping...")
        self._send_command_with_check(data)

    def trade_transaction(self, symbol, mode, trans_type, volume, **kwargs):
        """tradeTransaction command"""
        # check type
        if trans_type not in [x.sub_dict for x in TRANS_TYPES]:
            raise ValueError(f"Type must be in {[x for x in trans_type]}")
        # check kwargs
        accepted_values = ['order', 'price', 'expiration', 'customComment',
                           'offset', 'sl', 'tp']
        assert all([val in accepted_values for val in kwargs.keys()])
        _check_mode(mode)  # check if mode is acceptable
        volume = _check_volume(volume)  # check if volume is valid
        info = {
            'cmd': mode,
            'symbol': symbol,
            'type': trans_type,
            'volume': volume
        }
        info.update(kwargs)  # update with kwargs parameters
        data = _get_data("tradeTransaction", tradeTransInfo=info)
        name_of_mode = [x.name for x in MODES if x.sub_dict == mode][0]
        name_of_type = [x.name for x in TRANS_TYPES if x.sub_dict ==
                        trans_type][0]
        self.LOGGER.info(f"CMD: trade transaction of {symbol} of mode "
                         f"{name_of_mode} with type {name_of_type} of "
                         f"{volume}...")
        return self._send_command_with_check(data)

    def trade_transaction_status(self, order_id):
        """tradeTransactionStatus command"""
        data = _get_data("tradeTransactionStatus", order=order_id)
        self.LOGGER.info(f"CMD: trade transaction status for {order_id}...")
        return self._send_command_with_check(data)

    def get_user_data(self):
        """getCurrentUserData command"""
        data = _get_data("getCurrentUserData")
        self.LOGGER.info("CMD: get user data...")
        return self._send_command_with_check(data)


class Transaction(object):
    def __init__(self, trans_dict):
        self._trans_dict = trans_dict
        self.mode = {0: 'buy', 1: 'sell', 2: 'buy_limit', 3: 'sell_limit'}[trans_dict['cmd']]
        self.order_id = trans_dict['order']
        self.symbol = trans_dict['symbol']
        self.volume = trans_dict['volume']
        self.price = trans_dict['close_price']
        self.actual_profit = trans_dict['profit']
        self.timestamp = trans_dict['open_time'] / 1000
        LOGGER.debug(f"Transaction {self.order_id} inited")


class Client(BaseClient):
    """advanced class of client"""
    def __init__(self):
        super().__init__()
        self.trade_rec = {}
        self.LOGGER = logging.getLogger('XTBApi.api.Client')
        self.LOGGER.info("Client inited")

    def check_if_market_open(self, list_of_symbols):
        """check if market is open for symbol in symbols"""
        _td = datetime.today()
        actual_tmsp = _td.hour * 3600 + _td.minute * 60 + _td.second
        response = self.get_trading_hours(list_of_symbols)
        market_values = {}
        for symbol in response:
            today_values = [day for day in symbol['trading'] if day['day'] ==
                _td.isoweekday()][0]
            if today_values['fromT'] <= actual_tmsp <= today_values['toT']:
                market_values[symbol['symbol']] = True
            else:
                market_values[symbol['symbol']] = False
        return market_values

    def get_lastn_candle_history(self, symbol, timeframe_in_seconds, number):
        """get last n candles of timeframe"""
        acc_tmf = [60, 300, 900, 1800, 3600, 14400, 86400, 604800, 2592000]
        if timeframe_in_seconds not in acc_tmf:
            raise ValueError(f"timeframe not accepted, not in "
                             f"{', '.join([str(x) for x in acc_tmf])}")
        sec_prior = timeframe_in_seconds * number
        LOGGER.debug(f"sym: {symbol}, tmf: {timeframe_in_seconds},"
                     f" {time.time() - sec_prior}")
        res = {'rateInfos': []}
        while len(res['rateInfos']) < number:
            res = self.get_chart_last_request(symbol,
                timeframe_in_seconds // 60, time.time() - sec_prior)
            LOGGER.debug(res)
            res['rateInfos'] = res['rateInfos'][-number:]
            sec_prior *= 3
        candle_history = []
        for candle in res['rateInfos']:
            _pr = candle['open']
            op_pr = _pr / 10 ** res['digits']
            cl_pr = (_pr + candle['close']) / 10 ** res['digits']
            hg_pr = (_pr + candle['high']) / 10 ** res['digits']
            lw_pr = (_pr + candle['low']) / 10 ** res['digits']
            new_candle_entry = {'timestamp': candle['ctm'] / 1000, 'open':
                op_pr, 'close': cl_pr, 'high': hg_pr, 'low': lw_pr,
                                'volume': candle['vol']}
            candle_history.append(new_candle_entry)
        LOGGER.debug(candle_history)
        return candle_history

    def update_trades(self):
        """update trade list"""
        trades = self.get_trades()
        self.trade_rec.clear()
        for trade in trades:
            obj_trans = Transaction(trade)
            self.trade_rec[obj_trans.order_id] = obj_trans
        #values_to_del = [key for key, trad_not_listed in
        #                 self.trade_rec.items() if trad_not_listed.order_id
        #                 not in [x['order'] for x in trades]]
        #for key in values_to_del:
        #    del self.trade_rec[key]
        self.LOGGER.info(f"updated {len(self.trade_rec)} trades")
        return self.trade_rec

    def get_trade_profit(self, trans_id):
        """get profit of trade"""
        self.update_trades()
        profit = self.trade_rec[trans_id].actual_profit
        self.LOGGER.info(f"got trade profit of {profit}")
        return profit

    def open_trade(self, mode, symbol,volumen ,  dolars =0, custom_Messege ="", tp_per = 0.05, sl_per= 0.05, order_margin_per = 0, expiration_stamp = 0):
        """open trade transaction"""
        if mode in [MODES.BUY.value, MODES.SELL.value]:
            mode = [x for x in MODES if x.sub_dict == mode][0]
        elif mode in ['buy', 'sell']:
            modes = {'buy': MODES.BUY, 'sell': MODES.SELL}
            mode = modes[mode]
        else:
            raise ValueError("mode can be buy or sell")

        conversion_mode = {MODES.BUY.value: 'ask', MODES.SELL.value: 'bid'}
        price = self.get_symbol(symbol)[conversion_mode[mode.sub_dict]]

        if order_margin_per != 0:
            # https://www.xtb.com/int/education/xstation-5-pending-orders
            mode, mode_name = self.change_to_order_type_mode(mode.name)
        else:
            mode_name = mode.name
            mode = mode.sub_dict

        self.LOGGER.debug(f"opening trade of {symbol} of Dolars: {dolars} with {mode_name}  Expiration: {datetime.fromtimestamp(expiration_stamp/1000) }")
        price = round(price * (1 + order_margin_per) , 2) #No mas de 3 decimales

        if dolars != 0:
            volumen = round((dolars / price) , 0)

        # if dolars <= 0:
        #     raise ValueError(f"The value in dollars is lower than expected :{dolars}")
        if mode == MODES.BUY.value or mode == MODES.BUY_LIMIT.value:
            tp = round( price * (1+tp_per) , 2)
            sl = round( price * (1-sl_per) , 2)
        elif mode == MODES.SELL.value or mode == MODES.SELL_LIMIT.value:
            sl = round( price * (1+sl_per) , 2)
            tp = round( price * (1-tp_per) , 2)

        # trans_type = TRANS_TYPES.PENDING.value

        response = self.trade_transaction(symbol, mode, trans_type = 0,volume = volumen, price=price,
                                          customComment=custom_Messege,tp=tp, sl=sl, expiration = expiration_stamp  )
        # response = self.trade_transaction(symbol, MODES.SELL_LIMIT.value, 0, volume=volumen_with_dolars, price=price,customComment=custom_Messege, tp=tp, sl=sl)
        self.update_trades()
        status_rep = self.trade_transaction_status(response['order'])

        status = status_rep['requestStatus']
        status_messg = status_rep['message']

        self.LOGGER.debug(f"open_trade completed with status of {status} Message: {status_messg} Expiration: {datetime.fromtimestamp(expiration_stamp/1000)}")
        if status != 3:
            self.LOGGER.debug(f"FAIL. opening trade of {symbol} Message: {status_messg} of Dolars: {dolars} with {mode_name} Expiration: {datetime.fromtimestamp(expiration_stamp / 1000)}")
            # raise TransactionRejected(status)
        else:
            self.LOGGER.debug(f"Successfully. opening trade of {symbol} of Dolars: {dolars} with {mode_name} Expiration: {datetime.fromtimestamp(expiration_stamp/1000)}")
        return response

    def change_to_order_type_mode(self, mode_name):
        if mode_name == MODES.BUY.name:
            mode_name = MODES.BUY_LIMIT.name
            mode = MODES.BUY_LIMIT.value
        elif mode_name == MODES.SELL.name:
            mode_name = MODES.SELL_LIMIT.name
            mode = MODES.SELL_LIMIT.value
        return mode, mode_name

    def _close_trade_only(self, order_id):
        """faster but less secure"""
        trade = self.trade_rec[order_id]
        self.LOGGER.debug(f"closing trade {order_id}")
        try:
            response = self.trade_transaction(
                trade.symbol, 0, 2, trade.volume, order=trade.order_id,
                price=trade.price)
        except CommandFailed as e:
            if e.err_code == 'BE51':  # order already closed
                self.LOGGER.debug("BE51 error code noticed")
                return 'BE51'
            else:
                raise
        status = self.trade_transaction_status(
            response['order'])['requestStatus']
        self.LOGGER.debug(f"close_trade completed with status of {status}")
        if status != 3:
            raise TransactionRejected(status)
        return response

    def close_trade(self, trans):
        """close trade transaction"""
        if isinstance(trans, Transaction):
            order_id = trans.order_id
        else:
            order_id = trans
        self.update_trades()
        return self._close_trade_only(order_id)

    def close_all_trades(self):
        """close all trades"""
        self.update_trades()
        self.LOGGER.debug(f"closing {len(self.trade_rec)} trades")
        trade_ids = self.trade_rec.keys()
        for trade_id in trade_ids:
            self._close_trade_only(trade_id)


# - next features -
# TODO: withdraw
# TODO: deposit
