import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.ta import extract_features
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from typing import TypedDict
import pandas as pd
from datetime import datetime

# Double Sell means flipping Long / Flat to Sell
# Double Buy means flipping Short / Flat to Buy
class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.

def transform(position: Positions, action: int):
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:
        if position == Positions.LONG: return Positions.FLAT, False
        if position == Positions.FLAT: return Positions.SHORT, True

    if action == Actions.BUY:
        if position == Positions.SHORT: return Positions.FLAT, False
        if position == Positions.FLAT: return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False

class Trade(TypedDict):
    position: Positions
    openPrice: float
    closePrice: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class TradingEnvV2(gym.Env):

    metadata = {'render_modes': ['human']}

    # eps_length: To normalise the holding time of a position
    def __init__(
            self, df, window_size, symbol, spread, point, trade_fee, log_dir, flatten = False,
            eps_length = 253, render_mode=None, clear_trade=True,
            bar_limit=None, normalise=None, normalise_path=None
            ):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.flatten = flatten

        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.spread = spread / point
        self.point = point
        self.bar_limit = bar_limit
        self.trade_fee = trade_fee
        self.normalise = normalise
        self.normalise_path = normalise_path
        self.prices, self.signal_features, self.dates = self._process_data()

        if self.flatten == True:
            self.shape = (window_size * (self.signal_features.shape[1] + 2),)  # +2 for the position information
        else:
            self.shape = (window_size, self.signal_features.shape[1] + 2,)  # +2 for the position information
        self.eps_length = eps_length
        
        self.writer = tf.summary.create_file_writer(log_dir)
        self.episode = 0
        self.clear_trade = clear_trade

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._trades = None
        self._position = None
        self._position_history = None
        self._position_time_score = None
        self._profit_history = None
        self._pips = None
        self._total_reward = None

        # training variable
        self._best_score = 0

    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            position = self._position.value,
        )
    
    def terminated(self): return self._terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = ((self.window_size + 1) * [Positions.FLAT.value])
        self._position_time_score = ((self.window_size + 1) * [0])
        self._profit_history = [1.]
        self._total_reward = 0.
        self._pips = 0.
        self.episode += 1
        self._last_trades = self._trades
        if self.clear_trade: self._trades = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            # Draw the trades to tensorboard
            trades = self.trades()
            if trades.empty == False:
                # Calculate the trade profits
                trades.loc[trades['position'] == -1, 'profit'] = ((trades['openPrice'] - self.spread) - trades['closePrice']) * self.point
                trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + self.spread)) * self.point
                trades = trades.dropna()
                trades['cashflow'] = trades['profit'].cumsum()
                profit = trades['profit'].sum()
                # Draw the Cashflow to the tensorboard
                with self.writer.as_default():
                    for i, trade in trades['cashflow'].items():
                        tf.summary.scalar(f'trades/iteration-{self.episode}', trade, i)

                # Update Model
                if profit > self._best_score:
                    if hasattr(self, '_model') == True:
                        print(f'Saving Model... [+{round(profit, 2)} pips]')
                        self._model.save(f'output/best_{self.symbol}_v2.zip')

                    # Save the trades
                    self._best_score = profit
                    trades.to_csv(f'logs/{self.symbol}_trades_v2_train.csv', index=False)

            if self.render_mode == "human": self.render()
            self._terminated = True

        # Calculate step reward
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        # Compute the next position
        last_position = self._position
        self._position, trade = transform(self._position, action)

        # Update Trade History
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            trade: Trade = { 'position': last_position.value, 'openPrice': last_trade_price, 'closePrice': current_price,
                             'openTime': self.dates[self._last_trade_tick], 'closeTime': self.dates[self._current_tick] }
            
            # Add Order Pips
            if trade['position'] == -1: self._pips += ((trade['openPrice'] - self.spread) - trade['closePrice']) * self.point
            elif trade['position'] == 1: self._pips += (trade['closePrice'] - (trade['openPrice'] + self.spread)) * self.point

            # Append Trade History
            self._trades.append(trade)

            # Update Last Trade Tick
            self._last_trade_tick = self._current_tick

        # Update New Position History
        self._position_history.append(self._position)
        self._position_time_score.append((self._current_tick - self._last_trade_tick) / self.eps_length)
        self._profit_history.append(float(np.exp(self._total_reward)))

        # Return Observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, step_reward, self._terminated, False, info


    def _get_observation(self):
        # obs shape = (window_size, feature_size)
        obs = self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick+1]

        # Get the Position Holding Information
        positions = np.array(self._position_history[(self._current_tick - self.window_size + 1):self._current_tick+1])
        holding_time = np.array(self._position_time_score[(self._current_tick - self.window_size + 1):self._current_tick+1])

        # Add position info to the observation
        result = np.concatenate((obs, positions[:, np.newaxis], holding_time[:, np.newaxis]), axis=1)
        if self.flatten: return result.flatten()
        else: return result

    def render(self) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
        ax1.set_xlabel('trading days')
        ax1.set_ylabel('profit')
        ax1.plot(self._profit_history)

        ax2.set_xlabel('trading days')
        ax2.set_xlabel('close price')
        window_ticks = np.arange(len(self._position_history) - self.window_size)
        eps_price = self.prices[self._start_tick:self._end_tick + 1]
        ax2.plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        ax2.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        ax2.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        ax2.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        ax2.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

        plt.show()
    
    def trades(self):
        return pd.DataFrame(self._trades)
    
    def pips(self):
        return self._pips
    
    def total_steps(self):
        return self._end_tick - self._start_tick
        
    def close(self):
        plt.close()

    def set_model(self, model): self._model = model

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def total_steps(self): return self._end_tick - self._start_tick + 1

    def _calculate_reward(self, action):
        step_reward = 0  # pip

        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee) * (1 - self.trade_fee))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)
        return step_reward
    
    def _process_data(self):
        # Technical Indicator Processing
        prices = self.df.loc[:, 'close']
        
        if self.normalise is None: self.df = extract_features(self.df)
        elif isinstance(self.normalise, float): self.df = extract_features(self.df, normalise=self.normalise, normalise_path=self.normalise_path)
        elif self.normalise == True: self.df = extract_features(self.df, normalise=True, normalise_path=self.normalise_path)

        self.df = self.df.sort_index(ascending=True)
        if self.bar_limit is not None: self.df = self.df[-self.bar_limit:]

        prices = prices[prices.index.isin(self.df.index)]
        prices = prices.sort_index(ascending=True)
        prices = prices.to_numpy()
        dates = self.df.index
        # self.df.to_csv('test.csv')

        signal_features = self.df.values
        # print(signal_features.shape)
        # print(signal_features)
        print(f'Forex Environment with {len(prices)} bars and {signal_features.shape} feature shape.')
        return prices, signal_features, dates