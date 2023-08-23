import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from typing import TypedDict
import pandas as pd
from util.ta import extract_features

class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class Trade(TypedDict):
    position: Positions
    openPrice: float
    closePrice: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(
            self, df, window_size, symbol, spread, point, log_dir, flatten=False,
            bar_limit=None, render_mode=None, clear_trade=True,
            normalise=None, normalise_path=None
        ):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.point = point
        self.spread = spread/point
        self.bar_limit = bar_limit
        self.clear_trade = clear_trade
        self.normalise = normalise
        self.normalise_path = normalise_path
        self.prices, self.signal_features, self.dates = self._process_data()
        self.flatten = flatten
        if self.flatten: self.shape = (window_size * self.signal_features.shape[1],)
        else: self.shape = (window_size, self.signal_features.shape[1],)

        self.writer = tf.summary.create_file_writer(log_dir)
        self.episode = 0

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float64)
        print("Observation_space: ", self.observation_space)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        print("Tick_step \tStart: ", self._start_tick, " End: ", self._end_tick)
        self._terminated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._trades = []
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = None
        self._best_score = 0

        # create cache for faster training
        self._observation_cache = []
        for current_tick in range(self._start_tick, self._end_tick + 1):
            obs = self.signal_features[(current_tick-self.window_size+1):current_tick+1]
            self._observation_cache.append(obs)

    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            position = self._position.value
        )

    def reset(self, seed=None, options=None): #HERENCIA DE gym.Env
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self.episode += 1
        if self.clear_trade: self._trades = []
        self._first_rendering = True
        self.history = {}

        info = self._get_info()
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        print("\tDEBUG RESET_"+str(self._current_tick)+": ", info,  " current_tick-self.window_size: ",self._current_tick-self.window_size+1, ":", self._current_tick+1 ,"\tObservation.Shape: ", observation.shape, )
        return observation, info

    def step(self, action):#HERENCIA DE gym.Env
        self._terminated = False
        self._current_tick += 1

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            self._terminated = True

            # Draw the trades to tensorboard
            trades = self.trades()
            if trades.empty == False:
                # Calculate the trade profits
                trades.loc[trades['position'] == 0, 'profit'] = ((trades['openPrice'] - self.spread) - trades['closePrice']) * self.point
                trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + self.spread)) * self.point
                trades = trades.dropna()
                trades['cashflow'] = trades['profit'].cumsum()
                profit = trades['profit'].sum()# Que optimista
                # Draw the Cashflow to the tensorboard
                with self.writer.as_default():
                    for i, trade_cash in trades['cashflow'].items():
                        tf.summary.scalar(f'trades/iteration-{self.episode}', trade_cash, i)

                # Update Model
                if profit > self._best_score:
                    if hasattr(self, '_model') == True:
                        print(f'Saving Model... [+{round(profit, 2)} pips]')
                        self._model.save(f'output/best_{self.symbol}_v1.zip')

                    # Save the trades
                    self._best_score = profit
                    trades.to_csv(f'logs/{self.symbol}_trades_v1_train.csv', index=False)
                    print(f'logs/{self.symbol}_trades_v1_train.csv')

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        is_trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            is_trade = True
        # print("\tDEBUG STEP_"+str(self._current_tick)+": action: ",action ," position: ",self._position  , " is_trade: ", is_trade)

        if is_trade or self._terminated:
            # Update the trade result
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            trade_dict: Trade = { 'position': self._position.value, 'openPrice': last_trade_price, 'closePrice': current_price,
                             'openTime': self.dates[self._last_trade_tick], 'closeTime': self.dates[self._current_tick] }
            self._trades.append(trade_dict)

            # Update the last trade tick
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick
            print("\tDEBUG STEP_"+str(self._current_tick)+" Trade: ", trade_dict)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, self._terminated, False, info


    def _get_observation(self):
        result = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1] #TODO no entiendo
        if self.flatten: return result.flatten()
        else: return result

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):#HERENCIA DE gym.Env

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
        )

        plt.pause(0.01)


    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title: plt.title(title)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
        )
    
    def trades(self):
        return pd.DataFrame(self._trades)
        
    def close(self):#HERENCIA DE gym.Env
        print("Close HERENCIA DE gym.Env")
        plt.close()


    def save_rendering(self, filepath):
        print("save_rendering: ", filepath)
        plt.savefig(filepath)


    def pause_rendering(self):
        print("pause_rendering ")
        plt.show()

    def set_model(self, model): self._model = model

    def total_steps(self): return self._end_tick - self._start_tick + 1

    def _calculate_reward(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # Sum up the trade profit when we close the position
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Short:
                open_price = (last_trade_price - self.spread)
                # Use pips as reward
                # return (open_price - current_price) * self.point
                # Use returns as reward
                return (open_price - current_price) / open_price
            elif self._position == Positions.Long:
                open_price = (last_trade_price + self.spread)
                # Use pips as reward
                # return (current_price - open_price) * self.point
                return (current_price - open_price) / open_price
        else: return 0
    
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