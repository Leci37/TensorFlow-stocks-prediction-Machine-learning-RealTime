import os
import sys

import Pivot_point_util

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
    Nothing = 0
    Buy = 1
    Sell = 2

# class Reward(Enum):
#     Fail_buy_and_sell = -5
#     Zero  = 0
#     Accert_nothing = 0 #2200
#     Accert_buy = 10 #50
#     Accert_sell = 10 #50

# class Positions(Enum):
#     Short = 0
#     Long = 1
#
#     def opposite(self):
#         return Positions.Short if self == Positions.Long else Positions.Long

class Operation_Trade(TypedDict):
    openPrice: float
    closePrice: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class TradingEnv_luis(gym.Env):

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
        self.df_buy_sell = pd.DataFrame()
        print("Entrada df full data. Shape: ",df.shape, " Columns: ", str(df.columns))
        self.symbol = symbol
        self.window_size = window_size
        self.window_size_midle = self.window_size//2 # es por que hay que evaluar el valor medio de la ventana
        self.point = point
        self.spread = spread/point
        self.bar_limit = bar_limit
        self.clear_trade = clear_trade
        self.normalise = normalise
        self.normalise_path = normalise_path
        print("Process data ")
        self.Rewards_per_avg = pd.DataFrame()
        _, self.signal_features, self.dates, self.df_buy_sell, self.Rewards_per_avg  = self._process_data()
        self.Reward = self.created_balance_reward()

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
        self._end_tick = len( self.df_buy_sell) - 1
        print("Tick_step \tStart: ", self._start_tick, " End: ", self._end_tick)
        self._terminated = None
        self._current_tick = None
        self._last_trade_tick = None
        self.tried_trades = pd.DataFrame()
        # self._position = None
        # self._position_history = None
        # self._total_reward = None
        self._first_rendering = None
        self.history = None
        self._best_reward = 0

        # create cache for faster training TODO why ???
        # self._observation_cache = []
        # for current_tick in range(self._start_tick, self._end_tick + 1):
        #     #repety code in _get_observation(self):
        #     obs = self.signal_features[(current_tick-self.window_size+1):current_tick+1]
        #
        #     # result = self.signal_features[( self._current_tick - self.window_size + 1):self._current_tick + 1]  # Es la ventana de obserbacion
        #     self._observation_cache.append(obs)
        #     # print("Add observation ")

    def created_balance_reward(self):
        self.Rewards_per_avg = self.Rewards_per_avg['count']
        # ['Buy_True/Sell_True', 'Buy_True/Sell_False', 'Buy_False/Sell_True', 'Buy_False/Sell_False']

        class Reward_class(Enum):
            Fail_buy_and_sell = -5
            Zero = 0
            Accert_nothing = 1
            Accert_buy = self.Rewards_per_avg['Buy_True/Sell_True'] / self.Rewards_per_avg['Buy_True/Sell_False']  # +-41
            Accert_sell = self.Rewards_per_avg['Buy_True/Sell_True'] / self.Rewards_per_avg['Buy_False/Sell_True']  # +-39

        print("Balance Reward puntuation: \n\t", list(map(lambda x: str(x.name) +": "+str(x.value), Reward_class._member_map_.values())))
        return Reward_class

    def _get_info(self):
        return dict(
            # total_reward = self._total_reward#,
            # position = self._position.value
        )

    def reset(self, seed=None, options=None): #HERENCIA DE gym.Env
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        # self._position_history = (self.window_size * [None]) + [self._position]
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
        self._current_tick += 1 #the fisrth row are ignorated

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            self._terminated = True

            # Draw the trades to tensorboard
            trades = self.tried_trades
            if trades.empty == False:# si no esta vacio
                # Calculate the trade profits TODO
                # trades.loc[trades['position'] == 0, 'profit'] = ((trades['openPrice'] - self.spread) - trades['closePrice']) * self.point
                # trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + self.spread)) * self.point
                # trades = trades.dropna()
                # trades['cashflow'] = trades['profit'].cumsum()
                # profit = trades['profit'].sum()# Que optimista
                # # Draw the Cashflow to the tensorboard
                # with self.writer.as_default():
                #     for i, trade_cash in trades['cashflow'].items():
                #         tf.summary.scalar(f'trades/iteration-{self.episode}', trade_cash, i)

                # Update Model
                print("\nEND the cycle _total_reward: ",  self._total_reward , " _best_reward: ", self._best_reward)
                if self._total_reward > self._best_reward:
                    if hasattr(self, '_model') == True:
                        print(f'Saving Model... [+{round(self._total_reward, 2)} pips]    Puntuation: ', self._total_reward)
                        self._model.save(f'output/best_{self.symbol}_v1.zip')

                    # Save the trades
                    self._best_reward = self._total_reward
                    trades.to_csv(f'logs/{self.symbol}_trades_v1_train.csv', index=False)
                    print(f'logs/{self.symbol}_trades_v1_train.csv')
            #Avance a la siguiente ventana del ML
            # self._last_trade_tick = self._current_tick

        step_reward , dict_current_result = self._calculate_reward(action)
        if self._current_tick % 500 == 0:
            print("\t\t",self._current_tick, "_ reward : ",step_reward.name , "\t action: ", Actions(action), " PointReward: ", self._total_reward )
        info = {"Action": action, "Reward": step_reward.value, "Current_step": self._current_tick, 'date' : dict_current_result['close'].index[0],'close':dict_current_result['close'].values[0],
                'volume': dict_current_result['volume'].values[0],'touch_low': dict_current_result['touch_low'].values[0], 'touch_high':dict_current_result['touch_high'].values[0]}

        self.tried_trades = self.tried_trades.append(info, ignore_index=True)
        self._total_reward += step_reward.value
        # dict_current_result['close'],'date', 'volume','touch_low','touch_high' ]
        # is_trade = False
        # if ((action == Actions.Buy.value and self._position == Positions.Short) or
        #     (action == Actions.Sell.value and self._position == Positions.Long)):
        #     is_trade = True
        # print("\tDEBUG STEP_"+str(self._current_tick)+": action: ",action ," position: ",self._position  , " is_trade: ", is_trade)

        # if  self._terminated: #is_trade or
        #     # Update the trade result
        #     current_price = self.prices[self._current_tick]
        #     last_trade_price = self.prices[self._last_trade_tick]
        #     trade_dict: Operation_Trade = {'position': self._position.value, 'openPrice': last_trade_price, 'closePrice': current_price,
        #                      'openTime': self.dates[self._last_trade_tick], 'closeTime': self.dates[self._current_tick]}
        #
        #
        #     # Update the last trade tick
        #     # self._position = self._position.opposite()
        #     #Avance a la siguiente ventana del ML
        #     self._last_trade_tick = self._current_tick
        #     print("\tDEBUG STEP_"+str(self._current_tick)+" Trade: ", trade_dict)

        # self._position_history.append(self._position)
        observation = self._get_observation()
                  #  self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward.value, self._terminated, False, info


    def _get_observation(self):
        result = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1] #Es la ventana de obserbacion
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
            # start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        # plt.suptitle(
        #     "Total Reward: %.6f" % self._total_reward
        # )

        plt.pause(0.01)


    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        # for i, tick in enumerate(window_ticks):
        #     if self._position_history[i] == Positions.Short:
        #         short_ticks.append(tick)
        #     elif self._position_history[i] == Positions.Long:
        #         long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title: plt.title(title)
        # plt.suptitle(
        #     "Total Reward: %.6f" % self._total_reward
        # )
    
    def tried_trades(self):
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
        # trade = False
        # if ((action == Actions.Buy.value and self._position == Positions.Short) or
        #     (action == Actions.Sell.value and self._position == Positions.Long)):
        #     trade = True

        # # Sum up the trade profit when we close the position
    # if trade:

        current_position_validation =  self._current_tick - (self.window_size_midle)
        dict_current_result  = dict(self.df_buy_sell[current_position_validation:current_position_validation+1 ] )
        reward_return = self.Reward.Zero
        #compara abajo
        if dict_current_result['touch_low'].values[0] == True and dict_current_result['touch_high'].values[0] == True:
            # print(current_position_validation , "_ Warn Both of it positive BUY and SELL at the same time.  "  )
            reward_return = self.Reward.Fail_buy_and_sell
        elif action == Actions.Buy.value and dict_current_result['touch_low'].values[0] == True:
            # print(current_position_validation , "_ Actions.Buy result['touch_low'] ")
            reward_return =  self.Reward.Accert_buy #10 points
        elif action == Actions.Sell.value and dict_current_result['touch_high'].values[0] == True:
            # print(current_position_validation, "_ Actions.Sell result['touch_high'] ")
            reward_return =  self.Reward.Accert_sell #10 points
        elif action == Actions.Nothing.value and dict_current_result['touch_low'].values[0] == False and dict_current_result['touch_high'].values[0] == False:
            reward_return =  self.Reward.Accert_nothing

        return reward_return , dict_current_result


    def _process_data(self):
        # Technical Indicator Processing
        prices = self.df.loc[:, 'close']

        LEN_RIGHT = self.window_size
        LEN_LEFT = self.window_size # LEN_RIGHT * 2

        # if self.normalise is None:
        #     self.df = extract_features(self.df)
        # elif isinstance(self.normalise, float):
        #     self.df = extract_features(self.df, normalise=self.normalise, normalise_path=self.normalise_path)
        # elif self.normalise == True:
        #     self.df = extract_features(self.df, normalise=True, normalise_path=self.normalise_path)

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

        df_buy_sell , df_per = Pivot_point_util.get_HT_pp(self.df , int(self.window_size/2), self.window_size )



        return prices, signal_features, dates, df_buy_sell , df_per
