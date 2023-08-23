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
from typing import TypedDict, List, Dict, Tuple
import pandas as pd
from datetime import datetime
from util.ratio import *
from util.ta import extract_features

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

class Trade(TypedDict):
    symbol: str
    position: Positions
    openPrice: float
    closePrice: float
    profit: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class Memory(TypedDict):
    position: Positions
    equity: float
    last_trade_tick: int
    position_history: List[int]
    equity_history: List[float]
    returns_history: List[float]

class SymbolConfig(TypedDict):
    spread: float
    point: float

class RewardType(str, Enum):
    SHARPE = 'sharpe'
    OMEGA = 'omega'
    SORTINO = 'sortino'
    KAPPA = 'kappa'
    GAIN_LOSS = 'gain_loss'
    UPSIDE_POTENTIAL = 'upside_potential'
    CALMAR = 'calmar'
    EQUITY = 'equity'

class EnvConfig(TypedDict):
    symbols: Dict[str, SymbolConfig]
    window_size: int
    bar_limit: int

    reward_period: int
    reward_scaling: float
    reward_type: RewardType

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

class MultiTradeEnvV2(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data: Dict[str, pd.DataFrame], config: EnvConfig, log_dir, flatten=False, render_mode=None, clear_trade=True, normalise=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.clear_trade = clear_trade
        self.normalise = normalise

        # The Dataframe should be { [symbol]: ohlc dataframe }
        self.data = data

        self.bar_limit = config['bar_limit']
        self.window_size = config['window_size']
        self.symbols = config['symbols']
        self.flatten = flatten
        self.symbol_names = list(self.data.keys())
        # The Reward Calculation Period
        self.reward_period = config['reward_period']
        # The Reward Calculation Scaling
        self.reward_scaling = config['reward_scaling']
        self.reward_type = config['reward_type']

        # All spread need to divide by the point
        for symbol in self.symbols:
            self.symbols[symbol]['spread'] /= self.symbols[symbol]['point']
        
        self.prices, self.signal_features, self.dates = self._process_data()

        if self.flatten == True:
            self.shape = (self.window_size * self.signal_features.shape[1],)
        else:
            self.shape = (self.window_size, self.signal_features.shape[1],)

        self.writer = tf.summary.create_file_writer(log_dir)
        self.episode = 0

        # Environment Action SPace
        # We have one action space per symbol [0, 1, 2]
        self.action_space = spaces.MultiDiscrete([len(Actions)] * len(self.symbols))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # Episode variables
        self._start_tick = self.window_size
        self._end_tick = len(self.dates) - 1
        self._terminated = None
        self._current_tick = None
        self._trades = []
        self._best_score = 0
        self.reward_memory = (self.window_size + 1) * [0]

    def terminated(self): return self._terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self.reward_memory = (self.window_size + 1) * [0]

        # Create the env memory for positions/trades
        self.memory: Dict[str, Memory] = {}
        for symbol in self.symbol_names:
            self.memory[symbol] = {
                'position': Positions.FLAT,
                'equity': 0,
                'equity_history': (self.window_size + 1) * [0],
                'returns_history': (self.window_size + 1) * [0],
                'last_trade_tick': self._current_tick + 1,
                'position_history': (self.window_size + 1) * [Positions.FLAT.value],
            }
        
        self.episode += 1
        if self.clear_trade: self._trades = []

        return self._get_observation(), self._get_info()
    
    # Action is Multiple Discrete : [2,1,0,1,2,..,], len = len(symbols)
    def step(self, action: np.ndarray):
        step_returns = []
        for i, symbol in enumerate(self.symbol_names):
            # Get Symbol Action [0,1,2]
            # Compute the next position
            last_position = self.memory[symbol]['position']
            next_position, trade = transform(last_position, action[i])
            self.memory[symbol]['position'] = next_position
            # Update the New Position History
            self.memory[symbol]['position_history'].append(next_position.value)

            # ----- Calculate step returns & equity ----- #
            current_price = self.prices[symbol][self._current_tick]
            last_trade_tick = self.memory[symbol]['last_trade_tick']
            last_trade_price = self.prices[symbol][last_trade_tick]

            # Case 1 : Long Position -> Calculate Step return & equity
            if last_position == Positions.LONG:
                open_price = (last_trade_price + self.symbols[symbol]['spread'])
                self.memory[symbol]['equity'] = (current_price - open_price) * self.symbols[symbol]['point']
                self.memory[symbol]['returns_history'].append((current_price - open_price) / open_price)

            # Case 2 : Short Position -> Calculate Step return & equity
            elif last_position == Positions.SHORT:
                open_price = (last_trade_price - self.symbols[symbol]['spread'])
                self.memory[symbol]['equity'] = (open_price - current_price) * self.symbols[symbol]['point']
                self.memory[symbol]['returns_history'].append((open_price - current_price) / open_price)
            
            # Case 3 : Flat -> No equity
            else:
                self.memory[symbol]['equity'] = 0
                self.memory[symbol]['returns_history'].append(0)
            self.memory[symbol]['equity_history'].append(self.memory[symbol]['equity'])

            # Get the step returns array back to reward period
            step_returns.extend(self.memory[symbol]['returns_history'][-self.reward_period:]) 

            # Update the Trade History & Trade Tick
            if trade:
                if last_position != Positions.FLAT:
                    trade_data: Trade = {
                        'symbol': symbol,
                        'position': last_position.value, 'openPrice': open_price, 'closePrice': current_price,
                        'openTime': self.dates[last_trade_tick], 'closeTime': self.dates[self._current_tick], 'profit': self.memory[symbol]['equity'] }

                    self._trades.append(trade_data)
                self.memory[symbol]['last_trade_tick'] = self._current_tick

        # Calculate the reward based on the return
        step_reward = self._calculate_reward(step_returns)
        self.reward_memory.append(step_reward)

        self._terminated = False
        self._current_tick += 1

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            # Draw the trades to tensorboard
            trades = self.trades()
            if trades.empty == False:
                # Calculate the trade profits
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
                        self._model.save(f'output/best_multi_v2.zip')
                        
                    # Save the trades
                    self._best_score = profit
                    trades.to_csv(f'logs/multi_trades_v2_train.csv', index=False)
            self._terminated = True

        # Return the observation
        return self._get_observation(), step_reward, self._terminated, False, self._get_info()


    def _get_info(self):
        return {
            'memory': self.memory,
            'reward': self.reward_memory
        }
    
    def _calculate_reward(self, r: List[float]) -> float:
        '''
        returns: List[float]
        The list of last X bars returns ratio of all symbols
        '''

        e = np.mean(r) # Get the expected return
        f = 0.0 # Risk free rate (all trade at risk)

        # Calculate the reward
        if self.reward_type == RewardType.SHARPE: ratio = sharpe_ratio(e, r, f)
        elif self.reward_type == RewardType.OMEGA: ratio = omega_ratio(e, r, f)
        elif self.reward_type == RewardType.SORTINO: ratio = sortino_ratio(e, r, f)
        elif self.reward_type == RewardType.KAPPA: ratio = kappa_three_ratio(e, r, f)
        elif self.reward_type == RewardType.GAIN_LOSS: ratio = gain_loss_ratio(r)
        elif self.reward_type == RewardType.UPSIDE_POTENTIAL: ratio = upside_potential_ratio(r)
        elif self.reward_type == RewardType.CALMAR: ratio = calmar_ratio(e, r, f)
        elif self.reward_type == RewardType.EQUITY: ratio: float = np.mean(r)
        else: ratio = 0

        if math.isnan(ratio):
            print('Warning: Ratio returns NAN')
            return 0
        
        return ratio * self.reward_scaling
    
    def _get_observation(self):
        # obs shape = (window_size, feature_size)
        result = self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick+1]
        if self.flatten: return result.flatten()
        else: return result

    def set_model(self, model): self._model = model

    def trades(self):
        return pd.DataFrame(self._trades)
    
    def total_steps(self): return self._end_tick - self._start_tick + 1
    
    def _process_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DatetimeIndex]:
        # Technical Indicator Processing
        feature_df: pd.DataFrame = None
        price_df: pd.DataFrame = None
        for symbol in self.data:
            print(f'Processing Bar data on {symbol}...')
            df = self.data[symbol]
            # Get the close price
            close_price = df['close']

            # Extract features from the symbol data
            normalise_path=f'data/{symbol}_scaler.pkl'
            if self.normalise is None: df = extract_features(df)
            elif isinstance(self.normalise, float): df = extract_features(df, normalise=self.normalise, normalise_path=normalise_path)
            elif self.normalise == True: df = extract_features(df, normalise=True, normalise_path=normalise_path)

            df = df.sort_index(ascending=True)
            df.add_prefix(symbol)

            if feature_df is None:
                feature_df = df
                price_df = pd.DataFrame({ f'{symbol}': close_price })
            else:
                feature_df = pd.concat([feature_df, df], axis=1)
                price_df[symbol] = close_price

        # TODO: Check the alignment and missing bars
        if self.bar_limit is not None: feature_df = feature_df[-self.bar_limit:]
        
        feature_df = feature_df.dropna()
        price_df = price_df[price_df.index.isin(feature_df.index)]

        dates = feature_df.index
        signal_features = feature_df.values
        price_data = {}
        for symbol in self.data:
            price_data[symbol] = price_df[symbol].values

        print(f'Forex Environment with {len(dates)} bars and {signal_features.shape} feature shape.')
        print(f'Forex Environment start from {dates[0]} to {dates[-1]}')
        return price_data, signal_features, dates