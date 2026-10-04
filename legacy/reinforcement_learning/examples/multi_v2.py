import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from env.multi_env_v2 import MultiTradeEnvV2, EnvConfig, RewardType

import pandas as pd

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/DLR-RM/stable-baselines3
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO 

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

if __name__ == "__main__":
    # Setting on Multiple symbols
    config: EnvConfig = {
        'symbols': {
            'EURUSD': { 'point': 10000, 'spread': 0.4 },
            'GBPUSD': { 'point': 10000, 'spread': 0.6 },
            'USDJPY': { 'point': 100,   'spread': 1.4 },
            'USDCAD': { 'point': 10000, 'spread': 1.4 },
            'USDCHF': { 'point': 10000, 'spread': 1.4 },
            'GBPJPY': { 'point': 100,   'spread': 2.6 },
            'NZDUSD': { 'point': 10000, 'spread': 2.7 },
            'XAUUSD': { 'point': 10,    'spread': 10.0 },
        },
        'bar_limit': None,
        'reward_period': 20,
        'reward_scaling': 0.01,
        'reward_type': RewardType.SHARPE,
        'window_size': 20,
    }
    log_dir = f'./logs/multi_v2/{datetime.now().strftime("%Y%m%d_%H%M")}'

    # Train/Test Split Ratio
    split_ratio = 0.7

    # Training epochs
    epochs = 100

    # Get all bar data
    data = {}
    min_bars = np.inf
    for symbol in config['symbols']:
        data[symbol] = pd.read_csv(f'data/{symbol}_D1.csv', index_col='date', parse_dates=['date'])
        min_bars = min(min_bars, len(data[symbol]))
    
    seed = 888 #random seed
    set_random_seed(seed)

    print(f'Preparing Environment for {list(config["symbols"].keys())}...')
    train_size = int(min_bars * split_ratio)
    print(f'Training Period with ~{train_size} bars')
    print(f'Testing  Period: ~{min_bars - train_size} bars')
    
    # Create the Training Environment
    train_data = {}
    for symbol in data: train_data[symbol] = data[symbol][:train_size]
    train_env = MultiTradeEnvV2(train_data, config, log_dir=log_dir, flatten=True, clear_trade=True, normalise=1.0)

    # Start the Learning Process
    if config['bar_limit'] is not None: learning_step = epochs * config['bar_limit']
    else: learning_step = epochs * len(train_env.dates)

    model = RecurrentPPO(
                'MlpLstmPolicy', env=train_env, ent_coef=0.01, learning_rate=0.00025, batch_size=128,
                policy_kwargs={ 'n_lstm_layers': 3, 'lstm_hidden_size': 256 },
                tensorboard_log=f'./logs/multi_{datetime.now().strftime("%Y%m%d_%H%M")}')
    train_env.set_model(model)
    model.learn(total_timesteps=learning_step, log_interval=10, progress_bar=True)
    model.save(f'output/multi.zip')

    # Create the evaluation environment
    # For examples, we combine the evaluation script here
    # This part should be seperate from the training code
    model = RecurrentPPO.load(f'output/best_multi.zip')

    test_data = {}
    for symbol in data: test_data[symbol] = data[symbol][train_size:]
    eval_env = MultiTradeEnvV2(test_data, config, log_dir=log_dir, flatten=True, clear_trade=False, normalise=True)
    eval_env.set_model(model)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)

    # Get all trades
    trades = eval_env.trades()
    if trades.empty == True:
        print('The Model learned Nothing, no trades at all.')
        exit()

    # Calculate the trade profits
    trades['cashflow'] = trades['profit'].cumsum()

     # Export the trade logs
    trades.to_csv(f'logs/multi_trades.csv', index=False)

    # Plot the trade cashflow
    trades.index = pd.to_datetime(trades['openTime'])
    trades['cashflow'].plot()
    plt.xlabel('date')
    plt.ylabel('pips')
    plt.savefig(f'logs/multi_benchmark.png')
    plt.close()