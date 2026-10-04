import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import numpy as np
from env.mt_env import MtEnv, MtSimulator
from env.simulator.mt5 import Timeframe

from sb3_contrib import RecurrentPPO

if __name__ == "__main__":
    trading_symbols = ['EURUSD'] #, 'GBPUSD', 'USDJPY']
    log_dir = f'./logs/mt5/{datetime.now().strftime("%Y%m%d_%H%M")}'

    # Create the Simulator
    sim = MtSimulator(
        unit='USD',
        balance=10000.,
        leverage=100.,
        stop_out_level=0.2,
        hedge=True,
    )

    print('Downloading Symbol data...')
    sim.download_data(
        symbols=trading_symbols,
        time_range=(datetime(2019, 1, 1), datetime.now()), timeframe=Timeframe.H1
    )

    # Get the Training Timepoint
    training_time = sim.symbols_data[trading_symbols[0]].index.to_pydatetime().tolist()
    training_time = training_time[:int(len(training_time) * 0.7)]

    # Create the MetaTrader Environment
    env = MtEnv(
        original_simulator=sim,
        trading_symbols=trading_symbols,
        window_size=20,
        time_points=training_time,
        hold_threshold=0.5,
        close_threshold=0.5,
        fee=lambda symbol: {
            'EURUSD': max(0., np.random.normal(0.00001, 0.00001)),
            'GBPUSD': max(0., np.random.normal(0.00004, 0.00001)),
            'USDJPY': max(0., np.random.normal(0.01, 0.0005)),
        }[symbol],
        symbol_max_orders=2,
        normalise=1.0,
        multiprocessing_processes=2,
        log_dir=log_dir,
        early_stop_ratio=0.1    # Early Env Stopout (0.1 = current balance lower than 90% of init balance will reset env)
    )

    model = RecurrentPPO('MultiInputLstmPolicy', env, policy_kwargs={ 'n_lstm_layers': 2, 'lstm_hidden_size': 256 }, tensorboard_log=log_dir)

    # Calculate the Training Epochs
    learning_steps = env.total_steps() * 200
    model.learn(total_timesteps=learning_steps, progress_bar=True)
    model.save(f'output/MT5.zip')

# Future TODO
# Real time : https://github.com/AminHP/gym-mtsim/issues/25#issuecomment-1094134516