# Trading Reinforcement Learning Environments
#### based on https://github.com/nova-land

#### the model is trained with RL , instead of DL Examples of tests performed 

A collection of multiple Forex Reinforcement Learning Environments originated from [AminHP](https://github.com/AminHP).


## Requirements

- Python 3.10
- Anaconda
- Install packages from `requirements.txt`
- MetaTrader5 Terminal for `Collection 3 Env`.
- You need **Windows** to run `Collection 3 Env` as MetaTrader5 only supports Windows OS.

1. Install the [PyTorch](https://pytorch.org/get-started/locally/) correctly.
2. Install `stablebaseline3` and `stable-baselines3-contrib` with alpha version that supports gymnasium:
   - `pip install git+https://github.com/DLR-RM/stable-baselines3`
   - `pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib`

---
## Collection 1 : `env/forex_env.py`

This is a simple Reinforcement Learning Environment which contains 2-Discrete Action: `Buy` or `Sell`.

### Action Cycle

|Previous Action|Next Action|Next Position|
|---------------|-----------|-------------|
|Sell|Sell|Keep The Short Position|
|Sell|Buy|Close Short Position & Open Long Position|
|Buy|Buy|Keep The Long Position|
|Buy|Sell|Close Long Position & Open Short Position|

With the above action cycle, the strategy will be 100% time in the market. There will always be one long/short position.


## How to Use (no updated)


1. Gather your bar data into `./data/{your_bar_data}.csv`, the bar data should contains `date,open,high,low,close`.
2. Edit the `config.yml` to use your bar data.
3. Edit the `util/ta.py` to add your favourite features.
4. Run the Example Script to test it out.
   - Collection 1 : `python examples/simple.py`
   - Collection 2 : `python examples/intermediate.py`
   - Collection 3 : `python examples/advanced.py` 
   - Collection 4 : `python examples/multi.py`
   - - Collection 4 : `python examples/multi_v2.py`
5. Use `tensorboard --logdir=./logs` to get the real-time reward chart.

---

## References

- [gym-anytrading](https://github.com/AminHP/gym-anytrading)
- [gym-mtsim](https://github.com/AminHP/gym-mtsim)
