from gymnasium.envs.registration import register
from .trading_env import TradingEnv
from .trading_env_v2 import TradingEnvV2
from .mt_env import MtEnv
from .multi_env import MultiTradeEnv
from .multi_env_v2 import MultiTradeEnvV2

register(id='trade-v0', entry_point='env:TradingEnv')
register(id='trade-v2', entry_point='env:TradingEnvV2')
register(id='metatrader-v0', entry_point='env:MtEnv')
register(id='multi-v0', entry_point='env:MultiTradeEnv')
register(id='multi-v2', entry_point='env:MultiTradeEnvV2')