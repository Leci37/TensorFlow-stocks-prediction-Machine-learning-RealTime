# py_ti
A collection of 48 technical indicators. Suggestions are welcome.

# Current List:<br />
Accumulation/Distribution - acc_dist<br />
Average Directional Index - adx<br />
Average True Range - atr<br />
Average True Range Percent - atr_percent<br />
Bollinger Bands - bollinger_bands<br />
Camarilla Pivot Points - camarilla_pivots<br />
Chaikin Oscillator - chaikin_oscillator<br />
Choppiness Index - choppiness<br />
Classic Pivot Points - classic_pivots<br />
Commodity Channel Index - cci<br />
Coppock Curve - coppock<br />
Demark Pivot Points - demark_pivots<br />
Donchian Channels - donchian_channels<br />
Ease of Movement - ease_of_movement<br />
Exponential Moving Average - ema<br />
Fibonacci Moving Average - fma<br />
Fibonacci Pivot Points - fibonacci_pivots<br />
Force Index - force_index<br />
Historical Volatility - hvol<br />
Hull Moving Average - hma<br />
Kaufman's Adaptive Moving Average - kama<br />
Keltner Channels - keltner_channels<br />
KST Oscillator - kst<br />
Log Returns - returns(ret_method='log')<br />
MACD - macd<br />
Mass Index - mass_index<br />
Momentum - momentum<br />
Money Flow Index - money_flow_index<br />
On-Balance Volume - obv<br />
Parabolic Stop-and-Reverse - parabolic_sar<br />
Rate of Change - rate_of_change<br />
Relative Strength Index - rsi<br />
RSI-Stochastic Oscillator - rsi_stochastic<br />
Simple Moving Average - sma<br />
Simple Returns - returns(ret_method='simple')<br />
Stochastic Oscillator - stochastic<br />
Stochastic-RSI Oscillator - stochastic_rsi<br />
Supertrend - supertrend<br />
Traditional Pivot Points - trad_pivots<br />
Triangular RSI - triangular_rsi<br />
True Range - true_range<br />
TRIX - trix<br />
True Strength Index - tsi<br />
Ultimate Oscillator - ultimate_oscillator<br />
Vortex Indicator - vortex<br />
Weighted Moving Average - wma<br />
Wilder's Moving Average - wilders_ma<br />
Woodie Pivot Points - woodie_pivots<br />

# Data
Data should be in open/high/low/close/volume format in a Pandas DataFrame with the date as the index.<br />
ohlc = float<br />
volume = int<br />
date = Datetime<br />

Data Example:  
![data_example](https://user-images.githubusercontent.com/29778401/105869496-4b36a300-5fc5-11eb-8324-aaa0fc98f37d.png)

# Versions used:
python 3.8.10<br />
numpy 1.19.2<br />
pandas 1.2.3<br />
numba 0.53.0<br />
