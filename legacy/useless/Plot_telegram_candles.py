# Raw Package
import numpy as np
import pandas as pd
import talib
from pandas_datareader import data as pdr
from ta.trend import MACD
from ta.momentum import StochasticOscillator
# https://medium.com/@jsteinb/python-adding-features-to-your-stock-market-dashboard-with-plotly-4208d8bc3bd5
# Market Data
import yfinance as yf

# Graphing/Visualization
import datetime as dt
import plotly
import plotly.graph_objs as go

# Override Yahoo Finance
yf.pdr_override()

# Create input field for our desired stock
stock = input("Enter a stock ticker symbol: ")

# Retrieve stock data frame (df) from yfinance API at an interval of 1m
df = yf.download(tickers=stock, period='1d', interval='1m')

df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

# MACD
# macd = MACD(close=df['Close'],
#             window_slow=26,
#             window_fast=12,
#             window_sign=9)
# df_macd = pd.DataFrame()
df["mtum_MACD"], df["mtum_MACD_signal"] ,df["mtum_MACD_list"] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# stochastic
# stoch = StochasticOscillator(high=df['High'],
#                              close=df['Close'],
#                              low=df['Low'],
#                              window=14,
#                              smooth_window=3)
df["mtum_STOCH_k"], df["mtum_STOCH_d"] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# Declare plotly figure (go)
fig = go.Figure()

# add subplot properties when initializing fig variable
fig = plotly.subplots.make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.01,
                                    row_heights=[0.5, 0.1, 0.2, 0.2])

fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'], name='market data'))

fig.add_trace(go.Scatter(x=df.index,
                         y=df['MA5'],
                         opacity=0.7,
                         line=dict(color='blue', width=2),
                         name='MA 5'))

fig.add_trace(go.Scatter(x=df.index,
                         y=df['MA20'],
                         opacity=0.7,
                         line=dict(color='orange', width=2),
                         name='MA 20'))

# Plot volume trace on 2nd row
colors = ['green' if row['Open'] - row['Close'] >= 0
          else 'red' for index, row in df.iterrows()]
fig.add_trace(go.Bar(x=df.index,
                     y=df['Volume'],
                     marker_color=colors
                     ), row=2, col=1)

# Plot MACD trace on 3rd row
colorsM = ['green' if val >= 0
           else 'red' for val in df["mtum_MACD_list"]]
fig.add_trace(go.Bar(x=df.index,
                     y=df["mtum_MACD_list"],
                     marker_color=colorsM
                     ), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index,
                         y=df["mtum_MACD"],
                         line=dict(color='black', width=2)
                         ), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index,
                         y=df["mtum_MACD_signal"],
                         line=dict(color='blue', width=1)
                         ), row=3, col=1)

# Plot stochastics trace on 4th row
fig.add_trace(go.Scatter(x=df.index,
                         y=df["mtum_STOCH_k"],
                         line=dict(color='black', width=2)
                         ), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index,
                         y=df["mtum_STOCH_d"],
                         line=dict(color='blue', width=1)
                         ), row=4, col=1)

# update layout by changing the plot size, hiding legends & rangeslider, and removing gaps between dates
fig.update_layout(height=900, width=1200,
                  showlegend=False,
                  xaxis_rangeslider_visible=False)

# Make the title dynamic to reflect whichever stock we are analyzing
fig.update_layout(
    title=str(stock) + ' Live Share Price:',
    yaxis_title='Stock Price (USD per Shares)')

# update y-axis label
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
fig.update_yaxes(title_text="Stoch", row=4, col=1)

fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            # dict(count=45, label="45m", step="minute", stepmode="backward"),
            # dict(count=1, label="HTD", step="hour", stepmode="todate"),
            # dict(count=3, label="3h", step="hour", stepmode="backward"),
            # dict(step="all")
        ])
    )
)

fig.show()