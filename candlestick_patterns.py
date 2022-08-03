import talib_technical_funtions
import yhoo_history_stock


#https://github.com/CanerIrfanoglu/medium/blob/master/candle_stick_recognition/identify_candlestick.py
import talib
import yfinance as yf

from LogRoot.Logging import Logger

stockId = "MSFT"


df_y = yhoo_history_stock.get_historial_data_3y(stockId)

yho_stk = yf.Ticker(stockId)
df_m = yho_stk.history(period="7d", prepost=True, interval="5m") #yhoo_history_stock.get_historial_data_1_month(stockId)


df_y = talib_technical_funtions.gel_all_TALIB_funtion(df_m)
df_y = df_y.round(3)

df_y.to_csv("d_price/" + stockId + "_ALL_history_.csv", sep="\t")
Logger.logr.info("d_price/" + stockId + "_ALL_history_.csv  stock: " + stockId + " Shape: " + str(df_y.shape))
# df_candle = df_m

# candle_names = talib.get_function_groups()['Pattern Recognition']
# func_groups = ['Volume Indicators', 'Volatility Indicators',
#                'Overlap Studies', 'Momentum Indicators']

#candle_2 = talib.get_function_groups()['candle_rankings']

# extract OHLC
# op = df_candle['Open']
# hi = df_candle['High']
# lo = df_candle['Low']
# cl = df_candle['Close']
# # create columns for each pattern
# for candle in candle_names:
#     # below is same as;
#     # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
#     df_candle[candle] = getattr(talib, candle)(op, hi, lo, cl)


# df_candle.to_csv("d_price/" + stockId + "_candle_history_.csv", sep="\t")
# Logger.logr.info("d_price/" + stockId + "_candle_history_.csv  stock: " + stockId + " Shape: " + str(df_candle.shape))