from Utils import UtilsL
import yfinance as yf


from LogRoot.Logging import Logger
from _KEYS_DICT import Option_Historical

DICT_EXTERNAL_FACTORS ={
	"ZW=F" : "Wheat_F",
	"CL=F" : "Oil_F",
	"GC=F" : "Gold_F",
	"NG=F" : "NatGas_F" ,

	"EURUSD=X": "Euro" ,
	#Todo EURO STOXX 50  ={ ??
	"BTC-USD" : "BitCoin",
	"NQ=F":"Nasdaq_F",
	"^IXIC": "Nasdaq"
	#"SPX500": "^SPX",
	#"^IBEX" : "Ibex35",
}
list_external = list(DICT_EXTERNAL_FACTORS.keys())


def get_raw_stocks_values(opion):
	if opion == Option_Historical.YEARS_3:
		df_data = yf.download(tickers=list_external, period="3y",prepost=False, interval="1d")
	elif opion == Option_Historical.MONTH_3:
		df_data = yf.download(tickers=list_external, period="60d",prepost=False, interval="15m")


	# Plot the close prices
	df_data = df_data["Adj Close"]
	df_data['Date'] = df_data.index
	df_data.reset_index(drop=True, inplace=True)
	if opion == Option_Historical.YEARS_3:
		df_data = UtilsL.remove_weekend_data_values(df_data)
	for kcol, v in DICT_EXTERNAL_FACTORS.items():
		#df_dataR[kcol] = df_dataR[kcol].round(2)
		df_data = df_data.rename(columns={kcol: v})
	df_data.to_csv("d_external_factors/external_factors_hist_" + str(opion.name) + ".csv", sep="\t")
	Logger.logr.debug(" Generated File info: d_external_factors/external_factors_hist_"+str(opion.name)+".csv")

	# dict_j = Utils_Yfinance.prepare_df_to_json_by_date(df_dataR)
	# dict_json = {}
	# # dict_json['Date'] = dict_j
	# dict_json = dict_j
	# import json
	# with open("d_external_factors/external_factors_hist_"+str(opion.name)+".json", 'w') as fp:
	# 	json.dump(dict_json, fp, allow_nan=True)
	# Logger.logr.info("d_external_factors/external_factors_hist_"+str(opion.name)+".json  Numbres of Keys: " + str(len(dict_json)))


# get_raw_stocks_values(Option_Historical.YEARS_3)
# get_raw_stocks_values(Option_Historical.MONTH_3)

# df_dataR.Close.plot()
# plt.show()