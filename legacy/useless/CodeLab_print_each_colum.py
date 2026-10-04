"""https://github.com/Leci37/LecTrade LecTrade is a tool created by github user @Leci37. instagram @luis__leci Shared on 2022/11/12 .   . No warranty, rights reserved """
from matplotlib import pyplot
import pandas as pd

# load dataset

dataset = pd.read_csv("d_price/MELI_stock_history_MONTH_3.csv", index_col=False, sep='\t')
#read_csv('pollution.csv', header=0, index_col=0)


values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]

# plot each column
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

for y in range(0, len(dataset.index), len(groups) ) :
	#groups = [x+i for x in groups]
	i = 1
	pyplot.figure()
	for group in groups:
		pyplot.subplot(len(groups), 1, i)
		pyplot.plot(values[:, (group + y) ])
		pyplot.title(dataset.columns[(group + y)], y=0.5, loc='right')
		i += 1
	#pyplot.show()
	pyplot.savefig("d_price/MELI_img_" + dataset.columns[group + y])
	print("generate " + dataset.columns[group + y])