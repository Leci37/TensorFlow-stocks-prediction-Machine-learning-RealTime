


[INTRODUCTION](#_heading=h.gjdgxs)

[OBJECTIVE](#_heading=h.30j0zll)

[OPERATION](#_heading=h.1fob9te)

[1.1 Data collection](#_heading=h.3znysh7)

[1.2 Types of indicators](#_heading=h.2et92p0)

[2 Indicator filtering](#_heading=h.tyjcwt)

[3 Training TensorFlow, XGB and Sklearn models](#_heading=h.3dy6vkm)

[4.1 Assessing the QUALITY of these models](#_heading=h.1t3h5sf)

[4.2 Evaluating those real BENEFITS of models](#_heading=h.4d34og8)

[5.1 Making predictions for the past week](#_heading=h.2s8eyo1)

[5.2 Sending real-time alerts](#_heading=h.17dp8vu)

[Quick start-up](#_heading=h.3rdcrjn)

[Commissioning](#_heading=h.26in1rg)

[1 Historical data collection](#_heading=h.lnxbz9)

[1.0 (Recommended) alphavantage API](#_heading=h.35nkun2)

[1.1 The OHLCV history of the stock must be generated.](#_heading=h.1ksv4uv)

[2 Filtering technical indicators](#_heading=h.44sinio)

[3 Generate training of TensorFlow, XGB and Sklearn models](#_heading=h.2jxsxqh)

[4 Evaluate quality of predictive models](#_heading=h.z337ya)

[5 Predictions](#_heading=h.3j2qqm3)

[5.0 make predictions for the last week Optional Test](#_heading=h.1y810tw)

[5.1 Getting OHLCV data in real time](#_heading=h.4i7ojhp)

[5.2 Setting up chatIDs and tokens in Telegram](#_heading=h.2xcytpi)

[5.3 Sending real-time alerts Telegram](#_heading=h.1ci93xb)

[Possible improvements](#_heading=h.3whwml4)

[Improvements in predictive models, using multi-dimensional](#_heading=h.2bn6wsx)

[Review the way ground true is obtained](#_heading=h.qsh70q)

[Add news sentiment indicator](#_heading=h.3as4poj)

[Add balance sheets](#_heading=h.1pxezwc)

[List of suggested improvements:](#_heading=h.49x2ik5)

[Indicator names:](#_heading=h.2p2csry)





### INTRODUCTION
The stock market is moved by technical indicators, there are several types of volatility, cycle volume, candlesticks, supports, resistances, moving averages...

An excellent site to see all the stock market technical indicators is webull https://app.webull.com/trade?source=seo-google-home. 

Image: webull with Stochastic, MACD and RSI indicators

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.001.png)


On the stock market graphs have been invented EVERY possible way to predict the stock market, with mixed results, making clear the difficulty of predicting human behavior.

These indicators indicate where to buy and sell, there are many beliefs about them (we mean in beliefs, because if they always worked we would all be rich). 

Any technical indicator can be obtained by means of programmable mathematical operations.

Three examples:

**RSI** or Relative Strength Index is an oscillator that reflects relative strength

Greater than 70 overbought, indicates that it will go down.

Less than 70 oversold, indicates that it will go higher 

**MACD** is the acronym for Moving Average Convergence / Divergence. The MACD in the stock market is used to measure the robustness of the price movement. Through the crossing of the line of this indicator and the moving average

It operates on the basis of the crossovers between these two lines

Or it is operated when both exceed zero.

**Candlestick: Morning Star** The morning star pattern is considered a hopeful sign in a bearish market trend.

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.002.png)


These indicators are present in refuted and popular websites like investing.com to be analyzed by the market <https://es.investing.com/equities/apple-computer-inc-technical>

It is extremely difficult to predict the price of any stock. Inflation, wars, populism, all this conditions affect the economy, and it becomes difficult, if not impossible to predict what Vladimir Putin will do tomorrow. 

Here enters the self-fulfilling prophecy principle of explained is, at first, a "false" definition of the situation, which awakens a new behavior that makes the original false conception of the situation become "true". Example:


![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.004.png)

### OBJECTIVE
Understanding the principle of self-fulfilling prophecy, it is possible to obtain the pattern of the same, by means of the massive collection of technical patterns, their calculation and the study of their patterns.


` `For this, techniques such as big data will be used through Pandas Python libraries, machine learning through Sklearn, XGB and neural networks through the open google Tensor Flow library. 

The result will be displayed in a simple and friendly way through alerts on mobile or computer.

Example of a real-time alert via telegram bot https://t.me/Whale\_Hunter\_Alertbot 

The machine learnig models Sklearn, XGB and Tensor Flow , by means of the learning of the last months detect the point of sale. To detect this point of sale a series of indicators have been taken into account: olap\_VMAP, ma\_SMA\_50, ichi\_senkou\_a, olap\_BBAND\_dif ,mtum\_MACD\_ext, olap\_BBAND\_MIDDLE, mtum\_MACD\_ext\_signal, fibo\_s1, volu\_PVI\_1, ma\_KAMA\_5, etcetera.

The image shows: MACD, RSI , Stochastic and Balance of power (Elder Ray) 

The alert is sent on the vertical line, during the next 4 periods the stock decreases by 2.4%. Each candlestick period in the image indicates 15 minutes.

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.005.png)![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.006.png)




### OPERATION

#### **1.1** Data collection
Collect data to train the model

yhoo\_generate\_big\_all\_csv.py

The closing data is obtained through yahoo API finance, and hundreds of technical patterns are calculated using the pandas\_ta and talib libraries. 

yhoo\_history\_stock.get\_SCALA\_csv\_stocks\_history\_Download\_list()

The model to be able to train in detecting points of purchase and sale, creates the column buy\_seel\_point has value of: 0, -100, 100. These are detected according to the maximum changes, (positive 100, negative -100) in the history of the last months, this point will be with which the training is trained, also called the *ground* true. 

Value will be assigned in buy\_seel\_point if the increase or decrease of the stock is greater than 2.5% in a period of 3 hours, using the get\_buy\_sell\_points\_Roll function.

Once the historical data of the stock has been obtained and all the technical indicators have been calculated, a total of 1068, files of type AAPL\_stock\_history\_MONTH\_3\_AD.csv are generated.

Example of the file with the first eight indicators:

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.007.png)

This data collection is customizable, you can obtain and train models of any Nasdaq stock, for other indicators or crypto-assets, it is also possible by making small changes.

**Through the Option\_Historical** class it is possible to create historical data files: annual, monthly and daily.

**class Option\_Historical**(Enum): YEARS\_3 = 1, MONTH\_3 = 2, MONTH\_3\_AD = 3, DAY\_6 = 4, DAY\_1 = 5

The files *\d\_price\_maxAAPL\_min\_max\_stock\_MONTH\_3.csv* are generated, which store the max and min value of each column, to be read in Model\_predictions\_Nrows.py for a quick fit\_scaler() (this is the "cleaning" process that the data requires before entering the AI training models) . This operation is of vital importance for a correct optimization in reading data in real time.




#### **1.2** Types of indicators
During the generation of the data collection file of point 1 AAPL\_stock\_history\_MONTH\_3\_AD.csv 1068 technical indicators are calculated, which are divided into subtypes, based on **prefixes** in the name.

List of prefixes and an example of the name of one of them.

- Overlap: **olap\_**

olap\_BBAND\_UPPER, olap\_BBAND\_MIDDLE, olap\_BBAND\_LOWER, 

- Momentum: **mtum\_**

mtum\_MACD, mtum\_MACD\_signal, mtum\_RSI, mtum\_STOCH\_k,

- Volatility: **vola\_**

vola\_KCBe\_20\_2, vola\_KCUe\_20\_2, vola\_RVI\_14

- Cycle patterns: **cycl\_**

cycl\_DCPHASE, cycl\_PHASOR\_inph, cycl\_PHASOR\_quad

- Candlestick patterns: **cdl\_**

cdl\_RICKSHAWMAN, cdl\_RISEFALL3METHODS, cdl\_SEPARATINGLINES

- Statistics: **sti\_**

sti\_STDDEV, sti\_TSF, sti\_VAR

- Moving averages: **ma\_**

ma\_SMA\_100, ma\_WMA\_10, ma\_DEMA\_20, ma\_EMA\_100, ma\_KAMA\_10, 

- Trend: **tend\_** and **ti\_**

tend\_renko\_TR, tend\_renko\_brick, ti\_acc\_dist, ti\_chaikin\_10\_3

- Resistors and support suffixes: **\_s3, \_s2, \_s1, \_pp, \_r1, \_r2, \_r3**

fibo\_s3, fibo\_s2, fibo\_s1, fibo\_pp, fibo\_r1, fibo\_r2, fibo\_r3, fibo\_r2, fibo\_r3

demark\_s1, demark\_pp, demark\_r1

- Intersection point with resistance or support: **pcrh\_.**

pcrh\_demark\_s1, pcrh\_demark\_pp, pcrh\_demark\_r1

- Intersection point with moving average or of moving averages between them: **mcrh\_.**

mcrh\_SMA\_20\_TRIMA\_50, mcrh\_SMA\_20\_WMA\_50, mcrh\_SMA\_20\_DEMA\_100

- Indicators of changes in the stock index, nasdaq: **NQ\_.**

NQ\_SMA\_20, NQ\_SMA\_100

Note: To see the 1068 indicators used go to the attached sheets at the end of the document.


#### **2** Indicator filtering
Execute to find out which columns are relevant for the model output

Feature\_selection\_create\_json.py

It is necessary to know which of the hundreds of columns of technical data, is valid to train the neural model, and which are just noise. This will be done through correlations and Random Forest models.

Answer the question:

Which columns are most relevant for buy or sell points?

Generate the *best\_selection* files, which are a raking of the best technical data to train the model, it is intended to go from 1068 columns to about 120.

For example, for the Amazon stock, point-of-purchase detection, in the period June to October 2022, the most valuable indicators are:

- Senkuo of the Ichimoku Cloud
- Chaikin Volatility 
- On-balance volume

Example of *plots\_relations/best\_selection\_AMNZ\_pos.json* file

**"index"**: {

`  `**"12"**: [

`     `**"ichi\_chilou\_span"**

`  `],

`  `**"10"**: [

`     `**"volu\_Chaikin\_AD"**

`  `],

`  `**"9"**: [

`     `**"volu\_OBV"**

`  `],

Plots with the 3 best technical data are printed in the folder *plots\_relations/plot.*

Example name: *TWLO\_neg\_buy\_sell\_point\_\_ichi\_chikou\_span.png*

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.008.png)

#### **3** Training TensorFlow, XGB and Sklearn models 
Model\_creation\_models\_for\_a\_stock.py

this requires the selection of better columns from point #2

There are four types of predictive algorithms, AI models:

- **Gradient Boosting** consists of a set of individual [decision trees](https://www.cienciadedatos.net/documentos/py07_arboles_decision_python.html), trained sequentially, so that each new tree tries to improve on the errors of the previous trees. Sklearn Library
- **Random Forest** Random forests are an ensemble learning method for classification, regression, and other tasks that operates by constructing a multitude of decision trees at training time. Sklearn Library 
- **XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost Library 
- **TensorFlow TF** is an open source library for machine learning across a range of tasks, and developed by Google to meet their needs for systems capable of building and training neural networks to detect and decipher patterns and correlations, analogous to the learning and reasoning used by humans. TensorFlow Library 


There are POS (buy) or NEG (sell) models and there is a BOTH model (BOTH is discarded, since prediction models are binary, they only accept 2 positions, true or false).

This point generates prediction models .sav for XGB and Sklearn. .h5 for Tensor Flow.

Naming Examples: XGboost\_U\_neg\_vgood16\_s28.sav and TF\_AMZN\_pos\_low1\_s128.h5

Format of the names:

- Type of AI you train with can be:
  - ` `XGboost, TF, TF64, GradientBoost and RandomForest
- Stock ticker AMZN for amazon , AAPL for Apple ...
- Detects points of purchase or sale pos or neg
- How many indicators have been used in the learning, can be of 4 types depending on the relevance given by point *#2 Indicator filtering*. This ranking is organized in the **MODEL\_TYPE\_COLM** class, 
  - vgood16 the best 16 indicators
  - good9 the best 32 indicators
  - reg4 the best 64 indicators 
  - low1 the best 128 indicators 
- Only for TF models. Depending on the density of the neurons used, defined in the class a\_manage\_stocks\_dict. **MODEL\_TF\_DENSE\_TYPE\_ONE\_DIMENSI** can take value: s28 s64 and s128

These combinations imply that for each stock 5 types of IA are created, each in pos and neg, plus for each combination the 4 technical indicator configurations are added.  This generates 40 IA models, which will be selected in point: *#4 to evaluate the QUALITY of those models.*

Each time an AI template is generated, a log file is generated: *TF\_balance\_TF\_AAPL\_pos\_reg4.h5\_accuracy\_87.6%\_\_loss\_2.74\_\_epochs\_10[160].csv*

It contains the accuracy and loss data of the model, as well as the model training records.




#### **4.1** Assessing the QUALITY of these models 
Model\_creation\_scoring.py

To make a prediction with the AIs, new data is collected and the technical indicators with which it has been trained are calculated according to the *best\_selection* files.

When the .h5 and .sav models are queried:

` `Is this a point of sale? 

These answer a number that can vary between 0.1 and 4 

The higher the number the more likely it is to be a correct buy/sell point.

Each model has a rating scale on which it is considered point of sale. For some models with a rating of more than 0.4 will be enough (usually the XGboost), while for others require more than 1.5 (usually the TF).

How do you know what the threshold score is for each model?

The Model\_creation\_scoring.py class generates the threshold score *threshold* files, which tell which threshold point is considered the buy-sell point.

Each AI model will have its own type file:

*Models/Scoring/AAPL\_neg\_\_when\_model\_ok\_threshold.csv*

For each action in *#3 train the TF, XGB and Sklearn models*, 40 AI models are generated. This class evaluates and selects the most accurate models so that only the most accurate ones will be executed in real time (usually between 4 and 8 are selected).

*Models/Scoring/AAPL\_neg\_\_groupby\_buy\_sell\_point\_000.json*

**"list\_good\_params"**: [

`  `**"r\_rf\_AFRM\_pos\_low1\_"**,

`  `**"r\_TF64\_AFRM\_pos\_vgood16\_"**,

`  `**"r\_TF64\_AFRM\_pos\_good9\_"**,

`  `**"r\_TF\_AFRM\_pos\_reg4\_"**

],


#### **4.2** Evaluating those real BENEFITS of models
Model\_predictions\_N\_eval\_profits.py

Answer the question: 

If you leave it running for N days, how much hypothetical money do you make?

Note: this should be run on data that has not been used in the training model, preferably

*Models/eval\_Profits/\_AAPL\_neg\_ALL\_stock\_20221021\_\_20221014.csv*


#### **5.1** Making predictions for the past week
Model\_predictions\_Nrows.py

You can make predictions with the real-time data of the stock.

Through the function call every 10-12min, download the real-time stock data through the yahoo financial API.

df\_compare, df\_sell = get\_RealTime\_buy\_seel\_points()

This run generates the log file *d\_result/prediction\_results\_N\_rows.csv*

This file and the notifications (telegram and mail) contain information about each prediction that has been made. It contains the following columns:

- Date: date of the prediction 
- Stock: stock 
- buy\_sell: can be either NEG or POS, depending on whether it is a buy or sell transaction. 
- Close: This is the scaled value of the close value (not the actual value).
- Volume: This is the scaled value of the Volume (not the actual value).
- 88%: Fractional format ( **5/6** ) How many models have predicted a valid operating point above 88%? Five of the six analyzed 
- 93%: Fractional format ( **5/6** ), number of models above 93%.
- 95%: Fractional format ( **5/6** ), number of models above 95%.
- TF: Fractional format ( **5/6** ), number of models above 93%, whose prediction has been made with Tensor Flow models. 
- Models\_names: name of the models that have tested positive, with the hit % (88%, 93%, 95%) as suffix 

Registration example

` `**2022-11-07 16:00:00 MELI NEG -51.8 -85.80 5/6 0/6 0/6 0/6 1/2 TF\_reg4\_s128\_88%, rf\_good9\_88%, rf\_low1\_88%, rf\_reg4\_88%, rf\_vgood16\_88%,**

To be considered a valid prediction to trade, it must have at least half of the fraction score in the 93% and TF columns.

More than half of the models have predicted with a score above 93% which is a good point for trading 



#### **5.2** Sending real-time alerts
predict\_POOL\_enque\_Thread.py *multithreading glued 2s per action* 

It is possible to run it without configuring telegram point 5.2, in that case no alerts will be sent in telegram, but if the results were recorded in real time in: *d\_result/prediction\_real\_time.csv*

There is the possibility to send alerts of purchase and sale of the share, to telegram or mail.

the multiple AI trained models are evaluated, and only those greater than 96% probability (as previously trained) are reported.

Every 15 minutes, **all** necessary indicators are calculated in real time for each action and evaluated in the AI models.

The alert indicates which models are detecting the correct buy and sell points at which to execute the transaction. 

These buy and sell alerts expire in, plus or minus 7 minutes, given the volatility of the market.

Also attached is the price at which it was detected, the time, and links to news websites.

Note: financial news should always prevail over technical indicators. 

What is displayed in DEBUG alert, is the information from *d\_result/prediction\_results\_N\_rows.csv* of the Item: 5 make predictions of the last week Test

To understand the complete information of the alert see Point 5.1 Making predictions of the last week.

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.009.png)










### Quick start-up 
Install requirements 

pip install -r requirements.txt

Run Utils/API\_alphavantage\_get\_old\_history.py

Run yhoo\_generate\_big\_all\_csv.py

Run Model\_creation\_models\_for\_a\_stock.py

Run Model\_creation\_scoring.py

Run Model\_predictions\_Nrows.py Optional, last week predictions 

Real time forecasts:

Run Utils/Volume\_WeBull\_get\_tikcers.py Ignore in case of using default configuration 

Configure bot token see point 5**.2** Configuring chatID and tokens in Telegram

Run predict\_POOL\_inque\_Thread.py

It is possible to run it without configuring telegram point **5.2**, in that case no alerts will be sent in telegram, but if the results were recorded in real time in: *d\_result/prediction\_real\_time.csv*


### Commissioning 
(Running times are estimated for an intel i3 and 8GB of RAM)


**0.0**The interpreter with which the tutorial has been made is python 3.8 , IDE Pycharm, caution with the compatibility between versions of the library pandas and python
For example: today do not use python 3.10 , as it is incompatible with pandashttps://stackoverflow.com/questions/69586701/unable-to-install-pandas-for-python 


**0.1** Download and install requirements, the project is powerful and demanding in terms of libraries.

pip install -r requirements.txt

**0.2** Search all files for the string ***\*\*DOCU\*\*.***

this allows to watch all files that are executable from the startup tutorial easily 

**0.3** In the file a\_manage\_stocks\_dict.py all the configurations are stored, look at the document and know where it is.

In it there is the dictionary DICT\_COMPANYS

It contains the IDs (GameStops quotes with the ID: **GME**) of the companies to analyze. You can customize and create a class from the **nasdaq** tikers, by default the key **@FOLO3** will be used which will analyze these 39 companies.

**"@FOLO3:** 

`   `[**"UPST"**, **"MELI"**, "**TWLO"**, **"RIVN"**, **"SNOW"**, "**LYFT"**, **"ADBE"**, **"UBER"**, **"ZI"**, **"QCOM"**, **"PYPL"**, **"SPOT"**, **"RUN"**, "**GTLB"**, **"MDB"**, **"NVDA"**, **"AMD" ADSK"**, **"ADSK"**, **"AMZN"**, **"CRWD"**, **"NVST"**, **"HUBS"**, **"EPAM"**, **"PINS"**, **"TTD"**, **"SNAP"**, **"APPS"**, **"ASAN"**, **"AFRM"**, **"DOCN"**, **"ETSY"**, "**DDOG"**, **"SHOP"**, "**NIO"**, **"U"**, "**GME"**, "**RBLX"**, **"CRSR"**],

If a faster execution is desired, it is recommended to delete items from the list and leave three

#### 1 Historical data collection
##### **1.0** (Recommended) alphavantage API
` `The API yfinance , if you want price to price intervals in 15min intervals is limited to 2 months, to get more time data up to 2 years back (more data better predictive models) use the free version of the API https://www.alphavantage.co/documentation/  

Run Utils/API\_alphavantage\_get\_old\_history.py

The class is customizable: action intervals, months to ask, and ID action.

Note: being the free version, there is a portrait between request and request, to get a single 2 years history it takes 2-3 minutes per action. 

Once executed, the folder: *d\_price/RAW\_alpha* will be filled with historical OHLCV .csv of share prices. These files will be read in the next step. Example name: alpha\_GOOG\_15min\_20221031\_\_20201112.csv

Check that one has been generated for each action in *d\_price/RAW\_alpha*.


##### **1.1 The** OHLCV history of the stock must be generated.
As well as the history of technical patterns. It takes +-1 minute per share to calculate all technical patterns. 

Run yhoo\_generate\_big\_all\_csv.py

Once executed the folder: *d\_price* will be filled with historical OHLCV .csv of share prices.

Three types of files are generated (Example of name type for action: AMD):

- *AMD\_SCALA\_stock\_stock\_history\_MONTH\_3\_AD.csv* with all technical patterns calculated and applied a fit scaler(-100, 100), i.e. the stock prices are scaled (size: 30-90mb)
- *d\_price/min\_max/AMD\_min\_max\_stock\_MONTH\_3\_AD.csv* with scaling keys (size: 2-7kb)
- *AMD\_stock\_history\_MONTH\_3\_AD.csv* the pure history of the OHLCVs (size: 2-7mb)

Note: *MONTH\_3\_AD* means 3 months of *API* yfinance plus the history collected from alphavantage. Point 1.0

Check that one has been generated for each action.


#### 2 Filtering technical indicators
It is necessary to separate the technical indicators which are related to buy or sell points and which are noise. 20 seconds per share 

Run Model\_creation\_scoring.py

Three files are generated for each action in the folder: *plots\_relations* , relations for purchase "pos", relations for sale "neg" and relations for both "both".

- *plots\_relations/best\_selection\_AMD\_both.json*

These files contain a ranking of which technical indicator is best for each stock. 

Check that three .json have been generated for each action in *plots\_relations* .

#### 3 Generate TensorFlow, XGB and Sklearn model training 
Train the models, for each action 36 different models are trained.

15 minutes per share.

Run Model\_creation\_models\_for\_a\_stock.py

The following files are generated for each action:

*Models/Sklearn\_smote* folder:

- XGboost\_AMD\_yyy\_xxx\_.sav
- RandomForest\_AMD\_yyy\_xxx\_.sav
- XGboost\_AMD\_yyy\_xxx\_.sav

*Models/TF\_balance* folder:

- TF\_AMD\_yyy\_xxx\_zzz.h5
- TF\_AMD\_yyy\_xxx\_zzz.h5\_accuracy\_71.08%\_\_loss\_0.59\_\_epochs\_10[160].csv

xxx can take value vgood16 good9 reg4 and low1 

yyy can take value "pos" and "neg".

zzz can take value s28 s64 and s128

Check that all combinations of files exposed by each action have been generated in the /Models subfolders.


#### 4 Evaluate quality of predictive models 
From the 36 models created for each OHLCV history of each stock, only the best ones will be run in real time, in order to select and evaluate those best ones.

Run Model\_creation\_scoring.py

In the *Models/Scoring* folder

AMD\_yyy\_\_groupby\_buy\_sell\_point\_000.json

AMD\_yyy\_\_when\_model\_ok\_threshold.csv

Check that two have been generated for each action.


#### 5 Predictions
##### **5.0** make predictions of the last week Optional Test 
Run Model\_predictions\_Nrows.py

This run generates the log file *d\_result/prediction\_results\_N\_rows.csv*

Generates a sample file with predictions for the last week, data obtained with yfinance. 

Check that records exist 


##### **5.1** Getting OHLCV data in real time
In case you want to predict actions in the @FOLO3 list, ignore this point. 

It is difficult to get real time OHLCV, especially volume (yfinance gives real time volume, but this is not a correct value and after 1-2 hours it changes, making it unfeasible to use yfinance for real time predictions).

To get correct volumes in real time, queries are made to webull, for each stock every 2.5 minutes, a webull ID is required, the default ones @FOLO3 are cached and downloaded in *a\_manage\_stocks\_dict.py. DICT\_WEBULL\_ID*

But if you want to use actions outside the list @FOLO3 

In Utils/Volume\_WeBull\_get\_tikcers.py

Change the example list:

` `list\_stocks = [**"NEWS"**, "**STOCKS"**, "**WEBULL"**, "**IDs"**]

By the nasdaq ticker, of the webull ID you want to get.

Run Utils/Volume\_WeBull\_get\_tikcers.py

Once executed it will show a list on screen, that must be added in *a\_manage\_stocks\_dict.py.DICT\_WEBULL\_ID*

***"MELI"** : 913323000,*

***"TWLO"** : 913254000,*


##### **5.2** Setting up chatIDs and tokens in Telegram
You have to get the telegram token and create a channel. 

You can get the token by following the tutorial: [https:](https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token)//www.siteguarding.com/en/how-to-get-telegram-bot-api-token 

With the token update the variable of ztelegram\_send\_message\_handle.py

*#Get from telegram*

TOKEN = **"00000000xxxxxxx"**

Once the token has been obtained, the chatId of the users and administrator must be obtained. 

Users only receive purchase and startup sale alerts, while the administrator receives alerts from users as well as possible problems.

To get the chatId of each user run ztelegram\_send\_message\_UptateUser.py and then write any message to the bot, the chadID appears both in the console and in the user's chadID 

*[>>> BOT] Message Send on 2022-11-08 22:30:31:31*

`	`*Text: You "User nickname " send me:* 

*"Hello world"*

` `*ChatId: "5058733760".*

`	`*From: Bot name*

`	`*Message ID: 915*

`	`*CHAT ID: 500000760*

*-----------------------------------------------*

Pick up *CHAT ID: 500000760*

With the chatId of the desired users, add them to the LIST\_PEOPLE\_IDS\_CHAT list.

in ztelegram\_send\_message\_handle.py


##### **5.3** Sending real-time alerts Telegram
It is possible to run it without configuring telegram, in that case no alerts will be sent in telegram, but the results will be recorded in real time in: *d\_result/prediction\_real\_time.csv*

It will be reported in console via: 

*is\_token\_telegram\_configurated() - Results will be recorded in real time, but no alert will be sent on telegram. File: d\_result/prediction\_real\_time.csv*

*is\_token\_telegram\_configurated() - There is no value for the telegram TOKEN, telegram is required to telegram one*

The criteria to send alert or not is defined in the method ztelegram\_send\_message.will\_send\_alert(). If more than half of the models have a score greater than 93% or the TF models have a score greater than 93%, an alert is sent to the consumer users. 

Run predict\_POOL\_inque\_Thread.py

In this class there are 2 types of threads 

- Producer , constantly asks for OHLCV data, once it is obtained, it enters it into a queue. 
- Consumer (2 threads running simultaneously) are pulling OHLCV data from the queue, calculating technical parameters, making model predictions, registering them in zTelegram\_Registers.csv, and if they meet the requirements they are sent by telegram. 
#### Possible improvements
##### Improvements in predictive models, using multi-dimensional 
Improvements in TF predictive models using tensors (multiple matrices over time) and non-matrices (mono temporal, current design). 

In the class Model\_TF\_definitions.ModelDefinition.py

Through it, the model configurations, density, number of neurons, etc. are obtained.

There are two methods:

- get\_dicts\_models\_One\_dimension() is currently used and generates TF model configurations for arrays. 
- get\_dicts\_models\_multi\_dimension() is not in use, it is set to give multiple model configurations using tensors. 

There is the Utils.Utils\_model\_predict.df\_to\_df\_multidimension\_array(dataframe, BACHT\_SIZE\_LOOKBACK) method, which transforms 2-dimensional df [columns , rows] to 3-dimensional df [columns , files, BACHT\_SIZE\_LOOKBACK ].

BACHT\_SIZE\_LOOKBACK means how many records in the past tense are added to the df, the number is configurable and default value is eight.

To start the development must be to call the method with BACHT\_SIZE\_LOOKBACK with an integer value, the method will return a multidimensional df [columns, files, BACHT\_SIZE\_LOOKBACK ], with which to feed the TF models.

Utils\_model\_predict.scaler\_split\_TF\_onbalance(df, label\_name=Y\_TARGET, BACHT\_SIZE\_LOOKBACK=8)

**Improvement**: Once these multidimensional arrays are returned, models are obtained with get\_dicts\_models\_multi\_dimension(), it is not possible to train a model and make a prediction with multidimensional arrays. 

##### Review the way ground true is obtained 
Before training the models the intervals (of 15min) are classified as buy point 100 or 101, sell point -100 or .-101 or no trade point 0, these values are entered in the column Y\_TARGET = **'buy\_sell\_point'** through the method Utils.Utils\_buy\_sell\_points.get\_buy\_sell\_points\_Roll().  

The variation is calculated with respect to the following 12 windows (15min \* 12 = 3 hours), and from there the 8% points of greatest rise and greatest fall are obtained, and these points are assigned values other than 0.

To obtain the Y\_TARGET there are 2 methods that are responsible for the strategy to follow once you buy and sell, in case of loss will opt for Stop Loss.

rolling\_get\_sell\_price\_POS() and rolling\_get\_sell\_price\_NEG()

**Optional improvement**: the current system decides by percentages, i.e. the 16% highest rises and falls (8% each) are ground true. I.e. there are rises or falls greater than 3% that can be left out if the stock is very volatile.


##### Add news sentiment indicator
You get the news for each stock with news\_get\_data\_NUTS.get\_news\_sentiment\_data() this method gets all the associated news from: INVESTING.com, YAHOO.com and FINVIZ.COM.

( it uses investpy API , which recently october 2022 has started to fail , probably due to investing[.com](https://github.com/alvarobartt/investpy) blocking <https://github.com/alvarobartt/investpy> )

Once these news items are obtained, the method news\_sentiment\_va\_and\_txtBlod.get\_sentiment\_predictorS() proceeds to evaluate and score from -100 negative to 100 positive, using 4 models. It is convenient to introduce more news pages

The models are downloaded from the internet, either via AI models or libraries, you can find the references in:

news\_sentiment\_flair.get\_sentiment\_flair

news\_sentiment\_t5.get\_sentiment\_t5

news\_sentiment\_t5.get\_sentiment\_t5Be

get\_sentiment\_textBlod

Run news\_get\_data\_NUTS.get\_json\_news\_sentimet()

A .csv and .json file is generated, with action date the four models, the score and the news collected Example: *d\_sentiment/stock\_news\_DATE\_MELI.csv*

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.010.png)

**Improvement**: Once the sentiment-news score file is obtained, introduce it in the predictive models together with the technical indicators, it must be done in real time.


##### Add balance sheets
Economic balances can be added easily using the yahoo API

<https://github.com/ranaroussi/yfinance>

\# show financials

msft.financials

msft.quarterly\_financials

These balances are updated every quarter.

You can get the dates of publication of results in yahoo API

\# show next event (earnings, etc)

msft.calendar

\# show all earnings dates

msft.earnings\_dates


##### List of suggested improvements:

Allow to analyze stocks outside the nasdaq, change in :

yhoo\_history\_stock.\_\_select\_dowload\_time\_config()

Utils/API\_alphavantage\_get\_old\_history.py

Redirect remaining print() to Logger.logr.debug()

Translate through <https://www.deepl.com/> the possible remaining messages in Spanish to English. 

The plots generated in the *plots\_relations/plot* folder by 

Change the operation of the bot, that is enough to send the command \start, and remove the case of execution of ztelegram\_send\_message\_UptateUser.py described in point: 5.2

Send real time email alert

Revise Stock prediction fail LSTM 

LSTM time series + stock price prediction = FAI 

[https://www.kaggle.com/code/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail ](https://www.kaggle.com/code/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail)

Find the explanation of what indicators and values the AI model takes, to predict what it predicts and give a small explanation-schema, for example random forest models if you can print the sequence that makes the prediction. 

(green buy, red do not trade) https://stackoverflow.com/questions/40155128/plot-trees-for-a-random-forest-in-python-with-scikit-learn 

![](readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.011.png)


### Indicator names:

**Date	buy\_sell\_point	Open	High	Low	Close	Volume	per\_Close	per\_Volume	has\_preMarket	per\_preMarket	olap\_BBAND\_UPPER	olap\_BBAND\_MIDDLE	olap\_BBAND\_LOWER	olap\_BBAND\_UPPER\_crash	olap\_BBAND\_LOWER\_crash	olap\_BBAND\_dif	olap\_HT\_TRENDLINE	olap\_MIDPOINT	olap\_MIDPRICE	olap\_SAR	olap\_SAREXT	mtum\_ADX	mtum\_ADXR	mtum\_APO	mtum\_AROON\_down	mtum\_AROON\_up	mtum\_AROONOSC	mtum\_BOP	mtum\_CCI	mtum\_CMO	mtum\_DX	mtum\_MACD	mtum\_MACD\_signal	mtum\_MACD\_list	mtum\_MACD\_crash	mtum\_MACD\_ext	mtum\_MACD\_ext\_signal	mtum\_MACD\_ext\_list	mtum\_MACD\_ext\_crash	mtum\_MACD\_fix	mtum\_MACD\_fix\_signal	mtum\_MACD\_fix\_list	mtum\_MACD\_fix\_crash	mtum\_MFI	mtum\_MINUS\_DI	mtum\_MINUS\_DM	mtum\_MOM	mtum\_PLUS\_DI	mtum\_PLUS\_DM	mtum\_PPO	mtum\_ROC	mtum\_ROCP	mtum\_ROCR	mtum\_ROCR100	mtum\_RSI	mtum\_STOCH\_k	mtum\_STOCH\_d	mtum\_STOCH\_kd	mtum\_STOCH\_crash	mtum\_STOCH\_Fa\_k	mtum\_STOCH\_Fa\_d	mtum\_STOCH\_Fa\_kd	mtum\_STOCH\_Fa\_crash	mtum\_STOCH\_RSI\_k	mtum\_STOCH\_RSI\_d	mtum\_STOCH\_RSI\_kd	mtum\_STOCH\_RSI\_crash	mtum\_TRIX	mtum\_ULTOSC	mtum\_WILLIAMS\_R	volu\_Chaikin\_AD	volu\_Chaikin\_ADOSC	volu\_OBV	vola\_ATR	vola\_NATR	vola\_TRANGE	cycl\_DCPERIOD	cycl\_DCPHASE	cycl\_PHASOR\_inph	cycl\_PHASOR\_quad	cycl\_SINE\_sine	cycl\_SINE\_lead	cycl\_HT\_TRENDMODE	cdl\_2CROWS	cdl\_3BLACKCROWS	cdl\_3INSIDE	cdl\_3LINESTRIKE	cdl\_3OUTSIDE	cdl\_3STARSINSOUTH	cdl\_3WHITESOLDIERS	cdl\_ABANDONEDBABY	cdl\_ADVANCEBLOCK	cdl\_BELTHOLD	cdl\_BREAKAWAY	cdl\_CLOSINGMARUBOZU	cdl\_CONCEALBABYSWALL	cdl\_COUNTERATTACK	cdl\_DARKCLOUDCOVER	cdl\_DOJI	cdl\_DOJISTAR	cdl\_DRAGONFLYDOJI	cdl\_ENGULFING	cdl\_EVENINGDOJISTAR	cdl\_EVENINGSTAR	cdl\_GAPSIDESIDEWHITE	cdl\_GRAVESTONEDOJI	cdl\_HAMMER	cdl\_HANGINGMAN	cdl\_HARAMI	cdl\_HARAMICROSS	cdl\_HIGHWAVE	cdl\_HIKKAKE	cdl\_HIKKAKEMOD	cdl\_HOMINGPIGEON	cdl\_IDENTICAL3CROWS	cdl\_INNECK	cdl\_INVERTEDHAMMER	cdl\_KICKING	cdl\_KICKINGBYLENGTH	cdl\_LADDERBOTTOM	cdl\_LONGLEGGEDDOJI	cdl\_LONGLINE	cdl\_MARUBOZU	cdl\_MATCHINGLOW	cdl\_MATHOLD	cdl\_MORNINGDOJISTAR	cdl\_MORNINGSTAR	cdl\_ONNECK	cdl\_PIERCING	cdl\_RICKSHAWMAN	cdl\_RISEFALL3METHODS	cdl\_SEPARATINGLINES	cdl\_SHOOTINGSTAR	cdl\_SHORTLINE	cdl\_SPINNINGTOP	cdl\_STALLEDPATTERN	cdl\_STICKSANDWICH	cdl\_TAKURI	cdl\_TASUKIGAP	cdl\_THRUSTING	cdl\_TRISTAR	cdl\_UNIQUE3RIVER	cdl\_UPSIDEGAP2CROWS	cdl\_XSIDEGAP3METHODS	sti\_BETA	sti\_CORREL	sti\_LINEARREG	sti\_LINEARREG\_ANGLE	sti\_LINEARREG\_INTERCEPT	sti\_LINEARREG\_SLOPE	sti\_STDDEV	sti\_TSF	sti\_VAR	ma\_DEMA\_5	ma\_EMA\_5	ma\_KAMA\_5	ma\_SMA\_5	ma\_T3\_5	ma\_TEMA\_5	ma\_TRIMA\_5	ma\_WMA\_5	ma\_DEMA\_10	ma\_EMA\_10	ma\_KAMA\_10	ma\_SMA\_10	ma\_T3\_10	ma\_TEMA\_10	ma\_TRIMA\_10	ma\_WMA\_10	ma\_DEMA\_20	ma\_EMA\_20	ma\_KAMA\_20	ma\_SMA\_20	ma\_T3\_20	ma\_TEMA\_20	ma\_TRIMA\_20	ma\_WMA\_20	ma\_DEMA\_50	ma\_EMA\_50	ma\_KAMA\_50	ma\_SMA\_50	ma\_T3\_50	ma\_TEMA\_50	ma\_TRIMA\_50	ma\_WMA\_50	ma\_DEMA\_100	ma\_EMA\_100	ma\_KAMA\_100	ma\_SMA\_100	ma\_T3\_100	ma\_TEMA\_100	ma\_TRIMA\_100	ma\_WMA\_100	trad\_s3	trad\_s2	trad\_s1	trad\_pp	trad\_r1	trad\_r2	trad\_r3	clas\_s3	clas\_s2	clas\_s1	clas\_pp	clas\_r1	clas\_r2	clas\_r3	fibo\_s3	fibo\_s2	fibo\_s1	fibo\_pp	fibo\_r1	fibo\_r2	fibo\_r3	wood\_s3	wood\_s2	wood\_s1	wood\_pp	wood\_r1	wood\_r2	wood\_r3	demark\_s1	demark\_pp	demark\_r1	cama\_s3	cama\_s2	cama\_s1	cama\_pp	cama\_r1	cama\_r2	cama\_r3	ti\_acc\_dist	ti\_chaikin\_10\_3	ti\_choppiness\_14	ti\_coppock\_14\_11\_10	ti\_donchian\_lower\_20	ti\_donchian\_center\_20	ti\_donchian\_upper\_20	ti\_ease\_of\_movement\_14	ti\_force\_index\_13	ti\_hma\_20	ti\_kelt\_20\_lower	ti\_kelt\_20\_upper	ti\_mass\_index\_9\_25	ti\_supertrend\_20	ti\_vortex\_pos\_5	ti\_vortex\_neg\_5	ti\_vortex\_pos\_14	ti\_vortex\_neg\_14	cycl\_EBSW\_40\_10	mtum\_AO\_5\_34	mtum\_BIAS\_SMA\_26	mtum\_AR\_26	mtum\_BR\_26	mtum\_CFO\_9	mtum\_CG\_10	mtum\_CTI\_12	mtum\_DMP\_14	mtum\_DMN\_14	mtum\_ER\_10	mtum\_BULLP\_13	mtum\_BEARP\_13	mtum\_FISHERT\_9\_1	mtum\_FISHERTs\_9\_1	mtum\_INERTIA\_20\_14	mtum\_K\_9\_3	mtum\_D\_9\_3	mtum\_J\_9\_3	mtum\_PGO\_14	mtum\_PSL\_12	mtum\_PVO\_12\_26\_9	mtum\_PVOh\_12\_26\_9	mtum\_PVOs\_12\_26\_9	mtum\_QQE\_14\_5\_4236\_RSIMA	mtum\_QQEl\_14\_5\_4236	mtum\_QQEs\_14\_5\_4236	mtum\_RSX\_14	mtum\_STC\_10\_12\_26\_05	mtum\_STCmacd\_10\_12\_26\_05	mtum\_STCstoch\_10\_12\_26\_05	mtum\_SMI\_5\_20\_5	mtum\_SMIs\_5\_20\_5	mtum\_SMIo\_5\_20\_5	olap\_ALMA\_10\_60\_085	olap\_HWMA\_02\_01\_01	olap\_JMA\_7\_0	olap\_MCGD\_10	olap\_PWMA\_10	olap\_SINWMA\_14	olap\_SSF\_10\_2	olap\_SWMA\_10	olap\_VMAP	olap\_VWMA\_10	perf\_CUMLOGRET\_1	perf\_CUMPCTRET\_1	perf\_z\_30\_1	perf\_ha	sti\_ENTP\_10	sti\_KURT\_30	sti\_TOS\_STDEVALL\_LR	sti\_TOS\_STDEVALL\_L\_1	sti\_TOS\_STDEVALL\_U\_1	sti\_TOS\_STDEVALL\_L\_2	sti\_TOS\_STDEVALL\_U\_2	sti\_TOS\_STDEVALL\_L\_3	sti\_TOS\_STDEVALL\_U\_3	sti\_ZS\_30	tend\_LDECAY\_5	tend\_PSARl\_002\_02	tend\_PSARs\_002\_02	tend\_PSARaf\_002\_02	tend\_PSARr\_002\_02	tend\_VHF\_28	vola\_HWM	vola\_HWU	vola\_HWL	vola\_KCLe\_20\_2	vola\_KCBe\_20\_2	vola\_KCUe\_20\_2	vola\_RVI\_14	vola\_THERMO\_20\_2\_05	vola\_THERMOma\_20\_2\_05	vola\_THERMOl\_20\_2\_05	vola\_THERMOs\_20\_2\_05	vola\_TRUERANGE\_1	vola\_UI\_14	volu\_EFI\_13	volu\_NVI\_1	volu\_PVI\_1	volu\_PVOL	volu\_PVR	volu\_PVT	mtum\_murrey\_math	mtum\_td\_seq	mtum\_td\_seq\_sig	tend\_hh	tend\_hl	tend\_ll	tend\_lh	tend\_hh\_crash	tend\_hl\_crash	tend\_ll\_crash	tend\_lh\_crash	ichi\_tenkan\_sen	ichi\_kijun\_sen	ichi\_senkou\_a	ichi\_senkou\_b	ichi\_isin\_cloud	ichi\_crash	ichi\_chikou\_span	tend\_renko\_TR	tend\_renko\_ATR	tend\_renko\_brick	tend\_renko\_change	pcrh\_trad\_s3	pcrh\_trad\_s2	pcrh\_trad\_s1	pcrh\_trad\_pp	pcrh\_trad\_r1	pcrh\_trad\_r2	pcrh\_trad\_r3	pcrh\_clas\_s3	pcrh\_clas\_s2	pcrh\_clas\_s1	pcrh\_clas\_pp	pcrh\_clas\_r1	pcrh\_clas\_r2	pcrh\_clas\_r3	pcrh\_fibo\_s3	pcrh\_fibo\_s2	pcrh\_fibo\_s1	pcrh\_fibo\_pp	pcrh\_fibo\_r1	pcrh\_fibo\_r2	pcrh\_fibo\_r3	pcrh\_wood\_s3	pcrh\_wood\_s2	pcrh\_wood\_s1	pcrh\_wood\_pp	pcrh\_wood\_r1	pcrh\_wood\_r2	pcrh\_wood\_r3	pcrh\_demark\_s1	pcrh\_demark\_pp	pcrh\_demark\_r1	pcrh\_cama\_s3	pcrh\_cama\_s2	pcrh\_cama\_s1	pcrh\_cama\_pp	pcrh\_cama\_r1	pcrh\_cama\_r2	pcrh\_cama\_r3	mcrh\_DEMA\_5\_DEMA\_10	mcrh\_DEMA\_5\_EMA\_10	mcrh\_DEMA\_5\_KAMA\_10	mcrh\_DEMA\_5\_SMA\_10	mcrh\_DEMA\_5\_T3\_10	mcrh\_DEMA\_5\_TEMA\_10	mcrh\_DEMA\_5\_TRIMA\_10	mcrh\_DEMA\_5\_WMA\_10	mcrh\_DEMA\_5\_DEMA\_20	mcrh\_DEMA\_5\_EMA\_20	mcrh\_DEMA\_5\_KAMA\_20	mcrh\_DEMA\_5\_SMA\_20	mcrh\_DEMA\_5\_T3\_20	mcrh\_DEMA\_5\_TEMA\_20	mcrh\_DEMA\_5\_TRIMA\_20	mcrh\_DEMA\_5\_WMA\_20	mcrh\_DEMA\_5\_DEMA\_50	mcrh\_DEMA\_5\_EMA\_50	mcrh\_DEMA\_5\_KAMA\_50	mcrh\_DEMA\_5\_SMA\_50	mcrh\_DEMA\_5\_T3\_50	mcrh\_DEMA\_5\_TEMA\_50	mcrh\_DEMA\_5\_TRIMA\_50	mcrh\_DEMA\_5\_WMA\_50	mcrh\_DEMA\_5\_DEMA\_100	mcrh\_DEMA\_5\_EMA\_100	mcrh\_DEMA\_5\_KAMA\_100	mcrh\_DEMA\_5\_SMA\_100	mcrh\_DEMA\_5\_T3\_100	mcrh\_DEMA\_5\_TEMA\_100	mcrh\_DEMA\_5\_TRIMA\_100	mcrh\_DEMA\_5\_WMA\_100	mcrh\_DEMA\_5\_ti\_h20	mcrh\_EMA\_5\_DEMA\_10	mcrh\_EMA\_5\_EMA\_10	mcrh\_EMA\_5\_KAMA\_10	mcrh\_EMA\_5\_SMA\_10	mcrh\_EMA\_5\_T3\_10	mcrh\_EMA\_5\_TEMA\_10	mcrh\_EMA\_5\_TRIMA\_10	mcrh\_EMA\_5\_WMA\_10	mcrh\_EMA\_5\_DEMA\_20	mcrh\_EMA\_5\_EMA\_20	mcrh\_EMA\_5\_KAMA\_20	mcrh\_EMA\_5\_SMA\_20	mcrh\_EMA\_5\_T3\_20	mcrh\_EMA\_5\_TEMA\_20	mcrh\_EMA\_5\_TRIMA\_20	mcrh\_EMA\_5\_WMA\_20	mcrh\_EMA\_5\_DEMA\_50	mcrh\_EMA\_5\_EMA\_50	mcrh\_EMA\_5\_KAMA\_50	mcrh\_EMA\_5\_SMA\_50	mcrh\_EMA\_5\_T3\_50	mcrh\_EMA\_5\_TEMA\_50	mcrh\_EMA\_5\_TRIMA\_50	mcrh\_EMA\_5\_WMA\_50	mcrh\_EMA\_5\_DEMA\_100	mcrh\_EMA\_5\_EMA\_100	mcrh\_EMA\_5\_KAMA\_100	mcrh\_EMA\_5\_SMA\_100	mcrh\_EMA\_5\_T3\_100	mcrh\_EMA\_5\_TEMA\_100	mcrh\_EMA\_5\_TRIMA\_100	mcrh\_EMA\_5\_WMA\_100	mcrh\_EMA\_5\_ti\_h20	mcrh\_KAMA\_5\_DEMA\_10	mcrh\_KAMA\_5\_EMA\_10	mcrh\_KAMA\_5\_KAMA\_10	mcrh\_KAMA\_5\_SMA\_10	mcrh\_KAMA\_5\_T3\_10	mcrh\_KAMA\_5\_TEMA\_10	mcrh\_KAMA\_5\_TRIMA\_10	mcrh\_KAMA\_5\_WMA\_10	mcrh\_KAMA\_5\_DEMA\_20	mcrh\_KAMA\_5\_EMA\_20	mcrh\_KAMA\_5\_KAMA\_20	mcrh\_KAMA\_5\_SMA\_20	mcrh\_KAMA\_5\_T3\_20	mcrh\_KAMA\_5\_TEMA\_20	mcrh\_KAMA\_5\_TRIMA\_20	mcrh\_KAMA\_5\_WMA\_20	mcrh\_KAMA\_5\_DEMA\_50	mcrh\_KAMA\_5\_EMA\_50	mcrh\_KAMA\_5\_KAMA\_50	mcrh\_KAMA\_5\_SMA\_50	mcrh\_KAMA\_5\_T3\_50	mcrh\_KAMA\_5\_TEMA\_50	mcrh\_KAMA\_5\_TRIMA\_50	mcrh\_KAMA\_5\_WMA\_50	mcrh\_KAMA\_5\_DEMA\_100	mcrh\_KAMA\_5\_EMA\_100	mcrh\_KAMA\_5\_KAMA\_100	mcrh\_KAMA\_5\_SMA\_100	mcrh\_KAMA\_5\_T3\_100	mcrh\_KAMA\_5\_TEMA\_100	mcrh\_KAMA\_5\_TRIMA\_100	mcrh\_KAMA\_5\_WMA\_100	mcrh\_KAMA\_5\_ti\_h20	mcrh\_SMA\_5\_DEMA\_10	mcrh\_SMA\_5\_EMA\_10	mcrh\_SMA\_5\_KAMA\_10	mcrh\_SMA\_5\_SMA\_10	mcrh\_SMA\_5\_T3\_10	mcrh\_SMA\_5\_TEMA\_10	mcrh\_SMA\_5\_TRIMA\_10	mcrh\_SMA\_5\_WMA\_10	mcrh\_SMA\_5\_DEMA\_20	mcrh\_SMA\_5\_EMA\_20	mcrh\_SMA\_5\_KAMA\_20	mcrh\_SMA\_5\_SMA\_20	mcrh\_SMA\_5\_T3\_20	mcrh\_SMA\_5\_TEMA\_20	mcrh\_SMA\_5\_TRIMA\_20	mcrh\_SMA\_5\_WMA\_20	mcrh\_SMA\_5\_DEMA\_50	mcrh\_SMA\_5\_EMA\_50	mcrh\_SMA\_5\_KAMA\_50	mcrh\_SMA\_5\_SMA\_50	mcrh\_SMA\_5\_T3\_50	mcrh\_SMA\_5\_TEMA\_50	mcrh\_SMA\_5\_TRIMA\_50	mcrh\_SMA\_5\_WMA\_50	mcrh\_SMA\_5\_DEMA\_100	mcrh\_SMA\_5\_EMA\_100	mcrh\_SMA\_5\_KAMA\_100	mcrh\_SMA\_5\_SMA\_100	mcrh\_SMA\_5\_T3\_100	mcrh\_SMA\_5\_TEMA\_100	mcrh\_SMA\_5\_TRIMA\_100	mcrh\_SMA\_5\_WMA\_100	mcrh\_SMA\_5\_ti\_h20	mcrh\_T3\_5\_DEMA\_10	mcrh\_T3\_5\_EMA\_10	mcrh\_T3\_5\_KAMA\_10	mcrh\_T3\_5\_SMA\_10	mcrh\_T3\_5\_T3\_10	mcrh\_T3\_5\_TEMA\_10	mcrh\_T3\_5\_TRIMA\_10	mcrh\_T3\_5\_WMA\_10	mcrh\_T3\_5\_DEMA\_20	mcrh\_T3\_5\_EMA\_20	mcrh\_T3\_5\_KAMA\_20	mcrh\_T3\_5\_SMA\_20	mcrh\_T3\_5\_T3\_20	mcrh\_T3\_5\_TEMA\_20	mcrh\_T3\_5\_TRIMA\_20	mcrh\_T3\_5\_WMA\_20	mcrh\_T3\_5\_DEMA\_50	mcrh\_T3\_5\_EMA\_50	mcrh\_T3\_5\_KAMA\_50	mcrh\_T3\_5\_SMA\_50	mcrh\_T3\_5\_T3\_50	mcrh\_T3\_5\_TEMA\_50	mcrh\_T3\_5\_TRIMA\_50	mcrh\_T3\_5\_WMA\_50	mcrh\_T3\_5\_DEMA\_100	mcrh\_T3\_5\_EMA\_100	mcrh\_T3\_5\_KAMA\_100	mcrh\_T3\_5\_SMA\_100	mcrh\_T3\_5\_T3\_100	mcrh\_T3\_5\_TEMA\_100	mcrh\_T3\_5\_TRIMA\_100	mcrh\_T3\_5\_WMA\_100	mcrh\_T3\_5\_ti\_h20	mcrh\_TEMA\_5\_DEMA\_10	mcrh\_TEMA\_5\_EMA\_10	mcrh\_TEMA\_5\_KAMA\_10	mcrh\_TEMA\_5\_SMA\_10	mcrh\_TEMA\_5\_T3\_10	mcrh\_TEMA\_5\_TEMA\_10	mcrh\_TEMA\_5\_TRIMA\_10	mcrh\_TEMA\_5\_WMA\_10	mcrh\_TEMA\_5\_DEMA\_20	mcrh\_TEMA\_5\_EMA\_20	mcrh\_TEMA\_5\_KAMA\_20	mcrh\_TEMA\_5\_SMA\_20	mcrh\_TEMA\_5\_T3\_20	mcrh\_TEMA\_5\_TEMA\_20	mcrh\_TEMA\_5\_TRIMA\_20	mcrh\_TEMA\_5\_WMA\_20	mcrh\_TEMA\_5\_DEMA\_50	mcrh\_TEMA\_5\_EMA\_50	mcrh\_TEMA\_5\_KAMA\_50	mcrh\_TEMA\_5\_SMA\_50	mcrh\_TEMA\_5\_T3\_50	mcrh\_TEMA\_5\_TEMA\_50	mcrh\_TEMA\_5\_TRIMA\_50	mcrh\_TEMA\_5\_WMA\_50	mcrh\_TEMA\_5\_DEMA\_100	mcrh\_TEMA\_5\_EMA\_100	mcrh\_TEMA\_5\_KAMA\_100	mcrh\_TEMA\_5\_SMA\_100	mcrh\_TEMA\_5\_T3\_100	mcrh\_TEMA\_5\_TEMA\_100	mcrh\_TEMA\_5\_TRIMA\_100	mcrh\_TEMA\_5\_WMA\_100	mcrh\_TEMA\_5\_ti\_h20	mcrh\_TRIMA\_5\_DEMA\_10	mcrh\_TRIMA\_5\_EMA\_10	mcrh\_TRIMA\_5\_KAMA\_10	mcrh\_TRIMA\_5\_SMA\_10	mcrh\_TRIMA\_5\_T3\_10	mcrh\_TRIMA\_5\_TEMA\_10	mcrh\_TRIMA\_5\_TRIMA\_10	mcrh\_TRIMA\_5\_WMA\_10	mcrh\_TRIMA\_5\_DEMA\_20	mcrh\_TRIMA\_5\_EMA\_20	mcrh\_TRIMA\_5\_KAMA\_20	mcrh\_TRIMA\_5\_SMA\_20	mcrh\_TRIMA\_5\_T3\_20	mcrh\_TRIMA\_5\_TEMA\_20	mcrh\_TRIMA\_5\_TRIMA\_20	mcrh\_TRIMA\_5\_WMA\_20	mcrh\_TRIMA\_5\_DEMA\_50	mcrh\_TRIMA\_5\_EMA\_50	mcrh\_TRIMA\_5\_KAMA\_50	mcrh\_TRIMA\_5\_SMA\_50	mcrh\_TRIMA\_5\_T3\_50	mcrh\_TRIMA\_5\_TEMA\_50	mcrh\_TRIMA\_5\_TRIMA\_50	mcrh\_TRIMA\_5\_WMA\_50	mcrh\_TRIMA\_5\_DEMA\_100	mcrh\_TRIMA\_5\_EMA\_100	mcrh\_TRIMA\_5\_KAMA\_100	mcrh\_TRIMA\_5\_SMA\_100	mcrh\_TRIMA\_5\_T3\_100	mcrh\_TRIMA\_5\_TEMA\_100	mcrh\_TRIMA\_5\_TRIMA\_100	mcrh\_TRIMA\_5\_WMA\_100	mcrh\_TRIMA\_5\_ti\_h20	mcrh\_WMA\_5\_DEMA\_10	mcrh\_WMA\_5\_EMA\_10	mcrh\_WMA\_5\_KAMA\_10	mcrh\_WMA\_5\_SMA\_10	mcrh\_WMA\_5\_T3\_10	mcrh\_WMA\_5\_TEMA\_10	mcrh\_WMA\_5\_TRIMA\_10	mcrh\_WMA\_5\_WMA\_10	mcrh\_WMA\_5\_DEMA\_20	mcrh\_WMA\_5\_EMA\_20	mcrh\_WMA\_5\_KAMA\_20	mcrh\_WMA\_5\_SMA\_20	mcrh\_WMA\_5\_T3\_20	mcrh\_WMA\_5\_TEMA\_20	mcrh\_WMA\_5\_TRIMA\_20	mcrh\_WMA\_5\_WMA\_20	mcrh\_WMA\_5\_DEMA\_50	mcrh\_WMA\_5\_EMA\_50	mcrh\_WMA\_5\_KAMA\_50	mcrh\_WMA\_5\_SMA\_50	mcrh\_WMA\_5\_T3\_50	mcrh\_WMA\_5\_TEMA\_50	mcrh\_WMA\_5\_TRIMA\_50	mcrh\_WMA\_5\_WMA\_50	mcrh\_WMA\_5\_DEMA\_100	mcrh\_WMA\_5\_EMA\_100	mcrh\_WMA\_5\_KAMA\_100	mcrh\_WMA\_5\_SMA\_100	mcrh\_WMA\_5\_T3\_100	mcrh\_WMA\_5\_TEMA\_100	mcrh\_WMA\_5\_TRIMA\_100	mcrh\_WMA\_5\_WMA\_100	mcrh\_WMA\_5\_ti\_h20	mcrh\_DEMA\_10\_DEMA\_20	mcrh\_DEMA\_10\_EMA\_20	mcrh\_DEMA\_10\_KAMA\_20	mcrh\_DEMA\_10\_SMA\_20	mcrh\_DEMA\_10\_T3\_20	mcrh\_DEMA\_10\_TEMA\_20	mcrh\_DEMA\_10\_TRIMA\_20	mcrh\_DEMA\_10\_WMA\_20	mcrh\_DEMA\_10\_DEMA\_50	mcrh\_DEMA\_10\_EMA\_50	mcrh\_DEMA\_10\_KAMA\_50	mcrh\_DEMA\_10\_SMA\_50	mcrh\_DEMA\_10\_T3\_50	mcrh\_DEMA\_10\_TEMA\_50	mcrh\_DEMA\_10\_TRIMA\_50	mcrh\_DEMA\_10\_WMA\_50	mcrh\_DEMA\_10\_DEMA\_100	mcrh\_DEMA\_10\_EMA\_100	mcrh\_DEMA\_10\_KAMA\_100	mcrh\_DEMA\_10\_SMA\_100	mcrh\_DEMA\_10\_T3\_100	mcrh\_DEMA\_10\_TEMA\_100	mcrh\_DEMA\_10\_TRIMA\_100	mcrh\_DEMA\_10\_WMA\_100	mcrh\_DEMA\_10\_ti\_h20	mcrh\_EMA\_10\_DEMA\_20	mcrh\_EMA\_10\_EMA\_20	mcrh\_EMA\_10\_KAMA\_20	mcrh\_EMA\_10\_SMA\_20	mcrh\_EMA\_10\_T3\_20	mcrh\_EMA\_10\_TEMA\_20	mcrh\_EMA\_10\_TRIMA\_20	mcrh\_EMA\_10\_WMA\_20	mcrh\_EMA\_10\_DEMA\_50	mcrh\_EMA\_10\_EMA\_50	mcrh\_EMA\_10\_KAMA\_50	mcrh\_EMA\_10\_SMA\_50	mcrh\_EMA\_10\_T3\_50	mcrh\_EMA\_10\_TEMA\_50	mcrh\_EMA\_10\_TRIMA\_50	mcrh\_EMA\_10\_WMA\_50	mcrh\_EMA\_10\_DEMA\_100	mcrh\_EMA\_10\_EMA\_100	mcrh\_EMA\_10\_KAMA\_100	mcrh\_EMA\_10\_SMA\_100	mcrh\_EMA\_10\_T3\_100	mcrh\_EMA\_10\_TEMA\_100	mcrh\_EMA\_10\_TRIMA\_100	mcrh\_EMA\_10\_WMA\_100	mcrh\_EMA\_10\_ti\_h20	mcrh\_KAMA\_10\_DEMA\_20	mcrh\_KAMA\_10\_EMA\_20	mcrh\_KAMA\_10\_KAMA\_20	mcrh\_KAMA\_10\_SMA\_20	mcrh\_KAMA\_10\_T3\_20	mcrh\_KAMA\_10\_TEMA\_20	mcrh\_KAMA\_10\_TRIMA\_20	mcrh\_KAMA\_10\_WMA\_20	mcrh\_KAMA\_10\_DEMA\_50	mcrh\_KAMA\_10\_EMA\_50	mcrh\_KAMA\_10\_KAMA\_50	mcrh\_KAMA\_10\_SMA\_50	mcrh\_KAMA\_10\_T3\_50	mcrh\_KAMA\_10\_TEMA\_50	mcrh\_KAMA\_10\_TRIMA\_50	mcrh\_KAMA\_10\_WMA\_50	mcrh\_KAMA\_10\_DEMA\_100	mcrh\_KAMA\_10\_EMA\_100	mcrh\_KAMA\_10\_KAMA\_100	mcrh\_KAMA\_10\_SMA\_100	mcrh\_KAMA\_10\_T3\_100	mcrh\_KAMA\_10\_TEMA\_100	mcrh\_KAMA\_10\_TRIMA\_100	mcrh\_KAMA\_10\_WMA\_100	mcrh\_KAMA\_10\_ti\_h20	mcrh\_SMA\_10\_DEMA\_20	mcrh\_SMA\_10\_EMA\_20	mcrh\_SMA\_10\_KAMA\_20	mcrh\_SMA\_10\_SMA\_20	mcrh\_SMA\_10\_T3\_20	mcrh\_SMA\_10\_TEMA\_20	mcrh\_SMA\_10\_TRIMA\_20	mcrh\_SMA\_10\_WMA\_20	mcrh\_SMA\_10\_DEMA\_50	mcrh\_SMA\_10\_EMA\_50	mcrh\_SMA\_10\_KAMA\_50	mcrh\_SMA\_10\_SMA\_50	mcrh\_SMA\_10\_T3\_50	mcrh\_SMA\_10\_TEMA\_50	mcrh\_SMA\_10\_TRIMA\_50	mcrh\_SMA\_10\_WMA\_50	mcrh\_SMA\_10\_DEMA\_100	mcrh\_SMA\_10\_EMA\_100	mcrh\_SMA\_10\_KAMA\_100	mcrh\_SMA\_10\_SMA\_100	mcrh\_SMA\_10\_T3\_100	mcrh\_SMA\_10\_TEMA\_100	mcrh\_SMA\_10\_TRIMA\_100	mcrh\_SMA\_10\_WMA\_100	mcrh\_SMA\_10\_ti\_h20	mcrh\_T3\_10\_DEMA\_20	mcrh\_T3\_10\_EMA\_20	mcrh\_T3\_10\_KAMA\_20	mcrh\_T3\_10\_SMA\_20	mcrh\_T3\_10\_T3\_20	mcrh\_T3\_10\_TEMA\_20	mcrh\_T3\_10\_TRIMA\_20	mcrh\_T3\_10\_WMA\_20	mcrh\_T3\_10\_DEMA\_50	mcrh\_T3\_10\_EMA\_50	mcrh\_T3\_10\_KAMA\_50	mcrh\_T3\_10\_SMA\_50	mcrh\_T3\_10\_T3\_50	mcrh\_T3\_10\_TEMA\_50	mcrh\_T3\_10\_TRIMA\_50	mcrh\_T3\_10\_WMA\_50	mcrh\_T3\_10\_DEMA\_100	mcrh\_T3\_10\_EMA\_100	mcrh\_T3\_10\_KAMA\_100	mcrh\_T3\_10\_SMA\_100	mcrh\_T3\_10\_T3\_100	mcrh\_T3\_10\_TEMA\_100	mcrh\_T3\_10\_TRIMA\_100	mcrh\_T3\_10\_WMA\_100	mcrh\_T3\_10\_ti\_h20	mcrh\_TEMA\_10\_DEMA\_20	mcrh\_TEMA\_10\_EMA\_20	mcrh\_TEMA\_10\_KAMA\_20	mcrh\_TEMA\_10\_SMA\_20	mcrh\_TEMA\_10\_T3\_20	mcrh\_TEMA\_10\_TEMA\_20	mcrh\_TEMA\_10\_TRIMA\_20	mcrh\_TEMA\_10\_WMA\_20	mcrh\_TEMA\_10\_DEMA\_50	mcrh\_TEMA\_10\_EMA\_50	mcrh\_TEMA\_10\_KAMA\_50	mcrh\_TEMA\_10\_SMA\_50	mcrh\_TEMA\_10\_T3\_50	mcrh\_TEMA\_10\_TEMA\_50	mcrh\_TEMA\_10\_TRIMA\_50	mcrh\_TEMA\_10\_WMA\_50	mcrh\_TEMA\_10\_DEMA\_100	mcrh\_TEMA\_10\_EMA\_100	mcrh\_TEMA\_10\_KAMA\_100	mcrh\_TEMA\_10\_SMA\_100	mcrh\_TEMA\_10\_T3\_100	mcrh\_TEMA\_10\_TEMA\_100	mcrh\_TEMA\_10\_TRIMA\_100	mcrh\_TEMA\_10\_WMA\_100	mcrh\_TEMA\_10\_ti\_h20	mcrh\_TRIMA\_10\_DEMA\_20	mcrh\_TRIMA\_10\_EMA\_20	mcrh\_TRIMA\_10\_KAMA\_20	mcrh\_TRIMA\_10\_SMA\_20	mcrh\_TRIMA\_10\_T3\_20	mcrh\_TRIMA\_10\_TEMA\_20	mcrh\_TRIMA\_10\_TRIMA\_20	mcrh\_TRIMA\_10\_WMA\_20	mcrh\_TRIMA\_10\_DEMA\_50	mcrh\_TRIMA\_10\_EMA\_50	mcrh\_TRIMA\_10\_KAMA\_50	mcrh\_TRIMA\_10\_SMA\_50	mcrh\_TRIMA\_10\_T3\_50	mcrh\_TRIMA\_10\_TEMA\_50	mcrh\_TRIMA\_10\_TRIMA\_50	mcrh\_TRIMA\_10\_WMA\_50	mcrh\_TRIMA\_10\_DEMA\_100	mcrh\_TRIMA\_10\_EMA\_100	mcrh\_TRIMA\_10\_KAMA\_100	mcrh\_TRIMA\_10\_SMA\_100	mcrh\_TRIMA\_10\_T3\_100	mcrh\_TRIMA\_10\_TEMA\_100	mcrh\_TRIMA\_10\_TRIMA\_100	mcrh\_TRIMA\_10\_WMA\_100	mcrh\_TRIMA\_10\_ti\_h20	mcrh\_WMA\_10\_DEMA\_20	mcrh\_WMA\_10\_EMA\_20	mcrh\_WMA\_10\_KAMA\_20	mcrh\_WMA\_10\_SMA\_20	mcrh\_WMA\_10\_T3\_20	mcrh\_WMA\_10\_TEMA\_20	mcrh\_WMA\_10\_TRIMA\_20	mcrh\_WMA\_10\_WMA\_20	mcrh\_WMA\_10\_DEMA\_50	mcrh\_WMA\_10\_EMA\_50	mcrh\_WMA\_10\_KAMA\_50	mcrh\_WMA\_10\_SMA\_50	mcrh\_WMA\_10\_T3\_50	mcrh\_WMA\_10\_TEMA\_50	mcrh\_WMA\_10\_TRIMA\_50	mcrh\_WMA\_10\_WMA\_50	mcrh\_WMA\_10\_DEMA\_100	mcrh\_WMA\_10\_EMA\_100	mcrh\_WMA\_10\_KAMA\_100	mcrh\_WMA\_10\_SMA\_100	mcrh\_WMA\_10\_T3\_100	mcrh\_WMA\_10\_TEMA\_100	mcrh\_WMA\_10\_TRIMA\_100	mcrh\_WMA\_10\_WMA\_100	mcrh\_WMA\_10\_ti\_h20	mcrh\_DEMA\_20\_DEMA\_50	mcrh\_DEMA\_20\_EMA\_50	mcrh\_DEMA\_20\_KAMA\_50	mcrh\_DEMA\_20\_SMA\_50	mcrh\_DEMA\_20\_T3\_50	mcrh\_DEMA\_20\_TEMA\_50	mcrh\_DEMA\_20\_TRIMA\_50	mcrh\_DEMA\_20\_WMA\_50	mcrh\_DEMA\_20\_DEMA\_100	mcrh\_DEMA\_20\_EMA\_100	mcrh\_DEMA\_20\_KAMA\_100	mcrh\_DEMA\_20\_SMA\_100	mcrh\_DEMA\_20\_T3\_100	mcrh\_DEMA\_20\_TEMA\_100	mcrh\_DEMA\_20\_TRIMA\_100	mcrh\_DEMA\_20\_WMA\_100	mcrh\_EMA\_20\_DEMA\_50	mcrh\_EMA\_20\_EMA\_50	mcrh\_EMA\_20\_KAMA\_50	mcrh\_EMA\_20\_SMA\_50	mcrh\_EMA\_20\_T3\_50	mcrh\_EMA\_20\_TEMA\_50	mcrh\_EMA\_20\_TRIMA\_50	mcrh\_EMA\_20\_WMA\_50	mcrh\_EMA\_20\_DEMA\_100	mcrh\_EMA\_20\_EMA\_100	mcrh\_EMA\_20\_KAMA\_100	mcrh\_EMA\_20\_SMA\_100	mcrh\_EMA\_20\_T3\_100	mcrh\_EMA\_20\_TEMA\_100	mcrh\_EMA\_20\_TRIMA\_100	mcrh\_EMA\_20\_WMA\_100	mcrh\_KAMA\_20\_DEMA\_50	mcrh\_KAMA\_20\_EMA\_50	mcrh\_KAMA\_20\_KAMA\_50	mcrh\_KAMA\_20\_SMA\_50	mcrh\_KAMA\_20\_T3\_50	mcrh\_KAMA\_20\_TEMA\_50	mcrh\_KAMA\_20\_TRIMA\_50	mcrh\_KAMA\_20\_WMA\_50	mcrh\_KAMA\_20\_DEMA\_100	mcrh\_KAMA\_20\_EMA\_100	mcrh\_KAMA\_20\_KAMA\_100	mcrh\_KAMA\_20\_SMA\_100	mcrh\_KAMA\_20\_T3\_100	mcrh\_KAMA\_20\_TEMA\_100	mcrh\_KAMA\_20\_TRIMA\_100	mcrh\_KAMA\_20\_WMA\_100	mcrh\_SMA\_20\_DEMA\_50	mcrh\_SMA\_20\_EMA\_50	mcrh\_SMA\_20\_KAMA\_50	mcrh\_SMA\_20\_SMA\_50	mcrh\_SMA\_20\_T3\_50	mcrh\_SMA\_20\_TEMA\_50	mcrh\_SMA\_20\_TRIMA\_50	mcrh\_SMA\_20\_WMA\_50	mcrh\_SMA\_20\_DEMA\_100	mcrh\_SMA\_20\_EMA\_100	mcrh\_SMA\_20\_KAMA\_100	mcrh\_SMA\_20\_SMA\_100	mcrh\_SMA\_20\_T3\_100	mcrh\_SMA\_20\_TEMA\_100	mcrh\_SMA\_20\_TRIMA\_100	mcrh\_SMA\_20\_WMA\_100	mcrh\_T3\_20\_DEMA\_50	mcrh\_T3\_20\_EMA\_50	mcrh\_T3\_20\_KAMA\_50	mcrh\_T3\_20\_SMA\_50	mcrh\_T3\_20\_T3\_50	mcrh\_T3\_20\_TEMA\_50	mcrh\_T3\_20\_TRIMA\_50	mcrh\_T3\_20\_WMA\_50	mcrh\_T3\_20\_DEMA\_100	mcrh\_T3\_20\_EMA\_100	mcrh\_T3\_20\_KAMA\_100	mcrh\_T3\_20\_SMA\_100	mcrh\_T3\_20\_T3\_100	mcrh\_T3\_20\_TEMA\_100	mcrh\_T3\_20\_TRIMA\_100	mcrh\_T3\_20\_WMA\_100	mcrh\_TEMA\_20\_DEMA\_50	mcrh\_TEMA\_20\_EMA\_50	mcrh\_TEMA\_20\_KAMA\_50	mcrh\_TEMA\_20\_SMA\_50	mcrh\_TEMA\_20\_T3\_50	mcrh\_TEMA\_20\_TEMA\_50	mcrh\_TEMA\_20\_TRIMA\_50	mcrh\_TEMA\_20\_WMA\_50	mcrh\_TEMA\_20\_DEMA\_100	mcrh\_TEMA\_20\_EMA\_100	mcrh\_TEMA\_20\_KAMA\_100	mcrh\_TEMA\_20\_SMA\_100	mcrh\_TEMA\_20\_T3\_100	mcrh\_TEMA\_20\_TEMA\_100	mcrh\_TEMA\_20\_TRIMA\_100	mcrh\_TEMA\_20\_WMA\_100	mcrh\_TRIMA\_20\_DEMA\_50	mcrh\_TRIMA\_20\_EMA\_50	mcrh\_TRIMA\_20\_KAMA\_50	mcrh\_TRIMA\_20\_SMA\_50	mcrh\_TRIMA\_20\_T3\_50	mcrh\_TRIMA\_20\_TEMA\_50	mcrh\_TRIMA\_20\_TRIMA\_50	mcrh\_TRIMA\_20\_WMA\_50	mcrh\_TRIMA\_20\_DEMA\_100	mcrh\_TRIMA\_20\_EMA\_100	mcrh\_TRIMA\_20\_KAMA\_100	mcrh\_TRIMA\_20\_SMA\_100	mcrh\_TRIMA\_20\_T3\_100	mcrh\_TRIMA\_20\_TEMA\_100	mcrh\_TRIMA\_20\_TRIMA\_100	mcrh\_TRIMA\_20\_WMA\_100	mcrh\_WMA\_20\_DEMA\_50	mcrh\_WMA\_20\_EMA\_50	mcrh\_WMA\_20\_KAMA\_50	mcrh\_WMA\_20\_SMA\_50	mcrh\_WMA\_20\_T3\_50	mcrh\_WMA\_20\_TEMA\_50	mcrh\_WMA\_20\_TRIMA\_50	mcrh\_WMA\_20\_WMA\_50	mcrh\_WMA\_20\_DEMA\_100	mcrh\_WMA\_20\_EMA\_100	mcrh\_WMA\_20\_KAMA\_100	mcrh\_WMA\_20\_SMA\_100	mcrh\_WMA\_20\_T3\_100	mcrh\_WMA\_20\_TEMA\_100	mcrh\_WMA\_20\_TRIMA\_100	mcrh\_WMA\_20\_WMA\_100	mcrh\_DEMA\_50\_DEMA\_100	mcrh\_DEMA\_50\_EMA\_100	mcrh\_DEMA\_50\_KAMA\_100	mcrh\_DEMA\_50\_SMA\_100	mcrh\_DEMA\_50\_T3\_100	mcrh\_DEMA\_50\_TEMA\_100	mcrh\_DEMA\_50\_TRIMA\_100	mcrh\_DEMA\_50\_WMA\_100	mcrh\_DEMA\_50\_ti\_h20	mcrh\_EMA\_50\_DEMA\_100	mcrh\_EMA\_50\_EMA\_100	mcrh\_EMA\_50\_KAMA\_100	mcrh\_EMA\_50\_SMA\_100	mcrh\_EMA\_50\_T3\_100	mcrh\_EMA\_50\_TEMA\_100	mcrh\_EMA\_50\_TRIMA\_100	mcrh\_EMA\_50\_WMA\_100	mcrh\_EMA\_50\_ti\_h20	mcrh\_KAMA\_50\_DEMA\_100	mcrh\_KAMA\_50\_EMA\_100	mcrh\_KAMA\_50\_KAMA\_100	mcrh\_KAMA\_50\_SMA\_100	mcrh\_KAMA\_50\_T3\_100	mcrh\_KAMA\_50\_TEMA\_100	mcrh\_KAMA\_50\_TRIMA\_100	mcrh\_KAMA\_50\_WMA\_100	mcrh\_KAMA\_50\_ti\_h20	mcrh\_SMA\_50\_DEMA\_100	mcrh\_SMA\_50\_EMA\_100	mcrh\_SMA\_50\_KAMA\_100	mcrh\_SMA\_50\_SMA\_100	mcrh\_SMA\_50\_T3\_100	mcrh\_SMA\_50\_TEMA\_100	mcrh\_SMA\_50\_TRIMA\_100	mcrh\_SMA\_50\_WMA\_100	mcrh\_SMA\_50\_ti\_h20	mcrh\_T3\_50\_DEMA\_100	mcrh\_T3\_50\_EMA\_100	mcrh\_T3\_50\_KAMA\_100	mcrh\_T3\_50\_SMA\_100	mcrh\_T3\_50\_T3\_100	mcrh\_T3\_50\_TEMA\_100	mcrh\_T3\_50\_TRIMA\_100	mcrh\_T3\_50\_WMA\_100	mcrh\_T3\_50\_ti\_h20	mcrh\_TEMA\_50\_DEMA\_100	mcrh\_TEMA\_50\_EMA\_100	mcrh\_TEMA\_50\_KAMA\_100	mcrh\_TEMA\_50\_SMA\_100	mcrh\_TEMA\_50\_T3\_100	mcrh\_TEMA\_50\_TEMA\_100	mcrh\_TEMA\_50\_TRIMA\_100	mcrh\_TEMA\_50\_WMA\_100	mcrh\_TEMA\_50\_ti\_h20	mcrh\_TRIMA\_50\_DEMA\_100	mcrh\_TRIMA\_50\_EMA\_100	mcrh\_TRIMA\_50\_KAMA\_100	mcrh\_TRIMA\_50\_SMA\_100	mcrh\_TRIMA\_50\_T3\_100	mcrh\_TRIMA\_50\_TEMA\_100	mcrh\_TRIMA\_50\_TRIMA\_100	mcrh\_TRIMA\_50\_WMA\_100	mcrh\_TRIMA\_50\_ti\_h20	mcrh\_WMA\_50\_DEMA\_100	mcrh\_WMA\_50\_EMA\_100	mcrh\_WMA\_50\_KAMA\_100	mcrh\_WMA\_50\_SMA\_100	mcrh\_WMA\_50\_T3\_100	mcrh\_WMA\_50\_TEMA\_100	mcrh\_WMA\_50\_TRIMA\_100	mcrh\_WMA\_50\_WMA\_100	mcrh\_WMA\_50\_ti\_h20	mcrh\_DEMA\_100\_ti\_h20	mcrh\_EMA\_100\_ti\_h20	mcrh\_KAMA\_100\_ti\_h20	mcrh\_SMA\_100\_ti\_h20	mcrh\_T3\_100\_ti\_h20	mcrh\_TEMA\_100\_ti\_h20	mcrh\_TRIMA\_100\_ti\_h20	mcrh\_WMA\_100\_ti\_h20	NQ\_Close	NQ\_Volume	NQ\_per\_Close	NQ\_per\_Volume	NQ\_SMA\_20	NQ\_SMA\_100**


