  * [OPERATION](#operation)
    + **1.1** Data collection **1.2** Types of indicators
    + **2** Indicator filtering
    + **3** Training TF XGB and Sklearn
    + **4.1** Assessing the QUALITY of these models **4.2** Evaluating those real BENEFITS of models
    + **5.1** Making predictions for the past week **5.2** Sending real-time alerts





### OBJECTIVE
Understanding the principle of self-fulfilling prophecy, it is possible to obtain the pattern of the same, by means of the massive collection of technical patterns, their calculation and the study of their patterns.


` `For this, techniques such as big data will be used through Pandas Python libraries, machine learning through Sklearn, XGB and neural networks through the open google Tensor Flow library.

The result will be displayed in a simple and friendly way through alerts on mobile or computer.

The machine learnig models Sklearn, XGB and Tensor Flow , by means of the learning of the last months detect the point of sale. To detect this point of sale a series of indicators have been taken into account: olap_VMAP, ma_SMA_50, ichi_senkou_a, olap_BBAND_dif ,mtum_MACD_ext, olap_BBAND_MIDDLE, mtum_MACD_ext_signal, fibo_s1, volu_PVI_1, ma_KAMA_5, etcetera.

The image shows: MACD, RSI , Stochastic and Balance of power (Elder Ray)

The alert is sent on the **vertical line** (the only vertical line that crosses the whole image), during the next 4 periods the stock decreases (It will be indicated as _**SELL**_) by 2.4%. Each candlestick period in the image indicates 15 minutes.

![](../readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.006.png)



## OPERATION

**NOTE:** If you want to run better read    [Detailed start-up](#detailed-start-up)


#### **1.1** Data collection
Collect data to train the model

`1_Get_technical_indicators.py`


**Ground True is the variable** `buy_seel_point`
The model to be able to train in detecting points of purchase and sale, creates the column `buy_seel_point` has value of: 0, -100, 100. These are detected according to the maximum changes, (positive 100, negative -100) in the history of the last months, this point will be with which the training is trained, also called the *ground* true.

Defining the GT (Ground True) is a subjective task, these numbers can be obtained in 2 ways:
 - `Utils_buy_sell_points.get_buy_sell_points_Roll` (default). Value will be assigned in buy_seel_point if the increase or decrease of the stock is greater than 2.5% in a period of 3 hours, using the get_buy_sell_points_Roll function.

 - `Utils_buy_sell_points.get_buy_sell_points_HT_pp` (decomments the line) Inspired by the TraderView technical indicator "Pilots HL".

On the graphic, you can see the difference being:
**_Blue_** the candle Close
_**Red**_ the values obtained in `get_buy_sell_points_Roll()` the positive and negative peaks indicate sell and buy points.
The points obtained in `get_buy_sell_points_HT_pp()` are shown, the **_Orange_** peaks being the Buy points and the **_Green_** peaks being the Sell points.

![](../readme_img/GT_ways_3.PNG)

Once the historical data of the stock has been obtained and all the technical indicators have been calculated, a total of 1068, files of type `AAPL_stock_history_MONTH_3_AD.csv` are generated.

Example of the file with the first eight indicators:

![](../readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.007.png)

This data collection is customizable, you can obtain and train models of any Nasdaq stock, for other indicators or crypto-assets, it is also possible by making small changes.

**Through the Option_Historical** class it is possible to create historical data files: annual, monthly and daily.
```
**class Option_Historical**(Enum): YEARS_3 = 1, MONTH_3 = 2, MONTH_3_AD = 3, DAY_6 = 4, DAY_1 = 5
```

The files *\d_price_maxAAPL_min_max_stock_MONTH_3.csv* are generated, which store the max and min value of each column, to be read in `Model_predictions_Nrows.py` for a quick fit_scaler() (this is the "cleaning" process that the data requires before entering the AI training models) . This operation is of vital importance for a correct optimization in reading data in real time.




#### **1.2** Types of indicators
During the generation of the data collection file of point 1 `AAPL_stock_history_MONTH_3_AD.csv` 1068 technical indicators are calculated, which are divided into subtypes, based on **prefixes** in the name.

List of prefixes and an example of the name of one of them.

- Overlap: **olap_**

olap_BBAND_UPPER, olap_BBAND_MIDDLE, olap_BBAND_LOWER,

- Momentum: **mtum_**

mtum_MACD, mtum_MACD_signal, mtum_RSI, mtum_STOCH_k,

- Volatility: **vola_**

vola_KCBe_20_2, vola_KCUe_20_2, vola_RVI_14

- Cycle patterns: **cycl_**

cycl_DCPHASE, cycl_PHASOR_inph, cycl_PHASOR_quad

- Candlestick patterns: **cdl_**

cdl_RICKSHAWMAN, cdl_RISEFALL3METHODS, cdl_SEPARATINGLINES

- Statistics: **sti_**

sti_STDDEV, sti_TSF, sti_VAR

- Moving averages: **ma_**

ma_SMA_100, ma_WMA_10, ma_DEMA_20, ma_EMA_100, ma_KAMA_10,

- Trend: **tend_** and **ti_**

tend_renko_TR, tend_renko_brick, ti_acc_dist, ti_chaikin_10_3

- Resistors and support suffixes: **_s3, _s2, _s1, _pp, _r1, _r2, _r3**

fibo_s3, fibo_s2, fibo_s1, fibo_pp, fibo_r1, fibo_r2, fibo_r3, fibo_r2, fibo_r3

demark_s1, demark_pp, demark_r1

- Intersection point with resistance or support: **pcrh_.**

pcrh_demark_s1, pcrh_demark_pp, pcrh_demark_r1

- Intersection point with moving average or of moving averages between them: **mcrh_.**

mcrh_SMA_20_TRIMA_50, mcrh_SMA_20_WMA_50, mcrh_SMA_20_DEMA_100

- Indicators of changes in the stock index, nasdaq: **NQ_.**

NQ_SMA_20, NQ_SMA_100

Note: To see the 1068 indicators used go to the attached sheets at the end of the document.


#### **2** Indicator filtering
Execute to find out which columns are relevant for the model output

`2_Feature_selection_create_json.md`

It is necessary to know which of the hundreds of columns of technical data, is valid to train the neural model, and which are just noise. This will be done through correlations and Random Forest models.

Answer the question:

Which columns are most relevant for buy or sell points?

Generate the *best_selection* files, which are a raking of the best technical data to train the model, it is intended to go from 1068 columns to about 120.

For example, for the Amazon stock, point-of-purchase detection, in the period June to October 2022, the most valuable indicators are:

- Senkuo of the Ichimoku Cloud
- Chaikin Volatility
- On-balance volume

Example of *plots_relations/best_selection_AMNZ_pos.json* file
```json
"index": {
  "12": [
    "ichi_chilou_span"
    ],
  "10": [
    "volu_Chaikin_AD"
  ],
  "9": [
    "volu_OBV"
   ],
```
Plots with the 3 best technical data are printed in the folder *plots_relations/plot.*
Example name: *TWLO_neg_buy_sell_point__ichi_chikou_span.png*

![](../readme_img/Aspose.Words.b41e3638-ef34-4eaa-ac86-1fda8999e934.008.png)

#### **3** Training TF XGB and Sklearn
`3_Model_creation_models_for_a_stock.py`

this requires the selection of better columns from point #2

There are four types of predictive algorithms, AI models:

- **Gradient Boosting** consists of a set of individual [decision trees](https://www.cienciadedatos.net/documentos/py07_arboles_decision_python.html), trained sequentially, so that each new tree tries to improve on the errors of the previous trees. Sklearn Library
- **Random Forest** Random forests are an ensemble learning method for classification, regression, and other tasks that operates by constructing a multitude of decision trees at training time. Sklearn Library
- **XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost Library
- **TensorFlow TF** is an open source library for machine learning across a range of tasks, and developed by Google to meet their needs for systems capable of building and training neural networks to detect and decipher patterns and correlations, analogous to the learning and reasoning used by humans. TensorFlow Library


There are POS (buy) or NEG (sell) models and there is a BOTH model (BOTH is discarded, since prediction models are binary, they only accept 2 positions, true or false).

This point generates prediction models `.sav` for XGB and Sklearn. `.h5` for Tensor Flow.

Naming Examples: *XGboost_U_neg_vgood16_s28.sav* and *TF_AMZN_pos_low1_s128.h5*

Format of the names:

- Type of AI you train with can be:
  -  XGboost, TF, TF64, GradientBoost and RandomForest
- Stock ticker AMZN for amazon , AAPL for Apple ...
- Detects points of purchase or sale pos or neg
- How many indicators have been used in the learning, can be of 4 types depending on the relevance given by point *#2 Indicator filtering*. This ranking is organized in the **MODEL_TYPE_COLM** class,
  - vgood16 the best 16 indicators
  - good9 the best 32 indicators
  - reg4 the best 64 indicators
  - low1 the best 128 indicators
- Only for TF models. Depending on the density of the neurons used, defined in the class a_manage_stocks_dict. **MODEL_TF_DENSE_TYPE_ONE_DIMENSI** can take value: s28 s64 and s128

These combinations imply that for each stock 5 types of IA are created, each in pos and neg, plus for each combination the 4 technical indicator configurations are added.  This generates 40 IA models, which will be selected in point: *#4 to evaluate the QUALITY of those models.*

Each time an AI template is generated, a log file is generated: *TF_balance_TF_AAPL_pos_reg4.h5_accuracy_87.6%__loss_2.74__epochs_10[160].csv*

It contains the accuracy and loss data of the model, as well as the model training records.




#### **4.1** Assessing the QUALITY of these models
`4_Model_creation_scoring_multi.py`

To make a prediction with the AIs, new data is collected and the technical indicators with which it has been trained are calculated according to the *best_selection* files.

When the .h5 and .sav models are queried:

  Is this a point of sale?

These answer a number that can vary between 0.1 and 4

The higher the number the more likely it is to be a correct buy/sell point.

Each model has a rating scale on which it is considered point of sale. For some models with a rating of more than 0.4 will be enough (usually the XGboost), while for others require more than 1.5 (usually the TF).

How do you know what the threshold score is for each model?

The Model_creation_scoring.py class generates the threshold score *threshold* files, which tell which threshold point is considered the buy-sell point.

Each AI model will have its own type file:

*Models/Scoring/AAPL_neg__when_model_ok_threshold.csv*

For each action in *#3 train the TF, XGB and Sklearn models*, 40 AI models are generated. This class evaluates and selects the most accurate models so that only the most accurate ones will be executed in real time (usually between 4 and 8 are selected).

*Models/Scoring/AAPL_neg__groupby_buy_sell_point_000.json*
```json
"list_good_params": [

  "r_rf_AFRM_pos_low1_",

   "r_TF64_AFRM_pos_vgood16_",

   "r_TF64_AFRM_pos_good9_",

   "r_TF_AFRM_pos_reg4_"

],
```

#### **4.2** Evaluating those real BENEFITS of models
`Model_predictions_N_eval_profits.py`

Answer the question:

If you leave it running for N days, how much hypothetical money do you make?

Note: this should be run on data that has not been used in the training model, preferably

*Models/eval_Profits/_AAPL_neg_ALL_stock_20221021__20221014.csv*


#### **5.1** Making predictions for the past week
`Model_predictions_Nrows.py`

At this point the **file** `realtime_model_POOL_driver.py` **is required**, you must **ask for it** (if you wish you can reverse engineer it).

You can make predictions with the real-time data of the stock.

Through the function call every 10-12min, download the real-time stock data through the yahoo financial API.
```
df_compare, df_sell = get_RealTime_buy_seel_points()
```
~~This run generates the log file *d_result/prediction_results_N_rows.csv*~~



#### **5.2** Sending real-time alerts
`5_predict_POOL_enque_Thread.py `*multithreading glued 2s per action*

It is possible to run it without configuring telegram point 5.2, in that case no alerts will be sent in telegram, but if the results were recorded in real time in: *d_result/prediction_real_time.csv*

There is the possibility to send alerts of purchase and sale of the share, to telegram or mail.

the multiple AI trained models are evaluated, and only those greater than 96% probability (as previously trained) are reported.

Every 15 minutes, **all** necessary indicators are calculated in real time for each action and evaluated in the AI models.

The alert indicates which models are detecting the correct buy and sell points at which to execute the transaction.

These buy and sell alerts expire in, plus or minus 7 minutes, given the volatility of the market.

Also attached is the price at which it was detected, the time, and links to news websites.

Note: financial news should always prevail over technical indicators.

What is displayed in DEBUG alert, is the information from *d_result/prediction_results_N_rows.csv* of the Item: 5 make predictions of the last week Test

To understand the complete information of the alert see Point 5.1 Making predictions of the last week.

![](../readme_img/telegram_bot_alert_example_MONO_old_1.0.png.png)










