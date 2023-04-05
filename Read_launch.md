Explanations of the usefulness of each file in ReadMe.md



####  Run ***yhoo\_generate\_big\_all_csv.py***

Configure the list of desired actions, during all steps they should be the same 

`CSV_NAME = **"@FOLO3".**
list_stocks = DICT_COMPANYS [CSV_NAME]`

You also have to set the parameter 

`opion = _KEYS_DICT.Option_Historical.MONTH_3`

If done with `MONTH_3_AD` it takes more than 3 months, and requires running the **1.0 (Recommended) alphavantage API** before.

This execution has to generate the following files, for each action:

- d_price/GOOG_PLAIN_stock_history_MONTH_3.csv
- plots_relations/best_selection_GOOG_pos.json
- plots_relations/best_selection_GOOG_both.json
- plots_relations/best_selection_GOOG_neg.json



#### Run ***Model_creation_models_for_a_stock.py***

Generate all pattern combinations with the combination of multidimensional models, the .h5 are the TensorFlow models (the list of models is in *Model_TF_definitions.ModelDefinition.get_dicts_models_multi_dimension* ). :

- Models/TF_multi/TFm_GOOG_pos_reg4_mult_28.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_28.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_64.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_dense2.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_lstm.h5
- Models/TF_multi/TFm_GOOG_pos_low1_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_pos_good9__simp_cordo.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_linear.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_28.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_128.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_linear.h5
- Models/TF_multi/TFm_GOOG_neg_good9__simp_cordo.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_dense2.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_28.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_dense2.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_64.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_lstm.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_conv.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_128.h5
- Models/TF_multi/TFm_GOOG_pos_low1_mult_128.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16__mult_gru.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_128.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_dense2.h5
- Models/TF_multi/TFm_GOOG_pos_reg4__simp_cordo.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_linear.h5
- Models/TF_multi/TFm_GOOG_neg_reg4__simp_cordo.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16__simp_cordo.h5
- Models/TF_multi/TFm_GOOG_pos_low1_mult_28.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_linear.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_dense2.h5
- Models/TF_multi/TFm_GOOG_pos_low1_mult_64.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_conv.h5
- Models/TF_multi/TFm_GOOG_neg_low1_mult_64.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_pos_reg4__mult_gru.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_conv.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_64.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_128.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_conv.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_28.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_lstm.h5
- Models/TF_multi/TFm_GOOG_neg_good9_mult_lstm.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_conv2.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_linear.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_128.h5
- Models/TF_multi/TFm_GOOG_neg_low1_mult_28.h5
- Models/TF_multi/TFm_GOOG_neg_low1_mult_128.h5
- Models/TF_multi/TFm_GOOG_pos_good9__mult_gru.h5
- Models/TF_multi/TFm_GOOG_pos_good9_mult_64.h5
- Models/TF_multi/TFm_GOOG_pos_reg4_mult_conv.h5
- Models/TF_multi/TFm_GOOG_neg_reg4__mult_gru.h5
- Models/TF_multi/TFm_GOOG_pos_vgood16_mult_28.h5
- Models/TF_multi/TFm_GOOG_neg_reg4_mult_64.h5
- Models/TF_multi/TFm_GOOG_neg_vgood16_mult_lstm.h5


It also generates the _per_score files, these are scores in which the predictive model is considered valid, there are models that when asked for an answer answer answer with values between 0 and 0.5 and others between 0.4 and 0.9, to know the prediction weighting of each model are generated. 

- Models/TF_multi/GOOG_pos_pos_vgood16__per_score.csv
- Models/TF_multi/GOOG_neg_good9__per_score.csv
- Models/TF_multi/GOOG_pos_reg4__per_score.csv
- Models/TF_multi/GOOG_pos_good9__per_score.csv
- Models/TF_multi/GOOG_neg_vgood16__per_score.csv
- Models/TF_multi/GOOG_neg_reg4__per_score.csv
- Models/TF_multi/GOOG_pos_low1__per_score.csv
- Models/TF_multi/GOOG_neg_low1__per_score.csv




#### Run **Model_creation_scoring_multi.py**

to get the file **Models/TF_multi/_SCORE_ALL_T_multi_all.csv**

This file contains the hit thresholds for each model. 

TODO: I think it's not necessary at all. 
With some refactor it can be avoid, because:
"Models/TF_multi/_RESULTS_profit_multi_all.csv" Models/TF_multi/_SCORE_ALL_T_multi_all.csv"
this two files have the same values with diferent format





#### In the file **Model_predictions_Multi_N_eval_profits.py**

You have to execute the **functions** , for each desired action eval_profits_per_of_all_models()

`for S in list_stocks:
    eval_profits_per_of_all_models(S)`

this function generates the file *Models/Eval_multi/Eval_profict_GOOG.csv* , this file contains the predictions of all models **.h5** for the last months data 

Execute the function , for each desired action  


add_models_colum_by_best_Profits()
````
for S in list_stocks:

     path_csv_file_EVAL = **"Models/Eval_multi/Eval_profict_"** + S + **".csv"**

     add_models_colum_by_best_Profits(path_csv_file_EVAL, NUMBER_MACHTING_TEST_BEST= 3)
````
Select the models whose average profit per operation is greater than 2.2%, if not met in any of them are discarded. 

`m_highs_dif_per[m_highs_dif_per > 2.2].`

The list of these valid models is used to generate the file 

df_valids.to_csv(**"Models/TF_multi/_RESULTS_profit_multi_all.csv"**, sep=**'\t'**)

TODO make the execution in **Model_predictions_Multi_N_eval_profits.py** cleaner, and clearer. 




#### In the file **predict_POOL_enque_Thread.py,** 
run for real-time predictions, which you will see in the file *d_result/predi_MULTI_real_time_2022_12_29.csv*  

There are 2 threads:

one is in charge of downloading the data in real time and queueing it, it is associated to the **producer()** function.

`producer_thr = Thread(target=producer, args=(), name='PROD')`

And the consumer which queues and calculates the best models obtained from *Models/TF_multi/_RESULTS_profit_multi_all.csv* and record them in the *d_result/predi_MULTI_real_time_2022_12_29.csv* file.  

`consumer_thr_1 = threading.Thread(target=consumer, args=(1,), name='CONS_1')


## Thing to watch and debug to undertand:
Analyze, debug and **try to see small bugs** , all described **are critical** points of the code.


- Debug with  `Model_creation_models_for_a_stock.py` :
How GT is generated   IMPORTANT 
Another opinion , we have a lot of **_debate_** with it ? 
```python 
def rolling_get_sell_price_POS_next_value(rolling_col_slection):
def rolling_get_sell_price_NEG_next_value(rolling_col_slection):
```


- Debug with `Model_predictions_Multi_N_eval_profits.py` for LUIS 
How to evaluate the quality of the models in €€€€ IMPORTANT.
LUIS : take the best data from the df_test `get_best_proftis_models(df_eval, S, type_buy_sell :Op_buy_sell, df_molds_S_NP_columns, NUMBER_MACHTING_TEST_BEST = 3)`
SEBAS: take and calculate the buy-sell price and net profit `Model_predictions_Multi_N_eval_profits_2.py `

- Debug with  `Model_creation_models_for_a_stock.py` :
How to split (df_train, df_val, df_test ) and balance the SMOTE model
`Data_multidimension.Data_multidimension.load_split_data_multidimension() `

- Debug with  `Model_creation_models_for_a_stock.py` :
How to train: `early detection callback early_stopping = Utils_model_predict.CustomEarlyStopping(patience=8)`
How to obtain the neural model: `model = multi_data.get_dicts_models_multi_dimension(model_type) `

- Debug with  `Model_creation_models_for_a_stock.py` :
How to evaluate the quality of the models with df_test
`4.0 Eval de model with test_features this data had splited , and the .h5 model never see it`
`predit_test = model.predict(test_features).reshape(-1,)`

- Debug with ` Model_creation_models_for_a_stock.py` _**(the code are comment)**_:
skleanr and XGB have been removed, when switching from 2D monodimension to 3D multidimension, what is left of them, should they be used?
`def train_model_with_custom_columns(name_model, columns_list, csv_file_SCALA, op_buy_sell : _KEYS_DICT.Op_buy_sell):`

- Debug with `Model_creation_models_for_a_stock.py` :
How some data in .csv format is passed to 3D for TF. 
```python 
def df_to_df_multidimension_array_2D(dataframe, BACHT_SIZE_LOOKBACK, will_check_reshaped = True):
#must be displayed in 3D for TF by format of variable shape_imput_3d.
train_features = np.array(train_cleaned_df_target.drop(columns=[Y_TARGET]) ).reshape(shape_imput_3d)
```


- Debug with `gen_technical_indicators.py` :
When the technical patterns are generated 1068 are generated, of which are screened to plus or minus 120 in different leagues class MODEL_TYPE_COLM(Enum):
`def get_best_columns_to_train(cleaned_df, op_buy_sell : a_manage_stocks_dict.Op_buy_sell , num_best , CSV_NAME,path = None):`

###this is a sociological problem
You have to think in the principle of self-fulfilling prophecy, that is, use the same time frames that people in the city of london use,

the traders I know use 15 and 1 hour candles, never smaller, if the people of the city of london do not use 5 minutes or 1 minute, we should not use 5 minutes (I have thought a lot about making the models with 1 hour candles, since I understand that it is more common among the people of london). 

How many candles backwards se people in the city of london look  ? I don't think they look more than 20, I think that to put more than 20 is to put noise to the model.

Ask your trader friends, mine friends says, 15 minutes, 1 hour and daily.

this is a sociological problem, i.e. , 
to interpret the art painting, like the one in london, you have to look at the painting from the same position as london

