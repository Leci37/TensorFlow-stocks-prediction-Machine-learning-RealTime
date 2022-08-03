import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import Utils_model_predict
import Utils_plotter

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', #'val_recall', #'val_accuracy', #monitor='val_prc', monitor='recall',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

raw_df = pd.read_csv("d_price/FAV_SCALA_stock_history_E_MONTH_3.csv",index_col=False, sep='\t')
#raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')


Y_TARGET = 'buy_sell_point'
model_folder = "Models/TF_in_balance/"
model_h5_name = 'TF_in_balance.h5'

raw_df = Utils_model_predict.cast_Y_label_binary(raw_df,  label_name = Y_TARGET)
df = Utils_model_predict.clean_redifine_df(raw_df)
neg, pos = np.bincount(df[Y_TARGET])

Utils_plotter.plot_pie_countvalues(df,Y_TARGET , stockid= "", opion = "", path=model_folder )
print(df.isnull().sum())
# graficos de relaciones
# Utils_plotter.plot_relationdist_main_val_and_all_rest_val(df[["mtum_RSI","mtum_STOCH_k","mtum_STOCH_d", Y_TARGET]],Y_TARGET ,path = model_folder+"plot_relationdistplot_")

train_labels, val_labels, test_labels, train_features, val_features, test_features, bool_train_labels = Utils_model_predict.scaler_split_TF_onbalance(df, label_name = Y_TARGET)


EPOCHS = 100
BATCH_SIZE = 2048
initial_bias = np.log([pos/neg])
model = Utils_model_predict.make_model_TF_onbalance(shape_features = train_features.shape[-1])
print(model.summary())

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
print("Loss: {:0.4f}  without output_bias ".format(results[0]))
# model.predict(train_features[:10])


model = Utils_model_predict.make_model_TF_onbalance(shape_features = train_features.shape[-1], output_bias=initial_bias)
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=2)
print("Loss: {:0.4f} with output_bias".format(results[0]))
# model.predict(train_features[:10])

initial_weights = model_folder + "initial_weights/initial_weights"
model.save_weights(initial_weights)
print("model.save_weights initial_weights: ", initial_weights)


resampled_ds = Utils_model_predict.get_resampled_ds_onBalance(train_features, train_labels, bool_train_labels, BATCH_SIZE)
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

# Train on the oversampled data
# Now try training the model with the resampled data set instead of using class weights to see how these methods compare.
# Note: Because the data was balanced by replicating the positive examples, the total dataset size is larger, and each epoch runs for more training steps.
resampled_model = Utils_model_predict.make_model_TF_onbalance(shape_features = train_features.shape[-1])
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

resampled_history = resampled_model.fit(
    resampled_ds,
    epochs=EPOCHS,
    steps_per_epoch=resampled_steps_per_epoch,
    callbacks=[early_stopping],
    validation_data=val_ds)


resampled_model.save(model_folder +model_h5_name)
print(model_folder +'path_to_my_model.h5')
del model
resampled_model_2 = keras.models.load_model(model_folder + model_h5_name)

"""### Re-check training history"""
# plot_metrics(resampled_history)

"""### Evaluate metrics"""
train_predictions_resampled = resampled_model_2.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model_2.predict(test_features, batch_size=BATCH_SIZE)

resampled_results = resampled_model_2.evaluate(test_features, test_labels,
                                             batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resampled_model_2.metrics_names, resampled_results):
  print(name, ': ', value)
print()


Utils_plotter.plot_cm_TF_imbalance(test_labels, test_predictions_resampled,path= model_folder + "plot_confusion_matrix.png", p=0.5 )
# Utils_plotter.plot_confusion_matrix(cf_matrix, model_folder + "plot_confusion_matrix.png")
