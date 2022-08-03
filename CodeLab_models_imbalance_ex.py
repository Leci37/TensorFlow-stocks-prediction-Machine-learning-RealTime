import tensorflow as tf


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

INPUT_SHAPE = (train_features.shape[-1],)

dict_models_binary = {}
dict_models_NO_binary ={}
dict_models_complex = {}

model, name_model = get_model_sequen_cordobilla_LTMS()
dict_models_binary[name_model] = model
model, name_model = get_model_sequen_hassanamin_kaggle_LSTM()
dict_models_binary[name_model] = model
model, name_model = get_model_sequen_ML_recurrent_LSTM()
dict_models_binary[name_model] = model
model, name_model = get_model_sequen_simple_dense_steps()
dict_models_binary[name_model] = model
model, name_model = get_model_sequen_convolution_layer()
dict_models_binary[name_model] = model
model, name_model = get_model_sequen_simple_LTMS()
dict_models_binary[name_model] = model

model, name_model = get_model_multi_simple_dense()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_step_dense()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_simple_LSTM()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_single_shot()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_dense()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_conv_model()
dict_models_NO_binary[name_model] = model
model, name_model = get_model_multi_lstm_model()
dict_models_NO_binary[name_model] = model

model, name_model = get_model_multi_dnn_model(n_horizon, lr)
dict_models_complex[name_model] = model
model, name_model = get_model_multi_cnn_model( n_horizon, lr=3e-4)
dict_models_complex[name_model] = model
model, name_model = get_model_multi_lstm( n_horizon, lr)
dict_models_complex[name_model] = model
model, name_model = get_model_lstm_cnn_model( n_horizon,  lr)
dict_models_complex[name_model] = model
model, name_model = get_model_lstm_cnn_skip_model( n_horizon,  lr)
dict_models_complex[name_model] = model




#JORDI CORDOBILLA https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning
def get_model_sequen_cordobilla_LTMS():
    model = tf.keras.models.Sequential()
    # 1st layer with Dropout regularisation
    # * units = add 100 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    # * input_shape => Shape of the training dataset
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=INPUT_SHAPE)), #(x_train.shape[1], 1)))
    # 20% of the layers will be dropped
    model.add(tf.keras.layers.Dropout(0.2))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    # 20% of the layers will be dropped
    model.add(tf.keras.layers.Dropout(0.2))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    # 50% of the layers will be dropped
    model.add(tf.keras.layers.Dropout(0.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    model.add(tf.keras.layers.LSTM(units=50))
    # 50% of the layers will be dropped
    model.add(tf.keras.layers.Dropout(0.5))
    # Dense layer that specifies an output of one unit
    model.add(tf.keras.layers.Dense(units=1))
    # model.summary()
    # tf.keras.utils.plot_model(model, to_file=os.path.join(project_folder, 'model_lstm.png'), show_shapes=True,
    #                           show_layer_names=True)
    return model, "simple_LTMS_cordobilla"

#Time Series Analysis using LSTM Keras
#https://www.kaggle.com/code/hassanamin/time-series-analysis-using-lstm-keras/notebook
def get_model_sequen_hassanamin_kaggle_LSTM():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=INPUT_SHAPE)), #(x_train.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    return model , "simple_LSTM_kaggle_hassanamin"
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# create and fit the LSTM network
def get_model_sequen_ML_recurrent_LSTM():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(4, input_shape=INPUT_SHAPE)), #(x_train.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    return model , "simple_LSTM_ML_recurrent"
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# Trade time_series forecast.ipynb   GOOGLE
# LUIS
# https://colab.research.google.com/drive/1940aC6X6xyeJNTP1qnfxoHhQZwGa_yOx#scrollTo=jKq3eAIvH4Db
# Dense
# Before applying Models that actually operate on multiple time-steps, it's worth checking the performance of deeper, more powerful, single input step Models.
# Here's a model similar to the linear model, except it stacks several a few Dense layers between the input and the output:
def get_model_sequen_simple_dense_steps():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    return model, "simple_dense_steps"

# Convolution neural network
# A convolution layer (tf.keras.layers.Conv1D) also takes multiple time steps as input to each prediction.
# Below is the same model as multi_step_dense, re-written with a convolution.
# The tf.keras.layers.Flatten and the first tf.keras.layers.Dense are replaced by a tf.keras.layers.Conv1D.
# The tf.keras.layers.Reshape is no longer necessary since the convolution keeps the time axis in its output.
CONV_WIDTH = 3
# conv_window = WindowGenerator(
#     input_width=CONV_WIDTH,
#     label_width=1,
#     shift=1,
#     label_columns=['T (degC)'])
def get_model_sequen_convolution_layer():
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    return conv_model, "simple_convolution_layer"

# An important constructor argument for all Keras RNN layers, such as tf.keras.layers.LSTM, is the return_sequences argument. This setting can configure the layer in one of two ways:
# If False, the default, the layer only returns the output of the final time step, giving the model time to warm up its internal state before making a single prediction:
def get_model_sequen_simple_LTMS():
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return lstm_model, "simple_LTMS"
# Multi-output Models
# Multi-output Models
NUM_FEATURES = 2 #numero de cosas a predecir
def get_model_multi_simple_dense():
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=NUM_FEATURES)
    ])
    return dense ,"multi_simple_dense"

# Multi-step dense
# A single-time-step model has no context for the current values of its inputs. It can't see how the input features are changing over time.' \
# To address this issue the model needs access to multiple time steps when making predictions:
def get_model_multi_step_dense ():
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    return multi_step_dense, "multi_step_dense"

def get_model_multi_simple_LSTM():
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=NUM_FEATURES)
    ])
    return lstm_model, "multi_simple_LSTM"

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

# Single-shot Models
# One high-level approach to this problem is to use a "single-shot" model, where the model makes the entire sequence prediction in a single step.
# This can be implemented efficiently as a tf.keras.layers.Dense with OUT_STEPS*features output units. The model just needs to reshape that output to the required (OUTPUT_STEPS, features).
OUT_STEPS = 24

def get_model_multi_single_shot():
    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * NUM_FEATURES,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, NUM_FEATURES])
    ])
    return multi_linear_model, "multi_single_shot"
# Dense
# Adding a tf.keras.layers.Dense between the input and output gives the linear model more power, but is still only based on a single input time step.
def get_model_multi_dense():
    multi_dense_model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation='relu'),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * NUM_FEATURES,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, NUM_FEATURES])
    ])
    return multi_dense_model, "multi_dense_model"
# CNN
# A convolutional model makes predictions based on a fixed-width history, which may lead to better performance than the dense model since it can see how things are changing over time:
CONV_WIDTH = 3
def get_model_multi_conv_model():
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * NUM_FEATURES,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, NUM_FEATURES])
    ])
    return multi_conv_model , "multi_conv_model"
# RNN
# A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making. Here the model will accumulate internal state for 24 hours, before making a single prediction for the next 24 hours.
# In this single-shot format, the LSTM only needs to produce an output at the last time step, so set return_sequences=False in tf.keras.layers.LSTM.
def get_model_multi_lstm_model():
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS * NUM_FEATURES,kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, NUM_FEATURES])
    ])
    return multi_lstm_model , "multi_lstm_model"


#https://www.kaggle.com/code/nicholasjhana/multi-variate-time-series-forecasting-tensorflow/notebook
# Para mas info curso de coursera https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/EtewX/lstm-notebook-lab-2
# Multi-Variate Time Series Forecasting Tensorflow   KAGGLE
# Python · Hourly energy demand generation and weather
# A three layer DNN (one layer plus the common bottom two layers)
def get_model_multi_dnn_model(n_horizon, lr):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=INPUT_SHAPE),#input_shape=(n_steps, n_features)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='dnn')
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])

    return model, "multi_dnn_model"
# dnn = dnn_model(*get_params(multivar=True))
# dnn.summary()
# A CNN with two layers of 1D convolutions with max pooling.
def get_model_multi_cnn_model( n_horizon, lr=3e-4):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=INPUT_SHAPE),#input_shape=(n_steps, n_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name="CNN")
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    return model , "multi_cnn_model"
# cnn = cnn_model(*get_params(multivar=True))
# cnn.summary()
# A LSTM with two LSTM layers.
def get_model_multi_lstm( n_horizon, lr):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(72, activation='relu', input_shape=INPUT_SHAPE , return_sequences=True),#, input_shape=(n_steps, n_features),
        tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='lstm')
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    return model , "multi_lstm"
# lstm = lstm_model(*get_params(multivar=True))
# lstm.summary()
# A CNN stacked LSTM with layers from Models 2 and 3 feeding into the common DNN layer.
def get_model_lstm_cnn_model( n_horizon,  lr):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=INPUT_SHAPE ),#, input_shape=(n_steps, n_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(72, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name="lstm_cnn")
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    return model , "lstm_cnn_model"
# lstm_cnn = lstm_cnn_model(*get_params(multivar=True))
# lstm_cnn.summary()
# A CNN stacked LSTM with a skip connection to the common DNN layer.
def get_model_lstm_cnn_skip_model( n_horizon,  lr):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=INPUT_SHAPE ),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.LSTM(72, activation='relu', return_sequences=True),
      tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Flatten(),
      # tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(n_horizon)
    ], name="lstm_skip")
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    return model , "lstm_cnn_skip_model"
# lstm_skip = lstm_cnn_skip_model(*get_params(multivar=True))
# lstm_skip.summary()
# tf.keras.utils.plot_model(lstm_skip, show_shapes=True)
