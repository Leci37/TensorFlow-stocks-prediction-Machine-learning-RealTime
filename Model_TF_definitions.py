import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Flatten, MaxPooling1D, Bidirectional, LSTM, Dropout, TimeDistributed, MaxPool2D, GRU
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Sequential

from a_manage_stocks_dict import MODEL_TF_DENSE_TYPE, MODEL_TF_DENSE_TYPE_ONE_DIMENSI


class ModelDefinition:
    OUT_STEPS = 24
    CONV_WIDTH = 8
    shape_inputs = None
    num_features = None
    output_bias = None
    output_bias_k = None

    def __init__(self, shape_inputs_m, num_features_m, output_bias_m = None):
        self.shape_inputs = shape_inputs_m
        self.num_features = num_features_m
        self.output_bias = output_bias_m
        self.output_bias_k = None


    def get_dicts_models_multi_dimension(self):
        dict_models = {
            MODEL_TF_DENSE_TYPE.SIMP_DENSE28: self.__make_model_TF_onbalance_fine_28(),
            MODEL_TF_DENSE_TYPE.SIMP_DENSE64: self.__make_model_TF_onbalance_fine_64(),
            MODEL_TF_DENSE_TYPE.SIMP_DENSE128: self.__make_model_TF_onbalance_fine_128(),
            MODEL_TF_DENSE_TYPE.SIMP_CORDO: self.model_cordo_long_short_term_memory_model(),
            MODEL_TF_DENSE_TYPE.SIMP_DENSE: self.__model_dense_simple(),
            # MODEL_TF_DENSE_TYPE.MULT_DENSE: self.__model_dense_multi(), ggg
            MODEL_TF_DENSE_TYPE.SIMP_CONV: self.__model_conv(),
            # MODEL_TF_DENSE_TYPE.MULT_LINEAR: self.__model_multi_linear(), W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            # MODEL_TF_DENSE_TYPE.MULT_DENSE2: self.__model_multi_dense(),W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            #MODEL_TF_DENSE_TYPE.MULT_CONV: self.__model_multi_conv(),W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            # MODEL_TF_DENSE_TYPE.MULT_LSTM: self.__model_multi_lstm(),W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            # MODEL_TF_DENSE_TYPE.SIMPL_BIDI: self.__model_Embedding_Bidirectional(),FALLA
            # MODEL_TF_DENSE_TYPE.MULT_TIME: self.__model_time_multi(),This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.
            MODEL_TF_DENSE_TYPE.MULT_GRU: self.__model_gru()
        }
        return dict_models

    def get_dicts_models_One_dimension(self):
        dict_models = {
            MODEL_TF_DENSE_TYPE_ONE_DIMENSI.SIMP_28: self.make_model_TF_one_dimension_28(),
            MODEL_TF_DENSE_TYPE_ONE_DIMENSI.SIMP_64: self.make_model_TF_one_dimension_64(),
            MODEL_TF_DENSE_TYPE_ONE_DIMENSI.SIMP_128: self.make_model_TF_one_dimension_128(),
        }
        return dict_models

    METRICS_ACCU_PRE = [
        # keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.BinaryAccuracy(name='accuracy')
        # keras.metrics.Precision(name='precision')
    ]
    def __compiler_one_dimension(self, model , metrics=METRICS_ACCU_PRE):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        return model

    def make_model_TF_one_dimension_28(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(28, activation='relu', input_shape=(self.shape_inputs,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])

        return self.__compiler_one_dimension(model)

    def make_model_TF_one_dimension_64(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.shape_inputs,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])

        return self.__compiler_one_dimension(model)

    def make_model_TF_one_dimension_128(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.shape_inputs,)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])

        return self.__compiler_one_dimension(model)




    def __compiler_multi_dimensions(self, model, metrics=METRICS_ACCU_PRE):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        return model

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    def __make_model_TF_onbalance_fine_28(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)

        # input_shape = keras.layers.Input(batch_shape=[self.shape_inputs, ]).input_shape()[0]
        model = Sequential()
        model.add(keras.layers.Conv2D(32, (1, 1), input_shape=(self.shape_inputs[0],self.shape_inputs[1] ), data_format='channels_first' )),
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(32, (1, 1)))
        model.add(keras.layers.Activation('relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(keras.layers.Conv2D(64, (3, 3)))
        # model.add(keras.layers.Activation('relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(keras.layers.Dense(64))
        # model.add(keras.layers.Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(keras.layers.Dense(1, bias_initializer=self.output_bias_k),)
        model.add(keras.layers.Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model #self.compiler_m(model)

    # def __make_model_TF_onbalance_fine_28(self):
    #     if self.output_bias is not None:
    #         self.output_bias_k = keras.initializers.Constant(self.output_bias)
    #     model = keras.Sequential([
    #         keras.layers.Dense(28, activation='relu', input_shape=self.shape_inputs ),
    #         keras.layers.Dropout(0.2),
    #         keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
    #     ])
    #
    #     return self.compiler_m(model)

    def __make_model_TF_onbalance_fine_64(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=self.shape_inputs ),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(model)

    def __make_model_TF_onbalance_fine_128(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=self.shape_inputs ),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(model)


    # https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning
    def model_cordo_long_short_term_memory_model(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential()
        # 1st layer with Dropout regularisation
        # * units = add 100 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # * input_shape => Shape of the training dataset
        model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=self.shape_inputs))
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
        model.add(tf.keras.layers.Dense(units=1, bias_initializer=self.output_bias_k ))
        # tf.keras.utils.plot_model(model, to_file=os.path.join(project_folder, '__model_lstm.png'), show_shapes=True,
        #                           show_layer_names=True)
        return self.__compiler_multi_dimensions(model)

    # https://www.tensorflow.org/tutorials/structured_data/time_series
    # Antes de aplicar modelos que realmente operan en múltiples pasos de tiempo, vale la pena comprobar el rendimiento de modelos de paso de entrada única más profundos y potentes.
    # DENSO
    def __model_dense_simple(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu', input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1, bias_initializer=self.output_bias_k)
        ])
        return self.__compiler_multi_dimensions(dense)

    # Puede entrenar un modelo dense en una ventana de múltiples pasos de entrada agregando tf.keras.layers.Flatten como la primera capa del modelo:
    def __model_dense_multi(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu', input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, bias_initializer=self.output_bias_k),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])
        return self.__compiler_multi_dimensions(multi_step_dense)

    def __model_conv(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=(self.CONV_WIDTH,),
                                   activation='relu', input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(conv_model)

    def __model_multi_linear(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_linear_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :], input_shape=self.shape_inputs),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), bias_initializer=self.output_bias_k),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
        ])
        return self.__compiler_multi_dimensions(multi_linear_model)

    def __model_multi_dense(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_dense_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :], input_shape=self.shape_inputs),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), bias_initializer=self.output_bias_k),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
        ])
        return self.__compiler_multi_dimensions(multi_dense_model)

    def __model_multi_conv(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -self.CONV_WIDTH:, :], input_shape=self.shape_inputs),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(self.CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), bias_initializer=self.output_bias_k),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
        ])
        return self.__compiler_multi_dimensions(multi_conv_model)

    def __model_multi_lstm(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False, input_shape=self.shape_inputs),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), bias_initializer=self.output_bias_k),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features])
        ])
        return self.__compiler_multi_dimensions(multi_lstm_model)

    # def __model_dense_simple(self):
    #     # https://www.tensorflow.org/text/tutorials/text_classification_rnn
    #     model = tf.keras.Sequential([
    #         encoder,
    #         tf.keras.layers.Embedding(
    #             input_dim=len(encoder.get_vocabulary()),
    #             output_dim=64,
    #             # Use masking to handle the variable sequence lengths
    #             mask_zero=True),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dense(1)
    #     ])
    #     return model

    # def __model_Embedding_Bidirectional(self):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Embedding(len(self.num_features), 64, mask_zero=True, input_shape=self.shape_inputs),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dropout(0.5),
    #         tf.keras.layers.Dense(1, bias_initializer=self.output_bias)
    #     ])
    #     return model

    def __model_time_multi(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        # https://github.com/Soroush98/LSTM-CNN_Stock/blob/master/LSTM_CNN.py
        model = Sequential()
        # add model layers
        model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu', input_shape=self.shape_inputs)))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(200, return_sequences=False)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear', bias_initializer=self.output_bias_k))
        model.compile(optimizer='RMSprop', loss='mse')
        return self.__compiler_multi_dimensions(model)

    def __model_gru(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=self.shape_inputs))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=True))
        model.add(GRU(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, bias_initializer=self.output_bias_k))
        return self.__compiler_multi_dimensions(model)

# https://blog.mlreview.com/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
# Model architecture parameters
# Initializers
# sigma = 1
# weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
# bias_initializer = tf.zeros_initializer()
# n_stocks = 500
# n_neurons_1 = 1024
# n_neurons_2 = 512
# n_neurons_3 = 256
# n_neurons_4 = 128
# n_target = 1
# # Layer 1: Variables for hidden weights and biases
# W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
# bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# # Layer 2: Variables for hidden weights and biases
# W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
# bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# # Layer 3: Variables for hidden weights and biases
# W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
# bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# # Layer 4: Variables for hidden weights and biases
# W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
# bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
#
# # Output layer: Variables for output weights and biases
# W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
# bias_out = tf.Variable(bias_initializer([n_target]))
# # Hidden layer
# hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
# hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
# hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
# hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
#
# # Output layer (must be transposed)
# out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))