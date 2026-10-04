import tensorflow as tf
from keras.optimizers import RMSprop
from tensorflow import keras
from keras.layers import Conv1D, Flatten, MaxPooling1D, Bidirectional, LSTM, Dropout, TimeDistributed, MaxPool2D, GRU
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Sequential

from _KEYS_DICT import MODEL_TF_DENSE_TYPE_MULTI_DIMENSI, MODEL_TF_DENSE_TYPE_ONE_DIMENSI

ACTIVATION_1 = 'linear'
ACTIVATION_2_ReLU = tf.keras.layers.LeakyReLU(alpha=0.1)  # mejora un 20% los resultados
# ACTIVATION_3_end = 'sigmoid'

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
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_DENSE28: self.__make_model_TF_onbalance_fine_28(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_DENSE64: self.__make_model_TF_onbalance_fine_64(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_DENSE128: self.__make_model_TF_onbalance_fine_128(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_CORDO: self.model_cordo_long_short_term_memory_model(),
            # MODEL_TF_DENSE_TYPE.SIMP_DENSE: self.__model_dense_simple(),
            # MODEL_TF_DENSE_TYPE.MULT_DENSE: self.__model_dense_multi(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_CONV: self.__model_conv(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.SIMP_CONV2: self.__model_conv2(),
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_LINEAR: self.__model_multi_linear(),# W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_DENSE2: self.__model_multi_dense(), #W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            # MODEL_TF_DENSE_TYPE.MULT_CONV: self.__model_multi_conv(),#W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_LSTM: self.__model_multi_lstm(),#W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
            ####MODEL_TF_DENSE_TYPE.SIMPL_BIDI: self.__model_Embedding_Bidirectional(),#FALLA
            # MODEL_TF_DENSE_TYPE.MULT_TIME: self.__model_time_multi(),#This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.
            MODEL_TF_DENSE_TYPE_MULTI_DIMENSI.MULT_GRU: self.__model_gru()
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

    from keras.optimizers import RMSprop
    def __compiler_multi_dimensions(self, model,learning_rateR = 0.001, metrics=METRICS_ACCU_PRE):
        model.compile(
            #optimizer=keras.optimizers.Adam(learning_rate=0.001),
            optimizer=RMSprop(learning_rate=learning_rateR ),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        return model


    # MULTI DIMENSION
    def __make_model_TF_onbalance_fine_28(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(28, activation=ACTIVATION_1, input_shape=self.shape_inputs),
            keras.layers.Dense(8, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.2),
            Flatten(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(model)

    def __make_model_TF_onbalance_fine_64(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(64, activation=ACTIVATION_1, input_shape=self.shape_inputs),
            keras.layers.Dense(32, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation=ACTIVATION_1),
            keras.layers.Dropout(0.2),
            Flatten(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(model)

    def __make_model_TF_onbalance_fine_128(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = keras.Sequential([
            keras.layers.Dense(128, activation=ACTIVATION_1, input_shape=self.shape_inputs),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation=ACTIVATION_2_ReLU),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation=ACTIVATION_1),
            keras.layers.Dense(16, activation=ACTIVATION_1),
            keras.layers.Dropout(0.2),
            Flatten(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=self.output_bias_k),
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
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -1:, :], input_shape=self.shape_inputs) )
        model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
        # 20% of the layers will be dropped
        # model.add( tf.keras.layers.Dense(self.OUT_STEPS * self.num_features, kernel_initializer=tf.initializers.zeros(), activation=ACTIVATION_2_ReLU) )
        model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.Dense(units=32, activation=ACTIVATION_2_ReLU) )
        model.add(keras.layers.Dense(32, activation=ACTIVATION_1)),
        # model.add(Flatten())
        # 2nd LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(tf.keras.layers.LSTM(units=32, return_sequences=True ))
        # 20% of the layers will be dropped
        model.add(tf.keras.layers.Dropout(0.2))
        # 3rd LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
        model.add(keras.layers.Dense(32, activation=ACTIVATION_1)),
        # 50% of the layers will be dropped
        model.add(tf.keras.layers.Dropout(0.2))
        # 4th LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        model.add(tf.keras.layers.LSTM(units=16))
        # 50% of the layers will be dropped
        model.add(tf.keras.layers.Dropout(0.2))
        # Dense layer that specifies an output of one unit
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=self.output_bias_k))
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
            tf.keras.layers.Dense(units=64, activation=ACTIVATION_1, input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=64, activation=ACTIVATION_2_ReLU),
            tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=self.output_bias_k)
        ])
        return self.__compiler_multi_dimensions(dense)

    # Puede entrenar un modelo dense en una ventana de múltiples pasos de entrada agregando tf.keras.layers.Flatten como la primera capa del modelo:
    def __model_dense_multi(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)

            tf.keras.layers.Dense(units=32, activation=ACTIVATION_1, input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=32, activation=ACTIVATION_2_ReLU),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=self.output_bias_k),
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
                                   activation=ACTIVATION_1, input_shape=self.shape_inputs),
            tf.keras.layers.Dense(units=32, activation=ACTIVATION_2_ReLU),
            Flatten(),
            tf.keras.layers.Dense(units=1, activation="sigmoid", bias_initializer=self.output_bias_k),
        ])
        return self.__compiler_multi_dimensions(conv_model)

    def __model_conv2(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)

        # input_shape = keras.layers.Input(batch_shape=[self.shape_inputs, ]).input_shape()[0]
        model = Sequential()
        model.add(keras.layers.Conv2D(256, (16, 16), input_shape=(self.shape_inputs[0], self.shape_inputs[1], 1)  ,padding='same')),
        model.add(keras.layers.Activation(ACTIVATION_1))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dense(32, activation=ACTIVATION_1)),
        # model.add(keras.layers.Conv2D(32, (1, 1)))
        model.add(keras.layers.Activation(ACTIVATION_2_ReLU))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(Flatten())
        # model.add(keras.layers.Activation(ACTIVATION_1))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(keras.layers.Dense(64))
        # model.add(keras.layers.Activation(ACTIVATION_1))
        # model.add(Dropout(0.5))
        # tf.keras.layers.Dense(1, activation="sigmoid")
        model.add(keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k), )
        # model.add(keras.layers.Activation('sigmoid'))

        # model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

        return self.__compiler_multi_dimensions(model)

    def __model_multi_linear(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_linear_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :], input_shape=self.shape_inputs),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features]),
            Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k)

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
            tf.keras.layers.Dense(512, activation=ACTIVATION_1),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), activation=ACTIVATION_2_ReLU),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features]),
            Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k)
        ])
        return self.__compiler_multi_dimensions(multi_dense_model)

    def __model_multi_conv(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -self.CONV_WIDTH:, :], input_shape=self.shape_inputs),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation=ACTIVATION_1, kernel_size=(self.CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), activation=ACTIVATION_2_ReLU),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features]),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k)
        ])
        return self.__compiler_multi_dimensions(multi_conv_model)

    def __model_multi_lstm(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(128, return_sequences=False, input_shape=self.shape_inputs),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.OUT_STEPS * self.num_features,
                                  kernel_initializer=tf.initializers.zeros(), activation=ACTIVATION_2_ReLU),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([self.OUT_STEPS, self.num_features]),
            Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k)
        ])
        return self.__compiler_multi_dimensions(multi_lstm_model)

    # def __model_Embedding_Bidirectional(self):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Embedding(len(self.num_features), 64, mask_zero=True, input_shape=self.shape_inputs),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #         tf.keras.layers.Dense(64, activation=ACTIVATION_1),
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
        model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation=ACTIVATION_1,
                                         input_shape=(self.shape_inputs[0], self.shape_inputs[1], 1))))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation=ACTIVATION_1)))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation=ACTIVATION_1)))
        model.add(TimeDistributed(MaxPooling1D(2)))

        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(200, return_sequences=False)))
        model.add(TimeDistributed(Flatten()))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k))
        # model.compile(optimizer='RMSprop', loss='mse')
        return self.__compiler_multi_dimensions(model).build(
            input_shape=(self.shape_inputs[0], self.shape_inputs[1], 1))

    def __model_gru(self):
        if self.output_bias is not None:
            self.output_bias_k = keras.initializers.Constant(self.output_bias)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=self.shape_inputs))
        keras.layers.Dense(32, activation=ACTIVATION_1),
        # model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        # model.add(GRU(32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=True))
        model.add(GRU(32))
        model.add(keras.layers.Dense(64, activation=ACTIVATION_1))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid", bias_initializer=self.output_bias_k))
        return self.__compiler_multi_dimensions(model, learning_rateR = 0.001)
    #MULTI DIMENSION


    #MONO DIMENSION
    def __compiler_one_dimension(self, model , metrics=METRICS_ACCU_PRE):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
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
            keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01) ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])

        return self.__compiler_one_dimension(model)

    def make_model_TF_one_dimension_128(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.shape_inputs,)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='relu', bias_initializer=self.output_bias_k),
        ])

        return self.__compiler_one_dimension(model)
    #MONO DIMENSION



    # #https://stackoverflow.com/questions/48957094/how-can-i-use-leaky-relu-as-an-activation-in-tensorflow-tf-layers-dense
    # def my_leaky_relu(x):
    #     return tf.nn.leaky_relu(x, alpha=0.0001)
    #
    # ACTIVATION_ReLU = tf.keras.layers.LeakyReLU(alpha=0.1) #mejora un 20% los resultados
    # ACTIVATION = 'linear'
    # model.add(Dense(1024, input_shape=imput_shape, activation= ACTIVATION, ))
    # model.add(Dense(512, activation= ACTIVATION_ReLU, name="middle"))
    # model.add(Dense(128, input_shape=imput_shape, activation=ACTIVATION, ))
    # model.add(Flatten())
    # model.add(Dense(4, activation=ACTIVATION))
    # model.add(Dense(1, activation='sigmoid')) #sigmoid


