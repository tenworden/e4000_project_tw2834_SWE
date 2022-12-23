import tensorflow as tf


def get_lstm_model(n_neuron, input_shape, y_out):
    
    activation     = 'relu'
    learning_rate  = 0.001

    input_layer = tf.keras.layers.Input(shape=input_shape)
    lstm_1 = tf.keras.layers.LSTM(n_neuron,
               return_sequences=True, activation=activation)(input_layer)
    lstm_2 = tf.keras.layers.LSTM(n_neuron, return_sequences=True,
               activation=activation)(lstm_1)
    lstm_3 = tf.keras.layers.LSTM(n_neuron, return_sequences=False,
               activation=activation)(lstm_2)
    dense = tf.keras.layers.Dense(y_out, activation='relu')(lstm_3)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    model.summary()
    model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model