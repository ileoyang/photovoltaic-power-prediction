from tensorflow import keras
import numpy as np
from evaluation import test_model


def cnn_lstm_model1(n_filters, kernel_size, time_steps, features, hidden_units, conv_depth, lstm_depth):
    model = keras.Sequential()
    model.add(layers.TimeDistributed(layers.Conv1D(filters=n_filters,
                                                   kernel_size=kernel_size),
                                     input_shape=(time_steps, features, 1)))
    model.add(layers.LeakyReLU(0.01))
    for i in range(conv_depth - 1):
        model.add(layers.TimeDistributed(layers.Conv1D(filters=n_filters,
                                                       kernel_size=1)))
        model.add(layers.LeakyReLU(0.01))
    model.add(layers.TimeDistributed(layers.Flatten()))
    for i in range(lstm_depth - 1):
        model.add(layers.LSTM(hidden_units, return_sequences=True))
    model.add(layers.LSTM(hidden_units))
    model.add(layers.Dense(1))
    model.add(layers.LeakyReLU(0.01))
    Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=Adam,
                  loss='mse',
                  metrics=['mae'])
    return model


def cnn_lstm_model2(n_filters, kernel_size, time_steps, features, hidden_units, conv_depth, lstm_depth):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=n_filters,
                            kernel_size=kernel_size,
                            input_shape=(time_steps, features)))
    model.add(layers.LeakyReLU(0.01))
    for i in range(conv_depth - 1):
        model.add(layers.Conv1D(filters=n_filters,
                                kernel_size=1))
        model.add(layers.LeakyReLU(0.01))
    for i in range(lstm_depth - 1):
        model.add(layers.LSTM(hidden_units, return_sequences=True))
    model.add(layers.LSTM(hidden_units))
    model.add(layers.Dense(1))
    model.add(layers.LeakyReLU(0.01))
    Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=Adam,
                  loss='mse',
                  metrics=['mae'])
    return model


def train_model(model, x_train, y_train, x_valid, y_valid, filepath):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                                  patience=5, min_lr=0.0005)
    check_pointer = keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train,
                        batch_size=48,
                        epochs=300,
                        callbacks=[early_stopping, reduce_lr, check_pointer],
                        validation_data=(x_valid, y_valid))
    return history


def get_model(x_train, y_train, x_valid, y_valid, x_test, y_test):
    model1 = cnn_lstm_model1(5, 10, 48, 11, 10, 1, 2)
    model1.summary()
    history1 = train_model(model1, np.expand_dims(x_train, axis=3), y_train, np.expand_dims(x_valid, axis=3), y_valid,
                           "CNN-LSTM1.hdf5")
    test_model(model1, np.expand_dims(x_test, axis=3), y_test)
    model2 = cnn_lstm_model2(10, 1, 48, 11, 10, 1, 2)
    model2.summary()
    history2 = train_model(model2, x_train, y_train, x_valid, y_valid, "CNN-LSTM2.hdf5")
    test_model(model2, x_test, y_test)
    return model1, model2
