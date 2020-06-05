import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def mae(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return K.mean(K.abs(y_pred - y_true))


def rmae(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return K.sum(K.abs(y_pred - y_true)) / K.sum(y_true)


# Rectified mape to prevent divide overflow
def mape(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return K.mean(K.abs(y_pred - y_true) / (y_true + 1))


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    plt.plot(y_pred[:48])
    plt.plot(y_test[:48])
    print(rmae(y_test, y_pred), mape(y_test, y_pred))
