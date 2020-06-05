from preprocess import get_weather_data, get_power_data, data_slicing
from model import get_model


if __name__ == '__main__':
    wt = get_weather_data("data/weather.csv")
    pw = get_power_data("data/power.csv")
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_slicing(wt, pw)
    model1, model2 = get_model(x_train, y_train, x_valid, y_valid, x_test, y_test)
