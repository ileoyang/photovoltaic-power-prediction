import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from fbprophet import Prophet
from scipy.interpolate import CubicSpline


def line_weather_graph(wt):
    wt[['tempm', 'dewptm', 'hum']].plot(subplots=True)
    wt[['winx', 'winy', 'pressurem']].plot(subplots=True)


def line_power_graph(pw):
    plt.plot(pw[:96])
    plt.xlabel('Time (Hour)')
    plt.ylabel('PV Power (KW)')


def prophet_graph(pw):
    ppw = pw[['Date & Time', 'Solar+ [kW]']]
    ppw.rename(columns={'Date & Time': 'ds', 'Solar+ [kW]': 'y'}, inplace=True)
    m = Prophet()
    m.fit(ppw)
    future = m.make_future_dataframe(periods=120, freq='h')
    forecast = m.predict(future)
    fig = m.plot_components(forecast)


def heat_map(wt, pw):
    wt['power'] = pw
    corrmat = wt.corr('spearman')
    f, ax = plt.subplots(figsize=(12, 8))
    sns.set(font_scale=1.25)
    hm = sns.heatmap(corrmat, cbar=True, annot=True,
                     square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corrmat.columns,
                     xticklabels=corrmat.columns, linewidths=0)
    plt.show()
    wt.drop(['power'], axis=1, inplace=True)


def box_plot(df, attr):
    monthID = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
               'Sep', 'Oct', 'Nov', 'Dec']
    monthLen = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = np.array([])
    for i in range(12):
        month = np.append(month, np.tile(monthID[i], monthLen[i] * 24))
    df['month'] = month
    sns.boxplot(x="month", y=attr, data=df)
    df.drop(['month'], axis=1, inplace=True)


def cubic_curve(wt):
    x = np.arange(20)
    y = np.array(wt['tempm'])[:20]
    c = np.linspace(0, 20, 100)
    px = np.delete(x, (9, 10, 11))
    py = np.delete(y, (9, 10, 11))
    cs = CubicSpline(px, py, bc_type='natural')
    plt.plot(x, y, label='Real Data')
    plt.plot(px, py, label='Linear Interpolation')
    plt.plot(c, cs(c), label='Cubic Spline Interpolation')
    plt.legend()


def mountain_graph(df):
    z = np.array(df).reshape(365, 24)
    fig = go.Figure(data=[
        go.Surface(z=z, showscale=False, opacity=0.9)
    ])
    fig.show()
