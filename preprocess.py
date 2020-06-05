import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random


# Convert meteorological information
def icon_to_int(df):
    dict = {' clear': 10, ' partlycloudy': 9, ' mostlycloudy': 8, ' cloudy': 7, ' hazy': 6,
            ' fog': 5, ' rain': 4, ' tstorms': 3, ' sleet': 2, ' snow': 1, ' unknown': 0}
    df['status'] = df['icon'].map(dict)


# Convert wind information
def wind_to_vec(df):
    df['winx'] = np.sin(df['wdird'] / 180 * np.pi) * df['wspdm']
    df['winy'] = np.cos(df['wdird'] / 180 * np.pi) * df['wspdm']


# Get attributes with too many missing values
def get_defect(df):
    defect = []
    for col in df.columns:
        if sum(df[col] == -9999) > 1000:
            defect.append(col)
    return defect


# Get attributes with too low std
def get_inert(df):
    inert = []
    for col in df.columns:
        if np.std(df[col], ddof=1) < 0.1:
            inert.append(col)
    return inert


# Delete redundant features and reorder based on previous analysis
def del_spare(df):
    rm = ['tempm', 'dewptm', 'hum', 'pressurem', 'winx', 'winy', 'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado',
          'status']
    return df[rm]


def detect_outliers(df, attr):
    ret = np.array([])
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in attr:
        t = 0
        cur = np.array([])
        for j in months:
            tf = df[i].iloc[t: t + j * 24]
            # 1st quartile
            Q1 = np.percentile(tf, 25)
            # 3rd quartile
            Q3 = np.percentile(tf, 75)
            # Interquartile range
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_idx = (tf < Q1 - outlier_step) | (tf > Q3 + outlier_step)
            cur = np.hstack((cur, np.array(outlier_idx, dtype=bool)))
            t += j * 24
        j = 0
        while (cur[j]):
            cur[j] = 0
            j += 1
        ret = np.hstack((ret, cur))
    return ret.reshape(len(attr), -1)


def fill_missing_data(df, om, attr):
    for i in attr:
        a = om[attr.index(i),].astype(np.bool)
        b = (1 - a).astype(np.bool)
        x = np.arange(8760)[b]
        y = np.array(df[i][b])
        cs = CubicSpline(x, y, bc_type='natural')
        for j in np.arange(8760)[a]:
            df.iloc[j, attr.index(i)] = cs(j)


def set_pca(df, attr):
    x = df[attr].values
    # Data standardization
    x = preprocessing.StandardScaler().fit_transform(x)
    # Dimension reduction by PCA
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(x)
    # Calculate how much variance can be maintained
    perExplainedVariance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    if perExplainedVariance[0] > 90:
        df['pca'] = pca_data
        df.drop(attr, axis=1, inplace=True)


def get_weather_data(path):
    wt = pd.read_csv(path)
    wind_to_vec(wt)
    icon_to_int(wt)
    defect = get_defect(wt)
    # print(defect)
    wt = del_spare(wt)
    inert = get_inert(wt)
    # print(inert)
    wt.drop(inert, axis=1, inplace=True)
    attr = ['tempm', 'dewptm', 'hum', 'pressurem', 'winx', 'winy']
    om = detect_outliers(wt, attr)
    fill_missing_data(wt, om, attr)
    set_pca(wt, ['snow', 'status'])
    return wt


def get_power_data(path):
    pw = pd.read_csv(path)
    return pw[['Solar+ [kW]']]


def data_slicing(wt, pw):
    x = []
    y = []
    t = 48
    wa = np.array(wt)
    pa = np.array(pw)
    min_max_scaler = preprocessing.MinMaxScaler()
    for i in range(len(pa) - t):
        ws = wa[i: i + t, ]
        ws = min_max_scaler.fit_transform(ws)
        x.append(np.hstack((ws, pa[i: i + t])))
        y.append(pa[i + t])
    x = np.array(x)
    y = np.array(y)
    # Ordered test data
    ss = [40, 130, 220, 310]
    b = np.tile(False, len(x))
    for s in ss:
        for i in range(200):
            b[s * 24 + i] = True
    x_test = x[b]
    y_test = y[b]
    x = x[(1 - b).astype(np.bool)]
    y = y[(1 - b).astype(np.bool)]
    # Shuffle data
    index = np.arange(len(x))
    random.shuffle(index)
    x = x[index]
    y = y[index]
    # Split into train and valid data
    split = int(len(x) * 0.8)
    x_train = x[:split]
    x_valid = x[split:]
    y_train = y[:split]
    y_valid = y[split:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test
