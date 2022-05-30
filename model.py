import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import datetime as dt

np.random.seed(60)
window_len = 5
zero_base = True
lstm_neurons = 100
epochs = 10
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

start_time = dt.datetime(2018, 9, 1)
end_time = dt.datetime(2022, 2, 28)
pred_start = dt.datetime(2022, 3, 1)
pred_end = dt.datetime(2022, 6, 1)

def line_plot(line1, line2, line3=None, label1=None, label2=None, label3=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    if not (line3 is None):
      ax.plot(line3, label=label3, linewidth=lw)
    ax.set_ylabel('ETH/BTC', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()

def train_test_split(df, test_size=0.2, time=-1):
    train_data = df[start_time:end_time]
    test_data = df[pred_start:pred_end]
    return train_data, test_data

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy() # -1
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(train_data, test_data, target_col, window_len=10, zero_base=True):
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def main():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=BTC&tsym=ETH&limit=2000')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'

    hist = hist.drop(['conversionType', 'conversionSymbol'], axis=1)

    train, test = train_test_split(hist)

    train, test, X_train, X_test, y_train, y_test = prepare_data(train, test, target_col, window_len=window_len, zero_base=zero_base)
    model = build_lstm_model( X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    targets = test[target_col][window_len:]
    preds = model.predict(X_test).squeeze()

    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, None, 'real', 'prediction', lw=3)
    # line_plot(train[target_col], test[target_col], preds, 'real', 'real_pred', 'pred', lw=3)

if __name__ == "__main__":
    main()
