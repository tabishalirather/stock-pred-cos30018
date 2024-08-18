import sklearn as sk
import yfinance as yf
import datetime as dt
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler

default_end_date = dt.datetime.now().strftime('%Y-%m-%d')
default_start_date = (dt.datetime.now() - dt.timedelta(days=4 * 365)).strftime('%Y-%m-%d')


def read_data(ticker, feature_columns, start_date=default_start_date, end_date=default_end_date, scale=True,
              test_size=0.2, shuffle=False):
    data_df = yf.download(ticker, start_date, end_date).dropna()
    print(data_df)
    result = {'data': data_df.copy()}
    #     scaling the data if required.
    if (True==scale):
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            data_df[column] = scaler.fit_transform(np.expand_dims(data_df[column].values, axis=1))
            column_scaler[column] = scaler


read_data('AAPL', ['Open', 'Close', 'High', 'Low', 'Volume'])
