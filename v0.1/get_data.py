from collections import deque
from sklearn.model_selection import train_test_split
import os
import sklearn as sk
import yfinance as yf
import datetime as dt
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
default_end_date = dt.datetime.now().strftime('%Y-%m-%d')
default_start_date = (dt.datetime.now() - dt.timedelta(days=4 * 365)).strftime('%Y-%m-%d')

def get_data(ticker, feature_columns, start_date=default_start_date, end_date=default_end_date, scale=True,
             test_size=0.2, steps_to_predict=3, seq_train_length=50, save_data=False,
             split_by_date=False):
    print("I am read_data")


    data_df = load_or_download(ticker, start_date, end_date)
    missing_values_count = data_df.isna().sum()
    print(missing_values_count)


    print(data_df)
    data_df.dropna(inplace=True)
    result = {'data': data_df.copy()}

    #     add date as a column
    if "date" not in data_df.columns:
        data_df["date"] = data_df.index

    #     scaling the data if required.
    # print(data_df[])

    # print(nans_count)
    if (scale is True):
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            print(column)
            #   saves scaled data to each column in feature_columns
            data_df[column] = scaler.fit_transform(np.expand_dims(data_df[column].values, axis=1))
            #   saves the specific minMax instances used for each column in a dict for later use.
            column_scaler[column] = scaler
        # print(column_scaler)
        #     add the MinMaxScaler instances to the result returned to have all useful data/tools in one place
        result['column_scaler'] = column_scaler

    # save the data to csv
    if (save_data is True):
        save_data_to_csv(data_df, ticker)
    data_df['future'] = data_df['Close'].shift(-steps_to_predict)
    sequence_last_data = np.array(data_df[feature_columns].tail(steps_to_predict))

    data_df.dropna(inplace=True)
    data_in_sequence = []
    sequences = deque(maxlen=seq_train_length)

    for entry, target in zip(data_df[feature_columns + ['date']].values, data_df['future'].values):
        sequences.append(entry)
        if len(sequences) == seq_train_length:
            data_in_sequence.append([np.array(sequences), target])
    sequence_last_data = list([s[:len(feature_columns)] for s in sequences]) + list(sequence_last_data)

    result['last_sequence'] = sequence_last_data
    x = []
    y = []
    for seq, target in data_in_sequence:
        x.append(seq)
        y.append(target)

    x = np.array(x)
    y = np.array(y)

    if split_by_date:
        train_samples = int((1 - test_size) * len(x))
        result['X_train'] = x[:train_samples]
        result['y_train'] = y[:train_samples]
        result['X_test'] = x[train_samples:]
        result['y_test'] = y[train_samples:]
        # if shuffle:
        #     pass
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(x, y,test_size=test_size)
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    # print(result[data_df])
    return data_df




def save_data_to_csv(data_df, ticker):
    #    Check if the folder 'data' exists, if not, create it
    print("save data to file fxn is being called")
    if not os.path.exists('data'):
        os.makedirs('data')
        # Save the DataFrame to a CSV file, overwriting if it exists
    data_df.to_csv(f"data/{ticker}.csv", index=False)

def load_data(ticker):
    if(not os.path.exists(f"data/{ticker}.csv")):
        print("Data file does not exist, downloading it now from yfinance....")
        return None
    else:
        data_df = pd.read_csv(f"data/{ticker}.csv")
        return data_df

def load_or_download(ticker, start_date, end_date):
    if (os.path.exists(f"data/{ticker}.csv")):
        print("Data file already exists, loading it now....")
        data_df = load_data(ticker)
        return data_df
    else:
        print("Data file does not exist, downloading it now from yfinance....")
        data_df = yf.download(ticker, start_date, end_date)
        return data_df



get_data('AMZN', ['Open', 'Close', 'High', 'Low', 'Volume'], save_data=True, split_by_date=True)
