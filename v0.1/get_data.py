from collections import deque
from turtledemo.penrose import start

from fontTools.misc.plistlib import end_date
from sklearn.model_selection import train_test_split
import os
# import sklearn as sk
import yfinance as yf
import datetime as dt
import numpy as np
# import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# setup default end date and start date for convenience., this can be changed later.
default_end_date = dt.datetime.now().strftime('%Y-%m-%d')
default_start_date = (dt.datetime.now() - dt.timedelta(days=5 * 365)).strftime('%Y-%m-%d')


def get_data(ticker, feature_columns, start_date, end_date, seq_train_length,steps_to_predict, scale=True,
             test_size=0.2,  save_data=False,
             split_by_date=False):
	print("I am read_data")

	# load the data if it is saved already of download using yfinance.
	# data_df = load_or_download(ticker, start_date, end_date)
	data_df = yf.download(ticker, start_date, end_date)
	print(data_df.head())
	# count the missing values in the data. In case there's a high number of missing values, investigate why and how to mitigate the effects
	missing_values_count = data_df.isna().sum()
	# print(missing_values_count)

	# print(data_df)
	# drop nas
	data_df.dropna(inplace=True)

	#     add date as a column
	if "date" not in data_df.columns:
		data_df["date"] = data_df.index

	# Copy the dataframe for future use
	result = {'data_df': data_df.copy()}
	# scaling the data if the user wants it.
	# print(data_df[])
	if (scale is True):
		column_scaler = {}
		for column in feature_columns:
			# using min max scaler from scikit learn, normalises everything between 0 and 1
			scaler = MinMaxScaler()
			print(column)
			#   saves scaled data to each column in feature_columns, reshapes the data to be 2D as is a requirement for MinMaxScaler
			data_df[column] = scaler.fit_transform(np.expand_dims(data_df[column].values, axis=1))
			#   saves the specific minMax instances used for each column in a dict for later use, like descaling.
			column_scaler[column] = scaler
		# print(column_scaler)
		#     add the MinMaxScaler instances to the result returned to have all useful data/tools in one place
		result['column_scaler'] = column_scaler

	# save the data to csv if the user wants it.
	if (save_data is True):
		save_data_to_csv(data_df, ticker, start_date, end_date)
		# I need help explaining this step
		# this shifts the close price up by the number steps_to_predict. For example if steps_to_predict is 3, the future column row 1 will contain the close price from column 4.

		# data_df['future'] = data_df['Close'].shift(-steps_to_predict)

	future_columns = []

	for i in range(1, steps_to_predict + 1):
		future_col_name = f'future_{i}'
		data_df[future_col_name] = data_df['Close'].shift(-i)
		future_columns.append(future_col_name)

		# Drop rows with NaNs created by shifting
	data_df.dropna(inplace=True)  # Drop rows with NaNs created by shifting

	# Converts the latest data (of steps_to_predict length) in feature_columns to a numpy array and save to sequence_last_data. We want to get the recent data to predict the future.
	sequence_last_data = np.array(data_df[feature_columns].tail(steps_to_predict))
	result['last_sequence'] = sequence_last_data

	# drop NaNs that result from the shift operation.
	data_df.dropna(inplace=True)

	# This will contain sequences of data and their corresponding target values; Sequences + to predicts
	data_in_sequence = []
	# This initialises a double ended queue with a maximum length of seq_train_length. deque is a list-like data structure that removes the oldest value when it reaches the specified length.
	entry_sequences = deque(maxlen=seq_train_length)

	# get the feature columns and date data from the dataframe and save to a numpy array
	feature_and_date_data = data_df[feature_columns + ['date']].values

	# get the supposed future values from the dataframe and save to a numpy array
	future_values = data_df[future_columns].values

	for index in range(len(feature_and_date_data)):
		# Entry contains the feature and date data, target contains the future values that we aim to predict. We call it entry cuz it holds info about a specific time point.
		"""
		After processing 1st row:
		sequences contains: [[100, 1000, '2024-01-01']]
		
		After processing 2nd row:
		sequences contains: [[100, 1000, '2024-01-01'], [102, 1100, '2024-01-02']]
	
		After processing 3rd row:
		sequences contains: [[100, 1000, '2024-01-01'], [102, 1100, '2024-01-02'], [105, 1200, '2024-01-03']]
		
		Now the deque is full (length = 3).
		After processing 4th row:
		sequences contains: [[102, 1100, '2024-01-02'], [105, 1200, '2024-01-03'], [107, 1150, '2024-01-04']]
		
		The oldest entry ([100, 1000, '2024-01-01']) is removed, and the new entry is added.
		After processing 5th row:
		sequences contains: [[105, 1200, '2024-01-03'], [107, 1150, '2024-01-04'], [110, 1300, '2024-01-05']]
		Again, the oldest entry ([102, 1100, '2024-01-02']) is removed, and the new entry is added.
		"""
		entry = feature_and_date_data[index]
		target = future_values[index]
		entry_sequences.append(entry)
		if len(entry_sequences) == seq_train_length:
			# target_date = data_df['date'].values[index + steps_to_predict - 1]
			# Adjust the index to get the correct future date
			data_in_sequence.append([np.array(entry_sequences), target])

	# for entry, target in zip(data_df[feature_columns + ['date']].values, data_df['future'].values):
	#     sequences.append(entry)
	#     if len(sequences) == seq_train_length:
	#         data_in_sequence.append([np.array(sequences), target])
	# sequence_last_data = list([s[:len(feature_columns)] for s in sequences]) + list(sequence_last_data)

	x = []
	y = []
	# data in sequence is a list of lists, each list contains a sequence of data, entries and the target value.
	# we separate the entries and target values into x and y respectively.
	for entry_sequence, target in data_in_sequence:
		x.append(entry_sequence)
		y.append(target)

	# Convert to numpy arrays
	# x is input data, y is the target data
	x = np.array(x)
	y = np.array(y)
	print(f"y shape is: ", y.shape)
	# now we split the data into training and testing sets by date if the user has selected the option.
	if split_by_date:
		print("Splitting by date")
		train_samples = int((1 - test_size) * len(x))
		# print(f"result is {result}")
		# slices x from the beginning to the train_samples index and saves to X_train and same for y_train.
		# print("x[:train_samples] is ", x[:train_samples])
		result['X_train'] = x[:train_samples]
		result['y_train'] = y[:train_samples]
		# slices x from the train_samples index to the end and saves to X_test and same for y_test.
		result['X_test'] = x[train_samples:]
		result['y_test'] = y[train_samples:]
	else:
		# Split the dataset randomly using train_test_split from scikit-learn.
		print("calling train_test_split")
		result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
			x, y, test_size=test_size)

		# Assuming the dates_X_test are in the last column of X_test
		dates_X_test = result["X_test"][:, -1, -1]

		# Debug prints
		# print(f"Keys in result: {result.keys()}")
		# print(f"Dates extracted from X_test: {dates_X_test}")

		# Retrieve test features from the original dataframe using the extracted dates_X_test
		"""
		result["data_df"] contains the original data.
		result["data_df"]['date'] extract the  date column in the original data.
		.loc() is used to select rows in data_df where the condition inside prackets is true. In this case, the condition is that the date column is in the dates_X_test list.
		
		"""
		result["test_df"] = result["data_df"].loc[result["data_df"]['date'].isin(dates_X_test)]

		"""
		In short the above line get the data from the original data frame that corresponds to the dates in the test set.
		"""

		# Remove duplicated dates in the testing dataframe
		result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

		# slice the result list to get only the feature columns and convert to float32 and stored them as X_train and X_test
		# print("Original way:")
		result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
		# print(result["X_train"])

		# print("New way:")
		result["X_train"] = result["X_train"].astype(np.float32)
		# print(result["X_train"])

		result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

		# Debug print the result dictionary after processing
		# print(f"Result after processing: {result}")

	return [data_df, result]


def save_data_to_csv(data_df, ticker, start_date, end_date):
	#    Check if the folder 'data' exists, if not, create it
	print("save data to file fxn is being called")
	if not os.path.exists('data'):
		os.makedirs('data')
		# Save the DataFrame to a CSV file, overwriting if it exists
		print("Data folder does not exist, creating it now....")
	filename = ticker + '_' + start_date + '_' + end_date
	data_df.to_csv(f"data/{filename}.csv", index=False)

# def load_or_download(ticker, start_date, end_date):
#     filename = ticker + '_' + start_date + '_' + end_date
#     # if (os.path.exists(f"data/{filename}.csv")):
#     #     print("Data file already exists, loading it now....")
#     # Fix loading data from system should work with plots as well.
#     #     data_df = pd.read_csv(f"data/{filename}.csv")
#     #     data_df = data_df
#     #     return data_df
#     # else:
#     print("Data file does not exist, downloading it now from yfinance....")
#     data_df = yf.download(ticker, start_date, end_date)
#     print(data_df.head())
#     print(data_df.tail())
#     return data_df


# get_data("AAPL", ['Open', 'High', 'Low', 'Close', 'Volume'], scale=True, save_data=True, split_by_date=True, start_date, end_date = "2017-01-16")
