from collections import deque
from sklearn.model_selection import train_test_split
import os
import yfinance as yf
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Default dates for convenience.
default_end_date = dt.datetime.now().strftime('%Y-%m-%d')
default_start_date = (dt.datetime.now() - dt.timedelta(days=5 * 365)).strftime('%Y-%m-%d')


def get_data_kan(ticker, feature_columns, start_date, end_date, seq_train_length, steps_to_predict,
             scale=True, test_size=0.2, save_data=False, split_by_date=False):
	print("I am read_data")

	# Load data (or download if not saved)
	data_df = load_or_download(ticker, start_date, end_date)
	print(f"data_df.head(): {data_df.head()}")
	data_df = data_df.reset_index()

	# Standardize the date column name.
	if 'index' in data_df.columns:
		data_df.rename(columns={'index': 'Date'}, inplace=True)
	elif 'date' in data_df.columns:
		data_df.rename(columns={'date': 'Date'}, inplace=True)
	elif 'Datetime' in data_df.columns:
		data_df.rename(columns={'Datetime': 'Date'}, inplace=True)

	# Store a copy of the original dataframe.
	result = {'data_df': data_df.copy()}

	# Scale the data if required.
	if scale:
		column_scaler = {}
		for column in feature_columns:
			scaler = MinMaxScaler()
			# print(f"Scaling column: {column}")
			vals = data_df[column].values
			# vals = vals[]
			# print(f"vals[1::] {vals[1::]}")
			# Only expand dims if necessary.
			if vals.ndim == 1:
				vals = np.expand_dims(vals, axis=1)
			elif vals.ndim > 2:
				vals = np.squeeze(vals)
				if vals.ndim == 1:
					vals = np.expand_dims(vals, axis=1)
			data_df[column] = scaler.fit_transform(vals)
			column_scaler[column] = scaler
		result['column_scaler'] = column_scaler

	# Save data to CSV if requested.
	if save_data:
		save_data_to_csv(data_df, ticker, start_date, end_date)

	# Create shifted target columns.
	data_df['future'] = data_df['Close'].shift(-steps_to_predict)
	future_columns = []
	for i in range(1, steps_to_predict + 1):
		future_col_name = f'future_{i}'
		data_df[future_col_name] = data_df['Close'].shift(-i)
		future_columns.append(future_col_name)

	# Drop rows with NaNs created by shifting.
	data_df.dropna(inplace=True)

	# Save the last sequence for later use.
	sequence_last_data = np.array(data_df[feature_columns].tail(steps_to_predict))
	result['last_sequence'] = sequence_last_data

	# Prepare sequences.
	data_in_sequence = []
	entry_sequences = deque(maxlen=seq_train_length)
	feature_and_date_data = data_df[feature_columns + ['Date']].values
	future_values = data_df[future_columns].values

	for index in range(len(feature_and_date_data)):
		entry = feature_and_date_data[index]
		target = future_values[index]
		entry_sequences.append(entry)
		if len(entry_sequences) == seq_train_length:
			data_in_sequence.append([np.array(entry_sequences), target])

	# Separate sequences and targets.
	x, y = [], []
	for entry_sequence, target in data_in_sequence:
		x.append(entry_sequence)
		y.append(target)
	x = np.array(x)
	y = np.array(y)
	print(f"y shape is: {y.shape}")

	# Split the data.
	if split_by_date:
		print("Splitting by date")
		train_samples = int((1 - test_size) * len(x))
		result['X_train'] = x[:train_samples]
		result['y_train'] = y[:train_samples]
		result['X_test'] = x[train_samples:]
		result['y_test'] = y[train_samples:]
	else:
		print("Calling train_test_split")
		from sklearn.model_selection import train_test_split
		result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
			x, y, test_size=test_size)
		# Optional: store test dates if available.
		sequence_dates = []
		for index in range(len(feature_and_date_data)):
			if index + steps_to_predict - 1 < len(data_df['Date'].values):
				date = data_df['Date'].values[index + steps_to_predict - 1]
				sequence_dates.append(date)
			else:
				break
		result["test_dates"] = sequence_dates[len(sequence_dates) - len(result["X_test"]):]

		# Ensure feature columns are of correct type.
		result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
		result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

	return [data_df, result]


def save_data_to_csv(data_df, ticker, start_date, end_date):
	print("save data to file fxn is being called")
	if not os.path.exists('data'):
		os.makedirs('data')
		print("Data folder does not exist, creating it now....")
	filename = f"{ticker}_{start_date}_{end_date}"
	data_df.to_csv(f"data/{filename}.csv", index=False)


def load_or_download(ticker, start_date, end_date):
	filename = f"{ticker}_{start_date}_{end_date}"
	if os.path.exists(f"data/{filename}.csv"):
		print("Data file already exists, loading it now....")
		data_df = pd.read_csv(f"data/{filename}.csv")
		if data_df.iloc[0]['Close'] == ticker:
			print("First row contains invalid data, dropping it...")
			data_df = data_df.iloc[1:]
		return data_df
	else:
		print("Data file does not exist, downloading it now from yfinance....")
		data_df = yf.download(ticker, start_date, end_date)
		if data_df.iloc[0]['Close'] == ticker:
			print("First row contains invalid data, dropping it...")
			data_df = data_df.iloc[1:]
		return data_df
