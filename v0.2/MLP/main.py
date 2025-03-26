"""
stock_forecasting.py
Authors: Bao Vo and Cheong Koo
Dates: 14/07/2021 (v1); 19/07/2021 (v2); 02/07/2024 (v3)

Description:
    This script trains an LSTM model on historical stock data (using your custom get_data and create_model functions),
    then uses the model to forecast future prices. It saves model performance metrics, applies inverse scaling,
    and prints a table of predicted versus actual 'Close' prices.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from tabulate import tabulate

from tensorflow.keras.layers import LSTM
from save_metrics import save_metrics

# Custom modules
from get_data import get_data  # Your function to download/process data
from create_model import create_model  # Your function to create an LSTM model

# ------------------------- Global Parameters -------------------------
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
# Feature columns for training (make sure these match what you want to forecast)
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
PREDICTION_DAYS = 20  # Number of days to look back for each prediction input
STEPS_TO_PREDICT = 1  # How many future steps to predict; using 'future' as target
TARGET_COLUMN = 'future'

# Model parameters
NUM_LAYERS = 4
UNITS_PER_LAYER = 10
LAYER_NAME = LSTM
NUM_TIME_STEPS = PREDICTION_DAYS
NUMBER_OF_FEATURES = len(FEATURE_COLUMNS)
ACTIVATION = "tanh"
LOSS = "mean_squared_error"
OPTIMIZER = "RMSprop"
METRICS = "mean_squared_error"

# Model saving folder
MODEL_DIR = "models"


# ------------------------- Data Loading & Preparation -------------------------
def load_training_data():
	"""
	Loads and prepares training data using your custom get_data function.
	Returns:
		data_df: Raw DataFrame (with extra columns dropped)
		result_df: Dictionary containing X_train, y_train, X_test, y_test, column_scaler, and test_dates.
	"""
	d_r = get_data(
		COMPANY,
		FEATURE_COLUMNS,
		save_data=True,
		split_by_date=False,
		start_date=TRAIN_START,
		end_date=TRAIN_END,
		seq_train_length=PREDICTION_DAYS,
		steps_to_predict=STEPS_TO_PREDICT
	)
	data_df = d_r[0]
	result_df = d_r[1]
	# Drop extra columns (e.g., 'date') if present
	if 'date' in data_df.columns:
		data_df = data_df.drop(columns=['date'])
	return data_df, result_df


# ------------------------- Model Building / Loading -------------------------
def get_model_path():
	"""
	Constructs a unique model name and path based on the parameters.
	"""
	model_name = (f"{LAYER_NAME.__name__}_layers{NUM_LAYERS}_units{UNITS_PER_LAYER}_steps{NUM_TIME_STEPS}_"
	              f"features{NUMBER_OF_FEATURES}_activation{ACTIVATION}_loss{LOSS}_optimizer{OPTIMIZER}_"
	              f"metrics{METRICS}_train{TRAIN_END}_to_{TRAIN_START}_predict{STEPS_TO_PREDICT}")
	model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
	return model_path, model_name


def build_or_load_model(X_train, y_train):
	"""
	Loads an existing model if available, or creates and trains a new one.
	Returns:
		model: Trained Keras model.
	"""
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	model_path, model_name = get_model_path()

	# if os.path.exists(model_path):
	# 	print(f"Loading existing model from {model_path}")
	# 	model = tf.keras.models.load_model(model_path)
	# else:
	print("Model does not exist, creating a new one.")
	model = create_model(
		num_layers=NUM_LAYERS,
		units_per_layer=UNITS_PER_LAYER,
		layer_name=LAYER_NAME,
		num_time_steps=NUM_TIME_STEPS,
		number_of_features=NUMBER_OF_FEATURES,
		activation=ACTIVATION,
		loss=LOSS,
		optimizer=OPTIMIZER,
		metrics=METRICS,
		steps_to_predict=STEPS_TO_PREDICT
	)
	model.fit(X_train, y_train, epochs=25, batch_size=30)
	model.save(model_path)
	return model, model_name


# ------------------------- Evaluation & Forecasting -------------------------
def evaluate_model(model, X_train, y_train, x_test, y_test, model_name):
	"""
	Saves model performance metrics using your custom save_metrics function.
	"""
	print("Saving metrics...")
	save_metrics(model, X_train, y_train, x_test, y_test, model_name)


def forecast_and_inverse_transform(model, x_data, y_data, column_scaler):
	"""
	Predicts future prices and applies inverse scaling to obtain actual values.
	Returns:
		predicted_close_prices: np.array of inverse-transformed predictions.
		actual_close_prices: np.array of inverse-transformed actual values.
	"""
	predicted_prices = model.predict(x_data)

	predicted_close_prices = []
	actual_close_prices = []
	for i in range(STEPS_TO_PREDICT):
		# Process predictions
		preds = predicted_prices[:, i].reshape(-1, 1)
		preds_inv = column_scaler['Close'].inverse_transform(preds)
		predicted_close_prices.append(preds_inv)

		# Process actual values
		actual = y_data[:, i].reshape(-1, 1)
		actual_inv = column_scaler['Close'].inverse_transform(actual)
		actual_close_prices.append(actual_inv)

	return np.array(predicted_close_prices), np.array(actual_close_prices)


def compute_differences(predicted, actual):
	"""
	Computes absolute and percentage differences between predictions and actual values.
	Returns:
		differences, avg_difference, avg_percentage_difference
	"""
	differences = np.abs(predicted - actual)
	avg_difference = np.mean(differences)
	percentage_diff = (differences / actual) * 100
	avg_percentage_difference = np.mean(percentage_diff)
	return differences, avg_difference, avg_percentage_difference


def print_results_table(predicted, actual, differences, test_dates):
	"""
	Prints a formatted table showing step number, date, predicted, actual, and difference.
	"""
	table_data = []
	steps = predicted.shape[0]
	# Loop over each prediction step and sample
	for step in range(steps):
		# Here, predicted[step] and actual[step] are arrays of shape (num_samples, 1)
		for j in range(predicted.shape[1]):
			predicted_val = round(float(predicted[step][j][0]), 3)
			actual_val = round(float(actual[step][j][0]), 3)
			diff_val = round(float(differences[step][j][0]), 3)
			# Align test date with prediction step (add step days to test date)
			prediction_date = pd.to_datetime(test_dates[j]) + pd.Timedelta(days=step)
			prediction_date_str = prediction_date.strftime('%Y-%m-%d')
			table_data.append([step + 1, f"Sample {j + 1}", prediction_date_str, predicted_val, actual_val, diff_val])
	headers = ["Step", "Sample", "Date", "Predicted 'Close' Price", "Actual 'Close' Price", "Difference"]
	print(tabulate(table_data, headers, tablefmt="pretty"))


# ------------------------- Real Data Forecasting -------------------------
def forecast_on_real_data(saved_model, column_scaler):
	"""
	Loads new (or same) data to simulate forecasting on real data.
	Returns:
		data_df_real, result_df_real from get_data, plus predictions.
	"""
	# For demonstration, here we reuse the training date range. In practice, update these dates.
	d_r_new = get_data(
		COMPANY,
		FEATURE_COLUMNS,
		save_data=True,
		split_by_date=False,
		start_date=TRAIN_START,
		end_date=TRAIN_END,
		seq_train_length=PREDICTION_DAYS,
		steps_to_predict=STEPS_TO_PREDICT
	)
	data_df_real = d_r_new[0]
	result_df_real = d_r_new[1]

	real_test = result_df_real['X_test']
	real_test_y = result_df_real['y_test']

	predicted_prices_real = saved_model.predict(real_test)

	predicted_close_prices_real = []
	actual_close_prices_real = []
	for i in range(STEPS_TO_PREDICT):
		preds_real = predicted_prices_real[:, i].reshape(-1, 1)
		preds_inv_real = column_scaler['Close'].inverse_transform(preds_real)
		predicted_close_prices_real.append(preds_inv_real)

		actual_real = real_test_y[:, i].reshape(-1, 1)
		actual_inv_real = column_scaler['Close'].inverse_transform(actual_real)
		actual_close_prices_real.append(actual_inv_real)

	return (data_df_real, result_df_real,
	        np.array(predicted_close_prices_real), np.array(actual_close_prices_real))


# ------------------------- Main Function -------------------------
def main():
	# Load training data
	data_df, result_df = load_training_data()
	# Extract training/test splits and scaler info
	X_train = result_df['X_train']
	y_train = result_df['y_train']
	x_test = result_df['X_test']
	y_test = result_df['y_test']
	column_scaler = result_df['column_scaler']
	test_dates = result_df.get("test_dates", None)  # Optional: may be present if split_by_date

	print(f"X_train shape: {X_train.shape}")
	print(f"y_train shape: {y_train.shape}")
	print(f"y_test shape: {y_test.shape}")

	# Build or load the model
	model, model_name = build_or_load_model(X_train, y_train)
	# Evaluate and save model metrics
	evaluate_model(model, X_train, y_train, x_test, y_test, model_name)

	# (Re)load the saved model to simulate a real-world scenario
	# model_path, _ = get_model_path()
	# saved_model = tf.keras.models.load_model(model_path)
	#
	# # Forecast on test data and apply inverse transformation
	# predicted_close, actual_close = forecast_and_inverse_transform(saved_model, x_test, y_test, column_scaler)
	# differences, avg_diff, avg_pct_diff = compute_differences(predicted_close, actual_close)
	#
	# print(f"\nTotal Average Difference: {avg_diff:.2f}")
	# print(f"Total Average Percentage Difference: {avg_pct_diff:.2f}%\n")

	# Print results table if test dates are available
	# if test_dates is not None:
	# 	print_results_table(predicted_close, actual_close, differences, test_dates)
	# else:
	# 	print("Test dates not provided in result_df.")

	# --------------------- Forecasting on 'Real' Data ---------------------
	# (data_df_real, result_df_real,
	#  predicted_close_real, actual_close_real) = forecast_on_real_data(saved_model, column_scaler)
	#
	# print("\nReal Data Forecasting:")
	# print(f"Real test y shape: {result_df_real['y_test'].shape}")
	# print(f"Number of real predictions: {len(predicted_close_real)}")
	# print(f"Predicted prices real shape: {predicted_close_real.shape}")
	#
	# # (Optional) Compute and print differences for real data forecast
	# differences_real, avg_diff_real, avg_pct_diff_real = compute_differences(predicted_close_real, actual_close_real)
	# print(f"\nReal Data - Average Difference: {avg_diff_real:.2f}")
	# print(f"Real Data - Average Percentage Difference: {avg_pct_diff_real:.2f}%")


# (Optional) Further visualizations and ensemble code can be called here.


if __name__ == "__main__":
	main()
