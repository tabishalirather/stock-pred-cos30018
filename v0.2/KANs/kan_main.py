import os
import torch
import numpy as np
from kan import KAN
from save_metrics_kans import *
from get_data_kans import get_data_kan
from get_commons import *  # Assumes this sets up 'config'
# Shift the sequence window for multi-day prediction.

# Other thing, how much of sequence length is needed.


if torch.cuda.is_available():
	print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
	print("No GPU available. Training will run on CPU.")
# ------------------------- Device Setup -------------------------
def get_device():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# if torch.cuda.is_available():
	# 	print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
	# else:
	# 	print("No GPU available. Training will run on CPU.")
	return device

# ------------------------- Data Loading & Preparation -------------------------
def load_and_prepare_data():
	# Data configuration from your config file
	COMPANY = config.get("COMPANY")
	TRAIN_START = config.get("TRAIN_START")
	TRAIN_END = config.get("TRAIN_END")
	PREDICTION_DAYS = int(config.get("PREDICTION_DAYS"))  # Look-back window
	STEPS_TO_PREDICT = int(config.get("STEPS_TO_PREDICT"))
	TARGET_COLUMN = config.get("TARGET_COLUMN")
	FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
	# NUM_OF_DAYS_TO_PREDICT = int(config.get("STEPS_TO_PREDICT"))
	# Additional parameters for data preparation
	# seq_train_length = 60  # Number of time steps for each input sequence
	# num_days_to_predict = 4  # Use last few days of test set for evaluation
	print(f"COMPANY: {COMPANY}")
	print(f"TRAIN_START: {TRAIN_START}")
	print(f"TRAIN_END: {TRAIN_END}")
	print(f"PREDICTION_DAYS: {PREDICTION_DAYS}")
	print(f"STEPS_TO_PREDICT: {STEPS_TO_PREDICT}")
	print(f"TARGET_COLUMN: {TARGET_COLUMN}")
	print(f"FEATURE_COLUMNS: {FEATURE_COLUMNS}")
	scale = True
	test_size = 0.2
	save_data = False
	split_by_date = False

	# Load data using your custom get_data_kan function
	d_r = get_data_kan(
		COMPANY,
		FEATURE_COLUMNS,
		save_data=save_data,
		split_by_date=split_by_date,
		start_date=TRAIN_START,
		end_date=TRAIN_END,
		seq_train_length=PREDICTION_DAYS,
		steps_to_predict=STEPS_TO_PREDICT
	)
	data_df, result_df = d_r[0], d_r[1]

	# Extract training and test splits plus auxiliary info
	x_train = result_df['X_train']
	y_train = result_df['y_train']
	x_test = result_df['X_test']
	y_test = result_df['y_test']
	test_dates = result_df["test_dates"]
	column_scaler = result_df['column_scaler']

	# Limit predictions to the last few days in the test set
	x_test_subset = x_test[-STEPS_TO_PREDICT:]
	y_test_subset = y_test[-STEPS_TO_PREDICT:]
	test_dates_subset = test_dates[-STEPS_TO_PREDICT:]

	# Reshape data for model input (flatten each sequence)
	x_train = x_train.reshape(x_train.shape[0], -1)
	x_test_subset = x_test_subset.reshape(x_test_subset.shape[0], -1)

	# Prepare dataset tensors for PyTorch
	device = get_device()
	dataset = {
		'train_input': torch.tensor(x_train).float().to(device),
		'test_input': torch.tensor(x_test_subset).float().to(device),
		'train_label': torch.tensor(y_train).float().to(device).unsqueeze(1),
		'test_label': torch.tensor(y_test_subset).float().to(device).unsqueeze(1)
	}
	return dataset, test_dates_subset, column_scaler


# ------------------------- Model Building/Training -------------------------
def train_model(dataset, model_path="kan_model.pth"):
	device = get_device()
	train_input = dataset['train_input']
	input_size = train_input.shape[1]
	# A simple heuristic for number of neurons:
	num_neurons = len(train_input) // 50
	output_size = int(config.get("STEPS_TO_PREDICT"))  # Predicting a single value (e.g. closing price)

	# Initialize the KAN model
	model = KAN(width=[input_size, num_neurons * 3, output_size], grid=3, k=3, seed=0, device=device)

	# For now, we always train a new model; you could add logic to load an existing model
	print("Training the model...")
	summary = model.fit(dataset, opt="LBFGS", steps=10)
	torch.save(model.state_dict(), model_path)
	print(f"Model saved to {model_path}")
	return model


# ------------------------- Prediction & Evaluation
def predict_and_evaluate(model, dataset, column_scaler, test_dates):
	device = get_device()
	with torch.no_grad():
		predictions = model(dataset['test_input']).cpu().numpy()

	# Inverse-transform the predicted and actual values using the scaler for 'Close'
	predicted_prices = column_scaler['Close'].inverse_transform(predictions)
	actual_prices = column_scaler['Close'].inverse_transform(
		dataset['test_label'].cpu().numpy().reshape(-1, 1)
	)
	# Clamp any negative predictions to zero
	predicted_prices = np.maximum(predicted_prices, 0)


	print("=" * 75)

	# Save evaluation metrics
	save_metrics_kans(model, dataset['train_input'], dataset['train_label'],
	             dataset['test_input'], dataset['test_label'], "KAN_Model")


# ------------------------- Main Function -------------------------
def main():
	dataset, test_dates, column_scaler = load_and_prepare_data()
	model = train_model(dataset)
	predict_and_evaluate(model, dataset, column_scaler, test_dates)


main()
