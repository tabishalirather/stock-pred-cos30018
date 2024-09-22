import numpy as np
import tensorflow as tf
from Tools.demo.sortvisu import steps

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from get_data import get_data  # Your custom function to get data
from create_model import create_model  # Your custom function to create the LSTM model
import yfinance as yf

# -------------------------------------------------------------------------------
# Load Data (Using your custom get_data function)
# -------------------------------------------------------------------------------
COMPANY = 'AMZN'  # Stock symbol
TRAIN_START = '2020-01-01'  # Training start date
TRAIN_END = '2023-08-01'  # Training end date
# // no need to specify test date, as that data is split by percentage.
# Feature columns
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PREDICTION_DAYS = 12
# target_column = 'Close'  # We are predicting the 'Close' price
# for mutlistep prediciton, we are predicting the future price num_steps_ahead days ahead
target_column = 'future'
STEPS_TO_PREDICT = 5

# Load the stock data using your custom `get_data` function
d_r = get_data(
	COMPANY,
	feature_columns,
	save_data=True,
	split_by_date=False,
	start_date=TRAIN_START,
	end_date=TRAIN_END,
	seq_train_length=PREDICTION_DAYS,
	steps_to_predict=STEPS_TO_PREDICT
)
data_df = d_r[0]
# print("data_df.head():", data_df.head())
result_df = d_r[1]
data_df = data_df.drop(columns=['date'])

# print("data_df.head():", data_df.head())


# # -------------------------------------------------------------------------------
# PREDICTION_DAYS = 15  # Number of days to look back
# target_column = 'Close'  # We are predicting the 'Close' price

X_train = result_df['X_train']
y_train = result_df['y_train']
x_test = result_df['X_test']
y_test = result_df['y_test']
print(f"y_test shape: {y_test.shape}")
# print(f"y_test head: {y_test}")
column_scaler = result_df['column_scaler']

print(f"x_train shape: {X_train.shape}")  # (samples, PREDICTION_DAYS, number_of_features)
print(f"y_train shape: {y_train.shape}")  # (samples,)

# -------------------------------------------------------------------------------
# Creating and Training the Model (Using your custom `create_model` function)
# -------------------------------------------------------------------------------

# Define variables for model parameters
num_layers = 5
units_per_layer = 100
layer_name = LSTM
num_time_steps = PREDICTION_DAYS
number_of_features = len(feature_columns)
activation = "linear"
loss = "mean_squared_error"
optimizer = "RMSprop"
metrics = "mean_squared_error"
STEPS_TO_PREDICT = 5

import os

# Create and train the model using the defined variables
model_name = f"{layer_name.__name__}_layers{num_layers}_units{units_per_layer}_steps{num_time_steps}_features{number_of_features}_activation{activation}_loss{loss}_optimizer{optimizer}_metrics{metrics}_train{TRAIN_END}_to_{TRAIN_START}_predict{STEPS_TO_PREDICT}"
model_path = f"models/{model_name}.keras"

if os.path.exists(model_path):
	print(f"Loading existing model from {model_path}")
	model = tf.keras.models.load_model(model_path)

else:
	print("model DNE, creating a new one.")
	model = create_model(
		num_layers=num_layers,
		units_per_layer=units_per_layer,
		layer_name=layer_name,
		num_time_steps=num_time_steps,
		number_of_features=number_of_features,
		activation=activation,
		loss=loss,
		optimizer=optimizer,
		metrics=metrics,
		steps_to_predict=STEPS_TO_PREDICT
	)

	# print("x_train shape:", X_train)
	# Train the model
	model.fit(X_train, y_train, epochs=50, batch_size=30)

	# Save the model for future use
	model.save(model_path)

# Load the saved model
saved_model = tf.keras.models.load_model(model_path)

# Predict the Close prices
predicted_prices = saved_model.predict(x_test)



predicted_close_prices = []
steps_to_predict = STEPS_TO_PREDICT
# Loop over each time step
for i in range(steps_to_predict):
    # Extract predictions for the i-th future step
    preds = predicted_prices[:, i].reshape(-1, 1)
    # Inverse transform
    preds_inv = column_scaler['Close'].inverse_transform(preds)
    predicted_close_prices.append(preds_inv)


# Extract the actual 'Close' prices from y_test (these are the actual values to compare with)
actual_close_prices = []

for i in range(steps_to_predict):
    actual = y_test[:, i].reshape(-1, 1)
    actual_inv = column_scaler['Close'].inverse_transform(actual)
    actual_close_prices.append(actual_inv)

actual_close_prices = np.array(actual_close_prices)


predicted_close_prices = np.array(predicted_close_prices)
actual_close_prices = np.array(actual_close_prices)

# Calculate the absolute differences
differences = np.abs(predicted_close_prices - actual_close_prices)

# Calculate the average difference
average_difference_actual = np.mean(differences)

percentage_difference_actual = (differences / actual_close_prices) * 100
average_percentage_difference_actual = np.mean(percentage_difference_actual)
print(f"Average Percentage Difference actual: {average_percentage_difference_actual}%")


from tabulate import tabulate
# Print the predicted, actual, and difference values in a formatted table
dates = result_df['test_df']['date'].values

table_data = []

# Iterate through each prediction step and store the results in a list
for i in range(len(predicted_close_prices)):
    for j in range(len(predicted_close_prices[i])):
        predicted = round(float(predicted_close_prices[i][j][0]), 3)
        actual = round(float(actual_close_prices[i][j][0]), 3)
        difference = round(float(differences[i][j][0]), 3)
        prediction_date = dates[j]  # Get the corresponding date from the test data

        table_data.append([i + 1, f'Sample {j + 1}', prediction_date, predicted, actual, difference])

# Define headers for the table
headers = ["Step", "Date","Predicted 'Close' Price", "Actual 'Close' Price", "Difference"]

# Print the table using tabulate
print("Table made here")
print(tabulate(table_data, headers, tablefmt="pretty"))

ticker = COMPANY
start_date = TRAIN_START
end_date = TRAIN_END

# Convert back to string in 'YYYY-MM-DD' format
start_date = '2024-01-19'
end_date = '2024-09-19'

print(f"Start date one day after TRAIN_END: {start_date}")
d_r = get_data(ticker, feature_columns, save_data=False, split_by_date=False, start_date=start_date, end_date=end_date,
               seq_train_length=PREDICTION_DAYS, steps_to_predict = STEPS_TO_PREDICT,  test_size=0.5)
data_df_real = d_r[0]
result_df_real = d_r[1]
print(data_df_real.head())
print(data_df_real.tail())
# real_data = real_data.drop(columns=['date', 'future'])

# print(f"Real data: {real_data}")
real_test = result_df_real['X_test']
real_test_y = result_df_real['y_test']
print(f"Real test y shape: {real_test_y.shape}")
# print(f"Real test y head: {real_test_y}")

predicted_prices_real = saved_model.predict(real_test)

print(len(predicted_prices_real))
print(f"Predicted prices real shape: {predicted_prices_real.shape}")
# Apply inverse transformation to the predicted 'Close' prices


predicted_close_prices_real = []
for i in range(steps_to_predict):
	# Extract predictions for the i-th future step
	preds_real = predicted_prices_real[:, i].reshape(-1, 1)
	# Inverse transform
	preds_inv_real = column_scaler['Close'].inverse_transform(preds_real)
	predicted_close_prices_real.append(preds_inv_real)

# predicted_close_prices_real = column_scaler['Close'].inverse_transform(predicted_prices_real)

# Extract the actual 'Close' prices from y_test (these are the actual values to compare with)
actual_close_prices_real = []
for j in range(steps_to_predict):
	actual_real = real_test_y[:, j].reshape(-1, 1)
	actual_inv_real = column_scaler['Close'].inverse_transform(actual_real)
	actual_close_prices_real.append(actual_inv_real)


# Optionally, you can print or plot the comparison
predicted_close_prices_real = np.array(predicted_close_prices_real)
actual_close_prices_real = np.array(actual_close_prices_real)

# print()

# Calculate the absolute differences
differences_real = np.abs(predicted_close_prices_real - actual_close_prices_real)

# Calculate the average difference
average_difference_real = np.mean(differences_real)



percentage_difference_real = (differences_real / actual_close_prices_real) * 100
average_percentage_difference_real = np.mean(percentage_difference_real)



print(f"\nAverage Difference actual: {float(average_difference_actual):.2f}")
print(f"Average Percentage Difference actual: {float(average_percentage_difference_actual):.2f}%")

print(f"\nAverage Difference real: {float(average_difference_real): .2f}")
print(f"Average Percentage Difference real: {float(average_percentage_difference_real):.2f}%")


# from simulate_trades import simulate_trades
# final_balance, profit_or_loss = simulate_trades(predicted_close_prices_real, actual_close_prices_real)
#
# print(f"Final Balance: {final_balance}")
# print(f"Profit or Loss: {profit_or_loss}")
