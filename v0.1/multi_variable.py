import numpy as np
import tensorflow as tf

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
# Load the stock data using your custom `get_data` function
d_r = get_data(
    COMPANY,
    feature_columns,
    save_data=True,
    split_by_date=False,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    seq_train_length= PREDICTION_DAYS,
    steps_to_predict=5
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
activation = "tanh"
loss = "mean_squared_error"
optimizer = "RMSprop"
metrics = "mean_squared_error"

import os
# Create and train the model using the defined variables
model_name = f"{layer_name.__name__}_layers{num_layers}_units{units_per_layer}_steps{num_time_steps}_features{number_of_features}_activation{activation}_loss{loss}_optimizer{optimizer}_metrics{metrics}_train{TRAIN_END}_to_{TRAIN_START}"
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
        metrics=metrics
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

# Apply inverse transformation to the predicted 'Close' prices
predicted_close_prices = column_scaler['Close'].inverse_transform(predicted_prices)

# Extract the actual 'Close' prices from y_test (these are the actual values to compare with)
actual_close_prices = y_test.reshape(-1, 1)  # Reshape if needed to match dimensions

# Apply inverse transformation to get the actual 'Close' prices in their original scale
actual_close_prices = column_scaler['Close'].inverse_transform(actual_close_prices)

# Print the predicted and actual 'Close' prices
print(f"Predicted 'Close' prices: {predicted_close_prices}")
print(f"Actual 'Close' prices: {actual_close_prices}")

# Optionally, you can print or plot the comparison
predicted_close_prices = np.array(predicted_close_prices)
actual_close_prices = np.array(actual_close_prices)

# Calculate the absolute differences
differences = np.abs(predicted_close_prices - actual_close_prices)

# Calculate the average difference
average_difference = np.mean(differences)

# Print the predicted, actual, and difference values along with the average
for i in range(len(predicted_close_prices)):
    print(f"Predicted: {predicted_close_prices[i]}, Actual: {actual_close_prices[i]}, Difference: {differences[i]}")

# Print the average difference
print(f"\nAverage Difference: {average_difference}")
percentage_difference = (differences / actual_close_prices) * 100
average_percentage_difference = np.mean(percentage_difference)
print(f"Average Percentage Difference: {average_percentage_difference}%")

# -------------------------------------------------------------------------------
# Predict the next day's `Close` price based on the last `PREDICTION_DAYS`
# -------------------------------------------------------------------------------
# real_data = model_inputs[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, len(feature_columns))
# next_day_prediction = saved_model.predict(real_data)
#
# next_day_prediction = next_day_prediction.reshape(-1, 1)
# print(f"Scaler min: {column_scaler['Close'].data_min_}")
# print(f"Scaler max: {column_scaler['Close'].data_max_}")
# print(f"Predicted next day 'Close' price before inverse transform: {next_day_prediction}")
# next_day_price = column_scaler['Close'].inverse_transform(next_day_prediction)
#
# print(f"Predicted next day 'Close' price: {next_day_price}")
#
# # print(f"Prediction: {next_day_prediction}")
# actual_prices = test_data["Close"].values
#
# print(f"Actual: {actual_prices[-1]}")
ticker = COMPANY
start_date = TRAIN_START
end_date = TRAIN_END


# Convert back to string in 'YYYY-MM-DD' format
start_date = '2024-01-19'
end_date = '2024-09-19'

print(f"Start date one day after TRAIN_END: {start_date}")
d_r = get_data(ticker, feature_columns, save_data=False, split_by_date=False, start_date=start_date, end_date=end_date, seq_train_length=PREDICTION_DAYS,test_size=0.5)
real_data = d_r[1]
data_df = d_r[0]
print(data_df.head())
print(data_df.tail())
# real_data = real_data.drop(columns=['date', 'future'])

# print(f"Real data: {real_data}")
real_test = real_data['X_test']
real_test_y = real_data['y_test']

predicted_prices_real = saved_model.predict(real_test)
print(len(predicted_prices_real))
print(f"Predicted prices real shape: {predicted_prices_real.shape}")
# Apply inverse transformation to the predicted 'Close' prices
predicted_close_prices_real = column_scaler['Close'].inverse_transform(predicted_prices_real)

# Extract the actual 'Close' prices from y_test (these are the actual values to compare with)
actual_close_prices_real = real_test_y.reshape(-1, 1)  # Reshape if needed to match dimensions

# Apply inverse transformation to get the actual 'Close' prices in their original scale
actual_close_prices_real = column_scaler['Close'].inverse_transform(actual_close_prices_real)

# Print the predicted and actual 'Close' prices
# print(f"Predicted real 'Close' prices: {predicted_close_prices_real}")
# print(f"Actual real 'Close' prices: {actual_close_prices_real}")

# Optionally, you can print or plot the comparison
predicted_close_prices_real = np.array(predicted_close_prices_real)
actual_close_prices_real = np.array(actual_close_prices_real)

print()

# Calculate the absolute differences
differences_real = np.abs(predicted_close_prices_real - actual_close_prices_real)

# Calculate the average difference
average_difference = np.mean(differences_real)

# Print the predicted, actual, and difference values along with the average
for i in range(len(predicted_close_prices_real)):
    print(f"Predicted: {predicted_close_prices_real[i]}, Actual: {actual_close_prices_real[i]}, Difference: {differences_real[i]}")

# Print the average difference
print(f"\nAverage Difference: {float(average_difference): .2f}")
percentage_difference = (differences_real / actual_close_prices_real) * 100
average_percentage_difference = np.mean(percentage_difference)
print(f"Average Percentage Difference: {float(average_percentage_difference):.2f}%")



# from simulate_trades import simulate_trades
# final_balance, profit_or_loss = simulate_trades(predicted_close_prices_real, actual_close_prices_real)
#
# print(f"Final Balance: {final_balance}")
# print(f"Profit or Loss: {profit_or_loss}")
