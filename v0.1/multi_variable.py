import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from get_data import get_data  # Your custom function to get data
from create_model import create_model  # Your custom function to create the LSTM model



# -------------------------------------------------------------------------------
# Load Data (Using your custom get_data function)
# -------------------------------------------------------------------------------
COMPANY = 'CBA.AX'  # Stock symbol
TRAIN_START = '2020-01-01'  # Training start date
TRAIN_END = '2023-08-01'  # Training end date
# // no need to specify test date, as that data is split by percentage.

# Feature columns
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PREDICTION_DAYS = 15
target_column = 'Close'  # We are predicting the 'Close' price
# Load the stock data using your custom `get_data` function
d_r = get_data(COMPANY, feature_columns, save_data=True, split_by_date=False, start_date=TRAIN_START, end_date=TRAIN_END, seq_train_length= PREDICTION_DAYS)
data_df = d_r[0]
print("data_df.head():", data_df.head())
result_df = d_r[1]
data_df = data_df.drop(columns=['date', 'future'])

print("data_df.head():", data_df.head())


# # -------------------------------------------------------------------------------
PREDICTION_DAYS = 15  # Number of days to look back
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
model = create_model(num_layers=5, units_per_layer=70, layer_name=LSTM, num_time_steps=PREDICTION_DAYS,
                     number_of_features=len(feature_columns), activation="tanh", loss="mean_absolute_error",
                     optimizer="adam", metrics="mean_absolute_error")

print("x_train shape:", X_train)
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10)

# Save the model for future use
model.save("multi_variable_model.keras")


# Load the saved model
saved_model = tf.keras.models.load_model("multi_variable_model.keras")

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