import numpy as np
import tensorflow as tf
from Tools.demo.sortvisu import steps
from save_metrics import save_metrics

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
STEPS_TO_PREDICT = 10

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
num_layers = 2
units_per_layer = 10
layer_name = LSTM
num_time_steps = PREDICTION_DAYS
number_of_features = len(feature_columns)
activation = "tanh"
loss = "mean_squared_error"
optimizer = "RMSprop"
metrics = "mean_squared_error"
# STEPS_TO_PREDICT = 1

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


model_name = f"{layer_name.__name__}_layers{num_layers}_units{units_per_layer}_steps{num_time_steps}_features{number_of_features}_activation{activation}_loss{loss}_optimizer{optimizer}_metrics{metrics}_train{TRAIN_END}_to_{TRAIN_START}_predict{STEPS_TO_PREDICT}"

# Evaluate the model and save errors to file
print("Before save metrics")
save_metrics(model, X_train, y_train, x_test, y_test, model_name)
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
# steps_to_predict = STEPS_TO_PREDICT
for i in range(steps_to_predict):
    print(f"i is {i} in line 132")
    actual = y_test[:, i].reshape(-1, 1)
    actual_inv = column_scaler['Close'].inverse_transform(actual)
    actual_close_prices.append(actual_inv)

actual_close_prices = np.array(actual_close_prices)


predicted_close_prices = np.array(predicted_close_prices)
actual_close_prices = np.array(actual_close_prices)

# Calculate the absolute differences
differences_actual = np.abs(predicted_close_prices - actual_close_prices)

# Calculate the average difference
total_average_difference_actual = np.mean(differences_actual)

percentage_difference_actual = (differences_actual / actual_close_prices) * 100
total_average_percentage_difference_actual = np.mean(percentage_difference_actual)
# print(f"Average Percentage Difference actual: {total_average_percentage_difference_actual}%")


from tabulate import tabulate
# Print the predicted, actual, and difference values in a formatted table
test_dates = result_df["test_dates"]


table_data = []
import pandas as pd
# Iterate through each prediction step and store the results in a list
for i in range(len(predicted_close_prices[:steps_to_predict])):
    for j in range(len(predicted_close_prices[i][:steps_to_predict])):
        predicted = round(float(predicted_close_prices[i][j][0]), 3)
        actual = round(float(actual_close_prices[i][j][0]), 3)
        difference = round(float(differences_actual[i][j][0]), 3)
        prediction_date =  pd.to_datetime(test_dates[j]) + pd.Timedelta(days=i)
        prediction_date = prediction_date.strftime('%Y-%m-%d')
        # Get the corresponding date from the test data

        table_data.append([i + 1, f'Sample {j + 1}', prediction_date, predicted, actual, difference])

# Define headers for the table
headers = ["Step", "Date","Predicted 'Close' Price", "Actual 'Close' Price", "Difference"]

# Calculate the absolute differences and percentage differences for each time step

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

# Calculate the averag
#
# e difference






# from simulate_trades import simulate_trades
# final_balance, profit_or_loss = simulate_trades(predicted_close_prices_real, actual_close_prices_real)
#
# print(f"Final Balance: {final_balance}")
# print(f"Profit or Loss: {profit_or_loss}")


for step in range(STEPS_TO_PREDICT):
    differences_step = np.abs(predicted_close_prices[step] - actual_close_prices[step])  # Absolute difference
    percentage_differences_step = (differences_step / actual_close_prices[step]) * 100  # Percentage difference

    # Calculate the averages
    average_difference = np.mean(differences_step)
    average_percentage_difference = np.mean(percentage_differences_step)

    # Print the results for each time step
    print(f"actual Time Step {step + 1}:")
    print(f" actual Average Difference: {average_difference:.2f}")
    print(f"actual Average Percentage Difference: {average_percentage_difference:.2f}%")
    print("-" * 50)




for step in range(STEPS_TO_PREDICT):
    differences_step_real = np.abs(predicted_close_prices_real[step] - actual_close_prices_real[step])  # Absolute difference
    percentage_differences_step_real = (differences_step_real / actual_close_prices_real[step]) * 100  # Percentage difference

    # Calculate the averages
    average_difference_real = np.mean(differences_step_real)
    average_percentage_difference_real = np.mean(percentage_differences_step_real)

    # Print the results for each time step
    print(f"real Time Step {step + 1}:")
    print(f"real Average Difference: {average_difference_real:.2f}")
    print(f"real   Average Percentage Difference: {average_percentage_difference_real:.2f}%")
    print("-" * 50)



total_average_difference_real = np.mean(differences_real)



total_percentage_difference_real = (differences_real / actual_close_prices_real) * 100

total_average_percentage_difference_real = np.mean(total_percentage_difference_real)




print(f"\n total Average Difference actual: {float(total_average_difference_actual):.2f}")
print(f" total Average Percentage Difference actual: {float(total_average_percentage_difference_actual):.2f}%")

print(f"\ntotal Average Difference real: {float(total_average_difference_real): .2f}")
print(f"total Average Percentage Difference real: {float(total_average_percentage_difference_real):.2f}%")




test_dates_real = result_df_real["test_dates"]

print("real table made here")
table_data_real = []
import pandas as pd
# Iterate through each prediction step and store the results in a list
for i in range(len(predicted_close_prices_real)):
    for j in range(len(predicted_close_prices_real[i])):
        predicted_real = round(float(predicted_close_prices_real[i][j][0]), 3)
        actual_real = round(float(actual_close_prices_real[i][j][0]), 3)
        difference_real = round(float(differences_real[i][j][0]), 3)
        prediction_date_real =  pd.to_datetime(test_dates_real[j]) + pd.Timedelta(days=i)
        prediction_date_real = prediction_date_real.strftime('%Y-%m-%d')
        # Get the corresponding date from the test data

        table_data_real.append([i + 1, f'Sample {j + 1}', prediction_date_real, predicted_real, actual_real, difference_real])

# Define headers for the table
headers = ["Step_real", "Date_real","Predicted_real 'Close_real' Price_real", "Actual_real 'Close_real' Price_real", "Difference_real"]


differences = np.abs(predicted_close_prices - actual_close_prices)

# Calculate average differences for each step and flatten to 1D
average_differences = np.mean(differences, axis=1).flatten()

# Calculate percentage differences for each step
percentage_differences = (differences / actual_close_prices) * 100

# Calculate average percentage differences for each step and flatten to 1D
average_percentage_differences = np.mean(percentage_differences, axis=1).flatten()

# Convert to Python lists (if needed)
average_differences = average_differences.tolist()
average_percentage_differences = average_percentage_differences.tolist()


from visualisations import (
    plot_predicted_vs_actual, plot_average_differences,
    display_prediction_table, plot_heatmap, display_summary_table
)


# Call the functions as needed
# plot_predicted_vs_actual(predicted_close_prices, actual_close_prices, STEPS_TO_PREDICT)
# plot_average_differences(average_differences, STEPS_TO_PREDICT, diff_type="absolute")
# plot_average_differences(average_percentage_differences, STEPS_TO_PREDICT, diff_type="percentage")
# display_prediction_table(predicted_close_prices, actual_close_prices, differences_actual, test_dates, STEPS_TO_PREDICT)
# plot_heatmap(differences_actual, STEPS_TO_PREDICT)
# display_summary_table(total_average_difference_actual, total_average_percentage_difference_actual)







# for now, let's make the ensemble here:
# from forecasting import random_forest, sarima_forecast, exponential_smoothing_forecast, return_final_forecast
'''
final_forecast = return_final_forecast()
# print(final_forecast)

# Assuming final_forecast already has a Date index or forecast dates
forecast_index = final_forecast.index  # Extract index (forecast dates)
predicted_close_prices = np.array(predicted_close_prices).reshape(-1)[:PREDICTION_DAYS]
actual_close_prices = np.array(actual_close_prices).reshape(-1)[:PREDICTION_DAYS]

# Ensure LSTM predicted values are a DataFrame with the same index
lstm_predictions_df = pd.DataFrame(
    np.array(predicted_close_prices).reshape(-1),  # Flatten the predictions
    index=forecast_index,  # Align with the forecast dates
    columns=["lstm_forecast"]  # Name of the new column for LSTM predictions
)
actual_values_df = pd.DataFrame(
    np.array(actual_close_prices).reshape(-1),  # Flatten the predictions
    index=forecast_index,  # Align with the forecast dates
    columns=["actual_values"]  # Name of the new column for LSTM predictions
)

# Concatenate the LSTM predictions with the final forecast
final_final_forecast = pd.concat([final_forecast, lstm_predictions_df, actual_values_df], axis=1)
# Print the updated forecast DataFrame with LSTM predictions included
pd.set_option('display.max_columns', None)
print(final_final_forecast)


import matplotlib.pyplot as plt

# Assuming `actual_values` is a DataFrame with the same index as `final_final_forecast`
# and contains the actual values for the forecast dates
# actual_values = actual
# actual_values = pd.DataFrame(actual_close_prices_real.reshape(-1), index=forecast_index, columns=["Actual Values"])

# Concatenate the actual values with the final forecast
# final_final_forecast_with_actuals = pd.concat([final_final_forecast, actual_values], axis=1)

# Calculate the ensemble value
ensemble_value = final_final_forecast.mean(axis=1)

# Plot the final_final_forecast DataFrame with actual values
plt.figure(figsize=(14, 7))
for column in final_final_forecast.columns:
    plt.plot(final_final_forecast.index, final_final_forecast[column], label=column)

# Plot the ensemble value
plt.plot(final_final_forecast.index, ensemble_value, label='Ensemble Value', color='black', linestyle='--', linewidth=2)

# Add title and labels
plt.title('Final Forecast with Ensemble and Actual Values')
plt.xlabel('Date')
plt.ylabel('Forecast Values')

# Add legend
plt.legend()

# Show the plot
plt.show()
'''