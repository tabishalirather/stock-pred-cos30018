import pandas as pd
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from tensorflow.python.ops.distributions.exponential import Exponential

from get_data import get_data
import pmdarima as pm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=ValueWarning)

print("working")

COMPANY = 'AMZN'  # Stock symbol
TRAIN_START = '2020-01-01'  # Training start date
TRAIN_END = '2023-08-01'  # Training end date
# // no need to specify test date, as that data is split by percentage.
# Feature columns
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PREDICTION_DAYS = 12
# target_column = 'Close'  # We are predicting the 'Close' price
# for mutlistep prediciton, we are predicting the future price num_steps_ahead days ahead
target_column = 'future_1'
STEPS_TO_PREDICT = 1
# Load the stock data using your custom `get_data` function
raw_d_r = get_data(
	COMPANY,
	feature_columns,
	save_data=True,
	split_by_date=False,
	start_date=TRAIN_START,
	end_date=TRAIN_END,
	seq_train_length=PREDICTION_DAYS,
	steps_to_predict=STEPS_TO_PREDICT,
	scale=False
)
# print(raw_d_r[0].head())

raw_data = raw_d_r[0]
print(raw_data.head())
print(raw_data.columns)
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
# raw_data = raw_data.drop(['Date'], axis=1)
# Fit an ARIMA model
sarima_model = pm.auto_arima(
	raw_data[target_column],
	start_p=1,
	start_q=1,
	test='adf',
	max_p=3,
	max_q=3,
	m=1,
	d=None,
	seasonal=True,
	start_P=0,
	D=0,
	trace=True,
	error_action='ignore',
	suppress_warnings=True,
	stepwise=True
)

sarima_model.summary()

forecast_index = pd.date_range(start=raw_data.index[-1], periods=PREDICTION_DAYS + 1, freq='D')[1:]
print("forecast index")
print(forecast_index)

sarima_forecast = pd.DataFrame(sarima_model.predict(n_periods=PREDICTION_DAYS))
sarima_forecast.columns = ['sarima_forecast']
# print(raw_data.index)
# sarima_forecast_df = pd.DataFrame(sarima_forecast, index=forecast_index, columns=['sarima_forecast'])
sarima_forecast.index = forecast_index
print("before forecast head")
print(sarima_forecast.head(6))