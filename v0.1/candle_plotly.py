from Tools.scripts.generate_re_casefix import alpha
from matplotlib.pyplot import ylabel
# pip install Ta-Lib wasn't working. Giving error: building 'talib._ta_lib' extension
#       error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
#       [end of output]
#
#   note: This error originates from a subprocess, and is likely not a problem with pip.
#   ERROR: Failed building wheel for TA-Lib
# Failed to build TA-Lib
# ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (TA-Lib)

# So I installed TA-lib .whl from https://github.com/cgohlke/talib-build/?tab=readme-ov-file and did pip install TA_Lib‑0.4.0‑cp39‑cp39‑win_amd64.whl which worked. otherwise I'd have needed to install visual studio and I am not gonna get that peice of crap on my machine.
import talib

print("working")
from get_data import get_data
# import mplfinance as fplt
import pandas as pd
import plotly.graph_objects as go
import plotly
import os
import plotly.io as pio

# set default renderer to browser instead of plots in pycharm.
pio.renderers.default = "browser"

# link to tutorial: https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2

# set ticker, feature columns, start date and end date to be passed as arguments to get_data function
ticker = 'CBA.AX'
feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
start_date = '2021-01-01'
end_date = '2021-03-30'


# get data from get_data function
data_df = get_data(ticker, feature_columns, start_date, end_date, save_data=True, split_by_date=True)

# define function to plot candlestick chart
def plot_candlestick(data_df, num_days_aggreate=1):
	# data_df = data_df.set_index('Date')
	#
	# print versions for talib and plotly, one for calculating technical indicators and the other for plotting the chart
	print("TA-Lib Version : {}".format(talib.__version__))
	print("Plotly Version : {}".format(plotly.__version__))

	# sets number of days to calculate sma, rsi and ema for.
	trade_days_month = 10

	# resample the data to aggregate it for given number of days and choose the first, max, min and last values for open, high, low and close respectively.
	resampled_data = data_df.resample(f"{num_days_aggreate}D").agg({
		'Open': 'first',
		'High': 'max',
		'Low': 'min',
		'Close': 'last',
		'Volume': 'sum'
	}).dropna()
	resampled_data["SMA"] = talib.SMA(resampled_data.Close, timeperiod=trade_days_month)
	resampled_data["RSI"] = talib.RSI(resampled_data.Close, timeperiod=trade_days_month)
	resampled_data["EMA"] = talib.EMA(resampled_data.Close, timeperiod=trade_days_month)
	# print("I should be printed here")
	# print(f"data_df.index: {data_df.index}")

	# create candlestick chart with plotly
	candlestick = go.Candlestick(
		x=resampled_data.index,
		open=resampled_data['Open'],
		high=resampled_data['High'],
		low=resampled_data['Low'],
		close=resampled_data['Close']
	)
	# create sma and ema lines
	sma = go.Scatter(
		x=resampled_data.index,
		y=resampled_data["SMA"],
		# mode='lines',
		yaxis='y1',
		name='SMA'
	)
	ema = go.Scatter(
		x=resampled_data.index,
		y=resampled_data["EMA"],
		# mode='lines',
		yaxis='y1',
		name='EMA'
	)

	# set title for the chart
	title = f"{ticker + start_date + end_date}"
	print("Before go.Figure")

	# create figure with candlestick, sma and ema
	fig = go.Figure(data=[candlestick, sma, ema])
	# to turn of slider use , xaxis_rangeslider_visible = False
	# print("Before fig.update_layoutm, after go.Figure")

	# update layout of the figure
	fig.update_layout(width=800, height=800, title=title, xaxis_title="Date", yaxis_title="Price")
	print("After fig.update_layout")

	# show the chart
	fig.show()

	# let's implement aggregation of data for candlestick chart.
	# we will aggregate data for a number of days and plot the candlestick chart.
	# resampled_data = data_df.resample(f"{num_days_aggreate}D").agg({
	# 	'Open': 'first',
	# 	'High': 'max',
	# 	'Low': 'min',
	# 	'Close': 'last',
	# 	'Volume': 'sum'
	# }).dropna()

	# candlestick_agg = go.Candlestick(
	# 	x=resampled_data.index,
	# 	open=data_df['Open'],
	# 	high=data_df['High'],
	# 	low=data_df['Low'],
	# 	close=data_df['Close']
	# )

	# fig = go.Figure(data=[candlestick_agg])
	# fig.show()
	# fig.write_html("candlestick_chart.html")



plot_candlestick(data_df)
