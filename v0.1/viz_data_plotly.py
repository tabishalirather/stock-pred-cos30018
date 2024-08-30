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

# link to tutorial: https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2

ticker = 'CBA.AX'
feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
start_date = '2021-01-01'
end_date = '2021-03-30'

data_df = get_data(ticker, feature_columns, start_date, end_date, save_data=True, split_by_date=True)


def plot_candlestick(data_df):
	# data_df = data_df.set_index('Date')
	# print("Candlestick Chart Styling from MPLFinance : {}".format(fplt.available_styles()))

	trade_days_month = 20
	# data_df['SMA'] = data_df['Close'].rolling(window=trade_days_month).mean()
	# data_df['EMA'] = data_df['Close'].ewm(span=trade_days_month, adjust=False).mean()
	# print(len(data_df))
	# data_df.index = pd.to_datetime(data_df.index)
	# data_df['RSI'] = ta.momentum.RSIIndicator(data_df['Close'], window=trade_days_month).rsi()
	# print("I am RSI")
	# print(data_df['RSI'])

	data_df["SMA"] = talib.SMA(data_df.Close, timeperiod=trade_days_month)
	data_df["RSI"] = talib.RSI(data_df.Close, timeperiod=trade_days_month)
	data_df["EMA"] = talib.EMA(data_df.Close, timeperiod=trade_days_month)

	print("TA-Lib Version : {}".format(talib.__version__))
	print("Plotly Version : {}".format(plotly.__version__))

	if not os.path.exists('images'):
		os.makedirs('images')


plot_candlestick(data_df)
