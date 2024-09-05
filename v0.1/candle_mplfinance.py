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
import mplfinance as fplt
import pandas as pd
import os


# sets ticker, feature columns, start date and end date to be passed as arguments to get_data function
ticker = 'CBA.AX'
feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
start_date = '2021-01-01'
end_date = '2021-03-30'


# get data from get_data function
data_df = get_data(ticker, feature_columns, start_date, end_date, save_data=True, split_by_date=True)

# define function to plot candlestick chart
def plot_candlestick(data_df):
	# data_df = data_df.set_index('Date')
	print("Candlestick Chart Styling from MPLFinance : {}".format(fplt.available_styles()))

	trade_days_month = 20
	# data_df['SMA'] = data_df['Close'].rolling(window=trade_days_month).mean()
	# data_df['EMA'] = data_df['Close'].ewm(span=trade_days_month, adjust=False).mean()
	# print(len(data_df))
	# data_df.index = pd.to_datetime(data_df.index)
	# data_df['RSI'] = ta.momentum.RSIIndicator(data_df['Close'], window=trade_days_month).rsi()
	# print("I am RSI")
	# print(data_df['RSI'])

	# calculate simple moving average, relative strength index and exponential moving average for the given data using TA-Lib
	data_df["SMA"] = talib.SMA(data_df.Close, timeperiod=trade_days_month)
	data_df["RSI"] = talib.RSI(data_df.Close, timeperiod=trade_days_month)
	data_df["EMA"] = talib.EMA(data_df.Close, timeperiod=trade_days_month)

	print("TA-Lib Version : {}".format(talib.__version__))

	# create candlestick chart with mplfinance
	ema_plt = fplt.make_addplot(data_df["EMA"], color='red', width=1.2)
	sma_plt = fplt.make_addplot(data_df["SMA"], color='blue', width=1.7)
	sma_plt_scatter = fplt.make_addplot(data_df["SMA"], scatter=True, markersize=100, marker='^', color='green',alpha=0.5)

	rsi_plt = fplt.make_addplot(data_df["RSI"], color="grey", width=1.5, ylabel="RSI",
	                            secondary_y=True, linestyle='dashdot')
	volume = fplt.make_addplot(data_df["Volume"], color="purple",)

	# save the candlestick chart as an image in the images folder, create images folder if it doesn't exist
	if not os.path.exists('images'):
		os.makedirs('images')

	# set the style of the candlestick chart
	mc = fplt.make_marketcolors(
		up='tab:blue', down='tab:red',
		edge='lime',
		wick={'up': 'blue', 'down': 'red'},
		volume='lawngreen',
	)


	# set the style of the candlestick chart
	s = fplt.make_mpf_style(base_mpl_style="ggplot", marketcolors=mc)


	# save the candlestick chart as an image in the images folder
	savefig = dict(fname=f"images/{ticker}.png", dpi=1000, pad_inches=0.75)


	# plot the candlestick chart
	fplt.plot(
		data_df,
		type='candle',
		#  Add extra plots on top of main candle stick chart. These are technical indicators.
		addplot=[sma_plt, sma_plt_scatter, ema_plt, rsi_plt, volume],

		style=s,
		title=ticker,
		ylabel='Price($)',
		ylabel_lower='shares\nTraded',
		volume=True,
		mav=(3, 5, 7),
		# savefig='images/candlestick.png',
		savefig=savefig,

		show_nontrading=False
	)


plot_candlestick(data_df)
