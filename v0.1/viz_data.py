from matplotlib.pyplot import ylabel
import ta
print("working")
from get_data import get_data
import mplfinance as fplt
import pandas as pd
import os
ticker = 'CBA.AX'
feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
start_date = '2021-01-01'
end_date = '2021-03-30'

data_df = get_data(ticker, feature_columns, start_date, end_date, save_data=True, split_by_date=True)


def plot_candlestick(data_df):
    # data_df = data_df.set_index('Date')
    print("Candlestick Chart Styling from MPLFinance : {}".format(fplt.available_styles()))

    trade_days_month = 20
    data_df['SMA'] = data_df['Close'].rolling(window=trade_days_month).mean()
    data_df['EMA'] = data_df['Close'].ewm(span=trade_days_month, adjust=False).mean()
    print(len(data_df))
    data_df.index = pd.to_datetime(data_df.index)
    data_df['RSI'] = ta.momentum.RSIIndicator(data_df['Close'], window=trade_days_month).rsi()
    print("I am RSI")
    print(data_df['RSI'])


    ema = fplt.make_addplot(data_df["EMA"], color='red', width=1.2)
    sma = fplt.make_addplot(data_df["SMA"], color='blue', width=1.7)
    rsi = fplt.make_addplot(data_df["RSI"], color="grey", width=1.5, ylabel="RSI",
                            secondary_y=True, linestyle='dashdot')
    volume = fplt.make_addplot(data_df["Volume"], color="purple",
                               panel=1
                               )
    if not os.path.exists('images'):
        os.makedirs('images')
    fplt.plot(data_df,
              type='candle',
              addplot=[sma,ema,rsi, volume],
              style='charles',
              title=ticker,
              ylabel = 'Price($)',
              ylabel_lower='shares\nTraded',
              volume=True,
              savefig='images/candlestick.png',
               show_nontrading=False)



plot_candlestick(data_df)


