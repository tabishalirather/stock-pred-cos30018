import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_data import get_data

feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume']

data = get_data('CBA.AX', feature_cols, '2021-01-01', '2021-02-06', save_data=True, split_by_date=True, )
import os

# create boxplot for the data
# plt.figure(figsize=(10, 5))


def calc_roll_stats(data_df, feature_columns, num_days_roll=5):
	# print("Calc roll stats")
	rolling_min = data_df[feature_columns].rolling(window=num_days_roll).min().dropna()
	rolling_25 = data_df[feature_columns].rolling(window=num_days_roll).quantile(0.25).dropna()
	rolling_median = data_df[feature_columns].rolling(window=num_days_roll).median().dropna()
	rolling_75 = data_df[feature_columns].rolling(window=num_days_roll).quantile(0.75).dropna()
	rolling_max = data_df[feature_columns].rolling(window=num_days_roll).max().dropna()

	# aligned_index = rolling_min.index


	# print(f"rolling_min: {len(rolling_min)}")
	# print(f"rolling_25: {len(rolling_25)}")
	# print(f"rolling_median: {len(rolling_median)}")
	# print(f"rolling_75: {len(rolling_75)}")
	# print(f"rolling_max: {len(rolling_max)}")
	# print(f"Date: {len(data_df.index[num_days_roll - 1:])}")
	#
	# print(f"rolling_min after stacking: {len(rolling_min.stack())}")
	# print(f"rolling_25 after stacking: {len(rolling_25.stack())}")


	# concat the dataframes/series to create a single dataframe
	rolling_stats_df = pd.concat({
		"min": rolling_min,
		"25%": rolling_25,
		"median": rolling_median,
		"75%": rolling_75,
		"max": rolling_max
	})

	# print(f"rolling_data_df is: {rolling_stats_df} ")
	return rolling_stats_df
#


def plot_box(data_df=data, feature_columns=None, num_days_roll=5):
	rolling_data_df = calc_roll_stats(data_df, feature_columns, num_days_roll)
	# print(rolling_data_df.tail())
	# reset index to have the date as a column, adds default integer index as a column
	rolling_data_reset = rolling_data_df.reset_index()

	# Show the structure after resetting the index
	print(rolling_data_reset.head())

	# Creating the box plot.
	# Melt the data to have a single column for the stock feature and a single column for the value so that it's suitable for use in Seaborn. This is because Seaborn expects data to be in long format.
	rolling_data_long = pd.melt(rolling_data_reset, id_vars=['level_0'],
	                            value_vars=['Open', 'Close', 'High', 'Low', 'Volume'],
	                            var_name='Stock Feature', value_name='Value')

	# Create a Seaborn box plot
	plt.figure(figsize=(20, 16))
	sns.boxplot(x='level_0', y='Value', hue='Stock Feature', data=rolling_data_long)

	# Enhancing the plot
	plt.title(f'{num_days_roll} days rolling Boxplot', fontsize=16)
	plt.xlabel('Statistic Type', fontsize=12)
	plt.ylabel('Value', fontsize=12)
	plt.xticks(rotation=45)
	plt.legend(title='Stock Feature')

	# Display the plot
	if not os.path.exists('images'):
		os.makedirs('images')
	plt.savefig('images/boxplot.pdf', dpi=1200)
	plt.show()


plot_box(data, feature_cols, num_days_roll=5)