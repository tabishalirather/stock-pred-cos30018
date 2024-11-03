import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_predicted_vs_actual(predicted_prices, actual_prices, steps_to_predict):
	"""
	Plots predicted vs. actual 'Close' prices over multiple steps.
	"""
	plt.figure(figsize=(12, 6))
	for step in range(steps_to_predict):
		plt.plot(predicted_prices[step], label=f'Predicted Step {step + 1}')
		plt.plot(actual_prices[step], label=f'Actual Step {step + 1}', linestyle='--')

	plt.xlabel('Sample')
	plt.ylabel('Close Price')
	plt.title('Predicted vs. Actual Close Prices for Multiple Steps')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def plot_average_differences(average_differences, steps_to_predict, diff_type="absolute"):
	"""
	Plots average differences (absolute or percentage) across prediction steps.
	"""
	steps = range(1, steps_to_predict + 1)
	plt.figure(figsize=(10, 5))
	plt.bar(steps, average_differences, color='blue' if diff_type == "absolute" else 'orange', alpha=0.7)
	plt.xlabel('Step')
	plt.ylabel('Average Difference' if diff_type == "absolute" else 'Average Percentage Difference (%)')
	title = 'Average Absolute Differences Across Steps' if diff_type == "absolute" else 'Average Percentage Differences Across Steps'
	plt.title(title)
	plt.tight_layout()
	plt.show()


def display_prediction_table(predicted_prices, actual_prices, differences, test_dates, steps_to_predict):
	"""
	Displays a table of predicted vs. actual prices with differences for each sample and step.
	"""
	table_data = []
	for i in range(steps_to_predict):
		for j in range(len(predicted_prices[i])):
			predicted = round(float(predicted_prices[i][j][0]), 3)
			actual = round(float(actual_prices[i][j][0]), 3)
			difference = round(float(differences[i][j][0]), 3)
			prediction_date = pd.to_datetime(test_dates[j]) + pd.Timedelta(days=i)
			prediction_date = prediction_date.strftime('%Y-%m-%d')
			table_data.append([i + 1, f'Sample {j + 1}', prediction_date, predicted, actual, difference])

	# Convert to a DataFrame for better display
	df_table = pd.DataFrame(table_data, columns=["Step", "Sample", "Date", "Predicted 'Close' Price",
	                                             "Actual 'Close' Price", "Difference"])
	print("\nPredicted vs. Actual Close Prices Table:")
	print(df_table)


def plot_heatmap(differences, steps_to_predict):
	"""
	Plots a heatmap of prediction differences across steps and samples.
	"""
	differences_matrix = differences.squeeze()  # Convert to 2D matrix if needed
	plt.figure(figsize=(12, 8))
	sns.heatmap(differences_matrix, cmap='coolwarm', annot=True, fmt=".2f")
	plt.xlabel('Samples')
	plt.ylabel('Steps')
	plt.title('Heatmap of Prediction Differences Across Steps and Samples')
	plt.tight_layout()
	plt.show()


def display_summary_table(avg_diff, avg_perc_diff):
	"""
	Displays a summary table of average differences and average percentage differences.
	"""
	summary_data = {
		'Metric': ['Total Average Difference', 'Total Average Percentage Difference'],
		'Value': [avg_diff, f"{avg_perc_diff:.2f}%"]
	}
	summary_df = pd.DataFrame(summary_data)

	print("\nSummary of Prediction Errors:")
	print(summary_df)
