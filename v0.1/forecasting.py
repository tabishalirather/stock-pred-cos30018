import itertools

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from get_data import get_data
import warnings
import numpy as np
from get_data import get_data



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

COMPANY = 'AMZN'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PREDICTION_DAYS = 12
target_column = 'future_1'
STEPS_TO_PREDICT = 1

import pandas as pd
from statsmodels.tsa.stattools import adfuller


from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_errors(true_values, predicted_values, model_name="Model"):
    print("Predicted values shape:", predicted_values.shape)
    print("True values shape:", true_values.shape)
    true_values = true_values.reshape(-1)

    squared_errors = (predicted_values - true_values) ** 2

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(squared_errors)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    # print(f"Errors for {model_name}:")
    # print(f"Mean Absolute Error (MAE): {mae}")
    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"Mean Absolute Percentage Error (MAPE): {mape}%\n")
    return rmse





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

raw_data = raw_d_r[0]
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
forecast_index = pd.date_range(start=raw_data.index[-1], periods=PREDICTION_DAYS + 1, freq='D')[1:]

def sarima_forecast(raw_data, target_column, PREDICTION_DAYS):
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
    sarima_forecast = pd.DataFrame(sarima_model.predict(n_periods=PREDICTION_DAYS))
    sarima_forecast.columns = ['sarima_forecast']
    sarima_forecast.index = forecast_index

    return sarima_forecast

def exponential_smoothing_forecast(raw_data, target_column, PREDICTION_DAYS):
    expo_model = HWES(
        raw_data[target_column],
        seasonal='mul',
        seasonal_periods=5
    ).fit()
    expo_forecast = pd.DataFrame(expo_model.forecast(PREDICTION_DAYS))
    expo_forecast.columns = ['expo_forecast']
    expo_forecast.index = forecast_index
    return expo_forecast

def sarimax_forecast(raw_data, target_column, exog, PREDICTION_DAYS):
    sarimax_model = SARIMAX(
        raw_data[target_column],
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 5)
    ).fit()
    sarimax_forecast = pd.DataFrame(sarimax_model.forecast(steps=PREDICTION_DAYS, exog=exog[-PREDICTION_DAYS:]))
    sarimax_forecast.columns = ['sarimax_forecast']
    sarimax_forecast.index = forecast_index
    return sarimax_forecast







results = raw_d_r[1]
# print(results)
# n_estimators = [5000,10000,15000]
# max_depth = [10,50,100]
# # Minimum number of samples required to split a node
# min_samples_split = [50,100,200]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [2,4,10]
print(results['X_train'].shape)
print(results['y_train'].shape)
print(results['X_test'].shape)
print(results['y_test'].shape)

def random_forest(n_estimators, max_depth, min_sample_split, min_samples_leaf, results):
    '''


# 	random forest is a collection of decision tres, it takes different subsets of my data. helps reduce overfitting and bias. What are it's parameters? No of trees, node size and number of features. In finance, can predict likelihood of default, medicine, can predict survival rate based on treatment options and effective of policies in economics.
# How to use random forest in regression, not classification?
#Part of ensemble learning.
# 1) Pick k random points from the training set. 2) then build a decision tree that only considers those k points. 3) Choose number of trees to built and repeat 1 and 2
    '''
    # pass # Implement your random forest model here
    x_train_flat = results['X_train'].reshape(results['X_train'].shape[0], -1)
    y_train_flat = results['y_train']
    x_test_flat = results['X_test'].reshape(results['X_test'].shape[0], -1)
    y_test_flat = results['y_test']


    RF_Test_Accuracy_Data = pd.DataFrame(columns = ['n_estimators','max_depth','min_samples_split','min_samples_leaf','Train Accurcay','Test Accurcay'])
    print("Running decision trees")
    # creates all possible combinations of the parameters and iteratively selects the best parameters
    print("Total loops", len(list(itertools.product(n_estimators, max_depth, min_sample_split, min_samples_leaf))))
    count = 0
    for x in list(itertools.product(n_estimators, max_depth, min_sample_split, min_samples_leaf)):
                print("in for loop")
                print("count", count)
                count += 1
                rf = (RandomForestRegressor(
                    n_estimators=x[0],
                    max_depth=x[1],
                    min_samples_split=x[2],
                    min_samples_leaf=x[3],
                    random_state=42, # Set random seed to 42
                    n_jobs=-1, # Use all CPU cores
                    max_features='sqrt'
                ))
                rf.fit(x_train_flat, y_train_flat) # Train the model


                predict_train = rf.predict(x_train_flat)
                errors_train = abs(predict_train - y_train_flat)
                mape_train = 100 * (errors_train / y_train_flat)
                accuracy_train = 100 - np.mean(mape_train)


#         test data
                predict_test = rf.predict(x_test_flat)
                errors_test = abs(predict_test - y_test_flat)
                mape_test = 100 * (errors_test / y_test_flat)
                accuracy_test = 100 - np.mean(mape_test)

                RF_Test_Accuracy_Data_One = pd.DataFrame(
                    index = range(1),
                    columns = ['n_estimators','max_depth','min_samples_split','min_samples_leaf','Train Accurcay','Test Accurcay']
                )
                RF_Test_Accuracy_Data_One.loc[:, 'n_estimators'] = x[0]
                RF_Test_Accuracy_Data_One.loc[:, 'max_depth'] = x[1]
                RF_Test_Accuracy_Data_One.loc[:, 'min_samples_split'] = x[2]
                RF_Test_Accuracy_Data_One.loc[:, 'min_samples_leaf'] = x[3]
                RF_Test_Accuracy_Data_One.loc[:, 'Train Accurcay'] = accuracy_train
                RF_Test_Accuracy_Data_One.loc[:, 'Test Accurcay'] = accuracy_test
                RF_Test_Accuracy_Data = pd.concat([RF_Test_Accuracy_Data, RF_Test_Accuracy_Data_One], ignore_index=True)

    print(RF_Test_Accuracy_Data)
    best_fit_model = RF_Test_Accuracy_Data.loc[RF_Test_Accuracy_Data['Test Accurcay'] == RF_Test_Accuracy_Data['Test Accurcay'].max()]
    best_fit_model = best_fit_model.values.flatten().tolist()

    rf = (RandomForestRegressor(n_estimators=best_fit_model[0],
                                max_depth=best_fit_model[1],
                                min_samples_split=best_fit_model[2],
                                min_samples_leaf=best_fit_model[3],
                                random_state=42,
                                n_jobs=-1,
                                max_features='sqrt'))
    rf.fit(x_train_flat, y_train_flat)

    random_forest_forecast = pd.DataFrame(rf.predict(x_test_flat[:PREDICTION_DAYS]), columns=['random_forest_forecast'])
    print("Random Forest Forecast")
    print(random_forest_forecast.shape)
    print(random_forest_forecast)
    # rmse_errors_random_forest = calculate_errors(y_test_flat[:PREDICTION_DAYS], random_forest_forecast['random_forest_forecast'], model_name="Random Forest")

    random_forest_forecast.index = forecast_index
    # random_forest_forecast = pd.DataFrame({
    #     'random_forest_forecast': random_forest_forecast['random_forest_forecast'],
    #     'rmse': rmse_errors_random_forest * np.ones(PREDICTION_DAYS)
    # })
    # print("Random Forest Forecast")
    # print(random_forest_forecast)
    return random_forest_forecast





n_estimators = [50,100,200]
max_depth = [10,50,100]
# Minimum number of samples required to split a node
min_samples_split = [50,100,150]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2,4,10]

random_forest_result = random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf, results)
# print(random_forest_result)

sarima_result = sarima_forecast(raw_data, target_column, PREDICTION_DAYS)
# print(sarima_result)

expo_result = exponential_smoothing_forecast(raw_data, target_column, PREDICTION_DAYS)
# print(expo_result)

# sarimax_forecast = sarimax_forecast(raw_data, target_column, raw_data['Volume'], PREDICTION_DAYS)
# print(sarimax_forecast)

def return_final_forecast():
    final_forecast = pd.concat([sarima_result, random_forest_result, expo_result], axis=1)
    print(final_forecast)
    return final_forecast







