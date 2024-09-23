import pandas as pd
from kan import *
from sklearn.model_selection import train_test_split
from torch.cuda import device

from get_data import get_data
import torch
import pandas_datareader.data as web
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

# Get GDP data from FRED
gdp = web.DataReader('GDP', 'fred', start='1900-01-01', end='2023-01-01')
# Show the first few rows
print(gdp.head())
print(f"lelngth of gdp is {len(gdp)}")
gdp.reset_index(inplace=True)
gdp['DATE'] = pd.to_datetime(gdp['DATE'])
gdp['Num_date'] = (gdp['DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
print("Setting gdp head")
print(gdp.head())

x = gdp['Num_date']
y = gdp['GDP']
# print(x)
# print(y)
# print(gdp.head())
x_train, x_test, y_train, y_test = (train_test_split
                                    (x,
                                     y,
                                     test_size=0.2,
                                     random_state=42,
                                     shuffle=False)
                                    )

print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# we need the dataset in this shape.
dataset = {
    'train_input': torch.tensor(x_train.values).float().to(device).unsqueeze(1),
    'test_input': torch.tensor(x_test.values).float().to(device).unsqueeze(1),
    'train_label': torch.tensor(y_train.values).float().to(device).unsqueeze(1),
    'test_label': torch.tensor(y_test.values).float().to(device).unsqueeze(1)
}
print("Data shape is:")
print(dataset['train_input'].shape, dataset['train_label'].shape)

# data_df = get_data('')
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).

input_size = (dataset['train_input']).shape[1]
# input_size = 1
num_neurons = 10
output_size = 1
model = KAN(width=[input_size,num_neurons,output_size], grid=5, k=3, seed=0, device=device)
model(dataset['train_input'])
model.plot(beta=100)

# train the model
summary = model.fit(dataset, opt="LBFGS", steps=20);
print("Training summary:")
print(summary)

print("Model is trained, now plot")




