from keras.src.layers import SimpleRNN

from create_model import create_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN

create_model(4, 256, LSTM, num_time_steps=10, number_of_features=5, activation="ReLU", loss="mean_absolute_error", optimizer="Adam", metrics="mean_absolute_error")

create_model(4, 256, SimpleRNN, num_time_steps=10, number_of_features=5, activation="ReLU", loss="mean_absolute_error", optimizer="Adam", metrics="accuracy")

create_model(4, 256, GRU, num_time_steps=10, number_of_features=5, activation="ReLU", loss="mean_absolute_error", optimizer="Adam", metrics="sensitivity")
