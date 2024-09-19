from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout

import tensorflow as tf
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
# tf.compat.train.Optimizer

# num_time_steps in time series in number of rows(time points) to consider.
# number_of_features is the number of columns in the time series data
def create_model(num_layers, units_per_layer, layer_name, num_time_steps, number_of_features=5, activation="tanh", loss="mean_absolute_error", optimizer="rmsprop", metrics="mean_absolute_error"):
	print("Lock and load")
	# sequential allows us to stack layers on top of each other
	dl_model = Sequential()
	for layer in range(num_layers):
		# first layer, specify the shape of the input data, batch size for weight updation, num of steps to consider and number of features to expect.
		if (layer == 0):
			# input_shape=(batch_size, num_time_steps/data_seq_length/rows, number_of_features/columns)
			'''
			Sequence Length = 5:
				You choose a sequence length of 5, which means the model will look at 5 consecutive days of stock data at a time to make a prediction.
				For example, for a given training instance, if your data looks like this:
				Day	Open	High	Low	Close	Volume
				Day 1	100	105	98	102	200000
				Day 2	102	106	101	104	180000
				Day 3	104	107	102	106	170000
				Day 4	106	109	103	108	150000
				Day 5	108	110	104	109	140000
			
			
			Batch Size = 32:
				If your batch size is 32, it means the model processes 32 sequences at a time before it updates its weights.
				So, for a batch of size 32, the model will take 32 such sequences, each consisting of 5 consecutive days of data, and process them together.
				For instance, the first batch might contain these 32 sequences:
				
				Batch 1	Sequence Length (5 Days Each)
				Sequence 1	Day 1 to Day 5
				Sequence 2	Day 2 to Day 6
				Sequence 3	Day 3 to Day 7
				Sequence 4	Day 4 to Day 8
				...	...
				Sequence 32	Day 32 to Day 36
				
			How They Work Together:
				Sequence Length (5): Determines how much "history" the model looks at to make a prediction. In this case, 5 days of stock data.
				Batch Size (32): Determines how many sequences are processed together before the model updates its weights. In this case, 32 sequences, each 5 days long, are processed in one batch.
				Currently, batch size set to none, so that we can dynamically change the batch size.
			'''
			dl_model.add(layer_name(units_per_layer, batch_input_shape=(None, num_time_steps, number_of_features), return_sequences=True))

		elif (layer == num_layers - 1):
			# 	last layer, reutrn sequences is false cuz we not stacking anymore.
			dl_model.add(layer_name(units_per_layer, return_sequences=False))
		else:
			# hidden layers return sequences so that we can stack lstm layers on top of each other
			dl_model.add(layer_name(units_per_layer, return_sequences=True))


	# add dropout after each layer, which kicks out 20% of the neurons randomly to prevent overfitting
		dl_model.add(Dropout(0.2))
	# output layer, fully connected, 1 neuron layer for prediction value.
	dl_model.add(Dense(1, activation='tanh'))
	# compile the model with given loss function, optimizer and metrics. loss is the fxn to be minimised, i.e the goal for dl_model is to minimise the loss. optimizer is the algorithm to be used to minimise the loss. metrics is the fxn to be used to evaluate/track the model; often same as loss. .
	dl_model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
	# print(dl_model.summary())
	return dl_model


# create_model(4, 256, LSTM, num_time_steps=10, number_of_features=5, activation="ReLU", loss="mean_absolute_error", optimizer="Adam", metrics="mean_absolute_error")
