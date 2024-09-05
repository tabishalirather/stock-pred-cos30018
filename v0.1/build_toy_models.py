
import random
from create_model import create_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
# from tensorflow.keras import backend as k



# generate random parameters for the model
def rand_parameters():
	num_layers = random.randint(4, 10)
	print(num_layers)

	units_per_layer = random.randint(128, 256)
	print(units_per_layer)

	layer_name = random.choice([LSTM, GRU, SimpleRNN])
	print(layer_name)

	num_time_steps = random.randint(10, 50)
	print(num_time_steps)

	number_of_features = random.randint(5, 10)
	print(number_of_features)

	activation = random.choice(["ReLU", "tanh", "sigmoid"])
	print(activation)

	loss = random.choice(["mean_absolute_error", "mean_squared_error", "mean_squared_logarithmic_error"])
	print(loss)

	optimizer = random.choice(["Adam", "RMSprop", "SGD"])
	print(optimizer)

	metrics = random.choice(["mean_absolute_error", "mean_squared_error", "mean_squared_logarithmic_error"])
	print(metrics)
	return [num_layers, units_per_layer, layer_name, num_time_steps, number_of_features, activation, loss, optimizer, metrics]





def build_toy_models(num_of_models):
	list_of_models = []
	for model in range (num_of_models):

		[num_layers, units_per_layer, layer_name, num_time_steps, number_of_features, activation, loss, optimizer, metrics] = rand_parameters()
		# k.clear_session()  # Clear the session to reset layer naming

		dl_model = create_model(num_layers, units_per_layer, layer_name, num_time_steps, number_of_features, activation, loss, optimizer, metrics)
		list_of_models.append(dl_model)
	return list_of_models

num_of_models = random.randint(1, 3)
print(num_of_models)
model_list = build_toy_models(num_of_models)

[model.summary() for model in model_list]
