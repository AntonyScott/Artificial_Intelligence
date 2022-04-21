from math import exp
from random import seed
from random import random
import matplotlib.pyplot as plt

#variables
self_logged_error = [] #stored as an empty array to begin with

# Initialize a network
def init_Network(n_inputs, n_hidden, n_outputs):
        network = list() #entire network is defined as a list
        hidden_layer = [{'weights':[0.74, 0.8, 0.35, 0.9]},
                        {'weights':[0.13, 0.4, 0.97, 0.45]},
                        {'weights':[0.68, 0.1, 0.96, 0.36]}] #weights for hidden layer + bias as last element in each array
        network.append(hidden_layer) #above weights appended to hidden layer
        output_layer = [{'weights':[0.35, 0.5, 0.9, 0.98]},
                        {'weights':[0.8, 0.13, 0.8, 0.92]}] #weights for output layer + bias as last element in each array
        network.append(output_layer) #above weights appended to output layer
        return network
 
# Calculate neuron activation for an input
def activate_Neuron(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def sigmoid(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate through the network until an output is reached
def forward_Propagation(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate_Neuron(neuron['weights'], inputs)
			neuron['output'] = sigmoid(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of a neuron output
def transfer_Neuron_Derivative(output):
	return output * (1.0 - output)
 
# Backpropagates errors through the network
def back_Propagate_Error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list() #errors defined as a list
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_Neuron_Derivative(neuron['output'])
 
# Update network weights with error
def updating_Weights(network, row, learning_Rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= learning_Rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= learning_Rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def network_Training(network, train, learning_Rate, n_Epoch, n_Outputs):
	for epoch in range(n_Epoch):
		sum_error = 0
		for row in train:
			outputs = forward_Propagation(network, row)
			expected = [0 for i in range(n_Outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			back_Propagate_Error(network, expected)
			updating_Weights(network, row, learning_Rate)
		self_logged_error.append([epoch, sum_error])
		print('>epoch=%d, learning_rate=%.3f, error_value=%.3f' % (epoch, learning_Rate, sum_error))

def predict_Output(network, row): #predicts network output
	outputs = forward_Propagation(network, row)
	return outputs.index(max(outputs))

#backpropagation algorithm
def back_propagation(train, test, learning_Rate, n_Epoch, n_Hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = init_Network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, learning_Rate, n_Epoch, n_Hidden)
	predictions = list() #defined as a list
	for row in test:
		prediction = predict_Output(network, row)
		predictions.append(prediction) #output predictions appended into a array
	return(predictions)

def plot_Learning_Curve_Graph(self_logged_error):
        x_data = [] #empty array
        y_data = [] #empty array
        x_data.extend([self_logged_error[i][0] for i in range(0,len(self_logged_error))])
        y_data.extend([self_logged_error[i][1] for i in range(0,len(self_logged_error))])
        fig, ax = plt.subplots()
        ax.set(xlabel='Epoch', ylabel='Squared Error')
        ax.plot(x_data, y_data, 'tab:blue')
        plt.grid(True)
        plt.show()
        
#dataset of values we were given
dataset = [[0.50, 1.00, 0.75, 1], #first 3 values + expected output
           [1.00, 0.50, 0.75, 1],
           [1.00, 1.00, 1.00, 1],
           [-0.01, 0.50, 0.25, 0],
           [0.50, -0.25, 0.13, 0],
           [0.01, 0.02, 0.05, 0]]

network_Inputs = len(dataset[0]) - 1
network_Outputs = len(set([row[-1] for row in dataset]))
network = init_Network(network_Inputs, 2, network_Outputs) #network takes in network inputs, neurons in hidden layer and network outputs
network_Training(network, dataset, 0.1, 1000, network_Outputs) #takes in entire network, the dataset, learning rate (0.1), epochs, and the outputs of the network

for layer in network:
	print(layer) #prints out updated weights

for row in dataset: #prints out output predictions
        prediction = predict_Output(network, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))

plot_Learning_Curve_Graph(self_logged_error) #learning curve graph is called
