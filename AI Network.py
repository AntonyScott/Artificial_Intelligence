from random import random
import math
from math import exp
#import matplotlib

learning_Rate = 0.1

#initialises the network
def init_Network(net_Inputs, net_Hidden, net_Outputs):
    network = list() #network defined as a list
    
    hidden_Layer = {'weights for w14, w24, w34 + w04 bias': [0.74, 0.8, 0.35, 0.9]},
    {'weights for w15, w25, w35 + w05 bias': [0.13, 0.4, 0.97, 0.45]},
    {'weights for w16, w26, w36 + w06 bias': [0.68, 0.1, 0.96, 0.36]} #weights for hidden layer + bias as last element in each array
    network.append(hidden_Layer) #weight values added to hidden layer
    
    output_Layer = [{'weights for w47, w57, w67 + w07 bias': [0.35, 0.5, 0.9, 0.98]},
                    {'weights for w48, w58, w68 + w08 bias': [0.8, 0.13, 0.8, 0.92]}] #weights for output layer + bias as last element in each array
    network.append(output_Layer) #weight values added to output layer
    return network

#calculates activation for inputs
def activate_Neurons(weights, inputs): #takes in weight and input parameters
    activation = weights[-1] #activation equal to last element in weight array i.e. the bias/threshold
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] #activation is the sum of the weights * inputs
    return activation

#neuron activation transfer
def transfer_Neuron_Signals(activation):
    return 1.0 / (1.0 + exp(-activation))

#propagates neuron signals through the network
def forward_Propagation(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate_Neurons(neuron['weights'], inputs)
            neuron['output'] = transfer_Neuron_Signals(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

#calculate derivative of neuron output
def transfer_Neuron_Derivative(output):
    return output * (1.0 - output)

def backward_Propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
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

network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_Propagation(network, expected)
for layer in network:
	print(layer)
