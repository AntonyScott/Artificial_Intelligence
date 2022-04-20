from random import random
import math
from math import exp
import matplotlib

learning_Rate = 0.1

#initialises the network
def init_Network(net_Inputs, net_Hidden, net_Outputs):
    network = list() #network defined as a list
    #hidden_Layer = [{'weights':[random() for i in range(net_Inputs + 1)]} for i in range(net_Hidden)] #hidden layer is given random weight values
    
    #bias = [[0.9, 0.45, 0.36],[0.98, 0.92]]
    #network.append(bias)
    
    #hidden_Layer = [{'weights':[0.74, 0.13, 0.68],[0.8, 0.4, 0.10],[0.35, 0.97, 0.96]}] #weights of x1, x2 and x3
    hidden_Layer = [{'weights':[0.74, 0.13, 0.68]},
                    {'weights':[0.8, 0.4, 0.10]},
                    {'weights':[0.35, 0.97, 0.96]}] #weights of x1, x2 and x3
    network.append(hidden_Layer) #weight values added to hidden layer
    #output_Layer = [{'weights':[0.35, 0.8],[0.5, 0.13],[0.90, 0.8]}] #weights for a4, a5 and a6
    output_Layer = [{'weights':[0.35, 0.8]},
                    {'weights':[0.5, 0.13]},
                    {'weights':[0.90, 0.8]}] #weights for a4, a5 and a6
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

#network = init_Network(2, 2, 1)
#network = 
row = [1, 0, None]
output = forward_Propagation(network, row)
print(output)
