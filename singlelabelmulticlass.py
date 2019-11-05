# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:11:11 2019

@author: Chetan RS
"""

#Insidecity,Coast,Tallbuilding,Street,Highway
#coast - 0  [1,0,0,0,0]
# highway - 2 [0,1,0,0,0]
# indsidecity - 3 [0,0,1,0,0]
# street - 6 [0,0,0,1,0]
# tall building - 7 [0,0,0,0,1]


# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
 

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def train_test_split(dataset,split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
    
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset,l_rate, n_epoch, n_hidden):
    train,test_val = train_test_split(dataset,split=0.7)
    test,val=train_test_split(test_val,split=0.67)
    scores_train = list()
    scores_test = list()
    predicted = back_propagation(train, test,l_rate, n_epoch, n_hidden)
    actual_train = [row[60:] for row in train]
    scores_train = accuracy_metric(actual_train, predicted[0])
    actual_test = [row[60:0] for row in test]
    scores_test = accuracy_metric(actual_test, predicted[1])
    return scores_train,scores_test
 
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
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
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:60]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			for i in range(len(row[60:0])):
			    expected[i] = row[-5+i]
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	opt = list() 
	opt = [0 for _ in range(len(outputs))] 
	num = outputs.index(max(outputs))
	opt[num] = 1 
	return opt 

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 5
	n_outputs = 5
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions_test = list()
	predictions_train = list()
	for row in test:
		prediction = predict(network, row)
		predictions_test.append(prediction)
	for row in train:
		prediction = predict(network, row)
		predictions_train.append(prediction)
	return(predictions_train, predictions_test)
 
    
    

class_list = [0,2,3,6,7]
row_no = list()
fname = 'image_data_labels.txt'
f = open(fname,'r')
f2 = open('image_data_feat_dim60.txt')
X = list()
y1 = list()
y = list()
row = 1

for x in f.readlines():
    if int(x) in class_list:
        row_no.append(row)
        j = int(x)
        j = float(x)
        y1.append(j)
    row = row +1
f.close()

for val in y1:
    onehot = [0.0 for _ in range(len(class_list))]
    onehot[class_list.index(val)] = 1.0
    y.append(onehot)

row2 = 1
for line in f2.readlines():
    if row2 in row_no:
        X.append([float(x) for x in line.split()])
    row2 = row2 + 1
f2.close()
    
seed(1)
# load and prepare data
dataset = list()
i = 0
for i in range(len(y)):
    for j in range(len(y[0])):
        X[i].append(y[i][j])
    dataset.append(X[i])
    

# =============================================================================
# =============================================================================
# =============================================================================
# # =============================================================================
# =============================================================================
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# # # # # =============================================================================
# # # # # evaluate algorithm
# # # # # =============================================================================
# # 
l_rate = 0.1
n_epoch = 100
n_hidden = 50
scores_train, scores_test = evaluate_algorithm(dataset, l_rate, n_epoch, n_hidden)
print('Train Score: %s' % scores_train)
print('Test Score: %s' % scores_test)
# # =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
