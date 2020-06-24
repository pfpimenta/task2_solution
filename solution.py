#######
# Pedro Foletto Pimenta - June 2020
# Solution for the Task 2: application to the PhD position at I3S
#######
# Task 2 description: implementation of the solution proposed by
# Prof Ramon APARICIO BORGES to the capacitated shortest path problem
# reformulated as a supervised machine learning classification problem
#########

import json
import numpy as np
from sklearn.neural_network import MLPClassifier

NUM_DATASET_FILES = 15

# percentages of train/test/validation sets
TRAIN_RATIO = 70 
TEST_RATIO = 20
VAL_RATIO = 10

def load_topology_data():
	# load graph topology
	# topology_data (graph description) : dict with keys
	# = ['links', 'adjMat', 'incMat', 'list_paths',
	# 'sd_vs_path_bin_table', 'link_vs_path_bin_table',
	# 'node_vs_path_bin_table']
	print(f'...opening dataset/topology.json') #debug
	with open('dataset/topology.json') as f:
		topology_data = json.load(f)
		# print(topology_data)
		# print(type(topology_data))
		# print(topology_data.keys())
	return topology_data

def load_dataset():
	# load training samples
	# dataX (input features) : 2d array of floats
	# dataY (output labels)  : 2d array of floats

	dataX = []
	dataY = []

	for i in range(0,NUM_DATASET_FILES):
		filename = f'dataset/trace_{i}/dataset.json'
		
		print(f'...opening {filename}') #debug

		with open(filename) as f:
			data = json.load(f)
			# print(type(data)) # class 'dict'
			# print(data.keys()) # 'X', 'y'
			# print(type(data['y'])) # list of list of float
			# print(type(data['y'][0])) # list of float
			# print(type(data['y'][0][0])) # float

			dataX = dataX + data['X'] # features (input)
			dataY = dataY + data['y'] # labels (output)

	# put into numpy arrays
	x = np.array(dataX)
	y = np.array(dataY)

	return dataX, dataY

def dataset_split(dataX, dataY):
	# TODO : MUDAR PRA NUMPY ARRAYS
	# this function receives the dataset in two
	# list of lists (dataX: input features,
	# dataY: output labels) and splits it
	# in training/testing/validation sets

	# total number of samples
	assert(len(dataX) == len(dataY))
	num_samples = len(dataX)
	print(f'total number of samples : {num_samples}') # debug

	# train/test/val set sizes
	train_size = int(num_samples*TRAIN_RATIO/100)
	test_size = int(num_samples*TEST_RATIO/100)
	val_size = num_samples - train_size - test_size #int(num_samples*VAL_RATIO/100)
	print(f'train : {train_size}, test : {test_size}, val : {val_size}') # debug

	# do the split
	trainX = dataX[0:train_size]
	trainY = dataY[0:train_size]
	testX = dataX[train_size:train_size+test_size]
	testY = dataY[train_size:train_size+test_size]
	valX = dataX[train_size+test_size:]
	valY = dataY[train_size+test_size:]
	
	return trainX, trainY, testX, testY, valX, valY

def print_dataset_debug(dataX, dataY):
	# function for debugging

	# print total number of samples 
	num_total_samples = len(dataX)
	print(f'total number of samples : {num_total_samples}')
	# print feature and label size
	print(f'feature size : {len(dataX[0])}')
	print(f'label size : {len(dataY[0])}') 

	# aaaaa TODO arrumar essa merda
	
	# print feature basic statistics
	for i in range(len(x[0])):
		print(f'feature {i} : mean=={np.mean(x[:, i])}, (min,max):({np.min(x[:, i])},{np.max(x[:, i])})')

	# print label basic statistics
	for i in range(len(y[0])):
		print(f'label {i} : mean=={np.mean(y[:, i])}, (min,max):({np.min(y[:, i])},{np.max(y[:, i])})')

	# print some samples of the dataset
	# num_show_samples = 15 # num of samples to print
	# for i in range(int(num_total_samples/2)-num_show_samples, int(num_total_samples/2)):
	# 	print(f'dataX[{i}] : {dataX[i]} --- dataY[{i}] : {dataY[i]}')

def test_model(model, testX, testY):
	# calculates a score based on a comparison
	# between the predictions and the ground truth
	# of the samples in the test set

	num_total_samples = len(testX)

	# testing
	print("...testing")
	errors = []
	for i in range(num_total_samples):
		pred = model.predict([testX[i]])
		loss = np.sum(abs(pred - testY[i])) # TODO verificar se eh realmente uma 'loss'
		errors.append(loss)

		# debug : verify the wrong prediction cases
		if(loss > 0):
			print(f'example {i} - predicted: {pred}, ground truth: {testY[i]}') # debug
			print(f'loss : {loss}') # debug
			


	score = 1 - np.mean(errors)
	
	return score


#############################3

if(__name__ == "__main__"):

	### LOAD DATA
	# load graph topology
	topology_data = load_topology_data()
	# load training samples
	dataX, dataY = load_dataset()	

	# print data stats and metainfo
	#print_dataset_debug(dataX, dataY)

	### SPLIT DATASET : (training/test/validation)
	trainX, trainY, testX, testY, valX, valY = dataset_split(dataX, dataY)

	### TRAINING
	# create model
	model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 10, 5))#, random_state=1)

	# fit to data (training/learning)
	model.fit(trainX, trainY)

	### TESTING
	score = test_model(model, testX, testY)
	print(f'score: {score}')

	### TESTING
	score = test_model(model, valX, valY)
	print(f'validation score hahaha: {score}')

	###
	# VALIDATION