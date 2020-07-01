#######
# Pedro Foletto Pimenta - June 2020
# Solution for the Task 2: application to the PhD position at I3S
#######
# Task 2 description: implementation of the solution proposed by
# Prof Ramon APARICIO BORGES to the capacitated shortest path problem
# reformulated as a supervised machine learning classification problem
#########

### imports
import json
import numpy as np
from random import randint
from sklearn.neural_network import MLPClassifier

### defines
DATASET_PATH = 'dataset/'
NUM_DATASET_FILES = 15

# percentages of train/test/validation sets
TRAIN_RATIO = 80 
TEST_RATIO = 20

### functions

def load_topology_data():
	# load graph topology
	# topology_data (graph description) : dict with keys
	# = ['links', 'adjMat', 'incMat', 'list_paths',
	# 'sd_vs_path_bin_table', 'link_vs_path_bin_table',
	# 'node_vs_path_bin_table']

	print(f'...opening {DATASET_PATH}topology.json') #debug
	with open(f'{DATASET_PATH}topology.json') as f:
		topology_data = json.load(f)
	return topology_data

def load_dataset():
	# load training samples
	# dataX (input features) : 2d array of floats
	# dataY (output labels)  : 2d array of floats

	dataX = []
	dataY = []

	for i in range(0,NUM_DATASET_FILES):
		filename = f'{DATASET_PATH}trace_{i}/dataset.json'
		
		print(f'...opening {filename}') #debug

		with open(filename) as f:
			data = json.load(f)
			dataX = dataX + data['X'] # features (input)
			dataY = dataY + data['y'] # labels (output)

	# put into numpy arrays
	dataX = np.array(dataX)
	dataY = np.array(dataY)

	return dataX, dataY

def dataset_split(dataX, dataY):
	# this function receives the dataset in two
	# list of lists (dataX: input features,
	# dataY: output labels) and splits it
	# in training/testing/validation sets

	# total number of samples
	assert(len(dataX) == len(dataY))
	num_samples = len(dataX)
	print(f'total number of samples : {num_samples}') # debug

	# train/test set sizes
	train_size = int(num_samples*TRAIN_RATIO/100)
	test_size = num_samples - train_size
	print(f'train : {int(train_size*90/100)}, validation: {int(train_size*10/100)} test : {test_size}')

	# do the split
	trainX = dataX[0:train_size]
	trainY = dataY[0:train_size]
	testX = dataX[train_size:train_size+test_size]
	testY = dataY[train_size:train_size+test_size]
	
	return trainX, trainY, testX, testY

def print_dataset_debug(dataX, dataY):
	# function for debugging

	# print total number of samples 
	num_total_samples = len(dataX)
	print(f'total number of samples : {num_total_samples}')
	# print feature and label size
	print(f'feature size : {len(dataX[0])}')
	print(f'label size : {len(dataY[0])}') 
	
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

def sd_node_id_2_bin_array(sd_nodes, num_nodes):
	# this function receives:
	# - num_nodes : the number of nodes in the graph
	# - src_node_id, dst_node_id: IDs of the src and dst nodes
	# and returns a binary array with 1s in the corresponding positions
	num_samples = len(sd_nodes)
	sd_bin_array = np.zeros((num_samples, num_nodes))
	for i in range(num_samples):
		src_node_id = int(sd_nodes[i, 0])
		dst_node_id = int(sd_nodes[i, 1])
		sd_bin_array[i, src_node_id] = 1
		sd_bin_array[i, dst_node_id] = 1
	
	return sd_bin_array

def s_d_ids_2_sd_id(src_node_id, dst_node_id, num_nodes):
	# this function receives a source node id
	# and a destination node id
	# and returns the id corresponding to
	# the path src->dst

	sd_id = int((src_node_id+1)*(num_nodes-1) + dst_node_id - num_nodes + int(bool(src_node_id>dst_node_id)))
	return sd_id

def data_pre_processing(dataX, num_nodes):
	# this functions takes input samples,
	# changes the representation of the src and dst
	# nodes information,
	# and removes its unnecessary features:
	# - connection volume (always equal to 1)
	# - connection total duration (irrelevant to the problem)

	# get graph state
	graph_state = dataX[:,0:40]
	# get src-dst nodes information
	sd_nodes = dataX[:,40:42]
	sd_bin_array = sd_node_id_2_bin_array(sd_nodes,num_nodes)
	# concatenate graph state and src-dst information
	dataX = np.concatenate((graph_state, sd_bin_array), axis=1)

	return dataX

def label_post_processing(label):
	# this functions receives a predicted label,
	# corrects it if it is invalid,
	# and then returns it

	if(sum(label) != 1):
		# if invalid we say that there is no feasible path
		label = np.array([1, 0, 0, 0])
	return label

def classify_request(model, request_sample):
	# this function runs the received prediction model
	# with the received request_sample,
	# passes it through the post_processing,
	# and returns the predicted label

	# prediction
	predicted_label = model.predict([request_sample])[0]
	# post processing
	predicted_label = label_post_processing(predicted_label)

	return predicted_label

def test_model(model, testX, testY):
	# calculates a score based on a comparison
	# between the predictions and the ground truth
	# of the samples in the test set

	num_total_samples = len(testX)

	# testing
	errors = []
	for i in range(num_total_samples):
		#pred = model.predict([testX[i]])
		pred = classify_request(model, testX[i])
		loss = np.sum(abs(pred - testY[i])) # TODO verificar se eh realmente uma 'loss'
		errors.append(loss)

		# debug : verify the wrong prediction cases
		if(loss > 0):
			print(f'PREDICTION ERROR: example {i} - predicted: {pred}, ground truth: {testY[i]}, ---> loss : {loss}') # debug
			


	score = 1 - np.mean(errors)
	
	return score

def get_path(label, src_node_id, dst_node_id, path_list, num_nodes):
	# this function receives a label indicating
	# which of the 3 best paths (or none) was chosen
	# for a connection request,
	# the connection request source and destination nodes,
	# and the total number of nodes in the graph,
	# and then returns the list of nodes of the following path

	sd_id = s_d_ids_2_sd_id(src_node_id, dst_node_id, num_nodes)
	chosen_path_idx = int(np.argmax(label)) - 1
	if(chosen_path_idx == -1):
		# there is NO feasible path
		return "<there is no feasible path>"
	else:
		# there IS a feasible path
		path = path_list[sd_id][chosen_path_idx]
		return path

def readable_request(src_node, dst_node, connection_volume=None, connection_duration=None):
	# returns a readable version of a request
	readable_string = f"connection request from node {int(src_node)} to node {int(dst_node)}"
	if(connection_volume != None and connection_duration != None):
		readable_string += f" with connection volume {connection_volume} and duration {connection_duration}"
	return readable_string

#############################
### "main"

if(__name__ == "__main__"):

	### LOAD DATA
	# load graph topology
	topology_data = load_topology_data()
	# load training samples
	dataX, dataY = load_dataset()

	# preprocessing
	num_nodes = len(topology_data['adjMat'])
	pp_dataX = data_pre_processing(dataX, num_nodes)

	### SPLIT DATASET : (training/test)
	trainX, trainY, testX, testY = dataset_split(pp_dataX, dataY)

	### TRAINING
	# create model
	model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 10, 5), verbose=True, early_stopping=True, n_iter_no_change=20)
	print("\ntraining...")
	# fit to data (training/learning)
	model.fit(trainX, trainY)
	print("...training complete")

	### TESTING
	print("\ntesting...")
	score = test_model(model, testX, testY)
	print(f'... testing score: {score}')


	#################################################################
	#### EXAMPLE USE CASE

	# chosing a random sample
	random_example_index = randint(0, len(dataX))
	sample = dataX[random_example_index]
	expected_label = dataY[random_example_index]
	request = readable_request(sample[40], sample[41], sample[42], sample[43])
	# running the prediction on the chosen sample
	predicted_label = model.predict([pp_dataX[random_example_index]])[0]
	predicted_label = classify_request(model, pp_dataX[random_example_index])
	# getting the paths represented by the predicted and expected labels
	predicted_path = get_path(predicted_label, sample[40], sample[41], topology_data['list_paths'], num_nodes)
	expected_path = get_path(expected_label, sample[40], sample[41], topology_data['list_paths'], num_nodes)


	print(f'\n\n...random example:')
	print(f'*Input: {request} \n      and graph state={sample[0:40]}')
	print(f'*Predicted output: {predicted_label}\t-> path: {predicted_path} ')
	print(f'*Expected output: {expected_label}\t-> path: {expected_path} ')
