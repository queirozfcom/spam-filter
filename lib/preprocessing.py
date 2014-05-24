import numpy as np

def take_train_set(matrix,train_ratio):
	num_lines = len(matrix)
	size = int( train_ratio * num_lines )

	return np.split(matrix,[size])[0]

def take_test_set(matrix,train_ratio):
	num_lines = len(matrix)
	size = int(train_ratio * num_lines )

	return np.split(matrix,[size])[1]

def split_sets(matrix,train_ratio):
	train_set = take_train_set(matrix,train_ratio)
	test_set = take_test_set(matrix,train_ratio)

	return (train_set,test_set)
