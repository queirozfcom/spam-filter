import numpy as np
import sys

def take_train_set(matrix,train_ratio):
    num_lines = len(matrix)
    size = int( train_ratio * num_lines )

    return np.split(matrix,[size])[0]

def take_test_set(matrix,train_ratio):
    num_lines = len(matrix)
    size = int(train_ratio * num_lines )

    return np.split(matrix,[size])[1]

def split_sets(matrix,train_ratio,test_slice=0):

    assert train_ratio > 0 and train_ratio < 1, '\033[31mERROR: Train set to test set ratio should be between 0 and 1. \033[0m'
    # testing whether a variable is an integer: http://stackoverflow.com/a/3501408/436721
    assert isinstance(test_slice,(int,long)), '\033[31mERROR: When using cross-validation, the slice to use as test set should be an integer. \033[0m' 

    total_length = len(matrix)
    
    test_ratio = 1 - train_ratio

    # assuming the test set size is always smaller than the train set size
    # we'll find out the indexes at which to split the matrix
    boundaries = list()
    i = 0

    #always terminates because test_ratio is guaranteed to be within 0 and 1
    while True:
        i += test_ratio
        # to account for inaccuracies when casting float to int
        if abs(i-1) < 0.1:
            break
        else:
            boundaries.append(int(i*total_length))

    # i know some slices might have one or two extra elements
    # but that doesn't affect the overall results.
    # this happens because sometimes it's not possible to divide the data matrix
    # into exact parts

    slices = np.split(matrix,boundaries)

    train_set =  list()
    test_set = list()

    assert test_slice >= 0 and test_slice < len(slices),'\033[31mERROR: You cannot have more rounds than there are slices in your data matrix! \033[0m'

    for i in xrange(len(slices)):
        if test_slice == i:
            for row in slices[i]:
                test_set.append(row)
        else:
            for row in slices[i]:
                train_set.append(row)

    train_set = np.array(train_set)
    test_set = np.array(test_set)


    return (train_set,test_set)
