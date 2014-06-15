from config.constants import *
import math

def normalize(matrix,target_attribute_index):
    num_cols = matrix.shape[1]

    for col_idx in xrange(num_cols):

        # we don't want to normalize the target attribute
        if(col_idx != target_attribute_index):
            column = matrix[:,col_idx]
            feature_max = max(column)
            feature_min = min(column)

            # seach element in column gets replaced by its normalized version
            # using the maximum and minimum values, according to the following expression:
            #
            # new_value = (old_value - maximum_value) / (maximum_value - minimum_value)
            #
            normalized_column = map(lambda element: round( ( (element-feature_min)/(feature_max-feature_min) ),5), column)

            for idx,row in enumerate(matrix):
                row[col_idx] = normalized_column[idx]


    return matrix
