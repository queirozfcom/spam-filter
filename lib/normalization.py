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

            normalized_column = map(lambda element: round( ( (element-feature_min)/(feature_max-feature_min) ),5), column)

            for idx,row in enumerate(matrix):
                row[col_idx] = normalized_column[idx]


    return matrix
