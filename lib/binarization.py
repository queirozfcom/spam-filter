def binarize(matrix,target=57):
	for (i,row) in enumerate(matrix):
		for (j,elem) in enumerate(row):
			if elem != 0:
				matrix[i][j] = 1
				
	return matrix