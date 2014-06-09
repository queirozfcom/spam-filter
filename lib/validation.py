def validate_cross_validation(rounds,train_to_test_ratio):
	# the number of turns must be exactly equal to the number
	# of "parts" you'll split your data into

	res = rounds * (1 - train_to_test_ratio)

	# comparando floats na marra.
	assert ( abs(res - 1) < 0.0001 )
