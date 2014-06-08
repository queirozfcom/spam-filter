import numpy

prob_attr_spam_list_test = []
prob_attr_ham_list_test = []

prob_attr_spam_list_validation = []
prob_attr_ham_list_validation = []

def init_spam_list_test():
	print 'ok 1'

def init_ham_list_test():
	print 'ok 2'

def init_spam_list_validation():
	print 'ok 3'

def init_ham_list_validation():
	print 'ok 4'


def take_p_char_spam(data_matrix,char_index,spam_index):
	# probability of e-mail having a char given that it's spam
	spam_having_char = filter(lambda x: x[char_index] != 0 and x[spam_index] == 1 ,data_matrix)
        #spam_having_char = sum(spam_having_char[char_index])
	col_totals = [ sum(x) for x in zip(*spam_having_char) ]

	if (col_totals == []): return 0.0
	spam_having_char = col_totals[char_index]
	
	spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
	
	return (float(spam_having_char)/float(spam))

def take_p_spam(data_matrix,spam_index):
	# probabilily that an e-mail is spam
	spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
	ham = len(filter(lambda x: x[spam_index] == 0 ,data_matrix))

	return (float(spam)/float(ham))

def take_p_attribute_spam(data_matrix,attribute_index,spam_index):
	# probability of e-mail having an attribute given that it's spam
	# maybe this works for all kinds of attributes.
	spam_having_attribute = filter(lambda x: x[attribute_index] != 0 and x[spam_index] == 1 ,data_matrix)
	#spam_having_attribute = sum(spam_having_attribute[attribute_index])
	col_totals = [ sum(x) for x in zip(*spam_having_attribute) ]

	if (col_totals == []): return 0.0
	spam_having_attribute = col_totals[attribute_index]
	
	spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
	 
	return (float(spam_having_attribute)/float(spam))	

def take_p_attribute_ham(data_matrix,attribute_index,ham_index):
	# probability of e-mail having an attribute given that it's ham
	# maybe this works for all kinds of attributes.
	ham_having_attribute = filter(lambda x: x[attribute_index] != 0 and x[ham_index] == 1 ,data_matrix)

	col_totals = [ sum(x) for x in zip(*ham_having_attribute) ]

	if (col_totals == []): return 0.0
	ham_having_attribute = col_totals[attribute_index]

	ham = len(filter(lambda x: x[ham_index] == 0 ,data_matrix))

	return (float(ham_having_attribute)/float(ham))	


def	take_p_attribute(data_matrix,attribute_index,spam_index):
	# probability of an e-mail having an attribute
	# i know that i've already created a function for chars but I don't yet know whether 
	# i can treat all kinds of attributes (words,chars,capitals) the same
	
	p_attribute_spam = take_p_attribute_spam(data_matrix,attribute_index,spam_index)
	p_spam = take_p_spam(data_matrix,spam_index) 

	p_attribute_ham = take_p_attribute_ham(data_matrix,attribute_index,spam_index)
	p_ham = 1 - p_spam

	return (p_attribute_spam * p_spam) + (p_attribute_ham * p_ham) 

def attribute_in_row(row,attribute_index):
	return (row[attribute_index] != 0)

def attribute_not_in_row(row,attribute_index):
	return not attribute_in_row(row,attribute_index)

init_spam_list_test()
init_ham_list_test()
init_spam_list_validation()
init_ham_list_validation()
