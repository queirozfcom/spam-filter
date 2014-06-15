# let's find out which attribute we should use for cases 1 and 2
#
# call this script like this: 
#
# python -m helpers.which_attribute_case_1.py (while in the root)

from case1_naive_normal import case1 as case1_normal

# def which attribute_bernoulli():

def which_attribute_utility_normal():

	highest_accuracy = 0.0
	highest_accuracy_index = 0

	accuracies = list()

	for index in xrange(0,57):
		accuracy = case1_normal(index,False,'utility')
		accuracies.append(accuracy)
		if( accuracy > highest_accuracy ):
			highest_accuracy = accuracy
			highest_accuracy_index = index

	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print "HIGHEST UTILITY: "+str(highest_accuracy)
	print "HIGHEST UTILITY INDEX: "+str(highest_accuracy_index)
	print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
	print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
	print "ALL VALUES:"

	for (index,accuracy) in enumerate(accuracies):
		print str(index)+"=>"+str(accuracy)

def which_attribute_accuracy_normal():

	highest_accuracy = 0.0
	highest_accuracy_index = 0

	accuracies = list()

	for index in xrange(0,57):
		accuracy = case1_normal(index,False,'accuracy')
		accuracies.append(accuracy)
		if( accuracy > highest_accuracy ):
			highest_accuracy = accuracy
			highest_accuracy_index = index

	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
	print "HIGHEST ACCURACY: "+str(highest_accuracy)
	print "HIGHEST ACCURACY INDEX: "+str(highest_accuracy_index)
	print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
	print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"

	for (index,accuracy) in enumerate(accuracies):
		print str(index)+"=>"+str(accuracy)
