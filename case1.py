import numpy 
import sys

import lib.naive_bayes as nb 
import lib.preprocessing as prep

from config.constants import *


# does not distinguish between emails where an attribute appears more than
# once as opposed to those wher the attribute appears only once.

m = numpy.loadtxt(open("original_data/spambase.data","rb"),delimiter=',')

for i in xrange(NUMBER_OF_ROUNDS):
	shuffled = numpy.random.permutation(m)

	train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO)

	p_char_spam = nb.take_p_char_spam(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)
	p_spam = nb.take_p_spam(train_set,SPAM_ATTR_INDEX)
	p_char = nb.take_p_attribute(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

	p_spam_char = (p_char_spam * p_spam) / p_char

	print p_char_spam
