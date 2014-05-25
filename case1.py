import numpy 
import sys

import lib.naive_bayes as nb 
import lib.preprocessing as prep

from config.constants import *


# does not distinguish between emails where an attribute appears more than
# once as opposed to those where said attribute appears only once.

accuracy_in_each_turn = list()

m = numpy.loadtxt(open("original_data/spambase.data","rb"),delimiter=',')

for i in xrange(NUMBER_OF_ROUNDS):
    shuffled = numpy.random.permutation(m)

    train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO)

    p_char_spam = nb.take_p_char_spam(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)
    p_spam = nb.take_p_spam(train_set,SPAM_ATTR_INDEX)
    p_char = nb.take_p_attribute(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

    # whichever is greater - that will be our prediction
    p_spam_char = (p_char_spam * p_spam) / p_char
    p_ham_char = 1 - p_spam_char

    if p_spam_char > p_ham_char:
        guess = 1
    else:
        guess = 0

    hits = 0
    misses = 0

    # now we test the hypothesis against the test set
    for row in test_set:
        if(row[CASE_1_ATTRIBUTE_INDEX] == 0):
            if(row[SPAM_ATTR_INDEX] != guess):
                hits += 1
            else:
                misses += 1
        elif(row[CASE_1_ATTRIBUTE_INDEX] != 0):
            if(row[SPAM_ATTR_INDEX] == guess):
                hits += 1
            else:
                misses += 1

    accuracy = float(hits)/float(hits+misses)

    accuracy_in_each_turn.append(accuracy)

mean_accuracy = numpy.mean(accuracy_in_each_turn)
std_dev_accuracy = numpy.std(accuracy_in_each_turn)
variance_accuracy = numpy.var(accuracy_in_each_turn)

print ''
print 'CASE 1 - ONE ATTRIBUTE ONLY'
print 'MEAN_ACCURACY: '+str(mean_accuracy)
print 'POP. STD. DEV. OF ACCURACY: '+str(std_dev_accuracy)
print 'POP. VARIANCE OF ACCURACY: '+str(variance_accuracy)
print ''




    
