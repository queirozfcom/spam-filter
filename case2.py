import numpy 
import sys

import lib.naive_bayes as nb 
import lib.preprocessing as prep

from config.constants import *


def case2():

    # does not distinguish between emails where an attribute appears more than
    # once as opposed to those where said attribute appears only once.

    accuracy_in_each_turn = list()

    m = numpy.loadtxt(open("original_data/spambase.data","rb"),delimiter=',')

    for i in xrange(NUMBER_OF_ROUNDS):
        shuffled = numpy.random.permutation(m)

        train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO)

        total_p_spam_char = 0
        total_p_ham_char = 0

        for j in list(range(10)):

            # probability of having spam if the 10 attributes are set
            # is equal to the probabilities of having spam if each attribute is set
            # summed and divided by ten
            # but since what we want is to know what's more PROBABLE
            # a simple comparison of which is greater is enough

            p_char_spam = nb.take_p_char_spam(train_set,CASE_2_ATTRIBUTE_INDEX[j],SPAM_ATTR_INDEX)
            p_char = nb.take_p_attribute(train_set,CASE_2_ATTRIBUTE_INDEX[j],SPAM_ATTR_INDEX)

            # whichever is greater - that will be our prediction

            
            if j == 0:
                total_p_char_spam = p_char_spam
            else:
                total_p_char_spam *= p_char_spam
            total_p_ham_char += p_ham_char

        p_spam = nb.take_p_spam(train_set,SPAM_ATTR_INDEX)
        total_p_spam_char = (total_p_char_spam * p_spam) / p_char
        p_ham_char = 1 - total_p_spam_char
                    
        if total_p_spam_char > total_p_ham_char:
            guess = 1
        else:
            guess = 0

        hits = 0.0
        misses = 0.0

        # now we test the hypothesis against the test set
        for row in test_set:

            for j in list(range(10)):
        
                if(nb.attribute_not_in_row(row,CASE_2_ATTRIBUTE_INDEX[j])):
                    if(row[SPAM_ATTR_INDEX] != guess):
                        hits += 0.1
                    else:
                        misses += 0.1
                elif(nb.attribute_in_row(row,CASE_1_ATTRIBUTE_INDEX)):
                    if(row[SPAM_ATTR_INDEX] == guess):
                        hits += 0.1
                    else:
                        misses += 0.1

        accuracy = hits/(hits+misses)

        accuracy_in_each_turn.append(accuracy)

    mean_accuracy = numpy.mean(accuracy_in_each_turn)
    std_dev_accuracy = numpy.std(accuracy_in_each_turn)
    variance_accuracy = numpy.var(accuracy_in_each_turn)

    print ''
    print 'CASE 2 - TEN ATTRIBUTES'
    print 'MEAN_ACCURACY: '+str(mean_accuracy)
    print 'POP. STD. DEV. OF ACCURACY: '+str(std_dev_accuracy)
    print 'POP. VARIANCE OF ACCURACY: '+str(variance_accuracy)
    print ''

case2()


    
