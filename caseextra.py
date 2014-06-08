# -*- coding: cp1252 -*-
import numpy 
import sys

import lib.better_naive_bayes as bnb 
import lib.preprocessing as prep



from config.constants import *


def case3():

    # does not distinguish between emails where an attribute appears more than
    # once as opposed to those where said attribute appears only once.

    accuracy_in_each_turn = list()

    m = numpy.loadtxt(open("original_data/spambase.data","rb"),delimiter=',')

    for i in xrange(NUMBER_OF_ROUNDS):
        shuffled = numpy.random.permutation(m)

        train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO)

        total_p_spam_char = 0
        total_p_ham_char = 0

        for j in list(range(54)):

            # probability of having spam if the 10 attributes are set
            # is equal to the probabilities of having spam if each attribute is set
            # summed and divided by ten
            # but since what we want is to know what's more PROBABLE
            # a simple comparison of which is greater is enough

            p_char_spam = bnb.take_p_char_spam(train_set,j,SPAM_ATTR_INDEX)
            p_char = bnb.take_p_attribute(train_set,j,SPAM_ATTR_INDEX)
            p_ham_spam = 1 - p_char_spam

            # whichever is greater - that will be our prediction

            
            if j == 0:
                total_p_char_spam = p_char_spam
            else:
                total_p_char_spam *= p_char_spam
            total_p_ham_char += 1 - p_char_spam

        p_spam = bnb.take_p_spam(train_set,SPAM_ATTR_INDEX)
        total_p_spam_char = (total_p_char_spam * p_spam) / p_char
                    
        if total_p_spam_char > total_p_ham_char:
            guess = 1
        else:
            guess = 0

        hits = 0.0
        misses = 0.0

        # now we test the hypothesis against the test set
        for row in test_set:

            for j in list(range(54)):

                if(bnb.attribute_not_in_row(row,j)):
                    if(row[SPAM_ATTR_INDEX] != guess):
                        hits += (1.0/54)
                    else:
                        misses += (1.0/54)
                elif(bnb.attribute_in_row(row,j)):
                    if(row[SPAM_ATTR_INDEX] == guess):
                        hits += (1.0/54)
                    else:
                        misses += (1.0/54)

        accuracy = hits/(hits+misses)

        accuracy_in_each_turn.append(accuracy)

    mean_accuracy = numpy.mean(accuracy_in_each_turn)
    std_dev_accuracy = numpy.std(accuracy_in_each_turn)
    variance_accuracy = numpy.var(accuracy_in_each_turn)

    #com o uso do better_naive_bayes, o resultado foi de
    #0,3907310789(média de 10 aplicações do código), bem próximo aos (39.4%)
    #do dataset.

    print ''
    print 'CASE 3 - ALL ATTRIBUTES'
    print 'MEAN_ACCURACY: '+str(mean_accuracy)
    print 'POP. STD. DEV. OF ACCURACY: '+str(std_dev_accuracy)
    print 'POP. VARIANCE OF ACCURACY: '+str(variance_accuracy)
    print ''

case3()


    
