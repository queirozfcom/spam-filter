import numpy as np
import sys

import lib.naive_bayes as nb 
import lib.preprocessing as prep
import lib.validation as valid

from config.constants import *


def case2(indexes=CASE_2_ATTRIBUTE_INDEXES,output=True):

    # does not distinguish between emails where an attribute appears more than
    # once as opposed to those where said attribute appears only once.

    # note that we're not using equiprobabilities for the priors.
    # we'll simulate the population priors using the sample priors.

    accuracy_in_each_turn = list()

    precision_in_each_turn_spam = list()
    recall_in_each_turn_spam = list()

    precision_in_each_turn_ham = list()
    recall_in_each_turn_ham = list()

    m = np.loadtxt(open("resources/normalized_data.csv","rb"),delimiter=',')

    shuffled = np.random.permutation(m)

    valid.validate_cross_validation(NUMBER_OF_ROUNDS,TRAIN_TEST_RATIO)

    for i in xrange(NUMBER_OF_ROUNDS):

        train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO,i)

        numerator_product_spam = reduce(lambda acc,elem: acc * nb.take_p_attribute_spam(train_set,indexes[elem],SPAM_ATTR_INDEX) ,xrange(10),1)
        numerator_product_ham = reduce(lambda acc,elem: acc * nb.take_p_attribute_ham(train_set,indexes[elem],SPAM_ATTR_INDEX) ,xrange(10),1)

        prior_spam = nb.take_p_spam(train_set,SPAM_ATTR_INDEX)
        prior_ham = nb.take_p_ham(train_set,SPAM_ATTR_INDEX)


        # estas sao as estimativas finais levando em conta os 10 atributos
        # nao precisa dividir pelo p_attribute pois so estamos interessados em saber
        # qual o maior. Dividir por uma constante dos dois lados nao afeta o resultado.
        p_spam_attribute = numerator_product_spam * prior_spam
        p_ham_attribute = numerator_product_ham * prior_ham
        
        # whichever is greater - that will be our prediction

        if p_spam_attribute > p_ham_attribute:
            guess = 1
        else:
            guess = 0

        hits = 0.0
        misses = 0.0

        #number of instances correctly evaluated as spam
        correctly_is_spam = 0.0

        #total number of spam instances
        is_spam = 0.0

        #total number of instances evaluated as spam
        guessed_spam = 0.0

        #number of instances correctly evaluated as ham
        correctly_is_ham = 0.0

        #total number of ham instances
        is_ham = 0.0

        #total number of instances evaluated as ham
        guessed_ham = 0.0

        # now we test the hypothesis against the test set
        for row in test_set:

            for j in list(range(10)):
                
                # if the attribute isn't there, then our actual guess is the opposite of the calculated guess
                if (row[indexes[j]] == 0) and (guess == 0):
                    actual_guess = 1
                elif(row[indexes[j]] == 0) and (guess == 1):
                    actual_guess = 0
                else:
                    actual_guess = guess

                # all these values should in theory be divided by the number
                # of features we're using, but we just want to know which
                # of the two (spam or ham) is more likely so even though the result
                # will be skewed by a constant, both classes will be multiplied
                # by that constant hence the comparison between the two is
                # still valid
                if (row[SPAM_ATTR_INDEX] == 0) and (actual_guess == 0):
                    is_ham += 1
                    guessed_ham += 1
                    correctly_is_ham += 1
                    hits += 1
                elif (row[SPAM_ATTR_INDEX] == 0) and (actual_guess == 1):
                    is_ham += 1
                    guessed_spam += 1
                    misses += 1
                elif (row[SPAM_ATTR_INDEX] == 1) and (actual_guess == 0):
                    is_spam += 1
                    guessed_ham += 1
                    misses += 1
                elif (row[SPAM_ATTR_INDEX] == 1) and (actual_guess == 1):
                    is_spam += 1
                    guessed_spam += 1
                    correctly_is_spam += 1
                    hits += 1


        #accuracy = number of correctly evaluated instances/
        #           number of instances
        #
        #
        accuracy = hits/(hits+misses)


        #precision_spam = number of correctly evaluated instances as spam/
        #            number of spam instances
        #
        #
        # in order to avoid divisions by zero in case nothing was found
        if(is_spam == 0):
            precision_spam = 0
        else:
            precision_spam = correctly_is_spam/is_spam

        #recall_spam = number of correctly evaluated instances as spam/
        #         number of evaluated instances como spam
        #
        #
        # in order to avoid divisions by zero in case nothing was found
        if(guessed_spam == 0):
            recall_spam = 0
        else:
            recall_spam = correctly_is_spam/guessed_spam

        #precision_ham = number of correctly evaluated instances as ham/
        #            number of ham instances
        #
        #
        # in order to avoid divisions by zero in case nothing was found
        if(is_ham == 0):
            precision_ham = 0
        else:
            precision_ham = correctly_is_ham/is_ham

        #recall_ham = number of correctly evaluated instances as ham/
        #         number of evaluated instances como ham
        #
        #
        # in order to avoid divisions by zero in case nothing was found
        if(guessed_ham == 0):
            recall_ham = 0
        else:
            recall_ham = correctly_is_ham/guessed_ham

        accuracy_in_each_turn.append(accuracy)

        precision_in_each_turn_spam.append(precision_spam)
        recall_in_each_turn_spam.append(recall_spam)

        precision_in_each_turn_ham.append(precision_ham)
        recall_in_each_turn_ham.append(recall_ham)            


    mean_accuracy = np.mean(accuracy_in_each_turn)
    std_dev_accuracy = np.std(accuracy_in_each_turn)
    variance_accuracy = np.var(accuracy_in_each_turn)

    mean_precision_spam = np.mean(precision_in_each_turn_spam)
    std_dev_precision_spam = np.std(precision_in_each_turn_spam)
    variance_precision_spam = np.var(precision_in_each_turn_spam)

    mean_recall_spam = np.mean(recall_in_each_turn_spam)
    std_dev_recall_spam = np.std(recall_in_each_turn_spam)
    variance_recall_spam = np.var(recall_in_each_turn_spam)

    mean_precision_ham = np.mean(precision_in_each_turn_ham)
    std_dev_precision_ham = np.std(precision_in_each_turn_ham)
    variance_precision_ham = np.var(precision_in_each_turn_ham)

    mean_recall_ham = np.mean(recall_in_each_turn_ham)
    std_dev_recall_ham = np.std(recall_in_each_turn_ham)
    variance_recall_ham = np.var(recall_in_each_turn_ham)


    if output:
        print "\033[1;32m"
        print '============================================='
        print 'CASE 2 - TEN ATTRIBUTES - USING BERNOULLI MODEL'
        print '============================================='
        print "\033[00m"
        print 'MEAN ACCURACY: '+str(round(mean_accuracy,5))
        print 'STD. DEV. OF ACCURACY: '+str(round(std_dev_accuracy,5))
        print 'VARIANCE OF ACCURACY: '+str(round(variance_accuracy,8))
        print ''
        print 'MEAN PRECISION FOR SPAM: '+str(round(mean_precision_spam,5))
        print 'STD. DEV. OF PRECISION FOR SPAM: '+str(round(std_dev_precision_spam,5))
        print 'VARIANCE OF PRECISION FOR SPAM: '+str(round(variance_precision_spam,8))
        print ''
        print 'MEAN RECALL FOR SPAM: '+str(round(mean_recall_spam,5))
        print 'STD. DEV. OF RECALL FOR SPAM: '+str(round(std_dev_recall_spam,5))
        print 'VARIANCE OF RECALL FOR SPAM: '+str(round(variance_recall_spam,8))
        print ''
        print 'MEAN PRECISION FOR HAM: '+str(round(mean_precision_ham,5))
        print 'STD. DEV. OF PRECISION FOR HAM: '+str(round(std_dev_precision_ham,5))
        print 'VARIANCE OF PRECISION FOR HAM: '+str(round(variance_precision_ham,8))
        print ''
        print 'MEAN RECALL FOR HAM: '+str(round(mean_recall_ham,5))
        print 'STD. DEV. OF RECALL FOR HAM: '+str(round(std_dev_recall_ham,5))
        print 'VARIANCE OF RECALL FOR HAM: '+str(round(variance_recall_ham,8))

case2()


    
