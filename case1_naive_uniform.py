# -*- coding: cp1252 -*-
import numpy as np
import scipy.stats as stats
import sys

# lib eh a nossa biblioteca criada para este trabalho
import lib.naive_bayes as nb 
import lib.preprocessing as prep
import lib.validation as valid

from config.constants import *


def case1():

    accuracy_in_each_turn = list()
    precision_in_each_turn = list()
    recall_in_each_turn = list()

    m = np.loadtxt(open("original_data/spambase.data","rb"),delimiter=',')

    shuffled = np.random.permutation(m)

    valid.validate_cross_validation(NUMBER_OF_ROUNDS,TRAIN_TEST_RATIO)

    # equiprobable priors
    prior_spam = 0.5
    prior_ham = 0.5


    for i in xrange(NUMBER_OF_ROUNDS):

        # we're using cross-validation so each iteration we take a different
        # slice of the data to serve as test set
        train_set,test_set = prep.split_sets(shuffled,TRAIN_TEST_RATIO,i)

        #parameter estimation
        sample_mean_word_spam = nb.take_mean_spam(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

        sample_mean_word_ham = nb.take_mean_ham(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

        sample_variance_word_spam = nb.take_variance_spam(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

        sample_variance_word_ham = nb.take_variance_ham(train_set,CASE_1_ATTRIBUTE_INDEX,SPAM_ATTR_INDEX)

        #sample standard deviations from sampe variance
        sample_std_dev_spam = sample_variance_word_spam ** (1/2.0)
        
        sample_std_dev_ham = sample_variance_word_ham ** (1/2.0) 

        hits = 0.0
        misses = 0.0

        #number of correctly evaluated instances as spam
        correctly_is_spam = 0.0

        #number of spam instances
        is_spam = 0.0

        #number of evaluated instances como spam
        guessed_spam = 0.0
        

        # now we test the hypothesis against the test set
        for row in test_set:
            
            # nao precisa dividir pelo termo de normalizacao pois so queremos saber qual e o maior!
            posterior_spam = prior_spam * stats.uniform(sample_mean_word_spam, sample_std_dev_spam).pdf(row[CASE_1_ATTRIBUTE_INDEX])

            posterior_ham = prior_ham * stats.uniform(sample_mean_word_ham, sample_std_dev_ham).pdf(row[CASE_1_ATTRIBUTE_INDEX])
    
            # whichever is greater - that will be our prediction
            if posterior_spam > posterior_ham:
                guess = 1
            else:
                guess = 0

            if(row[SPAM_ATTR_INDEX] == guess):
                hits += 1
            else:
                misses += 1

            if (row[SPAM_ATTR_INDEX] == 1) and (guess == 1):
                    correctly_is_spam += 1
                    is_spam += 1
                    guessed_spam += 1
            elif row[SPAM_ATTR_INDEX] == 1:
                is_spam += 1
            elif guess == 1:
                guessed_spam += 1

            

            

        #accuracy = number of correctly evaluated instances/
        #           number of instances
        #
        #precision = number of correctly evaluated instances as spam/
        #            number of spam instances
        #
        #recall = number of correctly evaluated instances as spam/
        #         number of evaluated instances como spam

        accuracy = hits/(hits+misses)
        precision = correctly_is_spam/is_spam
        recall = correctly_is_spam/guessed_spam

        accuracy_in_each_turn.append(accuracy)
        precision_in_each_turn.append(precision)
        recall_in_each_turn.append(recall)

    mean_accuracy = np.mean(accuracy_in_each_turn)
    std_dev_accuracy = np.std(accuracy_in_each_turn)
    variance_accuracy = np.var(accuracy_in_each_turn)

    mean_precision = np.mean(precision_in_each_turn)
    std_dev_precision = np.std(precision_in_each_turn)
    variance_precision = np.var(precision_in_each_turn)

    mean_recall = np.mean(recall_in_each_turn)
    std_dev_recall = np.std(recall_in_each_turn)
    variance_recall = np.var(recall_in_each_turn)

    print posterior_spam
    print posterior_ham


    print ''
    print 'CASE 1 - ONE ATTRIBUTE ONLY - USING UNIFORM MODEL'
    print 'MEAN_ACCURACY: '+str(mean_accuracy)
    print 'STD. DEV. OF ACCURACY: '+str(std_dev_accuracy)
    print 'VARIANCE OF ACCURACY: '+str(variance_accuracy)
    print ''
    print 'MEAN_PRECISION: '+str(mean_precision)
    print 'STD. DEV. OF PRECISION: '+str(std_dev_precision)
    print 'VARIANCE OF PRECISION: '+str(variance_precision)
    print ''
    print 'MEAN_RECALL: '+str(mean_recall)
    print 'STD. DEV. OF RECALL: '+str(std_dev_recall)
    print 'VARIANCE OF RECALL: '+str(variance_recall)

case1()




    
