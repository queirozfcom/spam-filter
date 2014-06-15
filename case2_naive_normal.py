import numpy as np
import scipy.stats as stats
import sys

# lib eh a nossa biblioteca criada para este trabalho
import lib.naive_bayes as nb 
import lib.preprocessing as prep
import lib.validation as valid

from lib.general_utils import trunc

from config.constants import *


def case2():

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
        #but now we take 10 attributes into consideration
        sample_means_word_spam = list()
        sample_means_word_ham = list()

        sample_variances_word_spam = list()
        sample_variances_word_ham = list()

        for attr_index in CASE_2_ATTRIBUTE_INDEXES:

            sample_means_word_spam.append(nb.take_mean_spam(train_set,attr_index,SPAM_ATTR_INDEX))
            sample_means_word_ham.append(nb.take_mean_ham(train_set,attr_index,SPAM_ATTR_INDEX))

            sample_variances_word_spam.append(nb.take_variance_spam(train_set,attr_index,SPAM_ATTR_INDEX))
            sample_variances_word_ham.append(nb.take_variance_ham(train_set,attr_index,SPAM_ATTR_INDEX))


        #sample standard deviations from sample variances
        sample_std_devs_spam = map(lambda x: x ** (1/2.0), sample_variances_word_spam)
        sample_std_devs_ham = map(lambda x: x ** (1/2.0), sample_variances_word_ham)

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

            # ou seja, o produto de todas as prob. condicionais das palavras dada a classe   
            # eu sei que ta meio confuso, mas se olhar com cuidado eh bonito fazer isso tudo numa linha soh! =)         
            product_of_all_conditional_probs_spam = reduce(lambda acc,cur: acc * stats.norm(sample_means_word_spam[cur], sample_std_devs_spam[cur]).pdf(row[CASE_2_ATTRIBUTE_INDEXES[cur]]) , xrange(10), 1)
            # nao precisa dividir pelo termo de normalizacao pois so queremos saber qual e o maior!
            posterior_spam = prior_spam * product_of_all_conditional_probs_spam


            product_of_all_conditional_probs_ham = reduce(lambda acc,cur: acc * stats.norm(sample_means_word_ham[cur], sample_std_devs_ham[cur]).pdf(row[CASE_2_ATTRIBUTE_INDEXES[cur]]) , xrange(10), 1)
            posterior_ham = prior_ham * product_of_all_conditional_probs_ham
    
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

    print ''
    print '============================================='
    print 'CASE 2 - TEN ATTRIBUTES - USING NORMAL MODEL'
    print '============================================='
    print ''
    print 'MEAN_ACCURACY: '+str(round(mean_accuracy,5))
    print 'STD. DEV. OF ACCURACY: '+str(round(std_dev_accuracy,5))
    print 'VARIANCE OF ACCURACY: '+str(round(variance_accuracy,5))
    print ''
    print 'MEAN_PRECISION: '+str(round(mean_precision,5))
    print 'STD. DEV. OF PRECISION: '+str(round(std_dev_precision,5))
    print 'VARIANCE OF PRECISION: '+str(round(variance_precision,5))
    print ''
    print 'MEAN_RECALL: '+str(round(mean_recall,5))
    print 'STD. DEV. OF RECALL: '+str(round(std_dev_recall,5))
    print 'VARIANCE OF RECALL: '+str(round(variance_recall,5))

case2()    
