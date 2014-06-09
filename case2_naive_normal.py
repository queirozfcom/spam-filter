import numpy as np
import scipy.stats as stats
import sys

# lib eh a nossa biblioteca criada para este trabalho
import lib.naive_bayes as nb 
import lib.preprocessing as prep
import lib.validation as valid

from config.constants import *


def case2():

    accuracy_in_each_turn = list()

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

        accuracy = hits/(hits+misses)

        accuracy_in_each_turn.append(accuracy)

    mean_accuracy = np.mean(accuracy_in_each_turn)
    std_dev_accuracy = np.std(accuracy_in_each_turn)
    variance_accuracy = np.var(accuracy_in_each_turn)


    print ''
    print 'CASE 2 - TEN ATTRIBUTES - USING NORMAL MODEL'
    print 'MEAN_ACCURACY: '+str(mean_accuracy)
    print 'STD. DEV. OF ACCURACY: '+str(std_dev_accuracy)
    print 'VARIANCE OF ACCURACY: '+str(variance_accuracy)
    print ''

case2()    
