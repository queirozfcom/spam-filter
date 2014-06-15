import numpy as np
import scipy.stats as stats
import sys

# lib eh a nossa biblioteca criada para este trabalho
import lib.naive_bayes as nb 
import lib.preprocessing as prep
import lib.validation as valid

from config.constants import *


def case2(indexes=CASE_2_ATTRIBUTE_INDEXES,output=True):

    accuracy_in_each_turn = list()

    precision_in_each_turn_spam = list()
    recall_in_each_turn_spam = list()

    precision_in_each_turn_ham = list()
    recall_in_each_turn_ham = list()

    m = np.loadtxt(open("resources/normalized_data.csv","rb"),delimiter=',')

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

        for attr_index in indexes:

            sample_means_word_spam.append(nb.take_mean_spam(train_set,attr_index,SPAM_ATTR_INDEX))
            sample_means_word_ham.append(nb.take_mean_ham(train_set,attr_index,SPAM_ATTR_INDEX))

            sample_variances_word_spam.append(nb.take_variance_spam(train_set,attr_index,SPAM_ATTR_INDEX))
            sample_variances_word_ham.append(nb.take_variance_ham(train_set,attr_index,SPAM_ATTR_INDEX))


        #sample standard deviations from sample variances
        sample_std_devs_spam = map(lambda x: x ** (1/2.0), sample_variances_word_spam)
        sample_std_devs_ham = map(lambda x: x ** (1/2.0), sample_variances_word_ham)

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

            # ou seja, o produto de todas as prob. condicionais das palavras dada a classe   
            # eu sei que ta meio confuso, mas se olhar com cuidado eh bonito fazer isso tudo numa linha soh! =)         
            product_of_all_conditional_probs_spam = reduce(lambda acc,cur: acc * stats.norm(sample_means_word_spam[cur], sample_std_devs_spam[cur]).pdf(row[indexes[cur]]) , xrange(10), 1)
            # nao precisa dividir pelo termo de normalizacao pois so queremos saber qual e o maior!
            posterior_spam = prior_spam * product_of_all_conditional_probs_spam


            product_of_all_conditional_probs_ham = reduce(lambda acc,cur: acc * stats.norm(sample_means_word_ham[cur], sample_std_devs_ham[cur]).pdf(row[indexes[cur]]) , xrange(10), 1)
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

            # we'll use these to calculate metrics
            if (row[SPAM_ATTR_INDEX] == 1 ):
                is_spam += 1
                
                if guess == 1:
                    guessed_spam += 1
                    correctly_is_spam += 1
                else:
                    guessed_ham += 1
            else:
                is_ham += 1

                if guess == 1:
                    guessed_spam += 1
                else:
                    guessed_ham += 1
                    correctly_is_ham += 1

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


    # calculation of means for each metric at the end

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
        print 'CASE 2 - TEN ATTRIBUTES - USING NORMAL MODEL'
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