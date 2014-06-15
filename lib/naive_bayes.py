# def take_p_char_spam(data_matrix,char_index,spam_index):
#     # probability of e-mail having a char given that it's spam
#     spam_having_char = len(filter(lambda x: x[char_index] != 0 and x[spam_index] == 1 ,data_matrix))
#     spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
     
#     return (float(spam_having_char)/float(spam))

def take_p_spam(data_matrix,spam_index):
    # probabilily that an e-mail is spam
    spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
    ham = len(filter(lambda x: x[spam_index] == 0 ,data_matrix))

    return float(spam)/( float(ham)+float(spam) )

def take_p_ham(data_matrix,spam_index):
    # probabilily that an e-mail is spam
    spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
    ham = len(filter(lambda x: x[spam_index] == 0 ,data_matrix))

    return float(ham)/( float(ham)+float(spam) )

def take_p_attribute_spam(data_matrix,attribute_index,spam_index):
    # probability of e-mail having an attribute given that it's spam
    # this works for all kinds of attributes because the dataset is normalized
    spam_having_attribute = len(filter(lambda x: x[attribute_index] != 0 and x[spam_index] == 1 ,data_matrix))
    spam = len(filter(lambda x: x[spam_index] == 1 ,data_matrix))
     
    return (float(spam_having_attribute)/float(spam))   

def take_p_attribute_ham(data_matrix,attribute_index,ham_index):
    # probability of e-mail having an attribute given that it's ham
    # this works for all kinds of attributes because the dataset is normalized
    ham_having_attribute = len(filter(lambda x: x[attribute_index] != 0 and x[ham_index] == 0 ,data_matrix))
    ham = len(filter(lambda x: x[ham_index] == 0 ,data_matrix))
     
    return (float(ham_having_attribute)/float(ham)) 


def take_p_attribute(data_matrix,attribute_index,spam_index):
    # probability of an e-mail having an attribute
    # this works for all kinds of attributes because the dataset is normalized
    
    p_attribute_spam = take_p_attribute_spam(data_matrix,attribute_index,spam_index)
    p_spam = take_p_spam(data_matrix,spam_index) 

    p_attribute_ham = take_p_attribute_ham(data_matrix,attribute_index,spam_index)
    p_ham = 1 - p_spam

    return (p_attribute_spam * p_spam) + (p_attribute_ham * p_ham) 

# def attribute_in_row(row,attribute_index):
#     return (row[attribute_index] != 0)

# def attribute_not_in_row(row,attribute_index):
#     return not attribute_in_row(row,attribute_index)

def take_mean_spam(data,attribute_index,spam_attr_index):
    return take_mean(data,attribute_index,spam_attr_index,1)

def take_mean_ham(data,attribute_index,spam_attr_index):
    return take_mean(data,attribute_index,spam_attr_index,0)

def take_mean(data,attribute_index,spam_attr_index,spam_val = None):

    assert spam_val is None or spam_val == 1 or spam_val == 0 , 'Invalid value for spam class'

    sum = 0.0
    count = 0.0

    for row in data:
        if spam_val is None:
            count += 1
            sum += row[attribute_index]
        else:
            if row[spam_attr_index] == spam_val:
                count += 1
                sum += row[attribute_index]

    return (sum/count)

def take_variance_spam(data,attribute_index,spam_attr_index):
    return take_variance(data,attribute_index,spam_attr_index,1)

def take_variance_ham(data,attribute_index,spam_attr_index):
    return take_variance(data,attribute_index,spam_attr_index,0)

def take_variance(data,attribute_index,spam_attr_index,spam_val= None):

    assert spam_val is None or spam_val == 1 or spam_val == 0 , 'Invalid value for spam class'

    sum = 0.0
    count = 0.0
    mean = take_mean(data,attribute_index,spam_attr_index,spam_val)

    for row in data:
        if spam_val is None:
            count += 1
            sum += (row[attribute_index] - mean) ** 2
        else:
            if row[spam_attr_index] == spam_val:
                count += 1
                sum += (row[attribute_index] - mean) ** 2
                

    return (sum/count)