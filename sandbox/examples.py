import numpy as np 

m = np.loadtxt(open("../original_data/spambase.data","rb"),delimiter=',')

x = m[:,1:]
y = m[:,-1:]

print y.shape