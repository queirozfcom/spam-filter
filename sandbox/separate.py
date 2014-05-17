import numpy as np
import random as random


m = np.loadtxt(open("../original_data/spambase.data","rb"),delimiter=',')

x = m[:,1:]
y = m[:,-1:]
xx = x; yy = y;
i = 0; tammax = int(0.2*xx.shape[0])+1;
tx = xx; ty = yy;
vx = 0; vy = 0;
while i < tammax:
	rand = max(0, int(random.random()*(tx.shape[0]-1)))
	print rand
	if i == 0:
		vx = tx[rand:rand+1]
		vy = ty[rand:rand+1]
	else:
		vx = np.append(vx, tx[rand:rand+1], 0)
		vy = np.append(vy, ty[rand:rand+1], 0)
	tx = np.delete(tx, rand, 0)
	ty = np.delete(ty, rand, 0)
	i = i+1;
