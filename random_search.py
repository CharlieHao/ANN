#random search

from theano_ann import ANN
from util import get_spiral_data, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def random_search():
	#get the dataset
	X,T = get_spiral_data()
	X,T = shuffle(X,T)
	Ntrain = int(0.7*(len(T)))
	Xtrain,Ttrain = X[:Ntrain],T[:Ntrain]
	Xtest,Ttest = X[Ntrain:],T[Ntrain:]

	#the starting value of hyperpaameters
	#ps: hyperparameter x that should have a positive value we use the ln_x 
	M = 20
	nHidden = 2
	log_lr = -4
	log_l2 = -2 
    max_tries = 30

    # loops for searching the parameter combinition with best performence
    # by random walk
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for _ in range(max_tries):
    	model = ANN([M]*nHidden)
    	model.fit(
    		Xtrain,Ttrain,
    		learning_rate=10**log_lr, reg=10**log_l2,
    		mu=0.99, epochs=3000, show_fig=False
    		)
    	validation_accuracy = model.score(Xtest,Ttest)
    	train_accuracy = model.score(Xtrain,Ttrain)
    	print(
    		"validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" %
    		(validation_accuracy, train_accuracy, hls, lr, l2)
    		)
    	if validation_accuracy>best_validation_rate: #update of best or final choice
    		best_validation_rate = validation_accuracy
    		best_nHidden = nHidden
    		best_M = M
    		best_lr = lr
    		best_l2 = l2

    	#do the random walk in hyperparameter space
    	nHidden = best_nHidden+np.random.randint(-1,2)#-1,0,1
    	nHidden = max(1,nHidden)
    	M = best_M + 10 * np.random.randint(-1,2)
    	M = max(M,10)
    	log_lr = best_lr + np.random.randint(-1,2)
    	log_l2 = best_l2 + np.randint(-1,2)
    
    print("Best validation_accuracy:", best_validation_rate)
    print("Best settings:")
    print("best_M:", best_M)
    print("best_nHidden:", best_nHidden)
    print("learning_rate:", best_lr)
    print("l2:", best_l2)


if __name__ == '__main__':
	random_search()











