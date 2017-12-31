# theano
import numpy as np 
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
	return a*(a>0)

def main():
	# step1: get dataset and preprocess
    max_iter = 20 
    print_period = 10

    X, Y = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = int(N / batch_sz)

    M = 300
    K = 10
    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # step2: use theano varible to build structure and do training
    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init,'W1')
    b1 = theano.shared(b1_init,'b1')
    W2 = theano.shared(W2_init,'W2')
    b2 = theano.shared(b2_init,'b2')
    # structure expression
    thZ = relu(thX.dot(W1)+b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)
    #cost expression and prediction expression
    cost = -(thT*T.log(thY)).sum()+reg*((W1*W1).sum()+(b1*b1).sum()+(W2*W2).sum()+(b2*b2).sum())
    prediction = T.argmax(thY,axis = 1)
    # update expression
    W1_update = W1-lr*T.grad(cost,W1)
    b1_update = b1-lr*T.grad(cost,b1)
    W2_update = W2-lr*T.grad(cost,W2)
    b2_update = b2-lr*T.grad(cost,b2)
    # theano function for training,prediction and cost
    train = theano.function(
    	inputs = [thX,thT],
    	updates = [(W1,W1_update),(b1,b1_update),(W2,W2_update),(b2,b2_update)],
    )
    get_cost_and_prediction = theano.function(
    	inputs = [thX,thT],
    	outputs = [cost,prediction],
    )

    # step3: combine 1 and 2,use batch GD for optiization
    costs = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_cost_and_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print ("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                costs.append(cost_val)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
	main()


