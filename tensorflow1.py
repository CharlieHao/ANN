# tensorflow

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from util import get_normalized_data, y2indicator

def error_rate(y,t):
	return np.mean(y!=t)

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

    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / 28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)

    # step 2: use tensorflow to build construction and implement training
    # set placeholders, used in feed_dict later, equivalent to theano.tensor.variable 
    X = tf.placeholder(tf.float32,shape=(None,D),name='X')
    T = tf.placeholder(tf.float32,shape=(None,K),name='T' )
    # set tf.Variables, used for training parameters, equivalent to shared variable in theano
    # astype(np.float32), not tf.float32
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    # construct the structure
    # the output layer do not apply softmax 
    Z1 = tf.nn.relu(tf.matmul(X,W1)+b1)
    Z2 = tf.nn.relu(tf.matmul(Z1,W2)+b2)
    Y_a = tf.matmul(Z2,W3)+b3

    # set cost, train, prediction, init functions
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T,logits=Y_a))
    train_op = tf.train.RMSPropOptimizer(lr,decay=0.99,momentum=0.9).minimize(cost)
    prediction = tf.argmax(Y_a,1)
    init = tf.global_variables_initializer()

    # step 3: combine above
    costs = []
    with tf.Session() as session:
    	session.run(init)

    	for i in range(max_iter):
    		for n in range(n_batches):
    			Xbatch = Xtrain[n*batch_sz:(n*batch_sz + batch_sz),]
    			Ybatch = Ytrain_ind[n*batch_sz:(n*batch_sz + batch_sz),]

    			session.run(train_op,feed_dict={X:Xbatch,T:Ybatch})
    			if n%print_period == 0:
    				cost_val = session.run(cost,feed_dict={X:Xtest,T:Ytest_ind})
    				prediction_val = session.run(prediction,feed_dict={X:Xtest})
    				err = error_rate(prediction_val,Ytest)
    				print ('Cost / err at iteration i=%d, n=%d: %.3f / %.3f'%(i, n, cost_val, err))
    				costs.append(cost_val)

    plt.plot(costs)
    plt.show()



if __name__ == '__main__':
 	main()
























