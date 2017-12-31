# In this file we compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent
# 2) batch gradient descent
# 3) stochastic gradient descent
#
# We use the PCA-transformed data to keep the dimensionality down (D=300)
# I've tailored this example so that the training time for each is feasible.
# So what we are really comparing is how quickly each type of GD can converge,
# (but not actually waiting for convergence) and what the cost looks like at
# each iteration.
#
# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_pca_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator


def main():
    X,T,_,_ = get_pca_transformed_data()
    X = X[:,:300]

    # normalize X first
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std

    print("Performing logistic regression...")
    X_train,T_train = X[:-1000,],T[:-1000]
    X_test,T_test = X[-1000:,],T[-1000:]
    T_train_ind = y2indicator(T_train)
    T_test_ind = y2indicator(T_test)

    N,D = X_train.shape
    W = np.random.randn(D,10)/28
    b = np.zeros(10)

    


    # 1. full
    costs_test = []
    learning_rate = 0.0001
    reg = 0.01
    t0 = datetime.now()
    # use datatime to track time
    for n in range(200):
        Y_train = forward(X_train,W,b)

        W -= learning_rate*(gradW(T_train_ind,Y_train,X_train)+reg*W)
        b -= learning_rate*(gradb(T_train_ind,Y_train)+reg*b)
        
        Y_test = forward(X_test,W,b)
        c_test = cost(T_test_ind,Y_test)
        costs_test.append(c_test)

        if n%10 == 0:
            er_test = error_rate(T_test,Y_test)
            print('Cost at iteration %d is %.6f'%(n,c_test))
            print('error rate:',er_test)
    Y = forward(X_test,W,b)
    print('final error rate:',error_rate(T_test,Y))
    print ("Elapsted time for full GD:", datetime.now() - t0)


    # 2. stochastic
    W = np.random.randn(D,10)/28
    b = np.zeros(10)
    costs_test_stochastic = []
    learning_rate = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(1): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(X_train, T_train_ind)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            t = tmpY[n,:].reshape(1,10)
            y = forward(x, W, b)

            W -= learning_rate*(gradW(t, y, x) + reg*W)
            b -= learning_rate*(gradb(t, y) + reg*b)

            Y_test = forward(X_test,W,b)
            c_test = cost(T_test_ind,Y_test)
            costs_test_stochastic.append(c_test)

            if n % int(N/2) <= 1:
                er_test = error_rate(T_test,Y_test)
                print('Cost at iteration %d is %.6f'%(n,c_test))
                print('error rate:',er_test)
    Y = forward(X_test, W, b)
    print('final error rate:',error_rate(T_test,Y))
    print ("Elapsted time for Stochastic GD:", datetime.now() - t0)



    # 3. batch
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    costs_test_batch = []
    learning_rate = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = int(N / batch_sz)

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(X_train, T_train_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            t = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = forward(x, W, b)

            W -= learning_rate*(gradW(t, y, x) + reg*W)
            b -= learning_rate*(gradb(t, y) -+reg*b)

            Y_test = forward(X_test,W,b)
            c_test = cost(T_test_ind,Y_test)
            costs_test_batch.append(c_test)

            if j%int(n_batches/2) <= 1:
                er_test = error_rate(T_test,Y_test)
                print('Cost at iteration %d is %.6f'%(j,c_test))
                print('error rate:',er_test)
    Y = forward(X_test, W, b)
    print('final error rate:',error_rate(T_test,Y))
    print ("Elapsted time for batch GD:", datetime.now() - t0)



    x1 = np.linspace(0, 1, len(costs_test))
    plt.plot(x1, costs_test, label="full")
    x2 = np.linspace(0, 1, len(costs_test_stochastic))
    plt.plot(x2, costs_test_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(costs_test_batch))
    plt.plot(x3, costs_test_batch, label="batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()