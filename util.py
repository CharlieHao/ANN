import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


#csv files: first column, target variable. 1-724colum,pixel0 - pixel783, store 28x28 image 
#           pixel values, (0,255)
  

def get_pca_transformed_data():
	if not os.path.exists('/Users/zehaodong/research/digit_recognizer/large_files/train.csv'):
		print('There is no training dataset existing')
		print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
		print('Place train.csv in the folder large_files adjacent to the class folder')
		exit()

	df = pd.read_csv('/Users/zehaodong/research/digit_recognizer/large_files/train.csv')
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)	
	X = data[:,1:]
	Y = data[:,0].astype(np.int32)
	mu = np.mean(X,axis=0)
	X = X-mu
	pca = PCA()
	Z = pca.fit_transform(X)

	plot_cumulative_variance(pca)

	return Z,Y,pca,mu

def plot_cumulative_variance(pca):
	P = []
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p+P[-1])

	plt.plot(P)
	plt.show()

	return P

def get_normalized_data():
	if not os.path.exists('/Users/zehaodong/research/digit_recognizer/large_files/train.csv'):
		print('There is no traininf dataset existing')
		print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
		print('Place train.csv in the folder large_files adjacent to the class folder')
		exit()

	df = pd.read_csv('/Users/zehaodong/research/digit_recognizer/large_files/train.csv')
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)
	X = data[:,1:]
	mu = np.mean(X,axis=0)
	std = np.std(X,axis=0)
	np.place(std, std == 0, 1)
	X = (X-mu)/std
	Y = data[:,0]
	return X,Y

def forward(X,W,b):
	a = X.dot(W)+b
	expa = np.exp(a)
	return expa/expa.sum(axis = 1, keepdims = True)

def predict(Y):
	return np.argmax(Y,axis=1)

def error_rate(T,Y):
	prediction = predict(Y)
	return np.mean(prediction!=T)

def cost(T_ind,Y):
	return -(T_ind*np.log(Y)).sum()

def gradW(T_ind,Y,X):
	return X.T.dot(Y-T_ind)

def gradb(T_ind,Y):
	return (Y-T_ind).sum(axis=0)

def y2indicator(Y):
	N = len(Y)
	Y = Y.astype(np.int32)
	T = np.zeros((N,10))
	for n in range(N):
		T[n,Y[n]]=1
	return T

def benchmark():
	X,T = get_normalized_data()

	X_train,T_train = X[:-1000,],T[:-1000]
	X_test,T_test = X[-1000:,],T[-1000:]
	T_train_ind = y2indicator(T_train)
	T_test_ind = y2indicator(T_test)

	N,D = X_train.shape
	W = np.random.randn(D,10)/28
	b = np.zeros(10)

	costs_train = []
	costs_test = []
	error_rate_test = []

	learninf_rate = 0.00004
	reg = 0.01
	for n in range(500):
		Y_train = forward(X_train,W,b)
		
		c_train = cost(T_train_ind,Y_train)
		costs_train.append(c_train)

		Y_test = forward(X_test,W,b)
		c_test = cost(T_test_ind,Y_test)
		costs_test.append(c_test)

		er_test = error_rate(T_test,Y_test)
		error_rate_test.append(er_test)

		W -= learning_rate*(gradW(T_train_ind,Y_train,X_train)+reg*W)
		b -= learning_rate*(gradb(T_train_ind,Y_train)+reg*b)

		if n%10 == 0:
			print('Cost at iteration %d is %.6f'%(n,c_train))
			print('error rate:',er_test)

	Y = forward(X_test,W,b)
	print('final error rate:',error_rate(T_test_ind,Y))
	iters = len(costs_train)
	plt.plot(iters,costs_train,iters,costs_test)
	plt.show()
	plt.plot(error_rate_test)
	plt.show()


def benchmark_pca():
	X,T,_,_ = get_pca_transformed_data()
	X = X[:,:300]

	mu = X.mean(axis=0)
	std = X.std(axis = 0)
	X = (X-mu)/std

	X_train,T_train = X[:-1000,],T[:-1000]
	X_test,T_test = X[-1000:,],T[-1000:]
	T_train_ind = y2indicator(T_train)
	T_test_ind = y2indicator(T_test)

	N,D = X_train.shape
	W = np.random.randn(D,10)/28
	b = np.zeros(10)

	costs_train = []
	costs_test = []
	error_rate_test = []

	learninf_rate = 0.0001
	reg = 0.01
	for n in range(500):
		Y_train = forward(X_train,W,b)
		
		c_train = cost(T_train_ind,Y_train)
		costs_train.append(c_train)

		Y_test = forward(X_test,W,b)
		c_test = cost(T_test_ind,Y_test)
		costs_test.append(c_test)

		er_test = error_rate(T_test,Y_test)
		error_rate_test.append(er_test)

		W -= learning_rate*(gradW(T_train_ind,Y_train,X_train)+reg*W)
		b -= learning_rate*(gradb(T_train_ind,Y_train)+reg*b)

		if n%10 == 0:
			print('Cost at iteration %d is %.6f'%(n,c_train))
			print('error rate:',er_test)

	Y = forward(X_test,W,b)
	print('final error rate:',error_rate(T_test_ind,Y))
	iters = len(costs_train)
	plt.plot(iters,costs_train,iters,costs_test)
	plt.show()
	plt.plot(error_rate_test)
	plt.show()


if __name__ == '__main__':
	#benchmark()
	benchmark_pca()


def get_spiral_data():
	radius = np.linespace(1,10,100)
	theta = np.empty([6,100])
	for n in range(6):
		start_angle = np.pi*n/3
		end_angle = start_angle + np.pi/2
		points = np.linespace(start_angle,end_angle,100)
		theta[n] = points

	X1 = np.empty([6,100])
	X2 = np.empty([6,100])
	for n in range(6):
		X1[n] = radius*np.cos(theta[n])
		X2[n] = radius*np.sin(theta[n])

	X = np.empty([600,2])
	X[:,0]=X1.flatten()
	X[:,1]=X2.flatten()
	X += np.random.randn(600,2)*0.5

	Y = np.array([0]*100+[1]*100+[0]*100+[1]*100+[0]*100+[1]*100)
	return X,Y






