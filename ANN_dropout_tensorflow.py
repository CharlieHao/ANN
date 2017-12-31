# This file are structure for a generalized Neural Network
# Use Object Oriented Programming to create object of Hiddenlayer and ANN_dropout
# Based on tensorflow 
# Object Hiddenlayer: 1.attributes W, b (form attribute params) 2.method forward
# Object ANN_dropout: 1. attribute hidden_layer_sizes and dropout_rates, both are list
#                     len(second) = len(first)+1
# Neural Network: activation function: in Hiddenlayer, we choose relu here
#                 layers and size of each layer: in ANN_dropout.

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from util import get_normalized_data
from sklearn.utils import shuffle

class Hiddenlayer(object):
	def __init__(self,M1,M2):
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1,M2)/np.sqrt(2.0/M1)
		b = np.zeros(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self,X):
		return tf.nn.relu(tf.matmul(X,self.W)+self.b)


class ANN(object):
	def __init__(self,hidden_layer_sizes,p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropout_rates = p_keep

	def fit(self,X,Y,learning_rate=1e-4,mu=0.9,decay=0.9,epochs=8,batch_size=100,split=True,show_figure=True):
		# there will be two more attributes: hidden_layers and params
		# hidden layers: list of Hiddenlayer objects
		# params: list of theano.shared varables, each are a parameter matrix or vector
		# step1: get training set and validation set
		X,Y = shuffle(X,Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int64)	
		if split:
			Xtrain,Ytrain = X[:-1000],Y[:-1000]
			Xvalid,Yvalid = X[-1000:],Y[-1000:]
		else:
			Xvalid,Yvalid = X,Y

		# step2 initialize hidden layers 
		N,D = X.shape
		K = len(set(Y))
		self.hidden_layers=[]
		M1=D
		for M2 in self.hidden_layer_sizes:
			h = Hiddenlayer(M1,M2)
			self.hidden_layers.append(h)
			M1 = M2
		W = np.random.randn(M1,K)/np.sqrt(M1)
		b = np.zeros(K)
		self.W = tf.Variable(W.astype(np.float32)) 
		self.b = tf.Variable(b.astype(np.float32)) 
		# collect the parameters
		self.params = [self.W,self.b]
		for h in self.hidden_layers:
			self.params += h.params

		# step3: in tensorflow to set up training operation and predict operation
		inputs = tf.placeholder(tf.float32,shape=(None,D),name='inputs')
		labels = tf.placeholder(tf.int64,shape=(None,),name='labels')

		logits = self.forward(inputs)

		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
		)
		train_op = tf.train.RMSPropOptimizer(learning_rate,decay=decay,momentum=mu).minimize(cost)
		# just momentum 
		# train_op = tf.tain.RMSPropOptimizer(learning_rate,momentum=mu)
		predict_op = self.predict(inputs)

		# step4: combine data and tensorflow structure
		n_batches = int(N/batch_size)
		costs=[]
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
				for n in range(n_batches):
					Xbatch = Xtrain[n*batch_size:n*batch_size+batch_size]
					Ybatch = Ytrain[n*batch_size:n*batch_size+batch_size]

					session.run(train_op,feed_dict={inputs:Xbatch,labels:Ybatch})
					if n%20 == 0:
						p = session.run(predict_op,feed_dict={inputs:Xvalid})
						c = session.run(cost,feed_dict={inputs:Xvalid,labels:Yvalid})
						err = error_rate(Yvalid,p)
						costs.append(c)
						print('cost/err at iteration i=%d n=%d is %.3f/%.3f'%(i,n,c,err))
		
		if show_figure:
			plt.plot(costs)
			plt.show()

	# forward function defines the stucture
	def forward(self,X):
		Z = X
		Z = tf.nn.dropout(Z,self.dropout_rates[0])
		for h,p in zip(self.hidden_layers,self.dropout_rates[1:]):
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z,p)
		return tf.matmul(Z,self.W)+self.b

	def predict(self,X):
		Y = self.forward(X)
		return tf.argmax(Y,1)

def error_rate(p,t):
	return np.mean(p!=t)

def relu(a):
	return a * (a>0)


def main():
	X,Y = get_normalized_data()


	ann = ANN([500,300],[0.8,0.5,0.5])
	ann.fit(X,Y,show_figure=True)


if __name__ == '__main__':
	main()









































