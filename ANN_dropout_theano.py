# This file are structure for a generalized Neural Network
# Use Object Oriented Programming to create object of Hiddenlayer and ANN_dropout
# Based on theano 
# Object Hiddenlayer: 1.attributes W, b (form attribute params) 2.method forward
# Object ANN_dropout: 1. attribute hidden_layer_sizes and dropout_rates, both are list
#                     len(second) = len(first)+1
# Neural Network: activation function: in Hiddenlayer, we choose relu here
#                 layers and size of each layer: in ANN_dropout.


import numpy as np 
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 

from theano.tensor.shared_randomstreams import RandomStreams 
from sklearn.utils import shuffle
from util import get_normalized_data

class Hiddenlayer(object):
	def __init__(self,M1,M2,an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W = np.random.randn(M1,M2)/np.sqrt(2/M1)
		b = np.zeros(M2)
		self.W = theano.shared(W,'W_%d'%(self.id))
		self.b = theano.shared(b,'b_%d'%(self.id))
		self.params = [self.W, self.b]

	def forward(self,X):
		return T.nnet.relu(X.dot(self.W)+self.b)


class ANN_dropout(object):
	def __init__(self,hidden_layer_sizes,p_keep):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropout_rates = p_keep

	def fit(self,X,Y,learninf_rate=1e-4,mu=0.9,decay=0.9,epochs=8,batch_size=100,show_figue=False):
		# there will be two more attributes: hidden_layers and params
		# hidden layers: list of Hiddenlayer objects
		# params: list of theano.shared varables, each are a parameter matrix or vector
		# step1: get training set and validation set
		X,Y = shuffle(X,Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)
		Xtrain,Ytrain = X[:-1000],Y[:-1000]
		Xvalid,Yvalid = X[-1000:],Y[-1000:]

		self.rng = RandomStreams()
		# step2 initialize hidden layers and params
		# hidden_layers
		N,D = Xtrain.shape
		K = len(set(Y))
		M1 = D
		self.hidden_layers = []
		count = 0
		for M2 in self.hidden_layer_sizes:
			h = Hiddenlayer(M1,M2,count)
			count += 1
			M1 = M2
			self.hidden_layers.append(h)
		# params
		W = np.random.randn(M1,K)/np.sqrt(M1)
		b = np.zeros(K)
		self.W = theano.shared(W,'W_outlayer')
		self.b = theano.shared(b,'b_outlayer')
		self.params = [self.W,self.b]
		for h in self.hidden_layers:
			self.params += h.params

		# step 3: build the update function and cost_predict function
		thX = T.matrix('X')
		thT = T.ivector('T')
		
		Y = self.forward_train(thX)
		cost = -T.mean(T.log(Y[T.arange(thT.shape[0]),thT]))
		grads = T.grad(cost,self.params)
		# momentum initialization
		dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]
		# RMSprop initialization
		caches = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]
		# updates formation, in RMSprop, epsilon = 1e-10
		caches_new = [decay*c + (1-decay)*g*g for p,c,g in zip(self.params,caches,grads)]
		dparams_new = [mu*dp - learninf_rate*g/T.sqrt(c_new+1e-10) for p,c_new,g,dp in zip(self.params,caches_new,grads,dparams)]
		# updates
		updates = [
			(c,c_new) for c,c_new in zip(caches,caches_new)
		]+[
			(dp,dp_new) for dp,dp_new in zip(dparams,dparams_new)
		]+[
			(p,p+dp_new) for p,dp_new in zip(self.params,dparams_new)
		]

		train_op = theano.function(
			inputs = [thX,thT],
			updates = updates
		)
		# evaluation and cost
		Y_predict = self.forward_predict(thX)
		cost_predict = -T.mean(T.log(Y_predict[T.arange(thT.shape[0]),thT]))
		T_predict = self.predict(thX)
		cost_predict_op = theano.function(
			inputs = [thX,thT],
			outputs = [cost_predict,T_predict])

		#step4: apply theano strcuture in data and choose batch GD
		# actually, ooptimization method: batch GD + momentum + RMSprop
		n_batches = int(N/batch_size)
		costs=[]
		for i in range(epochs):
			Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
			for n in range(n_batches):
				Xbatch = Xtrain[n*batch_size:n*batch_size+batch_size]
				Ybatch = Ytrain[n*batch_size:n*batch_size+batch_size]

				train_op(Xbatch,Ybatch)
				if n%20 == 0:
					c,p = cost_predict_op(Xvalid,Yvalid)
					err = error_rate(Yvalid,p)
					costs.append(c)
					print('cost/err at iteration i=%d n=%d is %.3f/%.3f'%(i,n,c,err))
		if show_figue:
			plt.plot(costs)
			plt.show()

	# forward function defines the structure of this Neural Network
	def forward_train(self,X):
		Z = X
		for h,p in zip(self.hidden_layers,self.dropout_rates[:-1]):
			mask = self.rng.binomial(n=1,p=p,size=Z.shape)
			Z = Z*mask
			Z = h.forward(Z)
		mask = self.rng.binomial(n=1,p=self.dropout_rates[-1],size=Z.shape)
		Z = Z*mask
		return T.nnet.softmax(Z.dot(self.W)+self.b)

	def forward_predict(self,X):
		Z = X
		for h,p in zip(self.hidden_layers,self.dropout_rates[:-1]):
			Z = h.forward(p*Z)
		return T.nnet.softmax((Z*self.dropout_rates[-1]).dot(self.W)+self.b)

	def predict(self,X):
		Y = self.forward_predict(X)
		return T.argmax(Y,axis=1)

def error_rate(y,t):
	return np.mean(y!=t)

def relu(a):
	return a * (a>0)




def main():
	X,Y = get_normalized_data()

	ann = ANN_dropout([500,300],[0.8,0.5,0.5])
	ann.fit(X,Y,show_figue=True)


if __name__ == '__main__':
	main()




















		






























