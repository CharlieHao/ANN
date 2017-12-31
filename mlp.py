import numpy as np 

def forward(X,W1,b1,W2,b2):
	# hidden units activation function is relu
	Z = X.dot(W1)+b1
	Z[Z<0] = 0

	A = Z.dot(W2)+b2
	expa = np.exp(A)
	Y = expa/expa.sum(axis=1,keepdims =True)
	return Y,Z

def derivative_w2(Z,T,Y):
	return Z.T.dot(Y-T)

def derivative_b2(T,Y):
	return (Y-T).sum(axis=0)

def derivative_w1(X,Z,T,Y,W2):
	dZ = (Y-T).dot(W2.T)*(Z>0)
	return X.T.dot(dZ)

def derivative_b1(Z,T,Y,W2):
	dZ = (Y-T).dot(W2.T)*(Z>0)
	return dZ.sum(axis=0)


