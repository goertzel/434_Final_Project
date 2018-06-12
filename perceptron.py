import numpy as np
import random

class Perceptron:
	def __init__(self, training_data, learning_rate = 0.01, error_threshold = 1.0):
		self.training_data = training_data
		self.learning_rate = learning_rate
		self.error_threshold = error_threshold
		
		self.weights = self.learn_weights()
		self.predictions = self.predict()

	def learn_weights(self):
		# Split Rows x Cols matrix into two matrices
		# X = Rows x Cols-1
		# Y = Rows x 1
		# W = Cols-1 x 1
		# D = Cols-1 x 1
	
		X = np.asmatrix(self.training_data[:, [i for i in xrange(self.training_data.shape[1]-1)]])
		Y = np.matrix([[1] if r[0] > 0 else [-1] for r in np.matrix(self.training_data[:, self.training_data.shape[1]-1])])
		W = np.zeros((X.shape[1], 1))
		
		epoch = 0
		while 1:
			print epoch
			epoch += 1
			D = np.zeros((X.shape[1], 1))
			for i in xrange(X.shape[0]):
				prediction = np.matmul(W.T, X[i].T)
				if np.matmul(Y[i], prediction) <=0:
					D -= np.matmul(Y[i], X[i]).transpose()
			D /= X.shape[0]
			W -= self.learning_rate*D
			if np.linalg.norm(D) < self.error_threshold:
				return W
				
	def predict(self):
		X = np.asmatrix(self.training_data[:, [i for i in xrange(self.training_data.shape[1]-1)]])
		likelihoods = X * self.weights
		classifications = np.matrix([[1 if likelihoods[i,j] > 0 else 0 for i in range(likelihoods.shape[0])] for j in range(likelihoods.shape[1])]).T
		
		return np.hstack((likelihoods, classifications))
