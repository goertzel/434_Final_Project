class Perceptron:
	def __init__(self, training_data, learning_rate = 0.01, error_threshold = 0.01):
		self.training_data = training_data
		self.learning_rate = learning_rate
		self.error_threshold = error_threshold
		
		self.weights = self.learn_weights()
		self.predictions = self.predict()
		

	def learn_weights():
		# Split Rows x Cols matrix into two matrices
		# X = Rows x Cols-1
		# Y = Rows x 1
		# W = Cols-1 x 1
		# D = Cols-1 x 1
	
		X = np.asmatrix(self.data[:, [i for i in xrange(self.data.shape[1]-1)]])
		Y = np.asmatrix(self.data[:, self.data.shape[1]-1])
		W = np.zeros(X.shape[1], 1)
		D = np.zeros(X.shape[1], 1)
		
		while 1:
			for i in xrange(X.shape[0]):
				prediction = W.T * X[i].T
				if np.matmul(Y[i], prediction) <=0:
					D -= np.matmul(Y[i], X[i]).transpose()
			D /= X.shape[0]
			W -= self.learning_rate*D
			if np.linalg.norm(D) < self.error_threshold:
				return W
				
	def predict():
		X = np.asmatrix(self.data[:, [i for i in xrange(self.data.shape[1]-1)]])
		return X * self.weights
		
def write_predictions(predictions):
	f = open("pred.csv", 'w')
	map(lambda p: f.write(str(p)+'\n'), [p[0] for p in predictions])
	f.close()