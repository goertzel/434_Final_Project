def get_weights_perceptron(data, learning_rate = 0.01, error_threshold = 0.01):
	# Split Rows x Cols matrix into two matrices
	
	# X = Rows x Cols-1
	X = np.asmatrix(data[:, [i for i in xrange(data.shape[1]-1)]])
	
	# Y = Rows x 1
	Y = np.asmatrix(data[:, data.shape[1]-1])
	
	weights = np.zeros(X.shape[1], 1)
	deltas = np.zeros(X.shape[1], 1)
	
	while 1:
		for i in xrange(X.shape[0]):
			prediction = weights.T * X[i].T
			if np.matmul(Y[i], prediction) <=0:
				deltas -= np.matmul(Y[i], X[i]).transpose()
		deltas /= X.shape[0]
		weights -= learning_rate*deltas
		if np.linalg.norm(deltas) < error_threshold:
			return weights