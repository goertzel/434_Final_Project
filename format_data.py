import numpy as np

# np.set_printoptions(precision=4, suppress=True)

# Reads in a subject's data
def read_in_data(dir, subject):
	data_file = open(dir+'/'+'Subject_'+str(subject)+'.csv')
	X = np.genfromtxt( data_file , delimiter=',')
	data_file.close()
	ind_file = open(dir+'/'+'list_'+str(subject)+'.csv')
	I = np.asmatrix(np.genfromtxt( ind_file , delimiter=',')).T
	ind_file.close()
	X = np.hstack( (I, np.delete(X, 0, 1)) ) # Remove Timestamp and Prepend Indices 
	return X

# Merges morning, afternoon, evening, and night into a single feature
def join_time_cols(X):
	new_col = []
	for xi in X:
		# where = np.where( np.array(xi.tolist()[0][5:9]) == 1)[0]
		where = np.where( np.array(xi[5:9]) == 1)[0]

		new_col.append(where[0]+1 if len(where) else 0)

	# Replace columns 5,6,7,8 with combined version
	X = np.insert(np.delete(X, np.s_[5:9], axis=1), 5, new_col,  1)
	return X

# Creates the flattened 35-feature vectors (plus class label)
def create_instances(X):
	new_X = []
	i = 0
	while (i < len(X)-7):
		if ( (X.item(i+6,0) - X.item(i,0)) == 6 ):
			new_X.append((X[i:i+7][:,1:6]).flatten().tolist()[0] + [X.item(i+7, X.shape[1]-1)])
			# i = i + 6
		i += 1
	return new_X

# Returns joined instances of subjects provided in tuple
def get_data(subjects, dir='General_Population'):
	if isinstance(subjects, (int, long)):
		S = read_in_data(dir, subjects)
		S = join_time_cols(S)
		S = create_instances(S)
		return S

	else:
		X = []
		for i in subjects:
			S = read_in_data(dir,i)
			S = join_time_cols(S)
			S = create_instances(S)
			X += S
		return X

# Read in a sample from test data 
def read_in_sample(sample, dir='Sample_Test_Data'):
	test_file = open(dir+'/'+'sampleinstance_'+str(sample)+'.csv')
	T = np.genfromtxt( test_file , delimiter=',')
	test_file.close()
	return T

# Get samples
def get_samples(samples, dir='Sample_Test_Data'):
	if isinstance(samples, (int, long)):
		S = read_in_sample(samples, dir)
		S = join_time_cols(S)
		S = np.delete(S, 0, 1)
		S = S.flatten().tolist()
		return S
	else:
		X = []
		for i in samples:
			S = read_in_sample(i, dir)
			S = join_time_cols(S)
			S = np.delete(S, 0, 1)
			S = S.flatten().tolist()
			X.append(S)
		return X

# Read in and format 



