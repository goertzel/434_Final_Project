import numpy as np
import random

np.set_printoptions(precision=3, suppress=True)

# Reads in a subject's data
def read_in_data(dir, subject):
	data_file = open(dir+'/'+'Subject_'+str(subject)+'.csv')
	X = np.genfromtxt( data_file , delimiter=',')
	data_file.close()
	ind_file = open(dir+'/'+'list_'+str(subject)+'.csv')
	I = np.asmatrix(np.genfromtxt( ind_file , delimiter=',')).T
	ind_file.close()
	 # Remove Timestamp and Prepend Indices 
	X = np.append(I, np.delete(X, 0, 1), 1 )
	return X

# Merges morning, afternoon, and evening into a single feature
def join_time_cols(X, type='training'):
	new_col = []
	for xi in X:
		if (type == 'training'):
			where = np.where( np.array(xi.tolist()[0][5:9]) == 1)[0]
		elif (type == 'samples'):
			where = np.where( np.array(xi[5:9]) == 1)[0]
		elif (type == 'testing'):
			where = np.where( np.array(xi.tolist()[0][4:8]) == 1)[0]

		new_col.append(np.average(where)+1 if len(where) else 0)

	# Replace columns 5,6,7,8 with combined version
	X = np.insert(np.delete(X, np.s_[5:9], axis=1), 5, new_col,  1)
	return X

# Creates the flattened 35-feature vectors (plus class label)
def create_instances(X):
	new_X = []
	ground_truth = []
	i = 0
	while (i < len(X)-7):
		if ( (X.item(i+6,0) - X.item(i,0)) == 6 ):
			new_X.append((X[i:i+7][:,1:6]).flatten().tolist()[0] + [X.item(i+7, X.shape[1]-1)])
			ground_truth.append(X.item(i+7, X.shape[1]-1))
			# i = i + 6
		i += 1

	f = open("training_truth.csv", 'w')
	for g in ground_truth:
		f.write(str(g)+'\n')
	f.close()
	return new_X

# Returns joined instances of subjects provided in tuple
def get_data(subjects, dir='General_Population'):
	X = []
	for i in subjects:
		S = read_in_data(dir,i)
		S = join_time_cols(S, type="training")
		S = create_instances(S)
		X += S
	return X

# Read in a sample from test data 
def read_in_test(sample, dir='Sample_Test_Data', test=False):
	if test:
		file = open(dir+'/'+sample+'_test_instances.csv')
	else:
		file = open(dir+'/'+'sampleinstance_'+str(sample)+'.csv')
	T = np.genfromtxt( file , delimiter=',')
	file.close()
	return T

# Get samples
def get_samples(samples, dir='Sample_Test_Data'):
	X = []
	for i in samples:
		S = read_in_test(i, dir, test=False)
		S = join_time_cols(S, type="samples")
		S = np.delete(S, 0, 1).flatten().tolist()
		X.append(S)
	return X

# Get a subsample of the data
def get_subsample(X, neg_count = 900, pos_count = 100):
	negatives = np.matrix([row for row in X if row[-1] == 0])
	positives = np.matrix([row for row in X if row[-1] == 1])
	
	neg_row_indices = np.random.choice([i for i in xrange(negatives.shape[0])], neg_count)
	pos_row_indices = np.random.choice([i for i in xrange(positives.shape[0])], pos_count)
	
	sampled_negatives = np.vstack([negatives[index] for index in neg_row_indices])
	sampled_positives = np.vstack([positives[index] for index in pos_row_indices])
	
	subsample = np.vstack([sampled_negatives, sampled_positives])
	np.random.shuffle(subsample)
	
	return subsample
	
# Read in and format test data
def get_testing_data(data):
	T = read_in_test(data, dir='Final_Test_Data', test=True)
	Tests = []
	for i in xrange(T.shape[0]):
		tmp = np.asmatrix([ T[i][c*7:c*7+7] for c in xrange(9) ]).T
		tmp = np.delete(tmp, 0, 1)
		tmp = join_time_cols(tmp, type='testing')
		tmp = np.delete(tmp, 4, 1)
		Tests.append(tmp)
	return Tests


