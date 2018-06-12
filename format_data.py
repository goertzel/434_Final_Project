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
	tmp_X = []
	for xi in X:
		where = np.where( np.array(xi.tolist()[0][5:9]) == 1)[0]
		tmp_X.append(where[0]+1 if len(where) else 0)

	# Replace columns 5,6,7,8 with combined version
	X = np.insert(np.delete(X, np.s_[5:9], axis=1), 5, tmp_X,  1)
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

	return np.matrix(new_X)

# Returns joined instances of subjects provided in tuple
def get_data(subjects, filename='General_Population'):
	return np.vstack([
			create_instances(
			join_time_cols  (
			read_in_data    (filename, i)))
			for i in subjects])


#######################################
# def get_data(filename='General_Population',ind=1):
# 	X = read_in_data(filename,ind)

# 	X = join_time_cols(X)

# 	X = create_instances(X)

# 	print X.shape

# 	return X



# subjects = get_subjects((1,2,4,6,7,9))	

