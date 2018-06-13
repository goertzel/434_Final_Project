from format_data   import get_data, get_samples, get_testing_data, get_subsample
from perceptron    import Perceptron
from decision_tree import DecisionTree
from knn	   import KNN
import logistic_regression
import numpy as np
import os

def main():
	print "CS 434 Final"
	
	# Example of reading training data
	X = get_data((1,4))
	# print np.asmatrix(X[198][0:35]).reshape(7,5)

	# Example of reading samples
	# samples = get_samples((5,))
	# print np.asmatrix(samples[0][0:35]).reshape(7,5)

	# Example of reading testing data
	# T = get_testing_data('general')
	# print np.asmatrix(T[3][0:35]).reshape(7,5)

# -------

	tree = DecisionTree(X,6)
	tree.print_tree()

	my_knn = KNN(X)
	predictions = my_knn.get_predictions(X)

	X = get_data((1,4,6,9))
	training = get_subsample(X, 1000, 9000)
	validation = get_subsample(X, 100, 900)

	# perceptron = Perceptron(sub_x)
	# write_predictions(perceptron.training_predictions)
	# print len([p[0,1] for p in perceptron.training_predictions if p[0,1] > 0])

	# lr = logistic_regression.LogisticRegression(X)
	# lr.run()

	samples = get_samples((1,2,3,4,5))
	predictions = tree.get_predictions(samples)

	# samples = get_samples((1,2,3,4,5))
	# predictions = tree.get_predictions(samples)

	# evaluate(gold='/Sample_Test_Data/groundtruth')
	
	
	
# Expects predictions as a matrix of form:
# Rows x 2
def write_predictions(predictions):
	f = open("pred.csv", 'w')
	map(lambda p: f.write(str(p[0]) + ',' +	str(p[1]) +'\n'), [(p[0,0], int(p[0,1])) for p in predictions])
	f.close()

def evaluate(pred='pred', gold='gold'):
	os.system('python eval_simple.py -p '+pred+'.csv -g '+gold+'.csv')

if __name__ == "__main__":
	main()
 
