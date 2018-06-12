from format_data   import get_data, get_samples, get_testing_data
from perceptron    import Perceptron
from decision_tree import DecisionTree
import logistic_regression
import numpy as np
import os

def main():
	print "CS 434 Final"
	X = get_data(1)
	tree = DecisionTree(X,6)
	tree.print_tree()
	
	# perceptron = Perceptron(np.matrix(X))
	# write_predictions(perceptron.training_predictions)
	
	# lr = logistic_regression.LogisticRegression(X)
	# lr.run()

	# evaluate('pred', 'Sample_Test_Data/groundtruth')
	# predictions = []
	samples = get_samples((1,2,3,4,5))
	# print samples
	predictions = tree.get_predictions(samples)
	# for point in samples:
	# 	predictions.append(tree.get_choice(point))
	write_predictions(predictions)

	evaluate(gold='/Sample_Test_Data/groundtruth')
	
	
	
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
 
