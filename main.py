from format_data   import get_data
from perceptron    import Perceptron
from decision_tree import DecisionTree
import logistic_regression
import numpy as np
import os

def main():
	print "CS 434 Final"
	X = get_data((1,4,6,9))
	# tree = DecisionTree(X,6)
	# tree.print_tree()
	lr = logistic_regression.LogisticRegression(X)
	lr.run()
	# perceptron = Perceptron(np.matrix(X))
	# perceptron.predict()

	# evaluate('pred', 'Sample_Test_Data/groundtruth')
	
	
# Expects predictions as a matrix of form:
# Rows x 1
def write_predictions(predictions):
	f = open("pred.csv", 'w')
	map(lambda p: f.write(str(p)+'\n'), [p[0] for p in predictions])
	f.close()


def evaluate(pred='pred', gold='gold'):
	os.system('python eval_simple.py -p '+pred+'.csv -g '+gold+'.csv')


if __name__ == "__main__":
	main()
 
