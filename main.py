from format_data   import get_data
from perceptron    import Perceptron
from decision_tree import DecisionTree
import logistic_regression
import numpy as np

def main():
	print "CS 434 Final"
	X = get_data(1)
	tree = DecisionTree(X,6)
	tree.print_tree()
	# perceptron = Perceptron(np.matrix(X))
	# perceptron.predict()
	
	
# Expects predictions as a matrix of form:
# Rows x 1
def write_predictions(predictions):
	f = open("pred.csv", 'w')
	map(lambda p: f.write(str(p)+'\n'), [p[0] for p in predictions])
	f.close()
	
if __name__ == "__main__":
	main()
