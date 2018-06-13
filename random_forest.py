import numpy as np
import random
from decision_tree import DecisionTree
from format_data   import get_subsample


class RandomForest():
	def __init__(self, X, count, depth, neg, pos, ssf=False, numfeatures=35):
		self.ssf = ssf
		if (self.ssf == False):
			self.trees = [ DecisionTree( get_subsample(X,neg,pos).tolist() , depth) for _ in xrange(count) ]
		else:
			self.feature_subset =  [ np.random.choice([i for i in xrange(len(X[0])-1)], numfeatures).tolist()+[35] for _ in xrange(count) ]
			self.trees = [DecisionTree( np.asarray(get_subsample([[point[i] for i in fs] for point in X],neg,pos)) , depth) for fs in self.feature_subset ]

	def get_choice(self, point):
		if (self.ssf == False):
			votes = [ (t.get_choice(point)[1]+1)/2 for t in self.trees ]
		else:
			votes = [ (self.trees[i].get_choice([ point[x] for x 
				in self.feature_subset[i][:-1] ])[1]+1)/2 for i in xrange(len(self.trees)) ]

		d = {}
		for v in votes:
		    d[v] = d.get(v, 0) + 1
		count, pred = max([(j,i) for i,j in d.items()])

		return (float(count)/float(len(votes)), pred )

	def get_predictions(self, points):
		return np.matrix([self.get_choice(p) for p in points])
