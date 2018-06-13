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

			# NEED DIFFERENT SUBSETS OF FEATURES
			self.feature_subset =  [ np.random.choice([i for i in xrange(len(X[0])-1)], numfeatures) for _ in xrange(count) ]

			for fs in self.feature_subset:
				print fs
			# self.feature_subset =  np.random.choice([i for i in xrange(len(X[0])-1)], numfeatures)
			# modX = (np.asmatrix(X)[:,self.feature_subset]).tolist()
			# self.trees = [ DecisionTree( get_subsample(modX,neg,pos).tolist() , depth) for _ in xrange(count) ]
			self.trees = []
			for fs in self.feature_subset:
				modX = (np.asmatrix(X)[:,fs]).tolist()
				self.trees.append(DecisionTree( get_subsample(modX,neg,pos).tolist() , depth))


	def get_choice(self, point):
		print point
		# NEED DIFFERENT SUBSETS OF FEATURES

		if (self.ssf == False):
			votes = [ (t.get_choice(point)[1]+1)/2 for t in self.trees ]
		else:
			votes = []
			for i in xrange(len(self.trees)):
				mod_point = [ point[x] for x in self.feature_subset[i] ]
				votes.append( (self.trees[i].get_choice(mod_point)[1]+1)/2 )


		d = {}
		for v in votes:
		    d[v] = d.get(v, 0) + 1
		count, pred = max([(j,i) for i,j in d.items()])

		return (float(count)/float(len(votes)), pred )


	def get_predictions(self, points):
		# if (self.ssf == False):
		return np.matrix([self.get_choice(p) for p in points])
		# else:
		# 	modP = (np.asmatrix(points)[:,self.feature_subset]).tolist()
		# 	return np.matrix([self.get_choice(p) for p in modP])