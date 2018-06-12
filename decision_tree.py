#Decision_tree
#Expects training_data as a two-dimensional array, where each row is a data point, the last point of which is a -1 or 1
#tree = DecisionTree(training_data,depth)
#for point in test_data: #necessarily of same dimensionality as a point in the training_data
#	choice = tree.get_choice(point)

import math
import numpy as np

class DecisionTree():
   def __init__(self,training_data,depth):
      self.leaf = False
      if depth != 0:
	 base_entropy = calculate_entropy(training_data)
	 best_gain = -1
	 best_ind = -1
	 best_thresh = -1
	 for i in xrange(0, len(training_data[0])-1):
	    sorted_set = sorted(training_data, key = lambda point: point[i])
	    for p in xrange(0, len(sorted_set)-1):
	       if sorted_set[p+1][-1] != sorted_set[p][-1]:
		  split_ind = p
		  while split_ind < len(sorted_set)-1 and sorted_set[split_ind+1][i] == sorted_set[split_ind][i]:
			split_ind = split_ind + 1
		  if split_ind < len(sorted_set)-1:
		  	left_split = sorted_set[:split_ind+1]
			right_split = sorted_set[split_ind+1:]
			info_gain = base_entropy - float(split_ind)/float(len(sorted_set))*calculate_entropy(left_split) - float(len(sorted_set) - split_ind)/float(len(sorted_set))*calculate_entropy(right_split)
			if info_gain > best_gain:
				best_gain = info_gain
				best_ind = i
				best_thresh = (sorted_set[split_ind][i]+sorted_set[split_ind+1][i])/2.0
	 if best_ind == -1:
		self.leaf = True
	 else:
		self.feature_ind = best_ind
		self.thresh = best_thresh
		left_list = [point for point in training_data if point[best_ind] <= best_thresh]
		right_list = [point for point in training_data if point[best_ind] > best_thresh]
		self.left = DecisionTree(left_list,depth-1)
		self.right = DecisionTree(right_list,depth-1)
      else:
      	self.leaf = True
      if self.leaf == True:
      	total = sum([1 if point[-1] == 1 else -1 for point in training_data])
	if total == 0:
		self.choice = 0
	else:
		self.choice = 1 if total > 0 else -1
   def get_choice(self,point):
      if self.leaf == True:
	if (self.choice == 0):
	   return 1 if rand()%2 == 0 else -1
      	return self.choice
      else:
      	if point[self.feature_ind] <= self.thresh:
	  return self.left.get_choice(point)
	else:
	  return self.right.get_choice(point)
   def print_tree(self,depth=0):
      if self.leaf == True:
	 print "\t"*depth,self.choice
      else:
	 print "\t"*depth,"point["+str(self.feature_ind)+"] <=",self.thresh
	 self.left.print_tree(depth+1)
	 print "\t"*depth,"point["+str(self.feature_ind)+"] >",self.thresh
	 self.right.print_tree(depth+1)

def calculate_entropy(data):
   positive_count = 0.0
   negative_count = 0.0
   for i in xrange(0,len(data)):
      item = data[i]
      print item.shape
      if item[-1][0] == 1:
	 positive_count = positive_count + 1.0
      else:
	 negative_count = negative_count + 1.0
   pos = positive_count/(positive_count+negative_count)
   neg = negative_count/(positive_count+negative_count)
   return -(0 if pos == 0 else pos*math.log(pos,2))-(0 if neg == 0 else neg*math.log(neg,2))
