import numpy as np
import heapq

class KNN():
   def __init__(self,training_data,normalize=True):
      self.data = training_data
      self.features = len(self.data[0])-1
      if normalize is True:
	 mins = self.data[0][:-1]
	 maxs = self.data[0][:-1]
	 for point in self.data[1:]:
	    mins = [min(mins[i],point[i]) for i in xrange(0,self.features)]
	    maxs = [max(maxs[i],point[i]) for i in xrange(0,self.features)]
	 self.mins = mins
	 self.maxs = maxs
	 
	 for point in self.data:
	    for i in xrange(0,self.features):
	       point[i] = (point[i]-mins[i])/(maxs[i]-mins[i])
      print "initialized KNN for given data set"
   def normalize(self,point):
      temp_point = point
      for i in xrange(0,len(self.mins)):
	 temp_point[i] = (temp_point[i]-self.mins[i])/(self.maxs[i]-self.mins[i])
      return temp_point
   
   def get_knn(self,un_normalized_point,k):
      point = self.normalize(un_normalized_point)
      closest_k = heapq.nsmallest(k,[[np.linalg.norm([data_point[i]-point[i] for i in xrange(0,self.features)]),data_point[-1]] for data_point in self.data])
      total_positives = sum([points[1] for points in closest_k])
      if total_positives > k/2:
	 return [total_positives/k,1]
      else:
	 return [(1-total_positives)/k,0]

   def get_predictions(self,points,k=3):
      print "getting KNN predictions, k =",k
      predictions = []
      count = 0
      for point in points:
	 predictions.append(self.get_knn(point,k))
	 count += 1
	 if count%100 == 0:
	    print "processed",count,"points"
      return np.matrix(predictions)
      
