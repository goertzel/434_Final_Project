def calculate_w_BGD(X,Y,w,mu,Lambda):
   num_features = len(X[0])
   num_points = len(X)
   nabla = np.zeros( (num_features, 1) )
   for i in xrange(0, num_points):
      X_i = np.transpose(np.array([X[i]]))
      w_T_X_i = np.matmul(np.transpose(w),X_i)[0][0]
      if w_T_X_i < -700: #for underflows
	 y_i_hat = 0.0
      else:
	 y_i_hat = 1.0/(1.0+math.exp(-w_T_X_i))
      nabla = nabla + (y_i_hat-Y[i][0])*X_i
   w = w - mu*(nabla + Lambda*w)
   nabla_norm = np.linalg.norm(nabla)
   return (w, nabla_norm)

def load_X_and_Y(training_data):
   (rows, features) = training_data.shape
   X = np.zeros( (rows,features) )
   Y = np.zeros( (rows,1) )
   for i in xrange(0, rows):
      for x in xrange(0,features-1):
	 X[i][x] = training_data[i][x]
      Y[i][0] = training_data[i][-1]
      X[i][features-1] = 1
   return (X,Y)

class LogisticRegression():
   def __init__(self,training_data,MU = 0.001,Lambda = 128.0):
      (self.X,self.Y) = load_X_and_Y(training_data)
      self.MU = MU
      self.Lambda = Lambda
   def run(self, max_iterations = 250):
      w = np.zeros( (self.X.shape[1], 1) )
      count = 0
      (w, nabla_norm) = calculate_w_BGD(self.X,self.Y,w,self.MU,self.Lambda)
      while nabla_norm > 0.5 and count < 250:
	 count = count + 1
	 (w, nabla_norm) = calculate_w_BGD(self.X,self.Y,w,self.MU,self.Lambda)
      self.w = w
   def get_choice(self, point):
      X_i = np.transpose(point)
      w_T_X_i = np.matmul(np.transpose(self.w),X_i)[0][0]
      if w_T_X_i < -700:
	 P = 0
      else:
	 P = 1.0/(1.0+math.exp(-w_T_X_i))
      guess = 1 if P > 0.5 else 0
      return (guess, P)

