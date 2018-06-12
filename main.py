import perceptron

def main():
	
	
# Expects predictions as a matrix of form:
# Rows x 1
def write_predictions(predictions):
	f = open("pred.csv", 'w')
	map(lambda p: f.write(str(p)+'\n'), [p[0] for p in predictions])
	f.close()

if __name__ == "__main__":
	main()