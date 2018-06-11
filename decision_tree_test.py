from decision_tree import DecisionTree

training_data = [[3,2,3,1],[0,3,3,1],[1,4,1,0],[1,4,6,1],[-0.5,3,2,0],[-5.0,1,4,0]]

tree = DecisionTree(training_data,2)
tree.print_tree()
test_data = [[5,1,4,1],[6,2,1,0],[2,3,2,0]]
for point in test_data:
   print tree.get_choice(point)
