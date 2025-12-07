import pandas as pd
import numpy as np
import data_prep as dp

num_of_layers = 3
num_of_features = 64
inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
inputs = inputs.drop_duplicates()

np.random.seed(0)

weights = []
weights2 = []
for i in range (0, num_of_layers):
    weights.append(np.random.uniform(-1, 1, num_of_features))
    weights2.append(np.random.uniform(-1, 1, 3))

biases = np.random.randint(-2, 2, 3) # -2, 1, -1
biases2 = np.random.randint(-2, 2, 3) # 1, -1, -1

weights = np.array(weights)
weights2 = np.array(weights2)

layer1_outputs = np.dot(inputs, weights.T) + biases
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print(weights)

#print(layer1_outputs.shape) #205, 3
#print(layer1_outputs.ndim) #2



