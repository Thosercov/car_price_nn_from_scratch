import pandas as pd
import numpy as np
import data_prep as dp
import constants as c

inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
inputs = inputs.drop_duplicates()

np.random.seed(0)

weights = []
weights2 = []
for i in range (0, c.NUM_OF_NEURONS_L1):
    weights.append(np.random.uniform(-1, 1, c.NUM_OF_FEATURES))
    weights2.append(np.random.uniform(-1, 1, c.NUM_OF_NEURONS_L1))

biases = np.random.randint(-2, 2, c.NUM_OF_NEURONS_L1) # -2, 1, -1
biases2 = np.random.randint(-2, 2, c.NUM_OF_NEURONS_L2) # 1, -1, -1

weights = np.array(weights)
weights2 = np.array(weights2)

layer1_outputs = np.dot(inputs, weights.T) + biases
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print(weights)

#print(layer1_outputs.shape) #205, 3
#print(layer1_outputs.ndim) #2



