import pandas as pd
import numpy as np
import data_prep as dp
import constants as c

inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
inputs = inputs.drop_duplicates()

np.random.seed(0)

weights = 0.01 * np.random.randn(inputs.columns.size, c.NUM_OF_NEURONS_L1)
weights2 = 0.01 * np.random.randn(c.NUM_OF_NEURONS_L1, c.NUM_OF_NEURONS_L2)

biases = np.zeros((1, c.NUM_OF_NEURONS_L1))
biases2 = np.zeros((1, c.NUM_OF_NEURONS_L2))

weights = np.array(weights)
weights2 = np.array(weights2)

layer1_outputs = np.dot(inputs, weights) + biases
layer2_outputs = np.dot(layer1_outputs, weights2) + biases2




