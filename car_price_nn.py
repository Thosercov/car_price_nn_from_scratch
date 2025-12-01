import pandas as pd
import numpy as np
import data_prep as dp

num_of_layers = 3
inputs = dp.get_inputs()

np.random.seed(0)

weights = []
for i in range (0, num_of_layers):
    weights.append(np.random.uniform(-1, 1, 64))

biases = np.random.randint(-2, 2, 3) # -2, 1, -1

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer_outputs)
print(layer_outputs.shape) #205, 3
print(layer_outputs.ndim) #2

