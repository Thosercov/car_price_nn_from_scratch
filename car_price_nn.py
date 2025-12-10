import pandas as pd
import numpy as np
import data_prep as dp
import constants as c
from layer import Layer

inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
inputs = inputs.drop_duplicates()

np.random.seed(0)

layer1 = Layer(c.n_inputs, c.n_neurons_l1)

layer1.forward(inputs)

print(layer1.output)




